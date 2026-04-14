"""Tests for FatigueJudgeAgent and response parsing."""
import json
import pytest
from judge.judge_agent import FatigueJudgeAgent
from judge.response_parser import parse_verdict
from judge.prompt_template import build_user_prompt, SYSTEM_PROMPT
from llm.mock_llm_client import MockLLMClient
from models.judge_verdict import SeverityTier


class TestResponseParser:
    def _make_raw(self, composite, tier, text=0.3, image=0.5, audio=0.2):
        return json.dumps({
            "text_score":      text,
            "image_score":     image,
            "audio_score":     audio,
            "composite_score": composite,
            "severity_tier":   tier,
            "text_rationale":  "test text rationale",
            "image_rationale": "test image rationale",
            "audio_rationale": "test audio rationale",
            "reasoning":       "test reasoning",
        })

    def test_parse_none_tier(self):
        v = parse_verdict(self._make_raw(0.10, "NONE"))
        assert v.severity_tier == SeverityTier.NONE
        assert v.composite_score == pytest.approx(0.10)

    def test_parse_mild_tier(self):
        v = parse_verdict(self._make_raw(0.40, "MILD"))
        assert v.severity_tier == SeverityTier.MILD

    def test_parse_moderate_tier(self):
        v = parse_verdict(self._make_raw(0.60, "MODERATE"))
        assert v.severity_tier == SeverityTier.MODERATE

    def test_parse_severe_tier(self):
        v = parse_verdict(self._make_raw(0.80, "SEVERE"))
        assert v.severity_tier == SeverityTier.SEVERE

    def test_all_three_modality_scores_present(self):
        v = parse_verdict(self._make_raw(0.55, "MODERATE"))
        assert set(v.modality_scores.keys()) == {"text", "image", "audio"}

    def test_modality_score_values(self):
        v = parse_verdict(self._make_raw(0.55, "MODERATE", text=0.3, image=0.7, audio=0.2))
        assert v.modality_scores["image"].score == pytest.approx(0.7)
        assert v.modality_scores["text"].score  == pytest.approx(0.3)

    def test_context_tags_are_attached(self):
        v = parse_verdict(self._make_raw(0.4, "MILD"), context_tags=["highway", "night"])
        assert "highway" in v.context_tags

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            parse_verdict("not valid json")

    def test_unknown_tier_defaults_to_none(self):
        v = parse_verdict(self._make_raw(0.5, "UNKNOWN_TIER"))
        assert v.severity_tier == SeverityTier.NONE


class TestPromptTemplate:
    def test_prompt_contains_perclos(self, alert_context):
        prompt = build_user_prompt(alert_context)
        assert "0.350" in prompt or "0.35" in prompt

    def test_prompt_contains_road_type(self, alert_context):
        prompt = build_user_prompt(alert_context)
        assert "HIGHWAY" in prompt

    def test_prompt_contains_rest_stop(self, alert_context):
        prompt = build_user_prompt(alert_context)
        assert "G2 高速服务区" in prompt

    def test_prompt_contains_verbal_fatigue(self, alert_context):
        prompt = build_user_prompt(alert_context)
        assert "YES" in prompt

    def test_system_prompt_includes_severity_tiers(self):
        assert "NONE" in SYSTEM_PROMPT
        assert "MILD" in SYSTEM_PROMPT
        assert "MODERATE" in SYSTEM_PROMPT
        assert "SEVERE" in SYSTEM_PROMPT

    def test_prompt_shows_face_not_available_when_none(self, clear_context):
        prompt = build_user_prompt(clear_context)
        assert "NOT AVAILABLE" in prompt


class TestFatigueJudgeAgent:
    @pytest.mark.asyncio
    async def test_evaluate_returns_verdict(self, alert_context):
        agent = FatigueJudgeAgent(MockLLMClient())
        verdict = await agent.evaluate(alert_context)
        assert verdict.composite_score >= 0.0
        assert verdict.severity_tier in SeverityTier

    @pytest.mark.asyncio
    async def test_high_perclos_produces_higher_score(self, alert_context, clear_context):
        """Higher PERCLOS should yield a higher composite score."""
        agent = FatigueJudgeAgent(MockLLMClient())
        v_alert = await agent.evaluate(alert_context)
        v_clear = await agent.evaluate(clear_context)
        assert v_alert.composite_score > v_clear.composite_score

    @pytest.mark.asyncio
    async def test_context_tag_includes_road_type(self, alert_context):
        agent = FatigueJudgeAgent(MockLLMClient())
        verdict = await agent.evaluate(alert_context)
        assert "highway" in verdict.context_tags

    @pytest.mark.asyncio
    async def test_mock_client_zero_perclos_gives_none_tier(self, clear_context):
        agent = FatigueJudgeAgent(MockLLMClient())
        verdict = await agent.evaluate(clear_context)
        assert verdict.severity_tier == SeverityTier.NONE
