"""Tests for OrchestratorAgent pyramid dispatch logic."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from config import AppConfig, CooldownConfig
from orchestrator.orchestrator_agent import OrchestratorAgent
from orchestrator.cooldown_tracker import CooldownTracker
from models.judge_verdict import SeverityTier
from models.action_result import ActionResult, ActionStatus


def _make_action_mock(name: str) -> MagicMock:
    m = MagicMock()
    m.name = name
    m.execute = AsyncMock(return_value=ActionResult(name, ActionStatus.SUCCESS))
    return m


@pytest.fixture
def mocks():
    return {
        "screen":  _make_action_mock("screen"),
        "voice":   _make_action_mock("voice"),
        "video":   _make_action_mock("video"),
        "push":    _make_action_mock("push"),
        "context": _make_action_mock("context"),
    }


@pytest.fixture
def orch(mocks):
    cfg = AppConfig()
    cfg.cooldowns = CooldownConfig(mild_seconds=0, moderate_seconds=0, severe_seconds=0)
    return OrchestratorAgent(
        screen_agent  = mocks["screen"],
        voice_agent   = mocks["voice"],
        video_agent   = mocks["video"],
        push_agent    = mocks["push"],
        context_agent = mocks["context"],
        cfg           = cfg,
    )


class TestOrchestratorPyramid:
    @pytest.mark.asyncio
    async def test_none_tier_fires_no_actions(self, orch, mocks, clear_context):
        from models.judge_verdict import JudgeVerdict
        import time
        verdict = JudgeVerdict(
            composite_score=0.10,
            severity_tier=SeverityTier.NONE,
            modality_scores={},
            reasoning="",
            timestamp=time.time(),
        )
        await orch._dispatch(verdict, clear_context)
        for m in mocks.values():
            m.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_mild_fires_only_screen(self, orch, mocks, mild_verdict, clear_context):
        await orch._dispatch(mild_verdict, clear_context)
        mocks["screen"].execute.assert_called_once()
        mocks["voice"].execute.assert_not_called()
        mocks["video"].execute.assert_not_called()
        mocks["push"].execute.assert_not_called()
        mocks["context"].execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_moderate_fires_screen_and_voice(self, orch, mocks, alert_context):
        from models.judge_verdict import JudgeVerdict
        import time
        verdict = JudgeVerdict(
            composite_score=0.60,
            severity_tier=SeverityTier.MODERATE,
            modality_scores={},
            reasoning="",
            timestamp=time.time(),
        )
        await orch._dispatch(verdict, alert_context)
        mocks["screen"].execute.assert_called_once()
        mocks["voice"].execute.assert_called_once()
        mocks["video"].execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_severe_fires_all_five_actions(self, orch, mocks, severe_verdict, alert_context):
        await orch._dispatch(severe_verdict, alert_context)
        for name, m in mocks.items():
            assert m.execute.called, f"Expected {name} to be called for SEVERE tier"

    @pytest.mark.asyncio
    async def test_cooldown_blocks_repeat_dispatch(self, mocks, alert_context, mild_verdict):
        """With real cooldowns, second dispatch within window should be blocked."""
        cfg = AppConfig()
        cfg.cooldowns = CooldownConfig(mild_seconds=60, moderate_seconds=60, severe_seconds=120)
        orch = OrchestratorAgent(
            screen_agent  = mocks["screen"],
            voice_agent   = mocks["voice"],
            video_agent   = mocks["video"],
            push_agent    = mocks["push"],
            context_agent = mocks["context"],
            cfg           = cfg,
        )
        await orch._dispatch(mild_verdict, alert_context)  # fires
        await orch._dispatch(mild_verdict, alert_context)  # blocked by cooldown
        # screen should have been called exactly once
        assert mocks["screen"].execute.call_count == 1

    @pytest.mark.asyncio
    async def test_action_exception_does_not_crash_orchestrator(self, orch, mocks, mild_verdict, clear_context):
        """A failing action agent should be logged but not propagate."""
        mocks["screen"].execute = AsyncMock(side_effect=RuntimeError("HMI offline"))
        # Should not raise
        await orch._dispatch(mild_verdict, clear_context)

    @pytest.mark.asyncio
    async def test_run_consumes_from_queue(self, orch, mocks, mild_verdict, clear_context):
        verdict_queue: asyncio.Queue = asyncio.Queue()
        await verdict_queue.put((mild_verdict, clear_context))

        task = asyncio.create_task(orch.run(verdict_queue))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        mocks["screen"].execute.assert_called_once()
