"""
Parses the raw LLM JSON response into a structured JudgeVerdict.
"""
import json
import time
from models.judge_verdict import JudgeVerdict, SeverityTier, ModalityScore


TIER_MAP = {
    "NONE":     SeverityTier.NONE,
    "MILD":     SeverityTier.MILD,
    "MODERATE": SeverityTier.MODERATE,
    "SEVERE":   SeverityTier.SEVERE,
}


def parse_verdict(raw: str, context_tags: list = None) -> JudgeVerdict:
    data = json.loads(raw)

    tier_str = data.get("severity_tier", "NONE").upper()
    tier = TIER_MAP.get(tier_str, SeverityTier.NONE)

    modality_scores = {
        "text": ModalityScore(
            modality="text",
            score=float(data.get("text_score", 0.0)),
            rationale=data.get("text_rationale", ""),
            signals_used=["speed_change", "steering_correction", "brake_reaction_time"],
        ),
        "image": ModalityScore(
            modality="image",
            score=float(data.get("image_score", 0.0)),
            rationale=data.get("image_rationale", ""),
            signals_used=["perclos", "lane_deviation"],
        ),
        "audio": ModalityScore(
            modality="audio",
            score=float(data.get("audio_score", 0.0)),
            rationale=data.get("audio_rationale", ""),
            signals_used=["yawn_detected", "verbal_fatigue"],
        ),
    }

    return JudgeVerdict(
        composite_score=float(data.get("composite_score", 0.0)),
        severity_tier=tier,
        modality_scores=modality_scores,
        reasoning=data.get("reasoning", ""),
        timestamp=time.time(),
        context_tags=context_tags or [],
    )
