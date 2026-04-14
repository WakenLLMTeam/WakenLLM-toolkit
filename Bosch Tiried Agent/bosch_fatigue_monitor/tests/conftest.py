"""Shared pytest fixtures."""
import sys
import os
import time
import pytest

# Make sure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import AppConfig, ThresholdConfig, CooldownConfig
from models.fatigue_context import (
    FatigueContext, EnrichedFatigueContext,
    TextSignals, ImageSignals, AudioSignals, MapContext, RoadType, DriverProfile,
)
from models.driver_memory import DriverMemory, COFFEE_LOVER
from models.judge_verdict import JudgeVerdict, SeverityTier, ModalityScore


@pytest.fixture
def cfg():
    return AppConfig()


def _make_alert_fatigue(duration_min: float = 0.0) -> FatigueContext:
    return FatigueContext(
        text_signals=TextSignals(
            speed_change_variance=0.1,
            steering_correction_magnitude=20.0,
            brake_reaction_time_delta_ms=200.0,
        ),
        image_signals=ImageSignals(
            perclos_score=0.35,
            lane_deviation_count=5,
            face_frame_b64=None,
        ),
        audio_signals=AudioSignals(
            yawn_count_per_minute=3.5,
            verbal_fatigue_detected=True,
            transcript_snippet="我有点累了",
        ),
        window_seconds=60.0,
        timestamp=time.time(),
        driving_duration_min=duration_min,
    )


@pytest.fixture
def alert_context():
    """Highway long-haul context with severe fatigue signals."""
    return EnrichedFatigueContext(
        fatigue=_make_alert_fatigue(duration_min=0.0),
        map=MapContext(
            road_type=RoadType.HIGHWAY,
            nearest_rest_km=12.5,
            nearest_coffee_km=8.0,
            rest_spot_name="G2 高速服务区",
            coffee_shop_name="星巴克",
        ),
        driver_profile=DriverProfile.LONG_HAUL,
    )


@pytest.fixture
def alert_context_long_haul():
    """Long-haul driver with 3h+ continuous driving."""
    return EnrichedFatigueContext(
        fatigue=_make_alert_fatigue(duration_min=200.0),
        map=MapContext(
            road_type=RoadType.HIGHWAY,
            nearest_rest_km=12.5,
            nearest_coffee_km=8.0,
            rest_spot_name="G2 高速服务区",
            coffee_shop_name="星巴克",
        ),
        driver_profile=DriverProfile.LONG_HAUL,
        time_risk_multiplier=1.6,
    )


@pytest.fixture
def alert_context_commuter():
    """City commuter with mild fatigue."""
    return EnrichedFatigueContext(
        fatigue=_make_alert_fatigue(duration_min=25.0),
        map=MapContext(
            road_type=RoadType.CITY,
            nearest_rest_km=5.0,
            nearest_coffee_km=1.2,
            rest_spot_name="停车场",
            coffee_shop_name="星巴克",
        ),
        driver_profile=DriverProfile.COMMUTER,
        time_risk_multiplier=1.0,
    )


@pytest.fixture
def alert_context_city(alert_context):
    """Same fatigue signals but on a city road."""
    alert_context.map.road_type = RoadType.CITY
    alert_context.driver_profile = DriverProfile.COMMUTER
    return alert_context


@pytest.fixture
def alert_context_coffee_lover():
    """Commuter who likes coffee; Starbucks is 1.2 km away (within 2 km limit)."""
    return EnrichedFatigueContext(
        fatigue=_make_alert_fatigue(duration_min=25.0),
        map=MapContext(
            road_type=RoadType.CITY,
            nearest_rest_km=5.0,
            nearest_coffee_km=1.2,
            rest_spot_name="停车场",
            coffee_shop_name="星巴克",
            traffic_density=0.5,
        ),
        driver_profile=DriverProfile.COMMUTER,
        time_risk_multiplier=1.0,
        driver_memory=COFFEE_LOVER,
    )


@pytest.fixture
def alert_context_low_traffic():
    """Commuter on a quiet city road — pull-over is safe."""
    return EnrichedFatigueContext(
        fatigue=_make_alert_fatigue(duration_min=25.0),
        map=MapContext(
            road_type=RoadType.CITY,
            nearest_rest_km=5.0,
            nearest_coffee_km=3.0,   # too far to auto-order
            rest_spot_name="停车场",
            coffee_shop_name="星巴克",
            traffic_density=0.15,    # low traffic → pull-over path
        ),
        driver_profile=DriverProfile.COMMUTER,
        time_risk_multiplier=1.0,
        driver_memory=DriverMemory(likes_coffee=False, ok_to_pull_over_city=True),
    )


@pytest.fixture
def clear_context():
    """EnrichedFatigueContext well below all thresholds."""
    return EnrichedFatigueContext(
        fatigue=FatigueContext(
            text_signals=TextSignals(
                speed_change_variance=8.0,
                steering_correction_magnitude=2.0,
                brake_reaction_time_delta_ms=20.0,
            ),
            image_signals=ImageSignals(
                perclos_score=0.04,
                lane_deviation_count=0,
            ),
            audio_signals=AudioSignals(
                yawn_count_per_minute=0.0,
                verbal_fatigue_detected=False,
            ),
            window_seconds=60.0,
            timestamp=time.time(),
        ),
        map=MapContext(road_type=RoadType.CITY),
    )


@pytest.fixture
def severe_verdict():
    return JudgeVerdict(
        composite_score=0.82,
        severity_tier=SeverityTier.SEVERE,
        modality_scores={
            "text":  ModalityScore("text",  0.70, "Slow braking, big steering correction", []),
            "image": ModalityScore("image", 0.90, "PERCLOS=0.35 far above threshold", []),
            "audio": ModalityScore("audio", 0.60, "Frequent yawns + verbal confirmation", []),
        },
        reasoning="All three modalities indicate severe fatigue.",
        context_tags=["highway"],
    )


@pytest.fixture
def mild_verdict():
    return JudgeVerdict(
        composite_score=0.35,
        severity_tier=SeverityTier.MILD,
        modality_scores={
            "text":  ModalityScore("text",  0.30, "Slightly monotonous speed", []),
            "image": ModalityScore("image", 0.40, "PERCLOS borderline", []),
            "audio": ModalityScore("audio", 0.10, "No audio signals", []),
        },
        reasoning="Mild fatigue based on image signals.",
        context_tags=["city"],
    )
