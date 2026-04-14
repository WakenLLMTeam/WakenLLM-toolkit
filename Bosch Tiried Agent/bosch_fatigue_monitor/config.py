"""
Central configuration: thresholds, model names, cooldowns.
All tuneable parameters live here — nowhere else.
"""
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ThresholdConfig:
    # --- Text / telemetry ---
    speed_variance_low:              float = 0.5    # below = monotonous driving = fatigue signal
    steering_magnitude_high:         float = 15.0   # degrees; above = sudden correction
    brake_reaction_delta_ms_high:    float = 150.0  # ms slower than baseline

    # --- Image ---
    perclos_threshold:               float = 0.15   # clinical standard
    lane_deviation_per_min_high:     int   = 3

    # --- Audio ---
    yawn_per_min_high:               float = 2.0
    verbal_fatigue_weight:           float = 0.3    # boost to composite score when detected

    # --- Window ---
    observation_window_seconds:      float = 60.0   # rolling window duration

    # --- Driving duration fatigue milestones (minutes) ---
    # Reaching these thresholds triggers a judge evaluation even without
    # other sensor signals.  The Judge then assigns a tier based on all signals
    # combined (including the duration context passed via FatigueContext).
    driving_duration_mild_min:     float = 120.0  # 2 h → mild pre-check
    driving_duration_moderate_min: float = 240.0  # 4 h → moderate-level concern
    driving_duration_severe_min:   float = 360.0  # 6 h → severe-level concern

    # --- Composite scoring weights ---
    text_weight:                     float = 0.30
    image_weight:                    float = 0.50
    audio_weight:                    float = 0.20

    # --- Severity tiers (composite score 0-1) ---
    mild_threshold:                  float = 0.30
    moderate_threshold:              float = 0.55
    severe_threshold:                float = 0.75


@dataclass
class CooldownConfig:
    """Minimum seconds between repeated alerts per tier."""
    mild_seconds:     int = 30
    moderate_seconds: int = 60
    severe_seconds:   int = 120


@dataclass
class BoschAPIConfig:
    """Bosch internal LLM gateway settings."""
    api_key:  str = "4a52b2bf90254d29bfb86919924c6d7d"
    base_url: str = "https://aigc.bosch.com.cn/llmservice/api/v1"


@dataclass
class ModelConfig:
    # Judge: multimodal reasoning (text + face image via vision)
    judge_model:      str = "gpt-5"        # full reasoning + vision
    judge_max_tokens: int = 512

    # Lightweight inference (quick scoring, fallback)
    nano_model:       str = "gpt-4o-mini"  # fast, cheap sub-tasks

    # Audio ASR
    asr_model:        str = "qwen3-asr-flash"

    # Audio TTS
    tts_model:        str = "qwen3-tts-flash"
    tts_voice:        str = "default"


@dataclass
class AppConfig:
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    cooldowns:  CooldownConfig  = field(default_factory=CooldownConfig)
    models:     ModelConfig     = field(default_factory=ModelConfig)
    bosch_api:  BoschAPIConfig  = field(default_factory=BoschAPIConfig)

    # Sensor poll intervals (seconds)
    text_sensor_interval:             float = 1.0
    image_sensor_interval:            float = 0.5
    audio_sensor_interval:            float = 2.0
    driving_duration_sensor_interval: float = 60.0  # publish duration every 60 s

    # Phone push endpoint (mock)
    phone_push_url: str = "http://localhost:8080/push"


# Singleton default config — import this everywhere
config = AppConfig()


# ---------------------------------------------------------------------------
# Per-profile threshold presets
# Picked up by DriverProfileClassifier; override config.thresholds at runtime
# ---------------------------------------------------------------------------

LONG_HAUL_THRESHOLDS = ThresholdConfig(
    # Stricter image/audio thresholds — highway accident costs are severe
    perclos_threshold=0.12,
    lane_deviation_per_min_high=2,
    yawn_per_min_high=1.5,
    # Intervene earlier on composite score
    mild_threshold=0.25,
    moderate_threshold=0.48,
    severe_threshold=0.68,
    # Duration milestones same as default (2h/4h/6h)
    driving_duration_mild_min=120.0,
    driving_duration_moderate_min=240.0,
    driving_duration_severe_min=360.0,
)

COMMUTER_THRESHOLDS = ThresholdConfig(
    # Looser image threshold — reduce false alarms (dry eyes, morning blinks)
    perclos_threshold=0.20,
    lane_deviation_per_min_high=4,
    yawn_per_min_high=2.5,
    # Higher composite bar — don't disrupt commute heart rate
    mild_threshold=0.35,
    moderate_threshold=0.60,
    severe_threshold=0.80,
    # Commuters rarely drive >1h; check earlier but at lower stakes
    driving_duration_mild_min=60.0,
    driving_duration_moderate_min=90.0,
    driving_duration_severe_min=120.0,
)
