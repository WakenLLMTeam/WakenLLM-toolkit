from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from models.driver_memory import DriverMemory


class RoadType(str, Enum):
    CITY    = "city"
    HIGHWAY = "highway"
    UNKNOWN = "unknown"


class DriverProfile(str, Enum):
    LONG_HAUL = "long_haul"   # truck/bus driver; high-stakes; intervene early
    COMMUTER  = "commuter"    # daily driver; low-stakes; prefer gentle reminders
    UNKNOWN   = "unknown"


@dataclass
class TextSignals:
    speed_change_variance:         float = 0.0   # low variance = monotonous = fatigued
    steering_correction_magnitude: float = 0.0   # high = sudden large correction = fatigued
    brake_reaction_time_delta_ms:  float = 0.0   # positive delta = slower response = fatigued


@dataclass
class ImageSignals:
    perclos_score:        float = 0.0   # 0-1; clinical threshold >0.15
    lane_deviation_count: int   = 0
    face_frame_b64:       Optional[str] = None   # base64 JPEG for LLM vision input


@dataclass
class AudioSignals:
    yawn_count_per_minute:   float = 0.0
    verbal_fatigue_detected: bool  = False
    transcript_snippet:      Optional[str] = None


@dataclass
class FatigueContext:
    text_signals:        TextSignals
    image_signals:       ImageSignals
    audio_signals:       AudioSignals
    window_seconds:      float
    timestamp:           float
    driving_duration_min: float = 0.0  # cumulative minutes driven this session


@dataclass
class MapContext:
    road_type:         RoadType             = RoadType.UNKNOWN
    nearest_rest_km:   Optional[float]      = None
    nearest_coffee_km: Optional[float]      = None
    rest_spot_name:    Optional[str]        = None
    coffee_shop_name:  Optional[str]        = None
    # 0.0 = empty road; 1.0 = heavy traffic.  Used by city pull-over logic.
    traffic_density:   float                = 0.5


@dataclass
class EnrichedFatigueContext:
    fatigue:              FatigueContext
    map:                  MapContext
    time_risk_multiplier: float         = 1.0           # 1.0 normal; 1.3 post-lunch; 1.6 deep night (2-5 AM)
    driver_profile:       DriverProfile = DriverProfile.UNKNOWN
    driver_memory:        Optional["DriverMemory"] = None  # None = no personalisation
