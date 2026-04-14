from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import time


class Modality(str, Enum):
    TEXT  = "text"
    IMAGE = "image"
    AUDIO = "audio"


class SignalType(str, Enum):
    # Text / telemetry signals
    SPEED_CHANGE          = "speed_change"
    STEERING_CORRECTION   = "steering_correction"
    BRAKE_REACTION_TIME   = "brake_reaction_time"
    # Image signals
    PERCLOS               = "perclos"
    LANE_DEVIATION        = "lane_deviation"
    # Audio signals
    YAWN_DETECTED         = "yawn_detected"
    VERBAL_FATIGUE        = "verbal_fatigue"
    # Session signals
    DRIVING_DURATION      = "driving_duration"  # value = minutes driven this session


@dataclass
class SensorEvent:
    modality:    Modality
    signal_type: SignalType
    value:       Any            # float for numeric; bytes for image/audio frames
    timestamp:   float = field(default_factory=time.time)
    metadata:    dict  = field(default_factory=dict)
