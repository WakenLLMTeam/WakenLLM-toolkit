"""
DriverProfileClassifier: classifies the current driving session as
LONG_HAUL or COMMUTER and returns the matching ThresholdConfig.

Classification rules (evaluated in priority order):
  1. Driving duration >= 90 min              → LONG_HAUL
  2. Current hour in [22, 23, 0, 1, 2, 3, 4, 5] → LONG_HAUL (night shift)
  3. Road type is HIGHWAY                    → LONG_HAUL
  4. All other cases                         → COMMUTER

The classifier is stateless and cheap — called on every aggregator cycle.
"""
import datetime
from typing import Optional

from config import ThresholdConfig, LONG_HAUL_THRESHOLDS, COMMUTER_THRESHOLDS
from models.fatigue_context import DriverProfile, RoadType

_NIGHT_HOURS = {22, 23, 0, 1, 2, 3, 4, 5}
_LONG_HAUL_DURATION_MIN = 90.0


def classify_profile(
    driving_duration_min: float,
    road_type: RoadType = RoadType.UNKNOWN,
    hour: Optional[int] = None,
) -> DriverProfile:
    """Return the inferred DriverProfile for the current session."""
    if hour is None:
        hour = datetime.datetime.now().hour

    if driving_duration_min >= _LONG_HAUL_DURATION_MIN:
        return DriverProfile.LONG_HAUL
    if hour in _NIGHT_HOURS:
        return DriverProfile.LONG_HAUL
    if road_type == RoadType.HIGHWAY:
        return DriverProfile.LONG_HAUL
    return DriverProfile.COMMUTER


def get_thresholds(profile: DriverProfile) -> ThresholdConfig:
    """Return the ThresholdConfig for the given profile."""
    if profile == DriverProfile.LONG_HAUL:
        return LONG_HAUL_THRESHOLDS
    return COMMUTER_THRESHOLDS
