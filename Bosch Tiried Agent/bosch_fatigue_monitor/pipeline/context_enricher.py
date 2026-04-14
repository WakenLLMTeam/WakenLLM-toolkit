"""
ContextEnricher: injects map/location data and time-risk multiplier
into FatigueContext before it reaches the Judge.
Action agents never call maps directly.
"""
import datetime
from abc import ABC, abstractmethod
from typing import Optional
from models.fatigue_context import FatigueContext, EnrichedFatigueContext, MapContext, RoadType
from pipeline.driver_profile import classify_profile


class MapClientInterface(ABC):
    @abstractmethod
    async def get_map_context(self) -> MapContext: ...


class MockMapClient(MapClientInterface):
    """Returns a hardcoded highway scenario for testing."""

    async def get_map_context(self) -> MapContext:
        return MapContext(
            road_type=RoadType.HIGHWAY,
            nearest_rest_km=12.5,
            nearest_coffee_km=8.0,
            rest_spot_name="G2 高速服务区",
            coffee_shop_name="星巴克 (服务区店)",
        )


def _compute_time_risk_multiplier(hour: Optional[int] = None) -> float:
    """
    Return a risk multiplier based on time of day.

      2–5 AM  → 1.6  (human circadian sleep trough; highest crash risk)
      13–15   → 1.3  (post-lunch glycaemic dip)
      all else → 1.0
    """
    if hour is None:
        hour = datetime.datetime.now().hour
    if 2 <= hour < 5:
        return 1.6
    if 13 <= hour < 15:
        return 1.3
    return 1.0


class ContextEnricher:
    def __init__(self, map_client: MapClientInterface) -> None:
        self._map = map_client

    async def enrich(self, ctx: FatigueContext) -> EnrichedFatigueContext:
        map_ctx    = await self._map.get_map_context()
        multiplier = _compute_time_risk_multiplier()
        # Re-classify here with accurate road_type from map (aggregator used UNKNOWN)
        profile = classify_profile(
            driving_duration_min=ctx.driving_duration_min,
            road_type=map_ctx.road_type,
        )
        return EnrichedFatigueContext(
            fatigue=ctx,
            map=map_ctx,
            time_risk_multiplier=multiplier,
            driver_profile=profile,
        )
