"""Tests for ContextEnricher — verifies map context injection."""
import time
import pytest
from pipeline.context_enricher import ContextEnricher, MockMapClient
from models.fatigue_context import (
    FatigueContext, TextSignals, ImageSignals, AudioSignals, RoadType
)


@pytest.fixture
def bare_context():
    return FatigueContext(
        text_signals=TextSignals(),
        image_signals=ImageSignals(),
        audio_signals=AudioSignals(),
        window_seconds=60.0,
        timestamp=time.time(),
    )


class TestContextEnricher:
    @pytest.mark.asyncio
    async def test_enrich_adds_map_context(self, bare_context):
        enricher = ContextEnricher(MockMapClient())
        enriched = await enricher.enrich(bare_context)
        assert enriched.map is not None

    @pytest.mark.asyncio
    async def test_mock_returns_highway(self, bare_context):
        enricher = ContextEnricher(MockMapClient())
        enriched = await enricher.enrich(bare_context)
        assert enriched.map.road_type == RoadType.HIGHWAY

    @pytest.mark.asyncio
    async def test_fatigue_context_preserved(self, bare_context):
        enricher = ContextEnricher(MockMapClient())
        enriched = await enricher.enrich(bare_context)
        assert enriched.fatigue is bare_context

    @pytest.mark.asyncio
    async def test_rest_stop_name_present(self, bare_context):
        enricher = ContextEnricher(MockMapClient())
        enriched = await enricher.enrich(bare_context)
        assert enriched.map.rest_spot_name is not None
        assert len(enriched.map.rest_spot_name) > 0

    @pytest.mark.asyncio
    async def test_custom_map_client_is_used(self, bare_context):
        """Verify interface contract: a custom MapClient replaces MockMapClient cleanly."""
        from unittest.mock import AsyncMock
        from models.fatigue_context import MapContext
        from pipeline.context_enricher import MapClientInterface

        class CityMapClient(MapClientInterface):
            async def get_map_context(self) -> MapContext:
                return MapContext(
                    road_type=RoadType.CITY,
                    nearest_coffee_km=0.5,
                    coffee_shop_name="Luckin Coffee",
                )

        enricher = ContextEnricher(CityMapClient())
        enriched = await enricher.enrich(bare_context)
        assert enriched.map.road_type == RoadType.CITY
        assert enriched.map.coffee_shop_name == "Luckin Coffee"
