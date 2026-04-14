"""Tests for individual action agents — Phase 3 & 4 profile-aware behavior."""
import pytest
from models.action_result import ActionStatus
from models.fatigue_context import RoadType, DriverProfile
from actions.screen_display_agent import ScreenDisplayAgent
from actions.voice_broadcast_agent import VoiceBroadcastAgent
from actions.video_record_agent import VideoRecordAgent
from actions.phone_push_agent import PhonePushAgent
from actions.context_action_agent import ContextActionAgent


# ─────────────────────────────────────────────────────────────────────────────
# ScreenDisplayAgent
# ─────────────────────────────────────────────────────────────────────────────

class TestScreenDisplayAgent:
    @pytest.mark.asyncio
    async def test_returns_success(self, mild_verdict, clear_context):
        result = await ScreenDisplayAgent().execute(mild_verdict, clear_context)
        assert result.status == ActionStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_message_contains_score_percent(self, mild_verdict, clear_context):
        result = await ScreenDisplayAgent().execute(mild_verdict, clear_context)
        assert "35%" in result.message

    @pytest.mark.asyncio
    async def test_message_in_chinese(self, mild_verdict, clear_context):
        result = await ScreenDisplayAgent().execute(mild_verdict, clear_context)
        assert "疲劳" in result.message

    @pytest.mark.asyncio
    async def test_long_haul_message_shows_duration(self, severe_verdict, alert_context_long_haul):
        """Long-haul screen message must include driving duration."""
        result = await ScreenDisplayAgent().execute(severe_verdict, alert_context_long_haul)
        assert result.status == ActionStatus.SUCCESS
        # Duration 200 min → "3h20m" or similar
        assert "h" in result.message or "min" in result.message

    @pytest.mark.asyncio
    async def test_long_haul_message_shows_tier_label(self, severe_verdict, alert_context_long_haul):
        result = await ScreenDisplayAgent().execute(severe_verdict, alert_context_long_haul)
        assert "严重疲劳" in result.message

    @pytest.mark.asyncio
    async def test_commuter_message_no_duration(self, mild_verdict, alert_context_commuter):
        """Commuter message should not show driving duration (short trip)."""
        result = await ScreenDisplayAgent().execute(mild_verdict, alert_context_commuter)
        assert "h" not in result.message or "min" not in result.message


# ─────────────────────────────────────────────────────────────────────────────
# VoiceBroadcastAgent
# ─────────────────────────────────────────────────────────────────────────────

class TestVoiceBroadcastAgent:
    @pytest.mark.asyncio
    async def test_long_haul_highway_mentions_rest_stop(self, severe_verdict, alert_context_long_haul):
        result = await VoiceBroadcastAgent().execute(severe_verdict, alert_context_long_haul)
        assert result.status == ActionStatus.SUCCESS
        assert "G2 高速服务区" in result.message

    @pytest.mark.asyncio
    async def test_long_haul_message_includes_duration(self, severe_verdict, alert_context_long_haul):
        """Long-haul voice message must mention driving duration."""
        result = await VoiceBroadcastAgent().execute(severe_verdict, alert_context_long_haul)
        # 200 min → "3小时20分钟" or similar
        assert "小时" in result.message or "分钟" in result.message

    @pytest.mark.asyncio
    async def test_commuter_city_message_mentions_coffee(self, severe_verdict, alert_context_commuter):
        result = await VoiceBroadcastAgent().execute(severe_verdict, alert_context_commuter)
        assert "星巴克" in result.message

    @pytest.mark.asyncio
    async def test_commuter_message_is_gentle(self, severe_verdict, alert_context_commuter):
        """Commuter message should NOT use urgent exclamation-heavy tone."""
        result = await VoiceBroadcastAgent().execute(severe_verdict, alert_context_commuter)
        # Commuter path uses 提醒 (reminder), not 注意！ (urgent warning)
        assert "注意！" not in result.message

    @pytest.mark.asyncio
    async def test_legacy_highway_message_mentions_rest_stop(self, severe_verdict, alert_context):
        """Backwards compat: original alert_context (LONG_HAUL profile) still works."""
        result = await VoiceBroadcastAgent().execute(severe_verdict, alert_context)
        assert result.status == ActionStatus.SUCCESS
        assert "G2 高速服务区" in result.message

    @pytest.mark.asyncio
    async def test_city_message_mentions_coffee(self, severe_verdict, alert_context_city):
        result = await VoiceBroadcastAgent().execute(severe_verdict, alert_context_city)
        assert "星巴克" in result.message

    @pytest.mark.asyncio
    async def test_unknown_road_gives_generic_message(self, severe_verdict, alert_context):
        alert_context.map.road_type = RoadType.UNKNOWN
        alert_context.map.rest_spot_name = None
        result = await VoiceBroadcastAgent().execute(severe_verdict, alert_context)
        assert result.status == ActionStatus.SUCCESS
        assert len(result.message) > 0


# ─────────────────────────────────────────────────────────────────────────────
# VideoRecordAgent
# ─────────────────────────────────────────────────────────────────────────────

class TestVideoRecordAgent:
    @pytest.mark.asyncio
    async def test_returns_success_with_filename(self, severe_verdict, alert_context):
        result = await VideoRecordAgent().execute(severe_verdict, alert_context)
        assert result.status == ActionStatus.SUCCESS
        assert result.payload is not None
        assert result.payload["filename"].endswith(".mp4")

    @pytest.mark.asyncio
    async def test_payload_includes_score(self, severe_verdict, alert_context):
        result = await VideoRecordAgent().execute(severe_verdict, alert_context)
        assert result.payload["score"] == pytest.approx(0.82)


# ─────────────────────────────────────────────────────────────────────────────
# PhonePushAgent
# ─────────────────────────────────────────────────────────────────────────────

class TestPhonePushAgent:
    @pytest.mark.asyncio
    async def test_returns_success(self, severe_verdict, alert_context):
        agent = PhonePushAgent("http://localhost:8080/push")
        result = await agent.execute(severe_verdict, alert_context)
        assert result.status == ActionStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_payload_contains_tier(self, severe_verdict, alert_context):
        agent = PhonePushAgent("http://localhost:8080/push")
        result = await agent.execute(severe_verdict, alert_context)
        assert result.payload["tier"] == "SEVERE"

    @pytest.mark.asyncio
    async def test_message_contains_score_percentage(self, severe_verdict, alert_context):
        agent = PhonePushAgent("http://localhost:8080/push")
        result = await agent.execute(severe_verdict, alert_context)
        assert "82%" in result.message


# ─────────────────────────────────────────────────────────────────────────────
# ContextActionAgent
# ─────────────────────────────────────────────────────────────────────────────

class TestContextActionAgent:
    # --- Long-haul path ---

    @pytest.mark.asyncio
    async def test_long_haul_navigates_to_rest(self, severe_verdict, alert_context_long_haul):
        result = await ContextActionAgent().execute(severe_verdict, alert_context_long_haul)
        assert result.status == ActionStatus.SUCCESS
        assert result.payload["action"] == "navigate_to_rest"

    @pytest.mark.asyncio
    async def test_long_haul_message_mentions_duration(self, severe_verdict, alert_context_long_haul):
        """Long-haul context action must tell driver how long they've been driving."""
        result = await ContextActionAgent().execute(severe_verdict, alert_context_long_haul)
        assert "连续行驶" in result.message
        assert "小时" in result.message or "分" in result.message

    @pytest.mark.asyncio
    async def test_long_haul_message_mentions_eta(self, severe_verdict, alert_context_long_haul):
        """Long-haul message should include ETA to rest stop."""
        result = await ContextActionAgent().execute(severe_verdict, alert_context_long_haul)
        assert "分钟" in result.message    # ETA in minutes

    @pytest.mark.asyncio
    async def test_long_haul_payload_has_eta_and_duration(self, severe_verdict, alert_context_long_haul):
        result = await ContextActionAgent().execute(severe_verdict, alert_context_long_haul)
        assert "eta_min" in result.payload
        assert "driving_duration_min" in result.payload
        assert result.payload["driving_duration_min"] == pytest.approx(200.0)

    # --- Commuter path ---

    @pytest.mark.asyncio
    async def test_commuter_suggests_coffee(self, severe_verdict, alert_context_commuter):
        result = await ContextActionAgent().execute(severe_verdict, alert_context_commuter)
        assert result.status == ActionStatus.SUCCESS
        assert result.payload["action"] == "order_coffee"

    @pytest.mark.asyncio
    async def test_commuter_message_is_gentle(self, severe_verdict, alert_context_commuter):
        """Commuter message must not use long-haul urgent framing."""
        result = await ContextActionAgent().execute(severe_verdict, alert_context_commuter)
        assert "红灯" in result.message or "轻度" in result.message

    # --- Legacy / fallback ---

    @pytest.mark.asyncio
    async def test_highway_navigates_to_rest(self, severe_verdict, alert_context):
        """Original alert_context (LONG_HAUL) still dispatches navigate_to_rest."""
        result = await ContextActionAgent().execute(severe_verdict, alert_context)
        assert result.status == ActionStatus.SUCCESS
        assert result.payload["action"] == "navigate_to_rest"

    @pytest.mark.asyncio
    async def test_city_orders_coffee(self, severe_verdict, alert_context_city):
        result = await ContextActionAgent().execute(severe_verdict, alert_context_city)
        assert result.payload["action"] == "order_coffee"

    @pytest.mark.asyncio
    async def test_unknown_road_type_returns_generic(self, severe_verdict, alert_context):
        alert_context.map.road_type = RoadType.UNKNOWN
        alert_context.map.rest_spot_name = None
        alert_context.map.nearest_rest_km = None
        result = await ContextActionAgent().execute(severe_verdict, alert_context)
        assert result.status == ActionStatus.SUCCESS
