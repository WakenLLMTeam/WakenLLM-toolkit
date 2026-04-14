"""
MODERATE action: voice broadcast via TTS (Qwen3-tts-flash).
Message tone adapts to driver profile:
  LONG_HAUL → urgent, includes duration and nearest rest
  COMMUTER  → calm, avoids disruptive tone
"""
import logging
from models.judge_verdict import JudgeVerdict
from models.fatigue_context import EnrichedFatigueContext, RoadType, DriverProfile
from models.action_result import ActionResult, ActionStatus
from actions.base import ActionAgent

logger = logging.getLogger(__name__)


class VoiceBroadcastAgent(ActionAgent):
    @property
    def name(self) -> str:
        return "voice_broadcast"

    async def execute(
        self,
        verdict: JudgeVerdict,
        ctx: EnrichedFatigueContext,
    ) -> ActionResult:
        message = self._build_voice_message(verdict, ctx)
        # In production: call Qwen3-tts-flash API → audio → car speaker
        logger.info("[VOICE TTS] %s", message)
        print(f"\n[VOICE] {message}")
        return ActionResult(
            action_name=self.name,
            status=ActionStatus.SUCCESS,
            message=message,
        )

    def _build_voice_message(
        self,
        verdict: JudgeVerdict,
        ctx: EnrichedFatigueContext,
    ) -> str:
        m = ctx.map
        duration_min = ctx.fatigue.driving_duration_min

        if ctx.driver_profile == DriverProfile.LONG_HAUL:
            duration_label = (
                f"{int(duration_min // 60)}小时{int(duration_min % 60)}分钟"
                if duration_min >= 60
                else f"{int(duration_min)}分钟"
            )
            if m.road_type == RoadType.HIGHWAY and m.rest_spot_name:
                return (
                    f"注意！您已连续驾驶{duration_label}，检测到疲劳迹象。"
                    f"前方{m.nearest_rest_km:.0f}公里「{m.rest_spot_name}」，"
                    f"请立即驶入休息区。"
                )
            return f"注意！您已连续驾驶{duration_label}，请尽快靠边停车休息。"

        # Commuter — calm tone
        if m.road_type == RoadType.CITY and m.coffee_shop_name:
            return (
                f"提醒您注意休息，附近「{m.coffee_shop_name}」"
                f"距您约{m.nearest_coffee_km:.0f}公里，可以停车放松一下。"
            )
        return "您可能有些疲劳，请保持注意力，找到合适地点稍作休息。"
