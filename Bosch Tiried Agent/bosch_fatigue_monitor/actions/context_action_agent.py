"""
SEVERE context-aware action: profile-adapted smart suggestions.

Long-haul driver: show driving duration, time-to-rest, revenue framing
  ("15-min rest avoids a 30-min accident delay")
Commuter: gentle nudge with next red-light breathing tip
"""
import logging
from models.judge_verdict import JudgeVerdict
from models.fatigue_context import EnrichedFatigueContext, RoadType, DriverProfile
from models.action_result import ActionResult, ActionStatus
from actions.base import ActionAgent

logger = logging.getLogger(__name__)

# Rough km/h assumption for highway ETA estimate
_HIGHWAY_SPEED_KMH = 100.0


def _eta_minutes(km: float, speed_kmh: float = _HIGHWAY_SPEED_KMH) -> int:
    return max(1, round(km / speed_kmh * 60))


class ContextActionAgent(ActionAgent):
    @property
    def name(self) -> str:
        return "context_action"

    async def execute(
        self,
        verdict: JudgeVerdict,
        ctx: EnrichedFatigueContext,
    ) -> ActionResult:
        if ctx.driver_profile == DriverProfile.LONG_HAUL:
            return await self._long_haul_suggestion(verdict, ctx)
        return await self._commuter_suggestion(verdict, ctx)

    # ------------------------------------------------------------------
    # Long-haul: highway rest with ETA and revenue framing
    # ------------------------------------------------------------------

    async def _long_haul_suggestion(
        self,
        verdict: JudgeVerdict,
        ctx: EnrichedFatigueContext,
    ) -> ActionResult:
        m = ctx.map
        duration_h = ctx.fatigue.driving_duration_min / 60.0
        duration_label = (
            f"{int(duration_h)}小时{int(ctx.fatigue.driving_duration_min % 60)}分"
            if duration_h >= 1
            else f"{int(ctx.fatigue.driving_duration_min)}分钟"
        )

        if m.road_type == RoadType.HIGHWAY and m.rest_spot_name and m.nearest_rest_km:
            eta = _eta_minutes(m.nearest_rest_km)
            msg = (
                f"[长途预警] 已连续行驶 {duration_label}。\n"
                f"建议前往：{m.rest_spot_name}（{m.nearest_rest_km:.1f} km，约 {eta} 分钟）\n"
                f"15分钟休息可有效恢复注意力，避免事故耽误更多时间。"
            )
            payload = {
                "action": "navigate_to_rest",
                "destination": m.rest_spot_name,
                "eta_min": eta,
                "driving_duration_min": ctx.fatigue.driving_duration_min,
            }
        else:
            msg = (
                f"[长途预警] 已连续行驶 {duration_label}，请尽快寻找安全停靠点休息。"
            )
            payload = {"action": "find_rest_spot"}

        logger.warning("[CONTEXT-LONGHAUL] %s", msg)
        print(f"\n[NAV] {msg}")
        return ActionResult(
            action_name=self.name,
            status=ActionStatus.SUCCESS,
            message=msg,
            payload=payload,
        )

    # ------------------------------------------------------------------
    # Commuter: gentle nudge, no disruption
    # ------------------------------------------------------------------

    async def _commuter_suggestion(
        self,
        verdict: JudgeVerdict,
        ctx: EnrichedFatigueContext,
    ) -> ActionResult:
        m = ctx.map

        if m.road_type == RoadType.CITY and m.coffee_shop_name and m.nearest_coffee_km:
            msg = (
                f"[轻度疲劳提示] 下一个红灯时深呼吸 3 次。\n"
                f"附近：{m.coffee_shop_name}（{m.nearest_coffee_km:.1f} km），"
                f"可稍作休息。"
            )
            payload = {"action": "order_coffee", "shop": m.coffee_shop_name}
        else:
            msg = "[轻度疲劳提示] 建议在安全地点短暂停车，活动一下再继续。"
            payload = {"action": "rest_suggestion"}

        logger.info("[CONTEXT-COMMUTER] %s", msg)
        print(f"\n[ORDER] {msg}")
        return ActionResult(
            action_name=self.name,
            status=ActionStatus.SUCCESS,
            message=msg,
            payload=payload,
        )
