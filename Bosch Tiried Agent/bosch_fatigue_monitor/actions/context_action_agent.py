"""
SEVERE context-aware action: profile-adapted + memory-personalised suggestions.

Decision tree:
  LONG_HAUL → highway rest-stop navigation with ETA and revenue framing
  COMMUTER  → check driver_memory:
                likes_coffee + Starbucks ≤ coffee_max_km → AUTO-ORDER coffee
                ok_to_pull_over_city + traffic_density < 0.3 → pull-over suggestion
                fallback → generic gentle reminder
"""
import logging
from models.judge_verdict import JudgeVerdict
from models.fatigue_context import EnrichedFatigueContext, RoadType, DriverProfile
from models.action_result import ActionResult, ActionStatus
from actions.base import ActionAgent

logger = logging.getLogger(__name__)

_HIGHWAY_SPEED_KMH = 100.0
_LOW_TRAFFIC_THRESHOLD = 0.3   # traffic_density below this = safe to pull over


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

    # ──────────────────────────────────────────────────────────────────────────
    # Long-haul: highway rest with ETA and revenue framing
    # ──────────────────────────────────────────────────────────────────────────

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

    # ──────────────────────────────────────────────────────────────────────────
    # Commuter: memory-personalised suggestion
    # ──────────────────────────────────────────────────────────────────────────

    async def _commuter_suggestion(
        self,
        verdict: JudgeVerdict,
        ctx: EnrichedFatigueContext,
    ) -> ActionResult:
        m = ctx.map
        mem = ctx.driver_memory   # may be None

        # ── Path 1: driver likes coffee + Starbucks is close enough ──
        if (
            mem is not None
            and mem.likes_coffee
            and m.coffee_shop_name
            and m.nearest_coffee_km is not None
            and m.nearest_coffee_km <= mem.coffee_max_km
        ):
            order = mem.preferred_coffee_order
            msg = (
                f"[记忆偏好] 已为您自动下单：{m.coffee_shop_name} {order}（{m.nearest_coffee_km:.1f} km）。\n"
                f"前往取单，顺便活动一下，有助于恢复注意力。"
            )
            payload = {
                "action": "order_coffee",
                "shop": m.coffee_shop_name,
                "order": order,
                "distance_km": m.nearest_coffee_km,
                "auto_ordered": True,
            }
            logger.info("[CONTEXT-MEMORY] Auto-ordered coffee: %s @ %s", order, m.coffee_shop_name)
            print(f"\n[☕ AUTO-ORDER] {msg}")
            return ActionResult(
                action_name=self.name,
                status=ActionStatus.SUCCESS,
                message=msg,
                payload=payload,
            )

        # ── Path 2: city road, low traffic → safe pull-over ──
        if (
            m.road_type == RoadType.CITY
            and (mem is None or mem.ok_to_pull_over_city)
            and m.traffic_density < _LOW_TRAFFIC_THRESHOLD
        ):
            msg = (
                f"[轻度疲劳提示] 周边路段车流稀少（密度 {m.traffic_density:.0%}），"
                f"可在红灯时安全停车，深呼吸 3 次，活动颈部再继续行驶。"
            )
            payload = {
                "action": "pull_over_rest",
                "traffic_density": m.traffic_density,
            }
            logger.info("[CONTEXT-COMMUTER] Pull-over suggestion (low traffic)")
            print(f"\n[🅿 PULL-OVER] {msg}")
            return ActionResult(
                action_name=self.name,
                status=ActionStatus.SUCCESS,
                message=msg,
                payload=payload,
            )

        # ── Path 3: coffee shop present (no memory, or too far) ──
        if m.road_type == RoadType.CITY and m.coffee_shop_name and m.nearest_coffee_km:
            msg = (
                f"[轻度疲劳提示] 下一个红灯时深呼吸 3 次。\n"
                f"附近：{m.coffee_shop_name}（{m.nearest_coffee_km:.1f} km），"
                f"可稍作休息。"
            )
            payload = {"action": "order_coffee", "shop": m.coffee_shop_name, "auto_ordered": False}
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
