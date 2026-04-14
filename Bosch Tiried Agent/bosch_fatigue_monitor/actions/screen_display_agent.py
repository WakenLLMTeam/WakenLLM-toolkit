"""
MILD action: display fatigue warning on the car HMI screen.
Message adapts to driver profile and driving duration.
"""
import logging
from models.judge_verdict import JudgeVerdict, SeverityTier
from models.fatigue_context import EnrichedFatigueContext, DriverProfile
from models.action_result import ActionResult, ActionStatus
from actions.base import ActionAgent

logger = logging.getLogger(__name__)


class ScreenDisplayAgent(ActionAgent):
    @property
    def name(self) -> str:
        return "screen_display"

    async def execute(
        self,
        verdict: JudgeVerdict,
        ctx: EnrichedFatigueContext,
    ) -> ActionResult:
        message = self._build_message(verdict, ctx)
        # In production: send to car HMI via CAN bus / SOME/IP
        logger.info("[HMI SCREEN] %s", message)
        print(f"\n[CAR SCREEN] {message}")
        return ActionResult(
            action_name=self.name,
            status=ActionStatus.SUCCESS,
            message=message,
        )

    def _build_message(
        self,
        verdict: JudgeVerdict,
        ctx: EnrichedFatigueContext,
    ) -> str:
        score_pct = int(verdict.composite_score * 100)
        tier = verdict.severity_tier
        duration_min = ctx.fatigue.driving_duration_min

        tier_label = {
            SeverityTier.MILD:     "轻度疲劳",
            SeverityTier.MODERATE: "中度疲劳",
            SeverityTier.SEVERE:   "严重疲劳",
        }.get(tier, "疲劳预警")

        if ctx.driver_profile == DriverProfile.LONG_HAUL and duration_min > 0:
            duration_label = (
                f"{int(duration_min // 60)}h{int(duration_min % 60)}m"
                if duration_min >= 60
                else f"{int(duration_min)}min"
            )
            return f"⚠ {tier_label} ({score_pct}%) | 已行驶 {duration_label} — 建议休息"
        return f"疲劳预警 {tier_label} ({score_pct}%) — 请注意驾驶安全，建议适当休息。"
