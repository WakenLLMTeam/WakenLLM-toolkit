"""
SEVERE action: push notification + video clip to the driver's phone app.
"""
import logging
from models.judge_verdict import JudgeVerdict
from models.fatigue_context import EnrichedFatigueContext
from models.action_result import ActionResult, ActionStatus
from actions.base import ActionAgent

logger = logging.getLogger(__name__)


class PhonePushAgent(ActionAgent):
    def __init__(self, push_url: str) -> None:
        self._push_url = push_url

    @property
    def name(self) -> str:
        return "phone_push"

    async def execute(
        self,
        verdict: JudgeVerdict,
        ctx: EnrichedFatigueContext,
    ) -> ActionResult:
        payload = {
            "title": "严重疲劳预警",
            "body": f"疲劳评分 {verdict.composite_score:.0%}，请立即休息！",
            "score": verdict.composite_score,
            "tier": verdict.severity_tier.name,
            "reasoning": verdict.reasoning,
        }
        # In production: POST to phone push endpoint / FCM / APNs
        logger.warning("[PHONE PUSH] → %s | payload=%s", self._push_url, payload)
        print(f"\n[PHONE APP] Push sent: {payload['body']}")
        return ActionResult(
            action_name=self.name,
            status=ActionStatus.SUCCESS,
            message=payload["body"],
            payload=payload,
        )
