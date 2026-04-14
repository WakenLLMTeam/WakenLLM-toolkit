"""
SEVERE action: capture a video segment of the driver's fatigue state.
"""
import logging
import time
from models.judge_verdict import JudgeVerdict
from models.fatigue_context import EnrichedFatigueContext
from models.action_result import ActionResult, ActionStatus
from actions.base import ActionAgent

logger = logging.getLogger(__name__)


class VideoRecordAgent(ActionAgent):
    @property
    def name(self) -> str:
        return "video_record"

    async def execute(
        self,
        verdict: JudgeVerdict,
        ctx: EnrichedFatigueContext,
    ) -> ActionResult:
        filename = f"fatigue_{int(time.time())}.mp4"
        # In production: trigger dashcam/interior camera recording API
        logger.warning("[VIDEO RECORD] Capturing fatigue event → %s", filename)
        print(f"\n[VIDEO] Recording fatigue event → {filename}")
        return ActionResult(
            action_name=self.name,
            status=ActionStatus.SUCCESS,
            message=f"Video captured: {filename}",
            payload={"filename": filename, "score": verdict.composite_score},
        )
