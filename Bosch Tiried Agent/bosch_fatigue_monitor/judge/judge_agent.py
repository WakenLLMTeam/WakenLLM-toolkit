"""
FatigueJudgeAgent: stateless LLM-as-a-Judge.
Input:  EnrichedFatigueContext
Output: JudgeVerdict
"""
import logging
from models.fatigue_context import EnrichedFatigueContext
from models.judge_verdict import JudgeVerdict
from llm.base import LLMClientInterface
from judge.prompt_template import SYSTEM_PROMPT, build_user_prompt
from judge.response_parser import parse_verdict

logger = logging.getLogger(__name__)


class FatigueJudgeAgent:
    def __init__(self, llm_client: LLMClientInterface) -> None:
        self._llm = llm_client

    async def evaluate(self, ctx: EnrichedFatigueContext) -> JudgeVerdict:
        user_prompt = build_user_prompt(ctx)
        face_b64 = ctx.fatigue.image_signals.face_frame_b64

        tags = [ctx.map.road_type.value]

        try:
            raw = await self._llm.complete(
                system=SYSTEM_PROMPT,
                user=user_prompt,
                image_b64=face_b64,
            )
            verdict = parse_verdict(raw, context_tags=tags)
            logger.info(
                "Judge verdict: tier=%s composite=%.3f",
                verdict.severity_tier.name,
                verdict.composite_score,
            )
            return verdict
        except Exception as e:
            logger.error("Judge evaluation failed: %s", e)
            raise
