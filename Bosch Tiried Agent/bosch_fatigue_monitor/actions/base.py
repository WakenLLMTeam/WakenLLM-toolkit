"""ActionAgent ABC — all action agents implement this interface."""
from abc import ABC, abstractmethod
from models.judge_verdict import JudgeVerdict
from models.fatigue_context import EnrichedFatigueContext
from models.action_result import ActionResult


class ActionAgent(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def execute(
        self,
        verdict: JudgeVerdict,
        ctx: EnrichedFatigueContext,
    ) -> ActionResult: ...
