"""
OrchestratorAgent: the single authority for action dispatch.
Implements pyramid escalation — SEVERE includes all lower-tier actions.
No sensor or judge ever triggers an action directly.

Adaptive escalation:
  If the driver keeps ignoring alerts (score not improving), CooldownTracker
  will shorten cooldowns and escalate the effective tier automatically.
"""
import asyncio
import logging
from typing import List

from config import AppConfig
from models.judge_verdict import JudgeVerdict, SeverityTier
from models.fatigue_context import EnrichedFatigueContext
from actions.base import ActionAgent
from orchestrator.cooldown_tracker import CooldownTracker

logger = logging.getLogger(__name__)


class OrchestratorAgent:
    def __init__(
        self,
        screen_agent:   ActionAgent,
        voice_agent:    ActionAgent,
        video_agent:    ActionAgent,
        push_agent:     ActionAgent,
        context_agent:  ActionAgent,
        cfg:            AppConfig,
    ) -> None:
        self._screen  = screen_agent
        self._voice   = voice_agent
        self._video   = video_agent
        self._push    = push_agent
        self._context = context_agent
        self._cooldown = CooldownTracker(cfg.cooldowns)

    async def run(
        self,
        verdict_queue: asyncio.Queue,
    ) -> None:
        while True:
            item = await verdict_queue.get()
            verdict: JudgeVerdict
            ctx: EnrichedFatigueContext
            verdict, ctx = item
            await self._dispatch(verdict, ctx)

    async def _dispatch(
        self,
        verdict: JudgeVerdict,
        ctx: EnrichedFatigueContext,
    ) -> None:
        tier = verdict.severity_tier

        if tier == SeverityTier.NONE:
            logger.debug("Verdict NONE — no action required.")
            return

        # Update ignore tracking BEFORE cooldown check (uses last score as baseline)
        self._cooldown.update_ignored(tier, verdict.composite_score)

        if not self._cooldown.is_allowed(tier):
            logger.debug("Tier %s in cooldown — skipping.", tier.name)
            return

        # Effective tier may be escalated if driver kept ignoring previous alerts
        effective_tier = self._cooldown.effective_tier(tier)
        if effective_tier != tier:
            logger.warning(
                "Escalating tier %s → %s after %d consecutive ignores.",
                tier.name,
                effective_tier.name,
                self._cooldown.ignored_counts[tier],
            )

        logger.info(
            "Dispatching tier=%s (effective=%s) score=%.3f ignores=%s",
            tier.name,
            effective_tier.name,
            verdict.composite_score,
            self._cooldown.ignored_counts,
        )
        self._cooldown.record_fired(tier, verdict.composite_score)

        # Pyramid: higher tiers include all lower-tier actions
        actions: List[ActionAgent] = [self._screen]

        if effective_tier >= SeverityTier.MODERATE:
            actions.append(self._voice)

        if effective_tier >= SeverityTier.SEVERE:
            actions.extend([self._video, self._push, self._context])

        # Execute all actions for this tier
        results = await asyncio.gather(
            *[a.execute(verdict, ctx) for a in actions],
            return_exceptions=True,
        )
        for r in results:
            if isinstance(r, Exception):
                logger.error("Action failed: %s", r)
