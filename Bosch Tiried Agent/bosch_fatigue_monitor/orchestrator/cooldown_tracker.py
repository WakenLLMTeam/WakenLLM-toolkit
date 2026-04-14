"""
CooldownTracker: prevents alert spam AND escalates when the driver ignores alerts.

Adaptive logic:
  - Each time an alert fires for a tier, we compare the new composite score
    with the previous one.  If no improvement (score ≥ 90% of last score),
    we increment ignored_count for that tier.
  - Cooldown shortens progressively with ignored_count:
      1 ignore → 80% of base cooldown
      2 ignores → 60%
      3+ ignores → 40%
  - effective_tier() escalates MILD→MODERATE or MODERATE→SEVERE after 2 ignores,
    so the driver gets harder-to-miss interventions.
  - ignored_count resets when score improves meaningfully (< 80% of last score).
"""
import time
from models.judge_verdict import SeverityTier
from config import CooldownConfig

_IGNORE_THRESHOLD = 0.90   # score must drop below 90% of last to count as "responded"
_RESET_THRESHOLD  = 0.80   # score below 80% of last → driver responded, reset counter


class CooldownTracker:
    def __init__(self, cfg: CooldownConfig) -> None:
        self._cfg = cfg
        self._last_fired:   dict = {tier: 0.0 for tier in SeverityTier}
        self._last_score:   dict = {tier: 0.0 for tier in SeverityTier}
        self._ignored_count: dict = {tier: 0   for tier in SeverityTier}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_allowed(self, tier: SeverityTier) -> bool:
        if tier == SeverityTier.NONE:
            return False
        cooldown = self._effective_cooldown(tier)
        return (time.time() - self._last_fired[tier]) >= cooldown

    def record_fired(self, tier: SeverityTier, score: float = 0.0) -> None:
        """Call this AFTER dispatching, to stamp the fire time and score."""
        self._last_fired[tier] = time.time()
        self._last_score[tier] = score

    def update_ignored(self, tier: SeverityTier, current_score: float) -> None:
        """
        Call this BEFORE dispatching (when cooldown has just expired).
        Updates ignored_count based on whether score improved since last alert.
        """
        last = self._last_score[tier]
        if last <= 0.0:
            return  # first ever alert — no baseline yet

        if current_score >= last * _IGNORE_THRESHOLD:
            self._ignored_count[tier] += 1
        elif current_score < last * _RESET_THRESHOLD:
            self._ignored_count[tier] = 0  # driver responded; reset

    def effective_tier(self, tier: SeverityTier) -> SeverityTier:
        """
        Return a potentially escalated tier when the driver keeps ignoring alerts.
        Escalates after 2 consecutive ignores (caps at SEVERE).
        """
        if self._ignored_count[tier] >= 2 and tier < SeverityTier.SEVERE:
            return SeverityTier(int(tier) + 1)
        return tier

    @property
    def ignored_counts(self) -> dict:
        """Expose for logging / tests."""
        return dict(self._ignored_count)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _effective_cooldown(self, tier: SeverityTier) -> float:
        base = {
            SeverityTier.MILD:     self._cfg.mild_seconds,
            SeverityTier.MODERATE: self._cfg.moderate_seconds,
            SeverityTier.SEVERE:   self._cfg.severe_seconds,
        }.get(tier, 0.0)

        count = self._ignored_count[tier]
        if count >= 3:
            return base * 0.4
        if count >= 2:
            return base * 0.6
        if count >= 1:
            return base * 0.8
        return base
