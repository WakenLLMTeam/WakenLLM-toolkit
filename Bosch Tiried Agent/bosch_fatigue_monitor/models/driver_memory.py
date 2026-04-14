"""
DriverMemory: persistent per-driver preference store.

Loaded once at startup (from JSON or defaults); injected into EnrichedFatigueContext.
Action agents read it to personalise their suggestions — e.g. auto-order coffee
instead of just recommending it when the driver has that preference enabled.
"""
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class DriverMemory:
    # ── Beverage / caffeine ──
    likes_coffee: bool = False            # if True + Starbucks nearby → auto-order
    preferred_coffee_order: str = "拿铁"  # what to order (passed to coffee API)
    coffee_max_km: float = 2.0           # only auto-order if coffee shop ≤ this far

    # ── Rest style ──
    prefers_nap: bool = False            # prefer 20-min nap over coffee when severe
    ok_to_pull_over_city: bool = True    # allow "pull over at red light" suggestion

    # ── Notification preferences ──
    voice_enabled: bool = True
    push_enabled: bool = True

    # ── Historical context ──
    ignored_alert_streak: int = 0        # how many consecutive alerts were ignored
    last_rest_stop: Optional[str] = None # name of last rest stop used
    total_sessions: int = 0              # total driving sessions tracked

    # ── Tags / notes ──
    notes: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Pre-built profiles for the demo
# ─────────────────────────────────────────────────────────────────────────────

COFFEE_LOVER = DriverMemory(
    likes_coffee=True,
    preferred_coffee_order="美式（大杯）",
    coffee_max_km=2.0,
    ok_to_pull_over_city=True,
    notes=["司机偏好咖啡提神", "曾在G2服务区休息"],
)

NAP_PREFERS = DriverMemory(
    likes_coffee=False,
    prefers_nap=True,
    ok_to_pull_over_city=True,
    notes=["司机偏好短暂小憩"],
)

DEFAULT_MEMORY = DriverMemory()
