"""
Shared visual theme and font utilities for all renderers.
Works standalone (no DeerFlow dependency).
"""
from __future__ import annotations
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from typing import Optional

# ── CJK font auto-detection ──────────────────────────────────────────────────
_CJK_CANDIDATES = [
    "PingFang SC", "PingFang HK", "PingFang TC",
    "Heiti SC", "Heiti TC",
    "Hiragino Sans GB",
    "Noto Sans CJK SC", "Noto Sans SC", "Source Han Sans SC",
    "WenQuanYi Micro Hei",
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
]

def _detect_cjk_font() -> str:
    available = {f.name for f in fm.fontManager.ttflist}
    for candidate in _CJK_CANDIDATES:
        if candidate in available:
            return candidate
    return "sans-serif"

CJK_FONT = _detect_cjk_font()

def setup_matplotlib() -> None:
    """Call once before any rendering to set global font defaults."""
    matplotlib.rcParams["font.family"] = [CJK_FONT, "DejaVu Sans", "sans-serif"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["figure.dpi"] = 150


# ── Design tokens ─────────────────────────────────────────────────────────────

class Theme:
    """
    Central design token store.
    Light backgrounds, black text throughout.
    All colors are hex strings unless noted.
    """
    # Accent — used only for lines, arrows, borders, dots (never as bg fill)
    ACCENT       = "#2563eb"
    ACCENT_LIGHT = "#eff6ff"   # very light blue — lightest allowed bg tint
    ACCENT_MID   = "#93c5fd"   # medium blue for arrow/line highlights

    # All text is black or near-black
    INK          = "#111111"   # headings
    BODY         = "#111111"   # body text (same — no dark grey, pure black)
    MUTED        = "#444444"   # captions, sub-labels, secondary text

    # Backgrounds — all light / white
    BG           = "#ffffff"   # figure background
    SURFACE      = "#f5f7fa"   # card / node surface
    ALT_ROW      = "#eef2f7"   # alternating table row

    # Border
    BORDER       = "#c8d3e0"

    # Node / cell palette — all light pastels with black text
    GREEN        = "#e6f9ee"
    GREEN_BORDER = "#6dbf8a"
    YELLOW       = "#fdf8e1"
    YELLOW_BORDER= "#d4aa30"
    RED          = "#fdecea"
    RED_BORDER   = "#e07070"
    PURPLE       = "#f0eeff"
    PURPLE_BORDER= "#9b80d8"
    PINK         = "#fdeef6"
    PINK_BORDER  = "#d878b0"

    # Font sizes (pt)
    FS_TITLE  = 13
    FS_H1     = 11
    FS_H2     = 9.5
    FS_BODY   = 8.5
    FS_SMALL  = 7.5
    FS_MICRO  = 6.5

    # Figure defaults
    DPI       = 300
    FIG_W     = 13
    FIG_H     = 4.0


THEME = Theme()


# ── Bosch brand color palette ─────────────────────────────────────────────────

class BoschTheme(Theme):
    """
    Bosch corporate design language on top of the base Theme.
    Use for Bosch-specific presentations (ADAS, automotive, safety).
    """
    # Bosch Red — primary brand accent
    ACCENT       = "#E20015"
    ACCENT_LIGHT = "#fde8ea"
    ACCENT_MID   = "#f28090"

    # Bosch Navy — secondary accent for multi-series charts
    NAVY         = "#003366"
    NAVY_LIGHT   = "#e0e8f0"

    # Functional signal colors (ADAS / safety context)
    SAFE_GREEN   = "#00873D"   # system OK / L2+ operational
    WARN_AMBER   = "#F5A623"   # transitional / conditional
    ALERT_RED    = "#E20015"   # system limit / ASIL violation (same as accent)
    INFO_BLUE    = "#0057A8"   # informational / passive monitoring

    # Layer palette for arch diagrams (sensor→app order)
    LAYER_SENSOR = "#f3e8ff"   # purple tint — sensor / hardware
    LAYER_HAL    = "#dcfce7"   # green tint — hardware abstraction / BSP
    LAYER_MW     = "#fef9c3"   # yellow tint — middleware / OS
    LAYER_APP    = "#dbeafe"   # blue tint — application / algorithm


BOSCH_THEME = BoschTheme()
