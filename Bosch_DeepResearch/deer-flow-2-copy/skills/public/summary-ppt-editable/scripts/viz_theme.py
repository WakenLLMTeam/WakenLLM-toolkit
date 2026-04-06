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
    DPI       = 150
    FIG_W     = 13
    FIG_H     = 4.0


THEME = Theme()
