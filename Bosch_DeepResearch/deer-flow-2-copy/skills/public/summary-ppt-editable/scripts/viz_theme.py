"""
Shared visual theme and font utilities for all renderers.
Works standalone (no DeerFlow dependency).
"""
from __future__ import annotations
import colorsys
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
    matplotlib.rcParams["font.family"] = ["Comic Sans MS", CJK_FONT, "DejaVu Sans", "sans-serif"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["figure.dpi"] = 300
    # Bold everything by default
    matplotlib.rcParams["font.weight"] = "bold"
    matplotlib.rcParams["axes.labelweight"] = "bold"
    matplotlib.rcParams["axes.titleweight"] = "bold"
    matplotlib.rcParams["xtick.labelsize"] = 11
    matplotlib.rcParams["ytick.labelsize"] = 11
    matplotlib.rcParams["axes.labelsize"] = 12


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

    # Font sizes (pt) — sized for 300 DPI output, readable at screen zoom
    FS_TITLE  = 18
    FS_H1     = 15
    FS_H2     = 13
    FS_BODY   = 12
    FS_SMALL  = 11
    FS_MICRO  = 9.5

    # Figure defaults
    DPI       = 300
    FIG_W     = 13
    FIG_H     = 4.0




def fit_fontsize(text: str, box_width_in: float, box_height_in: float,
                 start_pt: float, min_pt: float = 5.5) -> float:
    """
    Estimate the largest font size (pt) that fits `text` in a box of
    `box_width_in` × `box_height_in` inches.

    Uses a simple character-width heuristic (no renderer needed):
      - average char width ≈ 0.55 × font_pt / 72  inches  (proportional font)
      - line height        ≈ 1.35 × font_pt / 72  inches

    Works offline (no Agg renderer required), so it can be called before
    fig/ax exist.  The result is an approximation — err on the conservative side.
    """
    if not text or box_width_in <= 0 or box_height_in <= 0:
        return max(start_pt, min_pt)

    lines = text.replace("\\n", "\n").split("\n")
    n_lines = len(lines)
    max_chars = max(len(l) for l in lines) if lines else 1

    pt = float(start_pt)
    while pt > min_pt:
        char_w = 0.55 * pt / 72       # inches per character
        line_h = 1.35 * pt / 72       # inches per line
        text_w = max_chars * char_w
        text_h = n_lines * line_h
        if text_w <= box_width_in * 0.92 and text_h <= box_height_in * 0.90:
            break
        pt -= 0.5
    return max(pt, min_pt)

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


# ── Morandi color palette ───────────────────────────────────────────────────────

def _hsluv_to_hex(h_deg: float, s: float, l: float) -> str:
    """
    Convert h/s/l (HSLuv-style) to sRGB hex.
    h: hue in degrees [0, 360)
    s: saturation in [0, 1]  (0.3-0.5 = Morandi muted)
    l: lightness in [0, 1]    (0.4-0.75 = medium range)
    """
    h_norm = (h_deg % 360) / 360.0
    r, g, b = colorsys.hls_to_rgb(h_norm, l, s)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def _oklch_to_hex(l: float, c: float, h_deg: float) -> str:
    """Convert OKLCH-style (l=lightness, c=chroma, h=hue_deg) to sRGB hex."""
    # For Morandi: l=0.65-0.78, c=0.28-0.42, h=evenly spaced
    # Scale c from [0.1-0.15] to [0.28-0.42] for visible saturation
    return _hsluv_to_hex(h_deg, c * 3.0, l)


# Pre-defined 18-color Morandi base palette (verified visually distinct)
# Blues, greens, yellows, oranges, reds, purples — all muted/desaturated
_MORANDI_BASE = [
    # Dusty blue family
    "#8EACCF",  # 0° - muted slate blue
    "#AAB7C4",  # 20°
    "#C4CED4",  # 40° - grey blue
    # Warm neutrals
    "#D4C4A8",  # 60° - warm sand
    "#C9B896",  # 80°
    "#BEA882",  # 100° - dusty tan
    # Olive/sage greens
    "#A8B89C",  # 120° - sage
    "#94A888",  # 140°
    "#7E9874",  # 160° - muted olive
    # Teal/aqua family
    "#8DB5A8",  # 180° - dusty teal
    "#7AA698",  # 200°
    "#689788",  # 220° - muted cyan-green
    # Lavender/purple
    "#ACA0C4",  # 240° - dusty lavender
    "#9C8AB0",  # 260°
    "#8C749C",  # 280° - muted purple
    # Rose/mauve
    "#B494A8",  # 300° - dusty rose
    "#A47E94",  # 320°
    "#946880",  # 340° - muted mauve
]


def _generate_extended_palette(target: int) -> list[str]:
    """Generate more colors by blending variants of base palette."""
    base = list(_MORANDI_BASE)
    while len(base) < target:
        i = len(base)
        base_color = _MORANDI_BASE[i % len(_MORANDI_BASE)]
        # Slightly vary lightness using colorsys
        r_val = int(base_color[1:3], 16) / 255.0
        g_val = int(base_color[3:5], 16) / 255.0
        b_val = int(base_color[5:7], 16) / 255.0
        h, l, s = colorsys.rgb_to_hls(r_val, g_val, b_val)
        # Alternate between lighter/darker
        l = max(0.40, min(0.85, l + (0.06 if i % 2 == 0 else -0.06)))
        r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
        base.append(f"#{int(r2*255):02x}{int(g2*255):02x}{int(b2*255):02x}")
    return base[:target]


_MORANDI_EXTENDED = _generate_extended_palette(30)


class MorandiPalette:
    """
    Morandi-style perceptual color palette.
    Low chroma, medium-high lightness, perceptual uniformity via OKLCH.
    Three modes: categorical (distinct), sequential (ordered), diverging (signed).
    """

    # Class-level base palette
    BASE = list(_MORANDI_BASE)          # 18 pre-generated
    EXTENDED = list(_MORANDI_EXTENDED)  # 30 colors

    @classmethod
    def generate_categorical(cls, n: int) -> list[str]:
        """
        Return n distinct colors for categorical data.
        Uses evenly-spaced hues at low chroma.
        """
        if n <= 0:
            return []
        if n <= len(cls.BASE):
            return cls.BASE[:n]
        return cls.EXTENDED[:n]

    @classmethod
    def generate_sequential(cls, n: int, base_hue: float = 30.0) -> list[str]:
        """
        Return n colors for sequential/ordered data.
        Lightness ramps from light to dark along a single hue direction.
        base_hue: starting hue in degrees (default 30° = warm ochre).
        """
        if n <= 0:
            return []
        L_start = 0.87
        L_end = 0.62
        C = 0.10
        colors = []
        for i in range(n):
            t = i / max(n - 1, 1)
            L = L_start + (L_end - L_start) * t
            colors.append(_oklch_to_hex(L, C, base_hue))
        return colors

    @classmethod
    def generate_diverging(cls, n: int) -> list[str]:
        """
        Return n colors for diverging data (e.g., negative → neutral → positive).
        Uses two hues meeting at a neutral center.
        n should be odd for a clear neutral midpoint; if even, center is omitted.
        """
        if n <= 0:
            return []
        half = n // 2
        left = cls.generate_sequential(half + (0 if n % 2 == 0 else 1), base_hue=250.0)  # blue-purple
        right = cls.generate_sequential(half + (1 if n % 2 == 0 else 0), base_hue=30.0)   # warm ochre
        if n % 2 == 0:
            # Even: no neutral center
            return left[:-1] + list(reversed(right))
        else:
            # Odd: neutral center
            return left[:-1] + right


# Convenience aliases
def get_categorical_palette(n: int) -> list[str]:
    """Return n distinct Morandi colors for categorical data."""
    return MorandiPalette.generate_categorical(n)


def get_sequential_palette(n: int, base_hue: float = 30.0) -> list[str]:
    """Return n ordered Morandi colors for sequential data."""
    return MorandiPalette.generate_sequential(n, base_hue)


def get_diverging_palette(n: int) -> list[str]:
    """Return n diverging Morandi colors."""
    return MorandiPalette.generate_diverging(n)


def get_series_colors(n: int, palette: str = "categorical", **kwargs) -> list[str]:
    """
    Return n colors from the Morandi palette.

    Args:
        n: Number of colors needed
        palette: "categorical" | "sequential" | "diverging"
        **kwargs: Passed to the specific palette generator
                  (e.g., base_hue for sequential)

    Returns:
        List of hex color strings
    """
    if palette == "sequential":
        return get_sequential_palette(n, **kwargs)
    elif palette == "diverging":
        return get_diverging_palette(n)
    else:
        return get_categorical_palette(n)


MORANDI_PALETTE = MorandiPalette  # alias for backward compat


def get_morandi_cmap(scheme: str = "blue"):
    """
    Return a Morandi-style matplotlib LinearSegmentedColormap for heatmaps.
    scheme: "blue" | "green" | "red" | "purple" | "diverging"
    """
    import matplotlib.colors as mcolors
    _HUE_MAP = {"blue": 220.0, "green": 130.0, "red": 5.0, "purple": 265.0}
    if scheme == "diverging":
        colors = get_diverging_palette(9)
    else:
        hue = _HUE_MAP.get(scheme, 220.0)
        colors = get_sequential_palette(9, base_hue=hue)
    return mcolors.LinearSegmentedColormap.from_list(f"morandi_{scheme}", colors)
