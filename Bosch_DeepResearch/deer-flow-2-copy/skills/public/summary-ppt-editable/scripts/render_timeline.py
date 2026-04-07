#!/usr/bin/env python3
"""
Render a high-quality horizontal timeline PNG from a JSON spec.
Standalone — no DeerFlow dependency required.

Usage:
  python render_timeline.py --spec timeline.json --output out.png
  python render_timeline.py --spec '{"stages":[...]}' --output out.png

Spec format:
{
  "type": "timeline",
  "title": "L2 to L3 Autonomous Driving Evolution",
  "stages": [
    {
      "label": "L2 Assisted Driving",
      "year": "2018-2021",
      "annotation": "ACC + LKA",
      "detail": "Driver remains responsible for monitoring"
    },
    {
      "label": "L2+ Extended ODD",
      "year": "2021-2024",
      "annotation": "Highway NOA",
      "detail": "Continuous assist within limited ODD"
    },
    {
      "label": "L3 Conditional Automation",
      "year": "2024+",
      "annotation": "Eyes-off within ODD",
      "detail": "System takes over driving task; takeover on ODD exit"
    }
  ],
  "highlight": [2],
  "accent_color": "#1a56db",
  "fig_width": 13,
  "fig_height": 4.0
}

Fields:
  stages[].label       Stage name (required)
  stages[].year        Time label below the dot (optional)
  stages[].annotation  Short tag above the dot (optional)
  stages[].detail      Longer description below year (optional)
  highlight            0-based indices to draw with accent color (default: all)
  accent_color         Hex, default "#1a56db"
  fig_width/fig_height Inches, defaults 13 x 4.0
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from viz_theme import THEME, setup_matplotlib

setup_matplotlib()


def _is_dark(hex_color: str) -> bool:
    """Return True if hex color is too dark for light-theme use."""
    h = hex_color.lstrip("#")
    if len(h) < 6:
        return False
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return luminance < 160


def render_timeline(spec: Dict[str, Any], output_path: str) -> str:
    stages: List[Dict[str, Any]] = spec.get("stages", [])
    if not stages:
        raise ValueError("spec.stages must be a non-empty list")

    title: Optional[str] = spec.get("title")
    highlight_raw = spec.get("highlight", list(range(len(stages))))
    if isinstance(highlight_raw, int):
        highlight_raw = [highlight_raw]
    highlight: List[int] = highlight_raw
    accent = spec.get("accent_color", THEME.ACCENT)
    if _is_dark(accent):
        accent = THEME.ACCENT

    n = len(stages)

    # ── Adaptive sizing based on number of stages ─────────────────────────────
    # More stages → wider figure, smaller fonts, no detail text
    fw = float(spec.get("fig_width", max(THEME.FIG_W, n * 1.9)))
    fh = float(spec.get("fig_height", THEME.FIG_H))
    label_fs = max(THEME.FS_MICRO + 0.5, THEME.FS_H2 - max(0, n - 4) * 0.5)
    anno_fs  = max(THEME.FS_MICRO,       THEME.FS_SMALL - max(0, n - 4) * 0.3)
    year_fs  = max(THEME.FS_MICRO,       THEME.FS_SMALL - max(0, n - 4) * 0.3)
    show_detail = n <= 5  # only show detail text when stages fit comfortably
    dot_size_outer = max(18, 28 - max(0, n - 4) * 2)
    dot_size_inner = max(14, 22 - max(0, n - 4) * 2)

    xs = np.linspace(0.07, 0.93, n)

    fig, ax = plt.subplots(figsize=(fw, fh))
    fig.patch.set_facecolor(THEME.BG)
    ax.set_facecolor(THEME.BG)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # ── Spine ─────────────────────────────────────────────────────────────────
    ax.hlines(0.50, xs[0], xs[-1], colors=THEME.BORDER, linewidths=2.5, zorder=1)
    ax.hlines(0.50, xs[0], xs[-1], colors=accent + "22", linewidths=6, zorder=1)

    for i, (x, stage) in enumerate(zip(xs, stages)):
        active = i in highlight
        ring_color = accent if active else THEME.BORDER

        # Alternate annotation above/below the spine to avoid overlap
        anno_above = (i % 2 == 0)

        # ── Shadow ────────────────────────────────────────────────────────────
        ax.plot(x + 0.002, 0.503, "o", markersize=dot_size_outer,
                color="#00000010", zorder=2)
        # ── Ring ──────────────────────────────────────────────────────────────
        ax.plot(x, 0.50, "o", markersize=dot_size_outer,
                color=ring_color, zorder=3)
        # ── Fill ──────────────────────────────────────────────────────────────
        ax.plot(x, 0.50, "o", markersize=dot_size_inner,
                color=THEME.ACCENT_LIGHT if active else THEME.SURFACE, zorder=4)
        # ── Number ────────────────────────────────────────────────────────────
        ax.text(x, 0.50, str(i + 1),
                ha="center", va="center",
                fontsize=anno_fs - 0.5, color=THEME.INK,
                fontweight="bold", zorder=5)

        # ── Connector tick ────────────────────────────────────────────────────
        tick_len = 0.10
        if anno_above:
            ax.vlines(x, 0.50 + 0.02, 0.50 + tick_len, colors=ring_color, linewidths=1.0, zorder=2)
        else:
            ax.vlines(x, 0.50 - tick_len, 0.50 - 0.02, colors=ring_color, linewidths=1.0, zorder=2)

        # ── Annotation pill (alternates above/below) ──────────────────────────
        anno = stage.get("annotation", "")
        if anno:
            pill_y = (0.50 + tick_len + 0.09) if anno_above else (0.50 - tick_len - 0.09)
            pill_bg = THEME.ACCENT_LIGHT if active else THEME.SURFACE
            ax.text(x, pill_y, anno,
                    ha="center", va="center",
                    fontsize=anno_fs, color=THEME.INK, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25",
                              facecolor=pill_bg, edgecolor=ring_color,
                              linewidth=0.6, alpha=0.95),
                    zorder=6)

        # ── Stage label (opposite side from annotation) ───────────────────────
        label_side = -1 if anno_above else 1   # below spine if anno is above
        label_y = 0.50 + label_side * (tick_len + 0.21)
        ax.text(x, label_y, stage.get("label", ""),
                ha="center", va="center",
                fontsize=label_fs, color=THEME.INK,
                fontweight="bold", zorder=5,
                multialignment="center")

        # ── Year badge ────────────────────────────────────────────────────────
        year = stage.get("year", "")
        if year:
            year_y = label_y + label_side * 0.10
            ax.text(x, year_y, year,
                    ha="center", va="center",
                    fontsize=year_fs, color=THEME.MUTED,
                    bbox=dict(boxstyle="round,pad=0.18",
                              facecolor=THEME.SURFACE, edgecolor=THEME.BORDER,
                              linewidth=0.5),
                    zorder=5)

        # ── Detail text (only when n <= 5, shown beyond year badge) ──────────────
        if show_detail:
            detail = (stage.get("detail") or "").strip()
            if detail:
                detail_fs = max(THEME.FS_MICRO, anno_fs - 1.0)
                # Place detail one step beyond year in the same spine direction
                detail_y = label_y + label_side * (0.10 + (0.13 if year else 0.03))
                # Wrap long detail text at ~30 chars
                if len(detail) > 30:
                    words = detail.split()
                    lines, cur = [], ""
                    for w in words:
                        if len(cur) + len(w) + 1 > 30 and cur:
                            lines.append(cur)
                            cur = w
                        else:
                            cur = (cur + " " + w).strip()
                    if cur:
                        lines.append(cur)
                    detail = "\n".join(lines)
                ax.text(x, detail_y, detail,
                        ha="center", va="center",
                        fontsize=detail_fs, color=THEME.MUTED,
                        style="italic",
                        multialignment="center",
                        linespacing=1.15,
                        zorder=5)

    # ── Arrows between dots ───────────────────────────────────────────────────
    for i in range(n - 1):
        gap = xs[i + 1] - xs[i]
        ax.annotate("",
                    xy=(xs[i + 1] - gap * 0.16, 0.50),
                    xytext=(xs[i] + gap * 0.16, 0.50),
                    arrowprops=dict(arrowstyle="-|>",
                                    color=THEME.ACCENT_MID,
                                    lw=1.4, mutation_scale=12),
                    zorder=3)

    # ── Title ─────────────────────────────────────────────────────────────────
    if title:
        fig.text(0.5, 0.98, title,
                 ha="center", va="top",
                 fontsize=THEME.FS_H1, color=THEME.INK,
                 fontweight="bold")

    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=THEME.DPI, bbox_inches="tight",
                facecolor=THEME.BG)
    plt.close(fig)
    return f"Timeline saved: {output_path}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Render horizontal timeline PNG")
    parser.add_argument("--spec", required=True,
                        help="JSON spec file path or inline JSON string")
    parser.add_argument("--output", required=True, help="Output PNG path")
    args = parser.parse_args()
    src = args.spec.strip()
    spec = json.loads(src) if src.startswith("{") else json.load(open(src, encoding="utf-8"))
    print(render_timeline(spec, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
