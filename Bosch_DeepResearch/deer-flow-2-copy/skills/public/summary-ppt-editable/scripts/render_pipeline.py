#!/usr/bin/env python3
"""
Render a high-quality horizontal pipeline / architecture block diagram PNG.
Standalone — no DeerFlow dependency required.

Usage:
  python render_pipeline.py --spec pipeline.json --output out.png

Spec format:
{
  "type": "pipeline",
  "title": "ADAS System Module Chain",
  "arrow_label": "data flow",
  "stages": [
    {"label": "Raw Sensors",     "sublabel": "Camera · Radar · LiDAR", "color": "#dbeafe"},
    {"label": "Perception",      "sublabel": "Detection · Lane",        "color": "#dcfce7"},
    {"label": "Prediction",      "sublabel": "Trajectory · Intent",     "color": "#fef9c3"},
    {"label": "Planning",        "sublabel": "Behavior · Path",         "color": "#fce7f3"},
    {"label": "Control",         "sublabel": "Longitudinal · Lateral",  "color": "#ede9fe"}
  ],
  "rows": 1,
  "accent_color": "#1a56db",
  "fig_width": 13,
  "fig_height": 3.0
}

Fields:
  stages[].label     Primary block label
  stages[].sublabel  Smaller secondary text (optional)
  stages[].color     Background hex (optional, default surface)
  stages[].badge     Small top-right badge text, e.g. "ASIL-D" (optional)
  arrow_label        Label shown above all arrows (optional)
  rows               Number of rows to wrap into (default 1)
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

from viz_theme import THEME, setup_matplotlib

setup_matplotlib()


_LIGHT_PALETTE = [
    "#eff6ff", "#f0fdf4", "#fefce8", "#fdf4ff", "#fff7ed",
    "#f0f9ff", "#fdf2f8", "#f7fee7", "#f8f9fa", "#eef2ff",
]

def _is_dark(hex_color: str) -> bool:
    h = hex_color.lstrip("#")
    if len(h) < 6:
        return False
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (0.299 * r + 0.587 * g + 0.114 * b) < 160

def _safe_color(color: str, fallback_idx: int) -> str:
    if not color or _is_dark(color):
        return _LIGHT_PALETTE[fallback_idx % len(_LIGHT_PALETTE)]
    return color


def render_pipeline(spec: Dict[str, Any], output_path: str) -> str:
    stages: List[Dict[str, Any]] = spec.get("stages", [])
    title: Optional[str] = spec.get("title")
    arrow_label: Optional[str] = spec.get("arrow_label")
    rows: int = max(1, int(spec.get("rows", 1)))
    accent = spec.get("accent_color", THEME.ACCENT)
    if _is_dark(accent):
        accent = THEME.ACCENT

    if not stages:
        raise ValueError("spec.stages must be non-empty")

    n = len(stages)
    cols_per_row = math.ceil(n / rows)

    # Adaptive figure size: taller when sublabels are long
    max_sublabel_len = max(
        (len(s.get("sublabel", "")) for s in stages), default=0)
    has_sublabel = any(s.get("sublabel") for s in stages)
    # Figure width: driven by longest sublabel so text fits without wrapping
    # ~0.11 inches per char across all stages in one row
    fw = float(spec.get("fig_width",
               max(THEME.FIG_W, cols_per_row * max(2.0, max_sublabel_len * 0.115))))
    fw = min(fw, 22.0)
    # Height: taller when sublabels are multi-word
    fh_per_row = float(spec.get("fig_height",
                                3.4 if has_sublabel else 2.6))

    fig, axes = plt.subplots(rows, 1, figsize=(fw, fh_per_row * rows),
                             squeeze=False)
    fig.patch.set_facecolor(THEME.BG)

    for row_idx in range(rows):
        ax = axes[row_idx][0]
        ax.set_facecolor(THEME.BG)
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        start = row_idx * cols_per_row
        end = min(start + cols_per_row, n)
        row_stages = stages[start:end]
        k = len(row_stages)
        if k == 0:
            continue

        # Block geometry — taller block when sublabels are long
        padding = 0.04
        arrow_frac = 0.030
        total_arrow = arrow_frac * (k - 1)
        block_w = (1.0 - padding * 2 - total_arrow) / k
        block_h = 0.58 if has_sublabel else 0.48
        block_y = 0.5 - block_h / 2

        # X positions (left edge of each block)
        xs = []
        cur = padding
        for i in range(k):
            xs.append(cur)
            cur += block_w + (arrow_frac if i < k - 1 else 0)

        # Adaptive font sizes based on number of stages per row
        label_fs  = max(THEME.FS_MICRO + 1, THEME.FS_H2  - max(0, k - 4) * 0.6)
        sub_fs    = max(THEME.FS_MICRO,     THEME.FS_SMALL - max(0, k - 4) * 0.5)
        badge_fs  = max(THEME.FS_MICRO - 0.5, THEME.FS_MICRO)

        for i, (x, stage) in enumerate(zip(xs, row_stages)):
            bg = _safe_color(stage.get("color", THEME.SURFACE), start + i)
            label = stage.get("label", "")
            sublabel = stage.get("sublabel", "")
            badge = stage.get("badge", "")

            # Shadow
            ax.add_patch(FancyBboxPatch(
                (x + 0.004, block_y - 0.006), block_w, block_h,
                boxstyle="round,pad=0.02",
                facecolor="#00000018", edgecolor="none",
                transform=ax.transAxes, zorder=2))
            # Card
            ax.add_patch(FancyBboxPatch(
                (x, block_y), block_w, block_h,
                boxstyle="round,pad=0.02",
                facecolor=bg, edgecolor=THEME.BORDER, linewidth=1.2,
                transform=ax.transAxes, zorder=3))

            # Accent top bar
            ax.add_patch(FancyBboxPatch(
                (x, block_y + block_h - 0.055), block_w, 0.055,
                boxstyle="square,pad=0",
                facecolor=accent + "33", edgecolor="none",
                transform=ax.transAxes, zorder=4, clip_on=True))

            # Step number circle — light fill, black number
            cx_ = x + 0.035
            cy_ = block_y + block_h - 0.028
            ax.plot(cx_, cy_, "o", markersize=13, color=THEME.ACCENT_LIGHT,
                    markeredgecolor=accent, markeredgewidth=1.2,
                    transform=ax.transAxes, zorder=5)
            ax.text(cx_, cy_, str(start + i + 1),
                    ha="center", va="center",
                    fontsize=THEME.FS_MICRO, color=THEME.INK, fontweight="bold",
                    transform=ax.transAxes, zorder=6)

            # Label
            label_y = 0.5 + (0.07 if sublabel else 0)
            ax.text(x + block_w / 2, label_y, label,
                    ha="center", va="center",
                    fontsize=label_fs, color=THEME.INK, fontweight="bold",
                    transform=ax.transAxes, zorder=4)

            # Sublabel — no manual wrap, figure width is sized to fit
            if sublabel:
                ax.text(x + block_w / 2, 0.5 - 0.10, sublabel,
                        ha="center", va="center",
                        fontsize=sub_fs, color=THEME.MUTED,
                        transform=ax.transAxes, zorder=4,
                        multialignment="center",
                        linespacing=1.2)

            # Badge
            if badge:
                ax.text(x + block_w - 0.008, block_y + block_h - 0.02,
                        badge,
                        ha="right", va="top",
                        fontsize=badge_fs, color=accent, fontweight="bold",
                        transform=ax.transAxes, zorder=5)

            # Arrow to next block
            if i < k - 1:
                ax_start = x + block_w + 0.004
                ax_end = xs[i + 1] - 0.004
                ay = 0.5
                ax.annotate("",
                            xy=(ax_end, ay), xytext=(ax_start, ay),
                            xycoords="axes fraction",
                            textcoords="axes fraction",
                            arrowprops=dict(arrowstyle="-|>",
                                            color=accent, lw=1.8,
                                            mutation_scale=14),
                            zorder=7)

    if title:
        fig.text(0.5, 0.99, title, ha="center", va="top",
                 fontsize=THEME.FS_H1, color=THEME.INK, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95], pad=0.5)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=THEME.DPI, bbox_inches="tight",
                facecolor=THEME.BG)
    plt.close(fig)
    return f"Pipeline diagram saved: {output_path}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Render pipeline block diagram PNG")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    src = args.spec.strip()
    spec = json.loads(src) if src.startswith("{") else json.load(open(src, encoding="utf-8"))
    print(render_pipeline(spec, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
