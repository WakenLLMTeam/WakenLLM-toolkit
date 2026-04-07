#!/usr/bin/env python3
"""
Render a waterfall chart PNG (incremental positive/negative bars).
Standalone — no DeerFlow dependency required.

Spec format:
{
  "type": "waterfall",
  "title": "FSD R&D Cost Breakdown (M$)",
  "items": [
    {"label": "Base Cost",      "value": 500,  "type": "start"},
    {"label": "Hardware",       "value": -120, "type": "negative"},
    {"label": "Software Dev",   "value": -80,  "type": "negative"},
    {"label": "Data & Compute", "value": -60,  "type": "negative"},
    {"label": "Partnerships",   "value": 40,   "type": "positive"},
    {"label": "Net Cost",       "value": 280,  "type": "total"}
  ],
  "unit": "M$",
  "fig_width": 11,
  "fig_height": 5
}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from viz_theme import THEME, setup_matplotlib

setup_matplotlib()


def render_waterfall(spec: Dict[str, Any], output_path: str) -> str:
    title: Optional[str] = spec.get("title")
    items: List[Dict[str, Any]] = spec.get("items", [])
    unit: str = spec.get("unit", "")

    if not items:
        raise ValueError("waterfall requires items")

    fw = float(spec.get("fig_width", 11.0))
    fh = float(spec.get("fig_height", 5.0))

    fig, ax = plt.subplots(figsize=(fw, fh))
    fig.patch.set_facecolor(THEME.BG)
    ax.set_facecolor(THEME.BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(THEME.BORDER)
    ax.spines["bottom"].set_color(THEME.BORDER)
    ax.tick_params(colors=THEME.MUTED, labelsize=THEME.FS_SMALL)
    ax.yaxis.grid(True, color=THEME.BORDER, linewidth=0.5, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    COLOR_POS   = "#16a34a"
    COLOR_NEG   = THEME.ACCENT
    COLOR_TOTAL = THEME.ACCENT + "cc"
    COLOR_START = "#64748b"

    x = np.arange(len(items))
    labels = [it.get("label", "") for it in items]
    running = 0.0
    bottoms = []
    heights = []
    colors = []

    for it in items:
        val = float(it.get("value", 0))
        kind = it.get("type", "positive")
        if kind in ("start", "total"):
            bottoms.append(0)
            heights.append(val)
            running = val if kind == "start" else running
            colors.append(COLOR_START if kind == "start" else COLOR_TOTAL)
        elif val >= 0:
            bottoms.append(running)
            heights.append(val)
            running += val
            colors.append(COLOR_POS)
        else:
            running += val
            bottoms.append(running)
            heights.append(-val)
            colors.append(COLOR_NEG)

    bars = ax.bar(x, heights, bottom=bottoms, color=colors, width=0.55,
                  edgecolor="white", linewidth=0.5, alpha=0.90)

    # Connector lines between bars
    for i in range(len(items) - 1):
        kind_next = items[i + 1].get("type", "")
        if kind_next != "total":
            top = bottoms[i] + heights[i]
            ax.plot([x[i] + 0.28, x[i + 1] - 0.28], [top, top],
                    color=THEME.BORDER, linewidth=0.8, linestyle="--", zorder=1)

    # Value labels
    for bar, it, bot, h in zip(bars, items, bottoms, heights):
        val = it.get("value", 0)
        label_y = bot + h + max(heights) * 0.015
        ax.text(bar.get_x() + bar.get_width() / 2, label_y,
                f"{val:+g}{unit}" if it.get("type") not in ("start", "total") else f"{val:g}{unit}",
                ha="center", va="bottom",
                fontsize=THEME.FS_SMALL, color=THEME.INK, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=THEME.FS_SMALL, color=THEME.INK)
    if unit:
        ax.set_ylabel(unit, fontsize=THEME.FS_SMALL, color=THEME.MUTED)

    if title:
        fig.text(0.5, 0.98, title, ha="center", va="top",
                 fontsize=THEME.FS_H1, color=THEME.INK, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.93 if title else 1.0], pad=0.4)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=THEME.DPI, bbox_inches="tight", facecolor=THEME.BG)
    pdf_path = str(output_path).replace(".png", ".pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor=THEME.BG)
    plt.close(fig)
    return f"Waterfall chart saved: {output_path}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Render waterfall chart PNG")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    src = args.spec.strip()
    spec = json.loads(src) if src.startswith("{") else json.load(open(src, encoding="utf-8"))
    print(render_waterfall(spec, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
