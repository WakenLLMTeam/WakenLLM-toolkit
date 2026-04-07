#!/usr/bin/env python3
"""
Render a funnel chart PNG.
Standalone — no DeerFlow dependency required.

Spec format:
{
  "type": "funnel",
  "title": "FSD Adoption Funnel",
  "stages": [
    {"label": "All Tesla Vehicles",  "value": 6000000, "color": "#dbeafe"},
    {"label": "FSD Capable HW",      "value": 3500000, "color": "#bfdbfe"},
    {"label": "FSD Purchased",        "value": 800000,  "color": "#93c5fd"},
    {"label": "Active FSD Users",    "value": 400000,  "color": "#60a5fa"},
    {"label": "Daily FSD Users",     "value": 150000,  "color": "#2563eb"}
  ],
  "show_percent": true,   // show conversion rate between stages
  "unit": "",             // unit suffix for values (e.g. "K", "M")
  "fig_width": 9,
  "fig_height": 6
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
import matplotlib.patches as mpatches
import numpy as np

from viz_theme import THEME, setup_matplotlib

setup_matplotlib()

_DEFAULT_COLORS = [
    "#dbeafe", "#bfdbfe", "#93c5fd", "#60a5fa",
    "#3b82f6", "#2563eb", "#1d4ed8", "#1e40af",
]


def _fmt_value(v: float, unit: str) -> str:
    if v >= 1_000_000:
        return f"{v/1_000_000:.1f}M{unit}"
    if v >= 1_000:
        return f"{v/1_000:.0f}K{unit}"
    return f"{v:,.0f}{unit}"


def render_funnel(spec: Dict[str, Any], output_path: str) -> str:
    title: Optional[str] = spec.get("title")
    stages: List[Dict[str, Any]] = spec.get("stages", [])
    show_percent: bool = spec.get("show_percent", True)
    unit: str = spec.get("unit", "")

    if not stages:
        raise ValueError("funnel requires stages")

    fw = float(spec.get("fig_width", 9.0))
    fh = float(spec.get("fig_height", 6.0))
    n = len(stages)

    fig, ax = plt.subplots(figsize=(fw, fh))
    fig.patch.set_facecolor(THEME.BG)
    ax.set_facecolor(THEME.BG)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    values = [float(s.get("value", 1)) for s in stages]
    max_val = max(values)
    bar_h = 0.80 / n
    gap = 0.02 / max(n - 1, 1)

    for i, (stage, val) in enumerate(zip(stages, values)):
        ratio = val / max_val
        bar_w = 0.2 + ratio * 0.6   # narrowest 20%, widest 80%
        bx = (1 - bar_w) / 2
        by = 0.92 - (i + 1) * (bar_h + gap)

        color = stage.get("color", _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)])
        rect = mpatches.FancyBboxPatch(
            (bx, by), bar_w, bar_h - 0.004,
            boxstyle="round,pad=0.005",
            facecolor=color, edgecolor="white", linewidth=0.8,
            transform=ax.transAxes, zorder=3
        )
        ax.add_patch(rect)

        # Stage label (left)
        ax.text(bx - 0.02, by + bar_h / 2,
                stage.get("label", ""),
                ha="right", va="center",
                fontsize=THEME.FS_SMALL, color=THEME.INK, fontweight="bold",
                transform=ax.transAxes)

        # Value (inside bar)
        ax.text(0.5, by + bar_h / 2,
                _fmt_value(val, unit),
                ha="center", va="center",
                fontsize=THEME.FS_BODY, color=THEME.INK, fontweight="bold",
                transform=ax.transAxes)

        # Conversion rate (right side, between stages)
        if show_percent and i > 0:
            pct = val / values[i - 1] * 100
            prev_by = 0.92 - i * (bar_h + gap)
            mid_y = (prev_by + by + bar_h) / 2
            right_x = (1 - 0.2 - 0.6) / 2 + 0.2 + 0.6 + 0.02
            ax.text(right_x, mid_y,
                    f"↓ {pct:.0f}%",
                    ha="left", va="center",
                    fontsize=THEME.FS_MICRO, color=THEME.MUTED, style="italic",
                    transform=ax.transAxes)

    if title:
        fig.text(0.5, 0.98, title, ha="center", va="top",
                 fontsize=THEME.FS_H1, color=THEME.INK, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=THEME.DPI, bbox_inches="tight", facecolor=THEME.BG)
    pdf_path = str(output_path).replace(".png", ".pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor=THEME.BG)
    plt.close(fig)
    return f"Funnel chart saved: {output_path}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Render funnel chart PNG")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    src = args.spec.strip()
    spec = json.loads(src) if src.startswith("{") else json.load(open(src, encoding="utf-8"))
    print(render_funnel(spec, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
