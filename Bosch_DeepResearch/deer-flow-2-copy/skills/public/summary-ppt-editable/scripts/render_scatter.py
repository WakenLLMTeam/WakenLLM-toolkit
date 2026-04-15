#!/usr/bin/env python3
"""
Render a scatter plot (with optional bubble sizing) PNG.
Standalone — no DeerFlow dependency required.

Spec format:
{
  "type": "scatter",
  "title": "Technology Maturity vs Market Readiness",
  "x_label": "Technical Maturity (TRL)",
  "y_label": "Market Readiness",
  "series": [
    {
      "name": "Camera",
      "points": [
        {"x": 8, "y": 7, "label": "Camera", "size": 200, "color": "#2563eb"}
      ]
    },
    {
      "name": "LiDAR",
      "points": [
        {"x": 6, "y": 5, "label": "LiDAR", "size": 400, "color": "#E20015"}
      ]
    }
  ],
  "quadrant_labels": ["Niche", "Leader", "Emerging", "Follower"],
  "fig_width": 8,
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
import numpy as np

from viz_theme import THEME, setup_matplotlib, get_categorical_palette

setup_matplotlib()

# High-contrast palette for scatter — Morandi is too muted for distinguishing points
_SCATTER_PALETTE = [
    "#E20015",  # red
    "#2563eb",  # blue
    "#16a34a",  # green
    "#f97316",  # orange
    "#9333ea",  # purple
    "#06b6d4",  # cyan
    "#eab308",  # amber
    "#ec4899",  # pink
    "#14b8a6",  # teal
    "#f43f5e",  # rose
]


def render_scatter(spec: Dict[str, Any], output_path: str) -> str:
    title: Optional[str] = spec.get("title")
    x_label: str = spec.get("x_label", "X")
    y_label: str = spec.get("y_label", "Y")
    series: List[Dict[str, Any]] = spec.get("series", [])
    quadrant_labels: List[str] = spec.get("quadrant_labels", [])

    fw = float(spec.get("fig_width", 8.0))
    fh = float(spec.get("fig_height", 6.0))

    fig, ax = plt.subplots(figsize=(fw, fh))
    fig.patch.set_facecolor(THEME.BG)
    ax.set_facecolor(THEME.BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(THEME.BORDER)
    ax.spines["bottom"].set_color(THEME.BORDER)
    ax.tick_params(colors=THEME.INK, labelsize=THEME.FS_SMALL)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    ax.yaxis.grid(True, color=THEME.BORDER, linewidth=0.5, linestyle="--", alpha=0.6)
    ax.xaxis.grid(True, color=THEME.BORDER, linewidth=0.5, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    all_x, all_y = [], []
    # Scatter always uses high-contrast palette — spec colors are ignored so that
    # different data types always get clearly distinct colors.
    # • Multiple series → one palette color per series (all points in a series share it)
    # • Single series   → one palette color per point (each point is a different category)
    n_series = len(series)

    def _scatter_color(si: int, pi: int) -> str:
        idx = si if n_series > 1 else pi
        return _SCATTER_PALETTE[idx % len(_SCATTER_PALETTE)]

    # Collect per-series representative color for legend
    series_colors = [_scatter_color(si, 0) for si in range(n_series)]

    import math as _math
    point_counters = [0] * n_series
    for si, ser in enumerate(series):
        for pt in ser.get("points", []):
            color = _scatter_color(si, point_counters[si])
            point_counters[si] += 1
            size  = pt.get("size", 150)
            x, y  = pt.get("x", 0), pt.get("y", 0)
            all_x.append(x)
            all_y.append(y)
            ax.scatter([x], [y], s=size, color=color, alpha=0.82,
                       edgecolors="white", linewidths=1.2, zorder=4)
            lbl = pt.get("label", "")
            # Labels shown in legend below — no dot-side annotations

    # Quadrant lines at midpoints
    if all_x and all_y and quadrant_labels:
        mx = (max(all_x) + min(all_x)) / 2
        my = (max(all_y) + min(all_y)) / 2
        ax.axvline(mx, color=THEME.BORDER, linewidth=1.0, linestyle="--", zorder=1)
        ax.axhline(my, color=THEME.BORDER, linewidth=1.0, linestyle="--", zorder=1)
        pad = 0.04
        xl, xr = ax.get_xlim() if ax.get_xlim()[0] != ax.get_xlim()[1] else (min(all_x)-1, max(all_x)+1)
        yl, yu = ax.get_ylim() if ax.get_ylim()[0] != ax.get_ylim()[1] else (min(all_y)-1, max(all_y)+1)
        ax.set_xlim(xl, xr)
        ax.set_ylim(yl, yu)
        qlabels = (quadrant_labels + [""] * 4)[:4]
        positions = [
            (xl + (mx - xl) * 0.1, yu - (yu - my) * 0.1),   # top-left
            (mx + (xr - mx) * 0.1, yu - (yu - my) * 0.1),   # top-right
            (xl + (mx - xl) * 0.1, yl + (my - yl) * 0.1),   # bottom-left
            (mx + (xr - mx) * 0.1, yl + (my - yl) * 0.1),   # bottom-right
        ]
        for ql, (qx, qy) in zip(qlabels, positions):
            if ql:
                ax.text(qx, qy, ql, fontsize=THEME.FS_SMALL,
                        color=THEME.MUTED, style="italic", alpha=0.7)

    ax.set_xlabel(x_label, fontsize=THEME.FS_SMALL, color=THEME.INK, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=THEME.FS_SMALL, color=THEME.INK, fontweight="bold")

    # Legend: one entry per point using its label + color, size proportional to bubble size
    legend_handles = []
    seen: set = set()
    leg_counters = [0] * n_series
    for si, ser in enumerate(series):
        for pt in ser.get("points", []):
            lbl = pt.get("label", "") or ser.get("name", "")
            color = _scatter_color(si, leg_counters[si])
            leg_counters[si] += 1
            size = pt.get("size", 150)
            if lbl and lbl not in seen:
                seen.add(lbl)
                ms = max(7, min(14, (size ** 0.5) / 2))
                legend_handles.append(plt.Line2D([0], [0], marker="o", color="w",
                                                 markerfacecolor=color, markersize=ms,
                                                 label=lbl))
    if legend_handles:
        ncol = min(len(legend_handles), 3)
        ax.legend(handles=legend_handles, fontsize=THEME.FS_BODY,
                  frameon=True, framealpha=0.95, edgecolor=THEME.BORDER,
                  loc="upper left", bbox_to_anchor=(0.01, 0.99),
                  ncol=ncol, handlelength=1.2)

    if title:
        fig.text(0.5, 0.98, title, ha="center", va="top",
                 fontsize=THEME.FS_TITLE, color=THEME.INK, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.93 if title else 1.0], pad=0.4)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=THEME.DPI, bbox_inches="tight", facecolor=THEME.BG)
    pdf_path = str(output_path).replace(".png", ".pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor=THEME.BG)
    plt.close(fig)
    return f"Scatter chart saved: {output_path}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Render scatter/bubble chart PNG")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    src = args.spec.strip()
    spec = json.loads(src) if src.startswith("{") else json.load(open(src, encoding="utf-8"))
    print(render_scatter(spec, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
