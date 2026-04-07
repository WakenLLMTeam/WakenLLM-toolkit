#!/usr/bin/env python3
"""
Render a 2x2 matrix (quadrant) PNG.
Standalone — no DeerFlow dependency required.

Spec format:
{
  "type": "matrix_2x2",
  "title": "Technology Priority Matrix",
  "x_label": "Market Readiness →",
  "y_label": "↑ Strategic Value",
  "quadrants": {
    "top_left":     {"label": "Invest", "color": "#fef9c3"},
    "top_right":    {"label": "Lead",   "color": "#dcfce7"},
    "bottom_left":  {"label": "Watch",  "color": "#f1f5f9"},
    "bottom_right": {"label": "Fast-follow", "color": "#dbeafe"}
  },
  "items": [
    {"label": "End-to-End AI", "x": 0.75, "y": 0.85, "color": "#E20015"},
    {"label": "V2X",           "x": 0.35, "y": 0.70, "color": "#2563eb"},
    {"label": "HD Map",        "x": 0.60, "y": 0.40, "color": "#f97316"},
    {"label": "Radar Fusion",  "x": 0.80, "y": 0.55, "color": "#16a34a"}
  ],
  "fig_width": 8,
  "fig_height": 7
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

from viz_theme import THEME, setup_matplotlib

setup_matplotlib()


def render_matrix_2x2(spec: Dict[str, Any], output_path: str) -> str:
    title: Optional[str] = spec.get("title")
    x_label: str = spec.get("x_label", "X →")
    y_label: str = spec.get("y_label", "↑ Y")
    quadrants: Dict[str, Any] = spec.get("quadrants", {})
    items: List[Dict[str, Any]] = spec.get("items", [])

    fw = float(spec.get("fig_width", 8.0))
    fh = float(spec.get("fig_height", 7.0))

    fig, ax = plt.subplots(figsize=(fw, fh))
    fig.patch.set_facecolor(THEME.BG)
    ax.set_facecolor(THEME.BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Quadrant backgrounds
    _DEFAULT_Q = {"label": "", "color": THEME.SURFACE}
    q_map = {
        "bottom_left":  (0.0, 0.0),
        "bottom_right": (0.5, 0.0),
        "top_left":     (0.0, 0.5),
        "top_right":    (0.5, 0.5),
    }
    for qkey, (qx, qy) in q_map.items():
        q = {**_DEFAULT_Q, **quadrants.get(qkey, {})}
        rect = mpatches.FancyBboxPatch(
            (qx + 0.005, qy + 0.005), 0.49, 0.49,
            boxstyle="round,pad=0.008",
            facecolor=q["color"], edgecolor=THEME.BORDER, linewidth=0.8,
            transform=ax.transAxes, zorder=1
        )
        ax.add_patch(rect)
        if q["label"]:
            lx = qx + 0.25
            ly = qy + 0.44
            ax.text(lx, ly, q["label"],
                    ha="center", va="top",
                    fontsize=THEME.FS_SMALL, color=THEME.MUTED,
                    fontweight="bold", style="italic",
                    transform=ax.transAxes, zorder=2)

    # Divider lines
    ax.plot([0.5, 0.5], [0.02, 0.98], color=THEME.BORDER, lw=1.2, zorder=3,
            transform=ax.transAxes)
    ax.plot([0.02, 0.98], [0.5, 0.5], color=THEME.BORDER, lw=1.2, zorder=3,
            transform=ax.transAxes)

    # Axis labels
    ax.text(0.5, 0.01, x_label, ha="center", va="bottom",
            fontsize=THEME.FS_SMALL, color=THEME.INK, fontweight="bold",
            transform=ax.transAxes)
    ax.text(0.01, 0.5, y_label, ha="left", va="center",
            fontsize=THEME.FS_SMALL, color=THEME.INK, fontweight="bold",
            rotation=90, transform=ax.transAxes)

    # Plot items
    for it in items:
        ix = float(it.get("x", 0.5)) * 0.96 + 0.02  # map [0,1] → [0.02, 0.98]
        iy = float(it.get("y", 0.5)) * 0.96 + 0.02
        color = it.get("color", THEME.ACCENT)
        label = it.get("label", "")
        ax.scatter([ix], [iy], s=160, color=color, zorder=5,
                   edgecolors="white", linewidths=1.2,
                   transform=ax.transAxes)
        ax.annotate(label, xy=(ix, iy), xytext=(6, 4),
                    textcoords="offset points",
                    fontsize=THEME.FS_SMALL, color=THEME.INK, fontweight="bold",
                    transform=ax.transAxes, zorder=6)

    if title:
        fig.text(0.5, 0.99, title, ha="center", va="top",
                 fontsize=THEME.FS_H1, color=THEME.INK, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95 if title else 1.0])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=THEME.DPI, bbox_inches="tight", facecolor=THEME.BG)
    pdf_path = str(output_path).replace(".png", ".pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor=THEME.BG)
    plt.close(fig)
    return f"2x2 Matrix saved: {output_path}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Render 2x2 matrix PNG")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    src = args.spec.strip()
    spec = json.loads(src) if src.startswith("{") else json.load(open(src, encoding="utf-8"))
    print(render_matrix_2x2(spec, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
