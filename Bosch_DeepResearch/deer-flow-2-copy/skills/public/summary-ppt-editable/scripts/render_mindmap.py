#!/usr/bin/env python3
"""
Render a radial mind map PNG.
Standalone — no DeerFlow dependency required.

Spec format:
{
  "type": "mindmap",
  "center": "FSD Strategy",
  "branches": [
    {
      "label": "Perception",
      "color": "#dbeafe",
      "children": ["Camera Vision", "BEV Fusion", "OccupancyNet"]
    },
    {
      "label": "Planning",
      "color": "#dcfce7",
      "children": ["End-to-End AI", "Behavior Predict", "Safety Guard"]
    },
    {
      "label": "Data Engine",
      "color": "#fef9c3",
      "children": ["6M Fleet", "Shadow Mode", "Auto Label"]
    },
    {
      "label": "Hardware",
      "color": "#f3e8ff",
      "children": ["HW4 Chip", "Dojo Cluster", "Redundant MCU"]
    }
  ],
  "fig_width": 10,
  "fig_height": 8
}
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
import matplotlib.patches as mpatches
import numpy as np

from viz_theme import THEME, setup_matplotlib

setup_matplotlib()

_DEFAULT_BRANCH_COLORS = [
    "#dbeafe", "#dcfce7", "#fef9c3", "#f3e8ff",
    "#fce7f3", "#ffedd5", "#ecfdf5", "#ede9fe",
]


def render_mindmap(spec: Dict[str, Any], output_path: str) -> str:
    center_label: str = spec.get("center", "Topic")
    branches: List[Dict[str, Any]] = spec.get("branches", [])
    fw = float(spec.get("fig_width", 10.0))
    fh = float(spec.get("fig_height", 8.0))

    if not branches:
        raise ValueError("mindmap requires at least one branch")

    fig, ax = plt.subplots(figsize=(fw, fh))
    fig.patch.set_facecolor(THEME.BG)
    ax.set_facecolor(THEME.BG)
    ax.axis("off")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")

    n = len(branches)
    # Distribute branches evenly, starting from top
    base_angles = [math.pi / 2 - 2 * math.pi * i / n for i in range(n)]

    R_branch = 0.42   # center → branch node
    R_child  = 0.72   # center → child node

    # Draw center node
    center_circle = plt.Circle((0, 0), 0.13, color=THEME.ACCENT,
                                 zorder=5, ec="white", lw=1.5)
    ax.add_patch(center_circle)
    ax.text(0, 0, center_label, ha="center", va="center",
            fontsize=THEME.FS_BODY, color="white", fontweight="bold",
            zorder=6, multialignment="center",
            wrap=True)

    for bi, branch in enumerate(branches):
        angle = base_angles[bi]
        bx = R_branch * math.cos(angle)
        by = R_branch * math.sin(angle)
        color = branch.get("color", _DEFAULT_BRANCH_COLORS[bi % len(_DEFAULT_BRANCH_COLORS)])
        label = branch.get("label", "")
        children: List[str] = branch.get("children", [])

        # Line: center → branch
        ax.plot([0, bx], [0, by], color=THEME.BORDER, lw=1.5, zorder=1)

        # Branch node (rounded rectangle via FancyBboxPatch)
        node_w, node_h = 0.20, 0.09
        rect = mpatches.FancyBboxPatch(
            (bx - node_w / 2, by - node_h / 2), node_w, node_h,
            boxstyle="round,pad=0.015",
            facecolor=color, edgecolor=THEME.ACCENT, linewidth=1.2,
            transform=ax.transData, zorder=4
        )
        ax.add_patch(rect)
        ax.text(bx, by, label, ha="center", va="center",
                fontsize=THEME.FS_SMALL, color=THEME.INK, fontweight="bold", zorder=5)

        # Children
        nc = len(children)
        if nc == 0:
            continue
        # Spread children around the branch angle, ±spread
        spread = math.pi / 6 if nc <= 2 else math.pi / 4
        child_angles = [angle + spread * (i - (nc - 1) / 2) for i in range(nc)]
        for ci, (child_label, cangle) in enumerate(zip(children, child_angles)):
            cx = R_child * math.cos(cangle)
            cy = R_child * math.sin(cangle)
            # Line: branch → child
            ax.plot([bx, cx], [by, cy], color=color,
                    lw=1.0, linestyle="--", zorder=2, alpha=0.8)
            # Child node (small pill)
            child_w, child_h = 0.18, 0.065
            crect = mpatches.FancyBboxPatch(
                (cx - child_w / 2, cy - child_h / 2), child_w, child_h,
                boxstyle="round,pad=0.012",
                facecolor=THEME.BG, edgecolor=color, linewidth=0.9,
                transform=ax.transData, zorder=3
            )
            ax.add_patch(crect)
            ax.text(cx, cy, child_label, ha="center", va="center",
                    fontsize=THEME.FS_MICRO + 0.5, color=THEME.INK, zorder=4,
                    multialignment="center")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=THEME.DPI, bbox_inches="tight", facecolor=THEME.BG)
    pdf_path = str(output_path).replace(".png", ".pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor=THEME.BG)
    plt.close(fig)
    return f"Mind map saved: {output_path}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Render mind map PNG")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    src = args.spec.strip()
    spec = json.loads(src) if src.startswith("{") else json.load(open(src, encoding="utf-8"))
    print(render_mindmap(spec, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
