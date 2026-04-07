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

from viz_theme import THEME, setup_matplotlib, fit_fontsize

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
    # Max safe spread = half the angle between adjacent branches, with a cap
    branch_gap = 2 * math.pi / max(n, 1)
    max_spread = branch_gap * 0.38   # never use more than 38% of branch gap

    R_branch = 0.42   # center → branch node
    R_child  = 0.82   # center → child node

    # ── Center node — size adapts to label length ────────────────────────────
    center_lines = center_label.replace("\\n", "\n").split("\n")
    max_center_chars = max(len(l) for l in center_lines)
    n_center_lines   = len(center_lines)
    # Radius: at least 0.13, grows with label
    R_c = max(0.13, max_center_chars * 0.012 + n_center_lines * 0.025)
    center_circle = plt.Circle((0, 0), R_c, color=THEME.ACCENT,
                                 zorder=5, ec="white", lw=1.5)
    ax.add_patch(center_circle)
    c_fs = fit_fontsize(center_label, R_c * 2 * fw * 0.82, R_c * 2 * fh * 0.82,
                        start_pt=THEME.FS_BODY + 2, min_pt=8.0)
    ax.text(0, 0, center_label.replace("\\n", "\n"), ha="center", va="center",
            fontsize=c_fs, color="white", fontweight="bold",
            zorder=6, multialignment="center")

    for bi, branch in enumerate(branches):
        angle = base_angles[bi]
        bx = R_branch * math.cos(angle)
        by = R_branch * math.sin(angle)
        color = branch.get("color", _DEFAULT_BRANCH_COLORS[bi % len(_DEFAULT_BRANCH_COLORS)])
        label = branch.get("label", "")
        children: List[str] = branch.get("children", [])

        # Line: center → branch — solid black
        ax.plot([0, bx], [0, by], color="black", lw=1.8, zorder=1)

        # Branch node — width adapts to label length
        b_chars = len(label)
        node_w = max(0.20, b_chars * 0.018 + 0.06)
        node_h = 0.095
        rect = mpatches.FancyBboxPatch(
            (bx - node_w / 2, by - node_h / 2), node_w, node_h,
            boxstyle="round,pad=0.015",
            facecolor=color, edgecolor=THEME.ACCENT, linewidth=1.5,
            transform=ax.transData, zorder=4
        )
        ax.add_patch(rect)
        b_fs = fit_fontsize(label, node_w * fw * 0.85, node_h * fh * 0.85,
                             start_pt=THEME.FS_BODY)
        ax.text(bx, by, label, ha="center", va="center",
                fontsize=b_fs, color=THEME.INK, fontweight="bold", zorder=5)

        # Children
        nc = len(children)
        if nc == 0:
            continue
        # Constrain spread so children of adjacent branches never overlap
        if nc == 1:
            spread = 0.0
        else:
            ideal = math.pi / 6 if nc <= 2 else math.pi / 5
            spread = min(ideal, max_spread)
        child_angles = [angle + spread * (i - (nc - 1) / 2) for i in range(nc)]
        for ci, (child_label, cangle) in enumerate(zip(children, child_angles)):
            cx = R_child * math.cos(cangle)
            cy = R_child * math.sin(cangle)
            # Line: branch → child — solid black
            ax.plot([bx, cx], [by, cy], color="black",
                    lw=1.2, linestyle="-", zorder=2)
            # Child node — width adapts to label length
            c_chars = len(child_label)
            child_w = max(0.18, c_chars * 0.016 + 0.05)
            child_h = 0.072
            crect = mpatches.FancyBboxPatch(
                (cx - child_w / 2, cy - child_h / 2), child_w, child_h,
                boxstyle="round,pad=0.012",
                facecolor=THEME.BG, edgecolor="black", linewidth=1.0,
                transform=ax.transData, zorder=3
            )
            ax.add_patch(crect)
            ax.text(cx, cy, child_label, ha="center", va="center",
                    fontsize=fit_fontsize(child_label, child_w * fw * 0.80, child_h * fh * 0.80,
                                             start_pt=THEME.FS_SMALL),
                    color=THEME.INK, fontweight="bold", zorder=4,
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
