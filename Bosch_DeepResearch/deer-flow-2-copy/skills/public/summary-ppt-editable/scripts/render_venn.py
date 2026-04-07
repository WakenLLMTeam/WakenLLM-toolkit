#!/usr/bin/env python3
"""
Render a Venn diagram PNG (2 or 3 circles).
Standalone — no DeerFlow dependency required.

Spec format:
{
  "type": "venn",
  "title": "Sensor Modality Overlap",
  "circles": [
    {"label": "Camera",  "color": "#dbeafe", "items": ["Color", "Lane Mark", "Sign"]},
    {"label": "Radar",   "color": "#dcfce7", "items": ["Velocity", "Metal Detect", "Rain OK"]},
    {"label": "LiDAR",   "color": "#fef9c3", "items": ["3D Point", "Precise Depth"]}
  ],
  "overlaps": [
    {"circles": [0, 1], "label": "Distance+Speed"},
    {"circles": [0, 2], "label": "Object Shape"},
    {"circles": [1, 2], "label": "Range Detect"},
    {"circles": [0, 1, 2], "label": "Full\nAwareness"}
  ],
  "fig_width": 9,
  "fig_height": 7
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

from viz_theme import THEME, setup_matplotlib

setup_matplotlib()


def render_venn(spec: Dict[str, Any], output_path: str) -> str:
    title: Optional[str] = spec.get("title")
    circles: List[Dict[str, Any]] = spec.get("circles", [])
    overlaps: List[Dict[str, Any]] = spec.get("overlaps", [])
    fw = float(spec.get("fig_width", 9.0))
    fh = float(spec.get("fig_height", 7.0))

    n = len(circles)
    if n not in (2, 3):
        raise ValueError("venn supports 2 or 3 circles only")

    fig, ax = plt.subplots(figsize=(fw, fh))
    fig.patch.set_facecolor(THEME.BG)
    ax.set_facecolor(THEME.BG)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")

    R = 0.65  # circle radius
    alpha = 0.35

    if n == 2:
        centers = [(-0.4, 0.0), (0.4, 0.0)]
    else:
        # Equilateral triangle arrangement
        r_arr = 0.45
        centers = [
            (r_arr * math.cos(math.pi / 2 + 2 * math.pi * i / 3),
             r_arr * math.sin(math.pi / 2 + 2 * math.pi * i / 3))
            for i in range(3)
        ]

    # Draw circles
    for ci, (circle, (cx, cy)) in enumerate(zip(circles, centers)):
        color = circle.get("color", THEME.SURFACE)
        patch = plt.Circle((cx, cy), R, color=color, alpha=alpha,
                            ec=THEME.BORDER, lw=1.2, zorder=2)
        ax.add_patch(patch)

        # Circle label (outside)
        label_angle = math.pi / 2 + 2 * math.pi * ci / n if n == 3 else (math.pi if ci == 0 else 0)
        lx = cx + (R + 0.22) * math.cos(label_angle)
        ly = cy + (R + 0.22) * math.sin(label_angle)
        ax.text(lx, ly, circle.get("label", ""),
                ha="center", va="center",
                fontsize=THEME.FS_BODY, color=THEME.INK, fontweight="bold")

        # Items inside the non-overlapping region (push outward)
        items = circle.get("items", [])
        for ii, item in enumerate(items[:3]):
            ix = cx + 0.28 * math.cos(label_angle)
            iy = cy + 0.28 * math.sin(label_angle) - ii * 0.15 + (len(items[:3]) - 1) * 0.075
            ax.text(ix, iy, f"· {item}",
                    ha="center", va="center",
                    fontsize=THEME.FS_MICRO, color=THEME.INK)

    # Overlap labels
    overlap_positions = {
        (0, 1): (0.0, 0.0) if n == 2 else ((centers[0][0] + centers[1][0]) / 2,
                                             (centers[0][1] + centers[1][1]) / 2),
    }
    if n == 3:
        overlap_positions[(0, 1)] = ((centers[0][0] + centers[1][0]) / 2,
                                     (centers[0][1] + centers[1][1]) / 2)
        overlap_positions[(0, 2)] = ((centers[0][0] + centers[2][0]) / 2,
                                     (centers[0][1] + centers[2][1]) / 2)
        overlap_positions[(1, 2)] = ((centers[1][0] + centers[2][0]) / 2,
                                     (centers[1][1] + centers[2][1]) / 2)
        overlap_positions[(0, 1, 2)] = (
            sum(c[0] for c in centers) / 3,
            sum(c[1] for c in centers) / 3,
        )

    for ov in overlaps:
        idxs = tuple(sorted(ov.get("circles", [])))
        pos = overlap_positions.get(idxs)
        if pos:
            ax.text(pos[0], pos[1], ov.get("label", ""),
                    ha="center", va="center",
                    fontsize=THEME.FS_MICRO + 0.5, color=THEME.INK,
                    fontweight="bold", multialignment="center",
                    zorder=5)

    if title:
        fig.text(0.5, 0.99, title, ha="center", va="top",
                 fontsize=THEME.FS_TITLE, color=THEME.INK, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95 if title else 1.0])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=THEME.DPI, bbox_inches="tight", facecolor=THEME.BG)
    pdf_path = str(output_path).replace(".png", ".pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor=THEME.BG)
    plt.close(fig)
    return f"Venn diagram saved: {output_path}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Render Venn diagram PNG")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    src = args.spec.strip()
    spec = json.loads(src) if src.startswith("{") else json.load(open(src, encoding="utf-8"))
    print(render_venn(spec, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
