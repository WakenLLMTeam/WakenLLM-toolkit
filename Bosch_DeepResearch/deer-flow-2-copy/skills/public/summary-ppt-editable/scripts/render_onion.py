#!/usr/bin/env python3
"""
Render a concentric onion (layer) diagram PNG.
Standalone — no DeerFlow dependency required.

Spec format:
{
  "type": "onion",
  "title": "FSD Value Proposition Layers",
  "layers": [
    {"label": "Safety",      "color": "#fee2e2", "description": "Zero-fatality goal"},
    {"label": "Autonomy",    "color": "#fef9c3", "description": "L2+ to L4 journey"},
    {"label": "Experience",  "color": "#dcfce7", "description": "Stress-free commute"},
    {"label": "Ecosystem",   "color": "#dbeafe", "description": "Robotaxi network"}
  ],
  "center_label": "FSD",
  "fig_width": 8,
  "fig_height": 7
}
Note: layers[0] is innermost, layers[-1] is outermost.
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


def render_onion(spec: Dict[str, Any], output_path: str) -> str:
    title: Optional[str] = spec.get("title")
    layers: List[Dict[str, Any]] = spec.get("layers", [])
    center_label: str = spec.get("center_label", "Core")
    fw = float(spec.get("fig_width", 8.0))
    fh = float(spec.get("fig_height", 7.0))

    if not layers:
        raise ValueError("onion requires at least one layer")

    fig, ax = plt.subplots(figsize=(fw, fh))
    fig.patch.set_facecolor(THEME.BG)
    ax.set_facecolor(THEME.BG)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1.18, 1)  # extra bottom space for stacked labels
    ax.set_aspect("equal")
    ax.axis("off")

    n = len(layers)
    # Outermost radius = 0.88, innermost ring width
    r_outer = 0.88
    r_center = 0.14
    ring_w = (r_outer - r_center) / n

    # Draw from outside in
    for i, layer in enumerate(reversed(layers)):
        r = r_outer - i * ring_w
        color = layer.get("color", THEME.SURFACE)
        circle = plt.Circle((0, 0), r, color=color,
                             ec=THEME.BORDER, lw=0.8,
                             zorder=i + 2)
        ax.add_patch(circle)

    # Center circle
    center_c = plt.Circle((0, 0), r_center, color=THEME.ACCENT,
                           ec="white", lw=1.2, zorder=n + 5)
    ax.add_patch(center_c)
    ax.text(0, 0, center_label, ha="center", va="center",
            fontsize=THEME.FS_SMALL, color="white", fontweight="bold",
            zorder=n + 6)

    # All labels placed inside their own ring band at the 6-o'clock (bottom) position.
    # Each label is at (0, -r_mid) where r_mid is the midpoint radius of that band.
    # High zorder ensures text renders above all circle patches.
    line_gap = 0.055   # vertical space between bold label and description

    for i, layer in enumerate(layers):
        r_inner  = r_center + i * ring_w
        r_outer_i = r_center + (i + 1) * ring_w
        r_mid    = (r_inner + r_outer_i) / 2

        label = layer.get("label", "")
        desc  = layer.get("description", "")

        lx        = 0.0
        ly_label  = -r_mid + line_gap * 0.5    # label slightly above band midpoint
        ly_desc   = -r_mid - line_gap * 0.5    # description slightly below

        ax.text(lx, ly_label, label,
                ha="center", va="center",
                fontsize=THEME.FS_SMALL, color=THEME.INK,
                fontweight="bold", zorder=n + 10)
        if desc:
            ax.text(lx, ly_desc, desc,
                    ha="center", va="center",
                    fontsize=THEME.FS_MICRO, color=THEME.MUTED,
                    fontweight="normal", zorder=n + 10)

    if title:
        fig.text(0.5, 0.99, title, ha="center", va="top",
                 fontsize=THEME.FS_TITLE, color=THEME.INK, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95 if title else 1.0])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=THEME.DPI, bbox_inches="tight", facecolor=THEME.BG)
    pdf_path = str(output_path).replace(".png", ".pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor=THEME.BG)
    plt.close(fig)
    return f"Onion diagram saved: {output_path}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Render onion/concentric diagram PNG")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    src = args.spec.strip()
    spec = json.loads(src) if src.startswith("{") else json.load(open(src, encoding="utf-8"))
    print(render_onion(spec, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
