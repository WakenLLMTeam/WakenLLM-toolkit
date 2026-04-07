#!/usr/bin/env python3
"""
Render a SWOT analysis diagram PNG.
Standalone — no DeerFlow dependency required.

Spec format:
{
  "type": "swot",
  "title": "Tesla FSD Strategic SWOT Analysis",
  "subject": "Tesla FSD",
  "quadrants": {
    "strengths": {
      "color": "#dcfce7",
      "items": ["6M vehicle fleet data advantage", "End-to-end AI leadership", "OTA update capability"]
    },
    "weaknesses": {
      "color": "#fee2e2",
      "items": ["No LiDAR redundancy", "Regulatory approval pending", "High R&D cost structure"]
    },
    "opportunities": {
      "color": "#dbeafe",
      "items": ["Robotaxi market $1T+", "FSD licensing to OEMs", "China market expansion"]
    },
    "threats": {
      "color": "#fef9c3",
      "items": ["Waymo L4 competition", "NHTSA recall risk", "Huawei ADS in China market"]
    }
  },
  "fig_width": 11,
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

_SWOT_LABELS = {
    "strengths":     ("Strengths",     "Internal · Positive"),
    "weaknesses":    ("Weaknesses",    "Internal · Negative"),
    "opportunities": ("Opportunities", "External · Positive"),
    "threats":       ("Threats",       "External · Negative"),
}
_SWOT_POSITIONS = {
    "strengths":     (0, 1),   # col, row  (0-indexed, row 0 = top)
    "weaknesses":    (1, 1),
    "opportunities": (0, 0),
    "threats":       (1, 0),
}
_SWOT_HEADER_COLORS = {
    "strengths":     "#16a34a",
    "weaknesses":    "#dc2626",
    "opportunities": "#2563eb",
    "threats":       "#d97706",
}


def render_swot(spec: Dict[str, Any], output_path: str) -> str:
    title: Optional[str] = spec.get("title")
    subject: str = spec.get("subject", "")
    quadrants: Dict[str, Any] = spec.get("quadrants", {})
    fw = float(spec.get("fig_width", 11.0))
    fh = float(spec.get("fig_height", 7.0))

    fig, ax = plt.subplots(figsize=(fw, fh))
    fig.patch.set_facecolor(THEME.BG)
    ax.set_facecolor(THEME.BG)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    gap = 0.012
    cell_w = (1 - gap * 3) / 2
    cell_h = (0.88 - gap * 3) / 2
    y_start = 0.06

    for key, (col, row) in _SWOT_POSITIONS.items():
        q = quadrants.get(key, {})
        color = q.get("color", THEME.SURFACE)
        items: List[str] = q.get("items", [])
        header_color = _SWOT_HEADER_COLORS.get(key, THEME.ACCENT)
        label, subtitle = _SWOT_LABELS.get(key, (key.title(), ""))

        bx = gap + col * (cell_w + gap)
        by = y_start + gap + row * (cell_h + gap)

        # Background
        rect = mpatches.FancyBboxPatch(
            (bx, by), cell_w, cell_h,
            boxstyle="round,pad=0.008",
            facecolor=color, edgecolor=THEME.BORDER, linewidth=0.8,
            transform=ax.transAxes, zorder=1
        )
        ax.add_patch(rect)

        # Header bar
        header_h = cell_h * 0.16
        hrect = mpatches.FancyBboxPatch(
            (bx, by + cell_h - header_h), cell_w, header_h,
            boxstyle="round,pad=0.005",
            facecolor=header_color, edgecolor="none",
            transform=ax.transAxes, zorder=2
        )
        ax.add_patch(hrect)

        # Header text
        ax.text(bx + cell_w / 2, by + cell_h - header_h / 2,
                label, ha="center", va="center",
                fontsize=THEME.FS_BODY, color="white",
                fontweight="bold", transform=ax.transAxes, zorder=3)

        # subtitle removed per user request

        # Items
        for ii, item in enumerate(items[:5]):
            iy = by + cell_h - header_h - 0.035 - ii * (cell_h * 0.162)
            ax.text(bx + 0.014, iy,
                    f"• {item}",
                    ha="left", va="top",
                    fontsize=THEME.FS_BODY, color=THEME.INK,
                    fontweight="bold",
                    transform=ax.transAxes, zorder=3)

    # subject and axis labels intentionally removed for cleaner look

    if title:
        fig.text(0.5, 0.99, title, ha="center", va="top",
                 fontsize=THEME.FS_TITLE, color=THEME.INK, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95 if title else 1.0])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=THEME.DPI, bbox_inches="tight", facecolor=THEME.BG)
    pdf_path = str(output_path).replace(".png", ".pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor=THEME.BG)
    plt.close(fig)
    return f"SWOT diagram saved: {output_path}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Render SWOT diagram PNG")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    src = args.spec.strip()
    spec = json.loads(src) if src.startswith("{") else json.load(open(src, encoding="utf-8"))
    print(render_swot(spec, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
