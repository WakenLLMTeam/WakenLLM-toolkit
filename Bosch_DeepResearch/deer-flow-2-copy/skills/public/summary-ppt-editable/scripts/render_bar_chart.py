#!/usr/bin/env python3
"""
Render a grouped or stacked bar chart PNG.
Standalone — no DeerFlow dependency required.

Spec format:
{
  "type": "bar_chart",
  "title": "季度营收对比",
  "mode": "grouped",        // "grouped" | "stacked" (default: grouped)
  "orientation": "vertical", // "vertical" | "horizontal" (default: vertical)
  "categories": ["Q1", "Q2", "Q3", "Q4"],
  "series": [
    {"name": "2023", "values": [42, 55, 61, 78], "color": "#2563eb"},
    {"name": "2024", "values": [50, 63, 70, 89], "color": "#E20015"}
  ],
  "unit": "亿元",           // label suffix (optional)
  "show_values": true,      // show value labels on bars (default: true)
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

from viz_theme import THEME, setup_matplotlib, get_categorical_palette

setup_matplotlib()


def render_bar_chart(spec: Dict[str, Any], output_path: str) -> str:
    title: Optional[str] = spec.get("title")
    mode = spec.get("mode", "grouped").lower()
    orientation = spec.get("orientation", "vertical").lower()
    categories: List[str] = spec.get("categories", [])
    series: List[Dict[str, Any]] = spec.get("series", [])
    unit: str = spec.get("unit", "")
    show_values: bool = spec.get("show_values", True)

    if not categories or not series:
        raise ValueError("bar_chart requires categories and series")

    fw = float(spec.get("fig_width", 11.0))
    fh = float(spec.get("fig_height", 5.0))

    fig, ax = plt.subplots(figsize=(fw, fh))
    fig.patch.set_facecolor(THEME.BG)
    ax.set_facecolor(THEME.BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(THEME.BORDER)
    ax.spines["bottom"].set_color(THEME.BORDER)
    ax.tick_params(colors=THEME.INK, labelsize=THEME.FS_SMALL, labelcolor=THEME.INK)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    ax.yaxis.grid(True, color=THEME.BORDER, linewidth=0.6, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    n_cat = len(categories)
    n_ser = len(series)
    x = np.arange(n_cat)
    morandi_colors = get_categorical_palette(n_ser)

    if mode == "stacked":
        bottoms = np.zeros(n_cat)
        for si, ser in enumerate(series):
            color = morandi_colors[si % len(morandi_colors)]
            vals = np.array(ser.get("values", [0] * n_cat), dtype=float)
            ser_label = ser.get("name") or f"Series {si + 1}"
            if orientation == "horizontal":
                bars = ax.barh(x, vals, left=bottoms, color=color,
                               label=ser_label, height=0.6, alpha=0.88)
            else:
                bars = ax.bar(x, vals, bottom=bottoms, color=color,
                              label=ser_label, width=0.6, alpha=0.88)
            if show_values:
                for bar, val in zip(bars, vals):
                    if val > 0:
                        if orientation == "horizontal":
                            ax.text(bar.get_x() + bar.get_width() / 2,
                                    bar.get_y() + bar.get_height() / 2,
                                    f"{val:g}{unit}", ha="center", va="center",
                                    fontsize=THEME.FS_MICRO, color="white", fontweight="bold")
                        else:
                            ax.text(bar.get_x() + bar.get_width() / 2,
                                    bar.get_y() + bar.get_height() / 2,
                                    f"{val:g}{unit}", ha="center", va="center",
                                    fontsize=THEME.FS_MICRO, color="white", fontweight="bold")
            bottoms += vals
    else:
        # grouped — bars within each category sit flush against each other
        # bar_gap: gap between individual bars (0 = no gap, default)
        # group_gap: fraction of x-unit left as space between category groups
        bar_gap    = float(spec.get("bar_gap", 0.0))       # gap between bars in same group
        group_gap  = float(spec.get("group_gap", 0.20))    # fraction of x reserved as inter-group space
        group_w    = 1.0 - group_gap                        # total bar area per x-unit
        bar_w      = (group_w - bar_gap * (n_ser - 1)) / n_ser  # width of each individual bar
        for si, ser in enumerate(series):
            color = morandi_colors[si % len(morandi_colors)]
            vals = np.array(ser.get("values", [0] * n_cat), dtype=float)
            ser_label = ser.get("name") or f"Series {si + 1}"
            # offset so the group is centred on x; bars touch each other
            offset = -group_w / 2 + si * (bar_w + bar_gap) + bar_w / 2
            if orientation == "horizontal":
                bars = ax.barh(x + offset, vals, color=color, height=bar_w,
                               label=ser_label, alpha=0.88)
            else:
                bars = ax.bar(x + offset, vals, color=color, width=bar_w,
                              label=ser_label, alpha=0.88)
            if show_values and n_ser <= 3:
                for bar, val in zip(bars, vals):
                    if orientation == "horizontal":
                        ax.text(bar.get_width() + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                                f"{val:g}{unit}", va="center", ha="left",
                                fontsize=THEME.FS_MICRO, color=THEME.INK)
                    else:
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + max(vals) * 0.01,
                                f"{val:g}{unit}", ha="center", va="bottom",
                                fontsize=THEME.FS_MICRO, color=THEME.INK)

    if orientation == "horizontal":
        ax.set_yticks(x)
        ax.set_yticklabels(categories, fontsize=THEME.FS_SMALL, color=THEME.INK)
        ax.set_xlabel(unit, fontsize=THEME.FS_SMALL, color=THEME.MUTED)
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=THEME.FS_SMALL, color=THEME.INK)
        if unit:
            ax.set_ylabel(unit, fontsize=THEME.FS_SMALL, color=THEME.MUTED)

    if n_ser > 1:
        # ── Smart legend placement: find the corner with most empty space ────
        # Build per-column height map so we can score each corner.
        all_vals = [np.array(s.get("values", [0] * n_cat), dtype=float) for s in series]
        if mode == "stacked":
            col_heights = np.sum(all_vals, axis=0)
        else:
            col_heights = np.max(all_vals, axis=0)

        ylo, yhi = ax.get_ylim()
        xlo, xhi = ax.get_xlim()
        y_span = max(yhi - ylo, 1e-9)
        x_span = max(xhi - xlo, 1e-9)

        if orientation == "vertical":
            # Vertical bars always start at 0, so lower corners have no free space.
            # Only compare upper-left vs upper-right (space above tallest bar on each side).
            mid = max(1, len(col_heights) // 2)
            left_max  = col_heights[:mid].max()
            right_max = col_heights[mid:].max()
            scores = {
                "upper left":  (yhi - left_max)  / y_span,
                "upper right": (yhi - right_max) / y_span,
            }
        else:  # horizontal bars — left side is category labels, right side is value space
            x_max_all = col_heights.max()
            # right side has free space beyond longest bar; left side is occupied by labels
            scores = {
                "upper right": (xhi - x_max_all) / x_span,
                "lower right": (xhi - x_max_all) / x_span,
            }

        best_loc = max(scores, key=scores.get)

        leg = ax.legend(
            fontsize=THEME.FS_SMALL,
            frameon=True,
            framealpha=0.72,            # semi-transparent background
            facecolor=THEME.BG,
            edgecolor=THEME.BORDER,
            loc=best_loc,
            borderpad=0.6,
            labelspacing=0.4,
            handlelength=1.4,
            handletextpad=0.5,
            borderaxespad=0.8,
        )
        leg.get_frame().set_linewidth(0.7)

    if title:
        fig.text(0.5, 0.98, title, ha="center", va="top",
                 fontsize=THEME.FS_TITLE, color=THEME.INK, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.93 if title else 1.0], pad=0.4)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=THEME.DPI, bbox_inches="tight", facecolor=THEME.BG)
    pdf_path = str(output_path).replace(".png", ".pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor=THEME.BG)
    plt.close(fig)
    return f"Bar chart saved: {output_path}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Render bar chart PNG")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    src = args.spec.strip()
    spec = json.loads(src) if src.startswith("{") else json.load(open(src, encoding="utf-8"))
    print(render_bar_chart(spec, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
