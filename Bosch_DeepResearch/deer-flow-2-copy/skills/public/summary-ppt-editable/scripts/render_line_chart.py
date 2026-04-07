#!/usr/bin/env python3
"""
Render a single or multi-line chart PNG with optional confidence band.
Standalone — no DeerFlow dependency required.

Spec format:
{
  "type": "line_chart",
  "title": "FSD里程累计增长趋势",
  "x_labels": ["2019", "2020", "2021", "2022", "2023", "2024"],
  "series": [
    {
      "name": "累计里程(亿英里)",
      "values": [1.0, 2.5, 5.0, 10.0, 22.0, 50.0],
      "color": "#E20015",
      "band": [0.8, 1.2, 2.1, 2.9, 8.5, 12.5, 20.0, 24.0, 46.0, 55.0],
      // band: pairs [lower0, upper0, lower1, upper1 ...] for confidence interval
      "marker": "o",   // "o" | "s" | "^" | "none" (default: "o")
      "linestyle": "-" // "-" | "--" | ":" (default: "-")
    }
  ],
  "unit": "亿英里",
  "show_grid": true,
  "log_scale": false,
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

_DEFAULT_COLORS = [
    THEME.ACCENT, "#16a34a", "#f97316", "#8b5cf6",
    "#0891b2", "#dc2626", "#65a30d", "#d97706",
]


def render_line_chart(spec: Dict[str, Any], output_path: str) -> str:
    title: Optional[str] = spec.get("title")
    x_labels: List[str] = spec.get("x_labels", [])
    series: List[Dict[str, Any]] = spec.get("series", [])
    unit: str = spec.get("unit", "")
    show_grid: bool = spec.get("show_grid", True)
    log_scale: bool = spec.get("log_scale", False)

    if not x_labels or not series:
        raise ValueError("line_chart requires x_labels and series")

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

    if show_grid:
        ax.yaxis.grid(True, color=THEME.BORDER, linewidth=0.6, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    if log_scale:
        ax.set_yscale("log")

    x = np.arange(len(x_labels))

    for si, ser in enumerate(series):
        color = ser.get("color", _DEFAULT_COLORS[si % len(_DEFAULT_COLORS)])
        vals = np.array(ser.get("values", []), dtype=float)
        name = ser.get("name", "")
        marker = ser.get("marker", "o")
        ls = ser.get("linestyle", "-")

        if marker == "none":
            marker = None

        ax.plot(x, vals, color=color, linewidth=2.2, marker=marker,
                markersize=6, label=name, linestyle=ls,
                markerfacecolor="white", markeredgewidth=1.8, zorder=4)

        # Confidence band
        band = ser.get("band", [])
        if len(band) == len(x) * 2:
            lower = np.array(band[::2])
            upper = np.array(band[1::2])
            ax.fill_between(x, lower, upper, color=color, alpha=0.12, zorder=2)

        # Annotate last point
        if len(vals):
            suffix = f" {unit}" if unit else ""
            ax.annotate(f"{vals[-1]:g}{suffix}",
                        xy=(x[-1], vals[-1]),
                        xytext=(6, 0), textcoords="offset points",
                        va="center", ha="left",
                        fontsize=THEME.FS_SMALL, color=color, fontweight="bold")

    # Extend x-axis right margin so end-of-line annotations don't get clipped
    if series:
        max_label_len = max(
            len(f"{s.get('values', [0])[-1]:g} {unit}") if s.get('values') else 0
            for s in series
        )
        x_pad = max(0.3, max_label_len * 0.055)
        ax.set_xlim(x[0] - 0.3, x[-1] + x_pad)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=THEME.FS_SMALL, color=THEME.INK, fontweight="bold")
    if unit and not log_scale:
        ax.set_ylabel(unit, fontsize=THEME.FS_SMALL, color=THEME.MUTED, fontweight='bold')

    if len(series) > 1:
        ax.legend(fontsize=THEME.FS_SMALL, frameon=True, framealpha=0.9,
                  edgecolor=THEME.BORDER, loc="upper left")

    if title:
        fig.text(0.5, 0.98, title, ha="center", va="top",
                 fontsize=THEME.FS_TITLE, color=THEME.INK, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.93 if title else 1.0], pad=0.4)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=THEME.DPI, bbox_inches="tight", facecolor=THEME.BG)
    pdf_path = str(output_path).replace(".png", ".pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor=THEME.BG)
    plt.close(fig)
    return f"Line chart saved: {output_path}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Render line chart PNG")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    src = args.spec.strip()
    spec = json.loads(src) if src.startswith("{") else json.load(open(src, encoding="utf-8"))
    print(render_line_chart(spec, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
