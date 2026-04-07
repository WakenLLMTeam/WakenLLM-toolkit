#!/usr/bin/env python3
"""
Render a Gantt chart PNG.
Standalone — no DeerFlow dependency required.

Spec format:
{
  "type": "gantt",
  "title": "FSD Development Roadmap 2024-2027",
  "x_labels": ["Q1 24", "Q2 24", "Q3 24", "Q4 24", "Q1 25", "Q2 25", "Q3 25", "Q4 25", "2026", "2027"],
  "tasks": [
    {
      "label": "FSD v12 Rollout",
      "start": 0, "end": 2,
      "color": "#dbeafe",
      "milestone": false
    },
    {
      "label": "FSD v13 Development",
      "start": 1, "end": 5,
      "color": "#dcfce7"
    },
    {
      "label": "Cybercab Validation",
      "start": 3, "end": 7,
      "color": "#fef9c3"
    },
    {
      "label": "Robotaxi Launch",
      "start": 8, "end": 8,
      "color": "#E20015",
      "milestone": true
    }
  ],
  "groups": [
    {"label": "Software", "rows": [0, 1]},
    {"label": "Hardware", "rows": [2, 3]}
  ],
  "fig_width": 13,
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
import matplotlib.patches as mpatches
import numpy as np

from viz_theme import THEME, setup_matplotlib

setup_matplotlib()


def render_gantt(spec: Dict[str, Any], output_path: str) -> str:
    title: Optional[str] = spec.get("title")
    x_labels: List[str] = spec.get("x_labels", [])
    tasks: List[Dict[str, Any]] = spec.get("tasks", [])
    groups: List[Dict[str, Any]] = spec.get("groups", [])

    if not tasks:
        raise ValueError("gantt requires at least one task")

    n_tasks = len(tasks)
    n_cols = max(len(x_labels), max((t.get("end", 0) for t in tasks), default=0) + 1)
    if not x_labels:
        x_labels = [str(i) for i in range(n_cols)]

    fw = float(spec.get("fig_width", max(11.0, n_cols * 1.1)))
    fh = float(spec.get("fig_height", max(3.5, n_tasks * 0.65 + 1.5)))

    fig, ax = plt.subplots(figsize=(fw, fh))
    fig.patch.set_facecolor(THEME.BG)
    ax.set_facecolor(THEME.BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(THEME.BORDER)
    ax.tick_params(left=False, colors=THEME.MUTED, labelsize=THEME.FS_SMALL)
    ax.xaxis.grid(True, color=THEME.BORDER, linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    row_h = 0.6
    label_w = 0.22  # fraction of x-axis reserved for task labels
    total_cols = n_cols

    ax.set_xlim(-0.5, total_cols - 0.5)
    ax.set_ylim(-0.5, n_tasks - 0.5)
    ax.set_xticks(range(total_cols))
    ax.set_xticklabels(x_labels[:total_cols], fontsize=THEME.FS_SMALL,
                       color=THEME.INK, fontweight="bold")
    ax.set_yticks(range(n_tasks))
    ax.set_yticklabels([t.get("label", "") for t in reversed(tasks)],
                       fontsize=THEME.FS_SMALL, color=THEME.INK, fontweight="bold")

    for ti, task in enumerate(reversed(tasks)):
        start = task.get("start", 0)
        end = task.get("end", start)
        color = task.get("color", THEME.ACCENT_LIGHT)
        is_milestone = task.get("milestone", False)
        y = ti

        if is_milestone:
            # Diamond marker
            ax.plot(start, y, marker="D", markersize=10,
                    color=task.get("color", THEME.ACCENT),
                    markeredgecolor="white", markeredgewidth=1.2,
                    zorder=5)
            ax.text(start, y + 0.35, task.get("label", ""),
                    ha="center", va="bottom",
                    fontsize=THEME.FS_MICRO, color=THEME.INK, fontweight="bold")
        else:
            width = end - start + 0.8
            bar = mpatches.FancyBboxPatch(
                (start - 0.4, y - row_h / 2 + 0.04), width, row_h - 0.08,
                boxstyle="round,pad=0.04",
                facecolor=color, edgecolor=THEME.BORDER, linewidth=0.7,
                zorder=3
            )
            ax.add_patch(bar)

    # Group separators
    if groups:
        for g in groups:
            rows = g.get("rows", [])
            if rows:
                sep_y = n_tasks - min(rows) - 0.5
                ax.axhline(sep_y, color=THEME.BORDER, lw=0.8, ls="--")
                ax.text(-0.48, n_tasks - sum(rows) / len(rows) - 1,
                        g.get("label", ""),
                        ha="left", va="center",
                        fontsize=THEME.FS_MICRO, color=THEME.MUTED,
                        style="italic")

    if title:
        fig.text(0.5, 0.99, title, ha="center", va="top",
                 fontsize=THEME.FS_TITLE, color=THEME.INK, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.94 if title else 1.0], pad=0.4)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=THEME.DPI, bbox_inches="tight", facecolor=THEME.BG)
    pdf_path = str(output_path).replace(".png", ".pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor=THEME.BG)
    plt.close(fig)
    return f"Gantt chart saved: {output_path}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Render Gantt chart PNG")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    src = args.spec.strip()
    spec = json.loads(src) if src.startswith("{") else json.load(open(src, encoding="utf-8"))
    print(render_gantt(spec, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
