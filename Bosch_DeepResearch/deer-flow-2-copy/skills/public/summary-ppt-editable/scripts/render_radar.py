#!/usr/bin/env python3
"""
Render a radar / spider chart for competitor capability comparison.
Standalone — no DeerFlow dependency required.

Designed for Bosch ADAS context: comparing players like Mobileye, Waymo,
Huawei, Continental, Bosch across multiple capability dimensions.

Usage:
  python render_radar.py --spec radar.json --output out.png

Spec format:
{
  "type": "radar",
  "title": "ADAS Capability Comparison",
  "dimensions": ["Perception", "Planning", "V2X", "Safety", "Cost"],
  "players": [
    {"name": "Bosch",      "scores": [8, 7, 6, 9, 7], "color": "#E20015", "highlight": true},
    {"name": "Mobileye",   "scores": [9, 8, 5, 8, 6], "color": "#2563eb"},
    {"name": "Waymo",      "scores": [9, 9, 7, 9, 3], "color": "#16a34a"},
    {"name": "Huawei",     "scores": [7, 8, 9, 7, 8], "color": "#f97316"}
  ],
  "score_range": [0, 10],
  "fig_width": 7,
  "fig_height": 7
}

Fields:
  dimensions      Capability axes (3-8 recommended)
  players[].name  Player label
  players[].scores One score per dimension (must match len(dimensions))
  players[].color Hex color for this player's polygon
  players[].highlight  If true, draws thicker line + filled polygon (for "our" player)
  score_range     [min, max] for the radar scale (default [0, 10])
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from viz_theme import THEME, setup_matplotlib

setup_matplotlib()


def _hex_to_rgba(hex_color: str, alpha: float):
    h = hex_color.lstrip("#")
    if len(h) < 6:
        return (0.1, 0.4, 0.9, alpha)
    r, g, b = int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255
    return (r, g, b, alpha)


def render_radar(spec: Dict[str, Any], output_path: str) -> str:
    dimensions: List[str] = spec.get("dimensions", [])
    players: List[Dict[str, Any]] = spec.get("players", [])
    title: Optional[str] = spec.get("title")
    score_min, score_max = spec.get("score_range", [0, 10])

    if len(dimensions) < 3:
        raise ValueError("radar requires at least 3 dimensions")
    if not players:
        raise ValueError("radar requires at least 1 player")

    n = len(dimensions)
    fw = float(spec.get("fig_width", 7.0))
    fh = float(spec.get("fig_height", 7.0))

    # Angles for each axis (evenly spaced, starting from top)
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False).tolist()
    angles_closed = angles + [angles[0]]  # close the polygon

    fig, ax = plt.subplots(figsize=(fw, fh), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(THEME.BG)
    ax.set_facecolor(THEME.BG)

    # ── Grid rings ────────────────────────────────────────────────────────────
    num_rings = 5
    ring_vals = np.linspace(score_min, score_max, num_rings + 1)[1:]
    for rv in ring_vals:
        ring_norm = (rv - score_min) / (score_max - score_min)
        ring_pts = [ring_norm] * n + [ring_norm]
        ax.plot(angles_closed, ring_pts, color=THEME.BORDER, lw=0.7, zorder=1)
        ax.fill(angles_closed, ring_pts, color=THEME.BORDER, alpha=0.04)

    # ── Axis spokes ───────────────────────────────────────────────────────────
    for angle in angles:
        ax.plot([angle, angle], [0, 1.0], color=THEME.BORDER, lw=0.8, zorder=1)

    # ── Player polygons ───────────────────────────────────────────────────────
    for player in players:
        scores = player.get("scores", [])
        if len(scores) != n:
            continue  # skip mismatched entries
        color = player.get("color", THEME.ACCENT)
        highlight = bool(player.get("highlight", False))

        # Normalise scores to [0, 1]
        norm = [(s - score_min) / (score_max - score_min) for s in scores]
        norm_closed = norm + [norm[0]]

        lw = 2.4 if highlight else 1.5
        alpha_fill = 0.18 if highlight else 0.08
        zorder = 5 if highlight else 3

        ax.plot(angles_closed, norm_closed,
                color=color, lw=lw, zorder=zorder, solid_capstyle="round")
        ax.fill(angles_closed, norm_closed,
                color=color, alpha=alpha_fill, zorder=zorder - 1)

        # Score dots
        dot_size = 7 if highlight else 5
        ax.scatter(angles, norm, s=dot_size ** 2,
                   color=color, zorder=zorder + 1, edgecolors="white", linewidths=0.6)

    # ── Axis labels ───────────────────────────────────────────────────────────
    label_pad = 1.18  # push labels slightly beyond the outer ring
    for angle, dim in zip(angles, dimensions):
        deg = math.degrees(angle)
        ha = "center"
        if 10 < deg < 170:
            ha = "left"
        elif 190 < deg < 350:
            ha = "right"
        ax.text(angle, label_pad, dim,
                ha=ha, va="center",
                fontsize=THEME.FS_BODY, color=THEME.INK,
                fontweight="bold")

    # ── Ring value labels (on the first spoke) ────────────────────────────────
    for rv in ring_vals:
        ring_norm = (rv - score_min) / (score_max - score_min)
        ax.text(angles[0], ring_norm + 0.03, str(int(rv)),
                ha="center", va="bottom",
                fontsize=THEME.FS_MICRO, color=THEME.MUTED)

    # ── Remove default polar decorations ─────────────────────────────────────
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)
    ax.set_ylim(0, 1.25)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = []
    for player in players:
        color = player.get("color", THEME.ACCENT)
        highlight = bool(player.get("highlight", False))
        lw = 2.4 if highlight else 1.5
        patch = mpatches.Patch(facecolor=color, edgecolor=color,
                               label=player.get("name", ""),
                               linewidth=lw, alpha=0.7)
        legend_handles.append(patch)

    legend = ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=min(len(players), 4),
        frameon=True,
        framealpha=0.92,
        edgecolor=THEME.BORDER,
        fontsize=THEME.FS_SMALL,
    )

    # ── Title ─────────────────────────────────────────────────────────────────
    if title:
        fig.text(0.5, 0.97, title,
                 ha="center", va="top",
                 fontsize=THEME.FS_H1, color=THEME.INK, fontweight="bold")

    plt.tight_layout(rect=[0.0, 0.08, 1.0, 0.95])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=THEME.DPI, bbox_inches="tight",
                facecolor=THEME.BG)
    # Also export as PDF (vector, lossless) alongside the PNG
    pdf_path = str(output_path).replace(".png", ".pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor=THEME.BG)
    plt.close(fig)
    return f"Radar chart saved: {output_path}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Render radar/spider chart PNG")
    parser.add_argument("--spec", required=True,
                        help="JSON spec file path or inline JSON string")
    parser.add_argument("--output", required=True, help="Output PNG path")
    args = parser.parse_args()
    src = args.spec.strip()
    spec = json.loads(src) if src.startswith("{") else json.load(open(src, encoding="utf-8"))
    print(render_radar(spec, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
