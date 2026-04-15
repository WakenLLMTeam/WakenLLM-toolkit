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

from viz_theme import THEME, setup_matplotlib, get_categorical_palette

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

    # Quadrant backgrounds — default to 4 distinct Morandi colors
    _mc = get_categorical_palette(4)
    _DEFAULT_Q_COLORS = {
        "top_left":     _mc[0],
        "top_right":    _mc[1],
        "bottom_left":  _mc[2],
        "bottom_right": _mc[3],
    }
    q_map = {
        "bottom_left":  (0.0, 0.0),
        "bottom_right": (0.5, 0.0),
        "top_left":     (0.0, 0.5),
        "top_right":    (0.5, 0.5),
    }
    for qkey, (qx, qy) in q_map.items():
        q = quadrants.get(qkey, {})
        color = q.get("color") or _DEFAULT_Q_COLORS[qkey]
        rect = mpatches.FancyBboxPatch(
            (qx + 0.005, qy + 0.005), 0.49, 0.49,
            boxstyle="round,pad=0.008",
            facecolor=color, edgecolor=THEME.BORDER, linewidth=0.8,
            transform=ax.transAxes, zorder=1
        )
        ax.add_patch(rect)
        label = q.get("label", "")
        if label:
            lx = qx + 0.25
            ly = qy + 0.46
            ax.text(lx, ly, label,
                    ha="center", va="top",
                    fontsize=THEME.FS_H1, color=THEME.INK,
                    fontweight="bold",
                    transform=ax.transAxes, zorder=2)

    # Divider lines
    ax.plot([0.5, 0.5], [0.02, 0.98], color=THEME.BORDER, lw=1.2, zorder=3,
            transform=ax.transAxes)
    ax.plot([0.02, 0.98], [0.5, 0.5], color=THEME.BORDER, lw=1.2, zorder=3,
            transform=ax.transAxes)

    # Axis labels
    ax.text(0.5, 0.01, x_label, ha="center", va="bottom",
            fontsize=THEME.FS_H1, color=THEME.INK, fontweight="bold",
            transform=ax.transAxes)
    ax.text(0.01, 0.5, y_label, ha="left", va="center",
            fontsize=THEME.FS_H1, color=THEME.INK, fontweight="bold",
            rotation=90, transform=ax.transAxes)

    # Spread clustered items so they don't overlap
    import math

    def _spread(raw_items, min_dist: float = 0.10, iterations: int = 80):
        """Repulsion pass: push items apart until no two are closer than min_dist."""
        pts = [[float(it.get("x", 0.5)), float(it.get("y", 0.5))] for it in raw_items]
        for _ in range(iterations):
            moved = False
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    dx = pts[j][0] - pts[i][0]
                    dy = pts[j][1] - pts[i][1]
                    dist = math.hypot(dx, dy)
                    if dist < min_dist:
                        push = (min_dist - dist) / max(dist, 1e-6) * 0.5
                        if dist < 1e-6:           # exact overlap → push radially
                            angle = math.pi * 2 * i / max(len(pts), 1)
                            dx, dy = math.cos(angle), math.sin(angle)
                            push = min_dist * 0.5
                        pts[i][0] = max(0.06, min(0.94, pts[i][0] - dx * push))
                        pts[i][1] = max(0.06, min(0.94, pts[i][1] - dy * push))
                        pts[j][0] = max(0.06, min(0.94, pts[j][0] + dx * push))
                        pts[j][1] = max(0.06, min(0.94, pts[j][1] + dy * push))
                        moved = True
            if not moved:
                break
        return pts

    spread_pts = _spread(items)
    morandi_items = get_categorical_palette(max(len(items), 1))

    # Map [0,1] → axes fraction [0.04, 0.96]
    def _to_ax(v):
        return float(v) * 0.92 + 0.04

    # ── Build label offsets with quadrant-centre reference ───────────────────
    # Using the whole-canvas centre (0.5, 0.5) as the repulsion origin causes
    # all items in the same quadrant to receive nearly identical angles → labels
    # pile up.  Instead, use the *quadrant centre* as origin so items within
    # the same quadrant fan out relative to each other.
    #
    # Quadrant centres (in _to_ax space):
    #   top-right   → (0.75, 0.75)   bottom-right → (0.75, 0.25)
    #   top-left    → (0.25, 0.75)   bottom-left  → (0.25, 0.25)
    #
    # After computing angles we also run a repulsion pass on the label
    # offsets themselves so that very close labels push each other apart.

    _OFF_DIST_X = 26   # base x offset (points)
    _OFF_DIST_Y = 20   # base y offset (points)

    # Step 1 – compute initial angle-based offset for each item
    raw_offsets = []   # [(off_x, off_y), ...]
    for idx, it in enumerate(items):
        ix = _to_ax(spread_pts[idx][0])
        iy = _to_ax(spread_pts[idx][1])
        # Choose quadrant centre as repulsion origin
        ref_x = 0.75 if ix >= 0.5 else 0.25
        ref_y = 0.75 if iy >= 0.5 else 0.25
        angle = math.atan2(iy - ref_y, ix - ref_x)
        # If item is exactly on quadrant centre, fall back to canvas centre
        if abs(ix - ref_x) < 1e-4 and abs(iy - ref_y) < 1e-4:
            angle = math.atan2(iy - 0.5, ix - 0.5)
        raw_offsets.append([math.cos(angle) * _OFF_DIST_X,
                             math.sin(angle) * _OFF_DIST_Y])

    # Step 2 – repulsion pass on label positions (in "offset points" space)
    # Convert axes coords to approx points for comparison
    _AX_TO_PT_X = fw * 72   # 1 axes unit ≈ fig_width * 72 pt
    _AX_TO_PT_Y = fh * 72

    for _ in range(40):
        moved = False
        for i in range(len(raw_offsets)):
            for j in range(i + 1, len(raw_offsets)):
                lx_i = _to_ax(spread_pts[i][0]) * _AX_TO_PT_X + raw_offsets[i][0]
                ly_i = _to_ax(spread_pts[i][1]) * _AX_TO_PT_Y + raw_offsets[i][1]
                lx_j = _to_ax(spread_pts[j][0]) * _AX_TO_PT_X + raw_offsets[j][0]
                ly_j = _to_ax(spread_pts[j][1]) * _AX_TO_PT_Y + raw_offsets[j][1]
                dist = math.hypot(lx_i - lx_j, ly_i - ly_j)
                _MIN_LABEL_SEP = 36   # minimum separation in points
                if dist < _MIN_LABEL_SEP:
                    push = (_MIN_LABEL_SEP - dist) / max(dist, 1e-3) * 0.5
                    dx = lx_i - lx_j
                    dy = ly_i - ly_j
                    if dist < 1e-3:
                        dx, dy = 1.0, 0.0
                    raw_offsets[i][0] += dx * push
                    raw_offsets[i][1] += dy * push
                    raw_offsets[j][0] -= dx * push
                    raw_offsets[j][1] -= dy * push
                    moved = True
        if not moved:
            break

    # Step 3 – draw items using the repulsion-adjusted offsets
    # Scale annotation font size down when there are many items
    ann_fs = THEME.FS_BODY if len(items) <= 5 else THEME.FS_SMALL

    for idx, it in enumerate(items):
        ix = _to_ax(spread_pts[idx][0])
        iy = _to_ax(spread_pts[idx][1])
        color = it.get("color") or morandi_items[idx % len(morandi_items)]
        label = it.get("label", "")
        ax.scatter([ix], [iy], s=220, color=color, zorder=5,
                   edgecolors="white", linewidths=1.4,
                   transform=ax.transAxes)
        off_x, off_y = raw_offsets[idx]
        ha = "left" if off_x >= 0 else "right"
        ax.annotate(
            label,
            xy=(ix, iy), xytext=(off_x, off_y),
            textcoords="offset points",
            fontsize=ann_fs, color=THEME.INK, fontweight="bold",
            ha=ha, va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85),
            arrowprops=dict(arrowstyle="-", color=THEME.BORDER, lw=0.7),
            transform=ax.transAxes, zorder=6,
        )

    if title:
        fig.text(0.5, 0.99, title, ha="center", va="top",
                 fontsize=THEME.FS_TITLE, color=THEME.INK, fontweight="bold")

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
