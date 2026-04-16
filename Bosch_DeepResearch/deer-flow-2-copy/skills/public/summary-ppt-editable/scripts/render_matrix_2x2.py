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
import math as _math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from viz_theme import THEME, setup_matplotlib, get_categorical_palette

setup_matplotlib()

# 16 candidate label offsets (dx, dy) in typographic points
_LABEL_CANDIDATES: List[Tuple[float, float]] = [
    (20,  12), (20, -12), (-20,  12), (-20, -12),   # primary diagonals
    (26,   0), (-26,  0), (  0,  20), (  0, -20),   # cardinals
    (16,  18), (16, -18), (-16,  18), (-16, -18),   # steep diagonals
    ( 8,  22), (-8,  22), (  8, -22), ( -8, -22),   # near-vertical
]


def _est_label_bbox(x_d: float, y_d: float,
                    dx_pt: float, dy_pt: float,
                    text: str, font_pt: float, dpi: float
                    ) -> Tuple[float, float, float, float]:
    """Estimate label bounding box in display pixels (no renderer needed)."""
    lines = text.split("\n")
    n_chars = max(len(l) for l in lines) if lines else 1
    n_lines = len(lines)
    char_w_px = font_pt * 0.60 * dpi / 72.0
    char_h_px = font_pt * 1.30 * dpi / 72.0
    w_px = n_chars * char_w_px + 6
    h_px = n_lines * char_h_px + 4
    dx_px = dx_pt * dpi / 72.0
    dy_px = dy_pt * dpi / 72.0
    ax_x = x_d + dx_px
    ax_y = y_d + dy_px
    if dx_pt >= 0:
        x0, x1 = ax_x, ax_x + w_px
    else:
        x0, x1 = ax_x - w_px, ax_x
    y0 = ax_y - h_px / 2
    y1 = ax_y + h_px / 2
    return (x0, y0, x1, y1)


def _overlap_area(b1: Tuple, b2: Tuple) -> float:
    ox = max(0.0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
    oy = max(0.0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
    return ox * oy


def _best_label_offset(
    x_d: float, y_d: float,
    text: str, font_pt: float, dpi: float,
    marker_bboxes: List[Tuple],
    placed_bboxes: List[Tuple],
    forbidden_bboxes: List[Tuple],  # title + quadrant labels
    axes_bbox: Tuple,
) -> Tuple[Tuple[float, float], Tuple]:
    """
    Greedy label placement. Lower score = better.
      • marker overlap      ×1
      • placed label overlap ×3
      • forbidden overlap   ×4  (title / quadrant labels)
      • out-of-axes penalty ×0.5
    """
    best_off = _LABEL_CANDIDATES[0]
    best_score = float("inf")
    best_bbox: Tuple = _est_label_bbox(x_d, y_d, *_LABEL_CANDIDATES[0], text, font_pt, dpi)
    ax0, ay0, ax1, ay1 = axes_bbox
    for dx, dy in _LABEL_CANDIDATES:
        lb = _est_label_bbox(x_d, y_d, dx, dy, text, font_pt, dpi)
        score = (
            sum(_overlap_area(lb, pb) for pb in marker_bboxes)
            + 3.0 * sum(_overlap_area(lb, pb) for pb in placed_bboxes)
            + 4.0 * sum(_overlap_area(lb, pb) for pb in forbidden_bboxes)
            + (max(0.0, ax0 - lb[0]) + max(0.0, lb[2] - ax1)
               + max(0.0, ay0 - lb[1]) + max(0.0, lb[3] - ay1)) * 0.5
        )
        if score < best_score:
            best_score = score
            best_off = (dx, dy)
            best_bbox = lb
    return best_off, best_bbox


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
        # quadrant labels intentionally suppressed — never rendered

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
    def _spread(raw_items, min_dist: float = 0.10, iterations: int = 80):
        """Repulsion pass: push items apart until no two are closer than min_dist."""
        pts = [[float(it.get("x", 0.5)), float(it.get("y", 0.5))] for it in raw_items]
        for _ in range(iterations):
            moved = False
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    dx = pts[j][0] - pts[i][0]
                    dy = pts[j][1] - pts[i][1]
                    dist = _math.hypot(dx, dy)
                    if dist < min_dist:
                        push = (min_dist - dist) / max(dist, 1e-6) * 0.5
                        if dist < 1e-6:           # exact overlap → push radially
                            angle = _math.pi * 2 * i / max(len(pts), 1)
                            dx, dy = _math.cos(angle), _math.sin(angle)
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

    # ── Draw scatter dots (no labels yet — need tight_layout first) ───────────
    ann_fs = THEME.FS_BODY if len(items) <= 5 else THEME.FS_SMALL
    item_colors = []
    for idx, it in enumerate(items):
        ix = _to_ax(spread_pts[idx][0])
        iy = _to_ax(spread_pts[idx][1])
        color = it.get("color") or morandi_items[idx % len(morandi_items)]
        item_colors.append(color)
        ax.scatter([ix], [iy], s=220, color=color, zorder=5,
                   edgecolors="white", linewidths=1.4,
                   transform=ax.transAxes)

    if title:
        fig.text(0.5, 0.99, title, ha="center", va="top",
                 fontsize=THEME.FS_TITLE, color=THEME.INK, fontweight="bold")

    # Finalise layout so transforms give accurate display coordinates
    plt.tight_layout(rect=[0, 0, 1, 0.95 if title else 1.0], pad=0.4)

    # ── Greedy label placement ─────────────────────────────────────────────────
    dpi = fig.get_dpi()
    fig_w_px = fig.get_figwidth() * dpi
    fig_h_px = fig.get_figheight() * dpi

    ax_disp = ax.get_window_extent()
    axes_bbox = (ax_disp.x0, ax_disp.y0, ax_disp.x1, ax_disp.y1)

    # Marker bboxes in display pixels
    marker_r_px = (_math.sqrt(220) / 2.0) * dpi / 72.0
    marker_bboxes: List[Tuple] = []
    item_display_coords: List[Tuple[float, float]] = []
    for idx in range(len(items)):
        ix = _to_ax(spread_pts[idx][0])
        iy = _to_ax(spread_pts[idx][1])
        xd, yd = ax.transAxes.transform((ix, iy))
        item_display_coords.append((xd, yd))
        marker_bboxes.append((xd - marker_r_px, yd - marker_r_px,
                               xd + marker_r_px, yd + marker_r_px))

    # Forbidden zones: title + quadrant header labels
    # Title: fig.text(0.5, 0.99, ...) — approximate bbox in display pixels
    forbidden_bboxes: List[Tuple] = []
    if title:
        t_fs = THEME.FS_TITLE
        t_w = len(title) * t_fs * 0.60 * dpi / 72.0
        t_h = t_fs * 1.5 * dpi / 72.0
        tx = fig_w_px * 0.5
        ty = fig_h_px * 0.99
        forbidden_bboxes.append((tx - t_w / 2, ty - t_h, tx + t_w / 2, ty))

    # (quadrant header labels are suppressed — no forbidden zones needed for them)

    # Greedy placement: process items in order, accumulate placed_bboxes
    placed_bboxes: List[Tuple] = []
    final_offsets: List[Tuple[float, float]] = []
    for idx, it in enumerate(items):
        xd, yd = item_display_coords[idx]
        label = it.get("label", "")
        off, bbox = _best_label_offset(
            xd, yd, label, ann_fs, dpi,
            marker_bboxes, placed_bboxes, forbidden_bboxes, axes_bbox,
        )
        placed_bboxes.append(bbox)
        final_offsets.append(off)

    # ── Draw annotations with greedy offsets ──────────────────────────────────
    for idx, it in enumerate(items):
        ix = _to_ax(spread_pts[idx][0])
        iy = _to_ax(spread_pts[idx][1])
        label = it.get("label", "")
        off_x, off_y = final_offsets[idx]
        ha = "left" if off_x >= 0 else "right"
        ax.annotate(
            label,
            xy=(ix, iy), xytext=(off_x, off_y),
            textcoords="offset points",
            fontsize=ann_fs, color=THEME.INK, fontweight="bold",
            ha=ha, va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc=THEME.BG, ec="none", alpha=0.35),
            arrowprops=dict(arrowstyle="-", color=THEME.BORDER, lw=0.7),
            transform=ax.transAxes, zorder=6,
        )
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
