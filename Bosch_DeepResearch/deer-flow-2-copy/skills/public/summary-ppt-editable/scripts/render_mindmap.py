#!/usr/bin/env python3
"""
Render a clean radial mind map PNG.
Professional mind map design: colored circular center, pastel branch nodes,
thin color-matched connectors, generous spacing.

Layout strategy (auto-selected per branch):
  - FAN layout: children spread radially — used when nc is small enough
    that all child nodes fit within the branch's angular sector without overlap.
  - LIST layout: children stacked vertically beside the branch node — used
    when fan would cause overlap (nc too large relative to branch_gap).

Standalone — no DeerFlow dependency required.

Spec format:
{
  "type": "mindmap",
  "center": "FSD Strategy",
  "branches": [
    {
      "label": "Perception",
      "color": "#A8C4D9",
      "children": ["Camera", "BEV Fusion", "Occupancy"]
    }
  ],
  "fig_width": 12,
  "fig_height": 9
}
"""
from __future__ import annotations

import argparse
import colorsys
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.path import Path as MplPath

from viz_theme import THEME, setup_matplotlib, fit_fontsize, get_categorical_palette

setup_matplotlib()

# Morandi palette from viz_theme (8 colors for branch rotation)
_MORANDI = get_categorical_palette(8)

# Node geometry constants (data coords, half-dimensions)
_BHW = 0.22   # branch half-width  (total 0.44)
_BHH = 0.065  # branch half-height (total 0.13)
_CHW = 0.19   # child half-width   (total 0.38)
_CHH = 0.05   # child half-height  (total 0.10)
_GAP = 0.06   # minimum gap between node edges


def _darken(hex_color: str, factor: float = 0.65) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255
    hue, lgt, sat = colorsys.rgb_to_hls(r, g, b)
    lgt = max(0.0, lgt * factor)
    r2, g2, b2 = colorsys.hls_to_rgb(hue, lgt, sat)
    return f"#{int(r2*255):02x}{int(g2*255):02x}{int(b2*255):02x}"


def _bezier(ax, x0, y0, x1, y1, color, lw, alpha=1.0, zorder=1):
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    cp1x = x0 * 0.45 + mx * 0.55
    cp1y = y0 * 0.45 + my * 0.55
    cp2x = x1 * 0.45 + mx * 0.55
    cp2y = y1 * 0.45 + my * 0.55
    path = MplPath(
        [(x0, y0), (cp1x, cp1y), (cp2x, cp2y), (x1, y1)],
        [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
    )
    ax.add_patch(mpatches.PathPatch(
        path, facecolor="none", edgecolor=color,
        lw=lw, alpha=alpha, zorder=zorder,
        capstyle="round", joinstyle="round",
    ))


def _rect_edge(cx, cy, hw, hh, dx, dy):
    """Point on rect boundary (cx±hw, cy±hh) in direction (dx, dy)."""
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return cx, cy
    tx = (hw / abs(dx)) if abs(dx) > 1e-12 else float("inf")
    ty = (hh / abs(dy)) if abs(dy) > 1e-12 else float("inf")
    t = min(tx, ty)
    L = math.sqrt(dx * dx + dy * dy)
    return cx + dx / L * t, cy + dy / L * t


def _draw_branch_node(ax, cx, cy, label, fill, border, fs, zorder=5):
    lines = label.replace("\\n", "\n").split("\n")
    n_lines = len(lines)
    max_chars = max(len(l) for l in lines) if lines else 1
    bw = max(0.24, min(0.44, max_chars * 0.031 + 0.14))
    bh = max(0.13, n_lines * 0.065 + 0.09)
    ax.add_patch(FancyBboxPatch(
        (cx - bw / 2, cy - bh / 2), bw, bh,
        boxstyle="round,pad=0.025",
        facecolor=fill, edgecolor=border,
        linewidth=2.5, zorder=zorder, antialiased=True,
    ))
    ax.text(cx, cy, label.replace("\\n", "\n"),
            ha="center", va="center", fontsize=fs,
            color=THEME.INK, fontweight="bold", zorder=zorder + 1,
            multialignment="center", antialiased=True)
    return bw / 2, bh / 2


def _draw_child_node(ax, cx, cy, label, border, fs, zorder=3):
    lines = label.replace("\\n", "\n").split("\n")
    n_lines = len(lines)
    max_chars = max(len(l) for l in lines) if lines else 1
    cdw = max(0.19, min(0.38, max_chars * 0.029 + 0.11))
    cdh = max(0.10, n_lines * 0.058 + 0.07)
    ax.add_patch(FancyBboxPatch(
        (cx - cdw / 2, cy - cdh / 2), cdw, cdh,
        boxstyle="round,pad=0.020",
        facecolor="white", edgecolor=border,
        linewidth=1.8, zorder=zorder, antialiased=True,
    ))
    ax.text(cx, cy, label.replace("\\n", "\n"),
            ha="center", va="center", fontsize=fs,
            color=THEME.INK, fontweight="bold", zorder=zorder + 1,
            multialignment="center", antialiased=True)
    return cdw / 2, cdh / 2


def _max_fan_children(branch_gap: float, R_CHILD: float) -> int:
    """
    Maximum number of children that can be arranged in FAN mode within
    the branch's angular sector (42% of branch_gap) without their node
    edges overlapping.
    chord = 2*R*sin(step/2) >= child_full_width + gap
    step = 2*asin((CHW*2+GAP)/(2*R))
    max_nc such that step*(nc-1) <= branch_gap*0.42
    """
    child_diam = _CHW * 2 + _GAP
    ratio = min(child_diam / (2.0 * R_CHILD), 0.999)
    step = 2.0 * math.asin(ratio)
    max_steps = (branch_gap * 0.42) / step
    return max(1, int(max_steps) + 1)   # nc = steps+1


def _fan_half_spread(nc: int, R_CHILD: float) -> float:
    """Angular half-spread for nc children with no overlap."""
    if nc <= 1:
        return 0.0
    child_diam = _CHW * 2 + _GAP
    ratio = min(child_diam / (2.0 * R_CHILD), 0.999)
    step = 2.0 * math.asin(ratio)
    return step * (nc - 1) / 2


def render_mindmap(spec: Dict[str, Any], output_path: str) -> str:
    center_label: str = spec.get("center", "Topic")
    branches: List[Dict[str, Any]] = spec.get("branches", [])
    fw = float(spec.get("fig_width", 12.0))
    fh = float(spec.get("fig_height", 9.0))

    if not branches:
        raise ValueError("mindmap requires at least one branch")

    n = len(branches)
    max_children = max(len(b.get("children", [])) for b in branches)

    # ── Geometry ──────────────────────────────────────────────────────────────
    R_CENTER  = max(0.14, min(0.21, 0.38 / math.sqrt(n)))
    R_BRANCH  = max(0.44, min(0.58, 0.30 + n * 0.018))
    R_CHILD   = R_BRANCH + max(0.28, min(0.40, 0.15 + max_children * 0.03))
    branch_gap = 2 * math.pi / max(n, 1)

    # How many children fit in FAN mode?
    fan_capacity = _max_fan_children(branch_gap, R_CHILD)

    # Canvas: in LIST mode children stack outward; estimate max extent
    list_h_per_child = _CHH * 2 + _GAP
    max_list_nc = max_children
    list_extent = R_CHILD + max_list_nc * list_h_per_child
    margin = max(R_CHILD + 0.32, list_extent + 0.18)
    ax_lim = margin

    fig, ax = plt.subplots(figsize=(fw, fh))
    fig.patch.set_facecolor(THEME.BG)
    ax.set_facecolor(THEME.BG)
    ax.axis("off")
    ax.set_xlim(-ax_lim, ax_lim)
    ax.set_ylim(-ax_lim, ax_lim)
    ax.set_aspect("equal")

    # ── Font sizes ────────────────────────────────────────────────────────────
    all_children = [
        c.get("label", "") if isinstance(c, dict) else str(c)
        for b in branches for c in b.get("children", [])
    ]
    longest_child  = max(all_children, key=len) if all_children else "Label"
    longest_branch = max((b.get("label", "") for b in branches), key=len)

    CHILD_FS = fit_fontsize(
        longest_child,
        box_width_in=(_CHW * 2) * fw * 0.65,
        box_height_in=(_CHH * 2) * fh * 0.90,
        start_pt=THEME.FS_SMALL, min_pt=7.5,
    )
    BRANCH_FS = fit_fontsize(
        longest_branch,
        box_width_in=0.30 * fw,
        box_height_in=0.10 * fh,
        start_pt=THEME.FS_BODY + 1, min_pt=9.5,
    )
    CENTER_FS = fit_fontsize(
        center_label,
        box_width_in=R_CENTER * 2 * fw * 0.78,
        box_height_in=R_CENTER * 2 * fh * 0.78,
        start_pt=THEME.FS_BODY + 3, min_pt=11.0,
    )

    # ── Center circle ─────────────────────────────────────────────────────────
    ax.add_patch(Circle(
        (0, 0), R_CENTER,
        facecolor=THEME.ACCENT,
        edgecolor=_darken(THEME.ACCENT, 0.75),
        linewidth=2.5, zorder=8,
    ))
    ax.text(0, 0, center_label.replace("\\n", "\n"),
            ha="center", va="center", fontsize=CENTER_FS,
            color="white", fontweight="bold",
            zorder=9, multialignment="center", antialiased=True)

    # ── Branches ──────────────────────────────────────────────────────────────
    for bi, branch in enumerate(branches):
        angle   = math.pi / 2 - 2 * math.pi * bi / n
        bc      = branch.get("color", _MORANDI[bi % len(_MORANDI)])
        bc_dark = _darken(bc, 0.65)
        label   = branch.get("label", "")
        children: List[str] = [
            c.get("label", "") if isinstance(c, dict) else str(c)
            for c in branch.get("children", [])
        ]
        nc = len(children)

        bx = R_BRANCH * math.cos(angle)
        by = R_BRANCH * math.sin(angle)

        # Center → branch connector
        sx = math.cos(angle) * R_CENTER
        sy = math.sin(angle) * R_CENTER
        _bezier(ax, sx, sy, bx, by, bc_dark, lw=2.6, zorder=2)

        # Branch node (drawn after line so it covers line terminus)
        bhw, bhh = _draw_branch_node(ax, bx, by, label, bc, bc_dark, BRANCH_FS, zorder=5)

        if nc == 0:
            continue

        use_fan = nc <= fan_capacity

        if use_fan:
            # ── FAN layout ────────────────────────────────────────────────────
            half_spread = _fan_half_spread(nc, R_CHILD)
            child_angles = [
                angle + half_spread * (2 * i / max(nc - 1, 1) - 1) if nc > 1 else angle
                for i in range(nc)
            ]
            for child_label, cangle in zip(children, child_angles):
                ccx = R_CHILD * math.cos(cangle)
                ccy = R_CHILD * math.sin(cangle)
                dx, dy  = ccx - bx, ccy - by
                b_ex, b_ey = _rect_edge(bx, by, bhw, bhh, dx, dy)
                chw, chh   = _CHW, _CHH
                c_ex, c_ey = _rect_edge(ccx, ccy, chw, chh, -dx, -dy)
                _bezier(ax, b_ex, b_ey, c_ex, c_ey, bc_dark, lw=2.0, alpha=0.85, zorder=2)
                _draw_child_node(ax, ccx, ccy, child_label, bc_dark, CHILD_FS, zorder=3)

        else:
            # ── LIST layout ───────────────────────────────────────────────────
            # Stack children in a column outward from the branch node,
            # aligned perpendicular to the radial direction.
            # Column anchor: radially outward from branch, offset so mid-child
            # aligns with branch centre.
            step_h = _CHH * 2 + _GAP
            total_h = nc * step_h - _GAP
            # Outward direction unit vector
            ux, uy = math.cos(angle), math.sin(angle)
            # Perpendicular (tangential) unit vector
            px, py = -uy, ux

            # Column x-centre: push outward by R_CHILD distance
            col_cx = bx + ux * (R_CHILD - R_BRANCH)
            col_cy = by + uy * (R_CHILD - R_BRANCH)

            for ci, child_label in enumerate(children):
                # Offset along perpendicular so list is centred on branch angle
                perp_offset = (ci - (nc - 1) / 2) * step_h
                ccx = col_cx + px * perp_offset
                ccy = col_cy + py * perp_offset

                dx, dy = ccx - bx, ccy - by
                b_ex, b_ey = _rect_edge(bx, by, bhw, bhh, dx, dy)
                chw, chh   = _CHW, _CHH
                c_ex, c_ey = _rect_edge(ccx, ccy, chw, chh, -dx, -dy)
                _bezier(ax, b_ex, b_ey, c_ex, c_ey, bc_dark, lw=2.0, alpha=0.85, zorder=2)
                _draw_child_node(ax, ccx, ccy, child_label, bc_dark, CHILD_FS, zorder=3)

    # ── Optional title ────────────────────────────────────────────────────────
    title = spec.get("title", "")
    if title:
        ax.text(0, ax_lim * 0.93, title,
                ha="center", va="top",
                fontsize=BRANCH_FS * 1.05,
                color=THEME.MUTED, fontweight="bold",
                zorder=10, antialiased=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=THEME.DPI, bbox_inches="tight", facecolor=THEME.BG)
    fig.savefig(str(output_path).replace(".png", ".pdf"), format="pdf",
                bbox_inches="tight", facecolor=THEME.BG)
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
