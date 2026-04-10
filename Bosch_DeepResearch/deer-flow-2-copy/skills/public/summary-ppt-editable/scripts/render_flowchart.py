#!/usr/bin/env python3
"""
Render a high-quality directed flowchart PNG from a JSON spec.
Standalone — no DeerFlow dependency required.

Usage:
  python render_flowchart.py --spec flowchart.json --output out.png

Spec format:
{
  "type": "flowchart",
  "title": "L3 ODD Decision Flow",
  "layout": "TB",
  "nodes": [
    {"id": "start",  "label": "Start",              "shape": "terminal", "color": "#1e3a5f"},
    {"id": "input",  "label": "Read Sensor Data",   "shape": "parallelogram", "color": "#dbeafe"},
    {"id": "proc",   "label": "Fuse & Classify",    "shape": "rect",     "color": "#dcfce7"},
    {"id": "decide", "label": "ODD Boundary\nCheck","shape": "diamond",  "color": "#fef9c3"},
    {"id": "sub",    "label": "Run Sub-routine",    "shape": "rounded",  "color": "#ede9fe"},
    {"id": "end",    "label": "End",                "shape": "terminal", "color": "#1e3a5f"}
  ],
  "edges": [
    {"from": "start",  "to": "input"},
    {"from": "input",  "to": "proc"},
    {"from": "proc",   "to": "decide"},
    {"from": "decide", "to": "sub",  "label": "Yes"},
    {"from": "decide", "to": "end",  "label": "No"},
    {"from": "sub",    "to": "end"}
  ],
  "fig_width": 11,
  "fig_height": 6
}

Shape semantics (standard flowchart convention):
  "terminal"      Oval/stadium  — Start / End node (use dark color for contrast)
  "rect"          Rectangle     — Process step / operation
  "diamond"       Diamond       — Decision / condition (Yes/No branch)
  "rounded"       Round-rect    — Sub-process / predefined process
  "parallelogram" Parallelogram — Input / Output data

Layout: "LR" (left-to-right) | "TB" (top-to-bottom)
Edge labels: use "Yes"/"No" on edges from diamond nodes for clarity
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

from viz_theme import THEME, setup_matplotlib, fit_fontsize

setup_matplotlib()


def _is_dark(hex_color: str) -> bool:
    h = hex_color.lstrip("#")
    if len(h) < 6:
        return False
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (0.299 * r + 0.587 * g + 0.114 * b) < 160


def _auto_layout(nodes: List[Dict], edges: List[Dict], layout: str) -> Dict[str, Tuple[float, float]]:
    """Layered BFS layout."""
    ids = [n["id"] for n in nodes]
    adj: Dict[str, List[str]] = {i: [] for i in ids}
    indeg: Dict[str, int] = {i: 0 for i in ids}
    for e in edges:
        s, d = e.get("from", ""), e.get("to", "")
        if s in adj and d in indeg:
            adj[s].append(d)
            indeg[d] += 1

    queue = [i for i in ids if indeg[i] == 0]
    layers: List[List[str]] = []
    visited: set = set()
    # Guard against cycles: track iterations to avoid infinite loop
    max_iters = len(ids) + 1
    iters = 0
    while queue and iters < max_iters:
        iters += 1
        layers.append(queue[:])
        nxt = []
        for n in queue:
            visited.add(n)
            for m in adj[n]:
                if m not in visited:
                    indeg[m] -= 1
                    if indeg[m] == 0:
                        nxt.append(m)
        queue = nxt
    # Append any unvisited nodes (isolated or part of cycles) to a final layer
    unvisited = [i for i in ids if i not in visited]
    if unvisited:
        if layers:
            layers[-1].extend(unvisited)
        else:
            layers.append(unvisited)

    pos: Dict[str, Tuple[float, float]] = {}
    nl = len(layers)
    for li, layer in enumerate(layers):
        for ni, nid in enumerate(layer):
            k = len(layer)
            if layout == "TB":
                x = (ni + 0.5) / k
                y = 1.0 - (li + 0.5) / nl
            else:
                x = (li + 0.5) / nl
                y = 1.0 - (ni + 0.5) / k
            pos[nid] = (x, y)
    return pos


def _draw_node(ax, x, y, label, shape, color, bw, bh, fontsize=None):
    bg = color or THEME.SURFACE
    if _is_dark(bg):
        bg = THEME.SURFACE
    text_color = THEME.INK

    if shape == "terminal":
        # Oval / stadium shape — Start / End
        # Use a wide rounded rectangle that looks like a stadium/pill
        bg_use = color or THEME.ACCENT
        # Detect if it's a dark fill → use white text
        h = bg_use.lstrip("#")
        if len(h) == 6:
            r2, g2, b2 = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            text_color = "white" if (0.299*r2 + 0.587*g2 + 0.114*b2) < 160 else THEME.INK
        # Shadow
        ax.add_patch(FancyBboxPatch(
            (x - bw / 2 + 0.003, y - bh / 2 - 0.003), bw, bh,
            boxstyle=f"round,pad={bh*0.40:.3f}",
            facecolor="#00000015", edgecolor="none",
            transform=ax.transAxes, zorder=2))
        # Main pill shape
        ax.add_patch(FancyBboxPatch(
            (x - bw / 2, y - bh / 2), bw, bh,
            boxstyle=f"round,pad={bh*0.40:.3f}",
            facecolor=bg_use, edgecolor="white" if text_color == "white" else THEME.BORDER,
            linewidth=1.5, transform=ax.transAxes, zorder=3))

    elif shape == "diamond":
        # Decision node — rhombus
        pts_shadow = np.array([
            [x + 0.003, y + bh * 0.65],
            [x + bw * 0.58, y],
            [x + 0.003, y - bh * 0.65],
            [x - bw * 0.55, y],
        ])
        ax.add_patch(plt.Polygon(pts_shadow, closed=True, facecolor="#00000015",
                                 edgecolor="none", transform=ax.transAxes, zorder=2))
        pts = np.array([
            [x,           y + bh * 0.65],
            [x + bw * 0.58, y],
            [x,           y - bh * 0.65],
            [x - bw * 0.58, y],
        ])
        ax.add_patch(plt.Polygon(pts, closed=True, facecolor=bg,
                                 edgecolor=THEME.BORDER, linewidth=1.2,
                                 transform=ax.transAxes, zorder=3))

    elif shape == "rounded":
        # Sub-process — rounded rectangle
        ax.add_patch(FancyBboxPatch(
            (x - bw / 2 + 0.003, y - bh / 2 - 0.003), bw, bh,
            boxstyle="round,pad=0.025", facecolor="#00000015",
            edgecolor="none", transform=ax.transAxes, zorder=2))
        ax.add_patch(FancyBboxPatch(
            (x - bw / 2, y - bh / 2), bw, bh,
            boxstyle="round,pad=0.025", facecolor=bg,
            edgecolor=THEME.ACCENT, linewidth=1.5,   # accent border = predefined process
            transform=ax.transAxes, zorder=3))

    elif shape == "parallelogram":
        # Input / Output — slanted rectangle
        skew = bw * 0.18   # horizontal skew offset
        pts_shadow = np.array([
            [x - bw/2 + skew + 0.003, y + bh/2 - 0.003],
            [x + bw/2 + skew + 0.003, y + bh/2 - 0.003],
            [x + bw/2 - skew + 0.003, y - bh/2 - 0.003],
            [x - bw/2 - skew + 0.003, y - bh/2 - 0.003],
        ])
        ax.add_patch(plt.Polygon(pts_shadow, closed=True, facecolor="#00000015",
                                 edgecolor="none", transform=ax.transAxes, zorder=2))
        pts = np.array([
            [x - bw/2 + skew, y + bh/2],
            [x + bw/2 + skew, y + bh/2],
            [x + bw/2 - skew, y - bh/2],
            [x - bw/2 - skew, y - bh/2],
        ])
        ax.add_patch(plt.Polygon(pts, closed=True, facecolor=bg,
                                 edgecolor=THEME.BORDER, linewidth=1.2,
                                 transform=ax.transAxes, zorder=3))

    else:
        # Default: rect — process step
        ax.add_patch(FancyBboxPatch(
            (x - bw / 2 + 0.003, y - bh / 2 - 0.003), bw, bh,
            boxstyle="square,pad=0.0", facecolor="#00000015",
            edgecolor="none", transform=ax.transAxes, zorder=2))
        ax.add_patch(FancyBboxPatch(
            (x - bw / 2, y - bh / 2), bw, bh,
            boxstyle="square,pad=0.0", facecolor=bg,
            edgecolor=THEME.BORDER, linewidth=1.2,
            transform=ax.transAxes, zorder=3))

    ax.text(x, y, label.replace("\\n", "\n"),
            ha="center", va="center",
            fontsize=fontsize if fontsize is not None else THEME.FS_BODY,
            color=text_color, fontweight="bold",
            transform=ax.transAxes, zorder=4,
            multialignment="center", linespacing=1.2)


def render_flowchart(spec: Dict[str, Any], output_path: str) -> str:
    nodes: List[Dict] = spec.get("nodes", [])
    edges: List[Dict] = spec.get("edges", [])
    title: Optional[str] = spec.get("title")
    layout = spec.get("layout", "LR").upper()

    if not nodes:
        raise ValueError("spec.nodes must be non-empty")

    pos = _auto_layout(nodes, edges, layout)
    num_layers = max(1, len(set(round(p[0], 2) for p in pos.values())))
    num_ranks  = max(1, len(set(round(p[1], 2) for p in pos.values())))

    # ── Adaptive figure size based on node count and label length ─────────────
    max_label_len = max(
        (len(n.get("label", "").replace("\\n", "\n").split("\n")[0]) for n in nodes),
        default=10)
    n_nodes = len(nodes)

    # Width: more layers → wider; longer labels → wider
    fw = float(spec.get("fig_width",
               max(13.0, num_layers * max(2.2, max_label_len * 0.13))))
    # Height: more ranks → taller; multi-line labels need more room
    max_lines = max((n.get("label","").count("\\n") + n.get("label","").count("\n") + 1
                     for n in nodes), default=1)
    fh = float(spec.get("fig_height",
               max(5.5, num_ranks * max(1.8, max_lines * 0.9))))
    fw = min(fw, 20.0)
    fh = min(fh, 12.0)

    # ── Node box size: fill more of each cell ────────────────────────────────
    # Give nodes ~60% of the cell width and ~55% of cell height
    bw = min(0.60 / num_layers, 0.22)
    bh = min(0.52 / num_ranks,  0.20)

    # ── Dynamic font size via fit_fontsize (physical inches) ─────────────────
    # Convert box axes-fraction → inches, then find largest pt that fits
    bw_in = bw * fw
    bh_in = bh * fh
    # Longest label across all nodes drives the sizing constraint
    longest_label = max(
        (n.get("label", "").replace("\\n", "\n") for n in nodes),
        key=lambda s: max(len(l) for l in s.split("\n")),
        default="Label"
    )
    node_fs = fit_fontsize(longest_label, bw_in * 0.82, bh_in * 0.75,
                           start_pt=22.0, min_pt=9.0)

    fig, ax = plt.subplots(figsize=(fw, fh))
    fig.patch.set_facecolor(THEME.BG)
    ax.set_facecolor(THEME.BG)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # ── Edges (draw BEFORE nodes so arrows go under node shapes) ─────────────
    # Pre-compute bidirectional pairs: if A→B and B→A both exist, offset them
    edge_pairs = set((e.get("from", ""), e.get("to", "")) for e in edges)

    for e in edges:
        src_id = e.get("from", "")
        dst_id = e.get("to", "")
        sx, sy = pos.get(src_id, (0, 0))
        dx, dy = pos.get(dst_id, (0, 0))
        if (sx, sy) == (0, 0) or (dx, dy) == (0, 0):
            continue
        angle = math.atan2(dy - sy, dx - sx)
        # Add extra padding so arrow starts/ends clearly outside the node boundary
        pad = 0.012
        sx2 = sx + math.cos(angle) * (bw / 2 + pad)
        sy2 = sy + math.sin(angle) * (bh / 2 + pad)
        dx2 = dx - math.cos(angle) * (bw / 2 + pad)
        dy2 = dy - math.sin(angle) * (bh / 2 + pad)

        # Offset bidirectional pairs so the two arrows run side-by-side
        if (dst_id, src_id) in edge_pairs:
            perp_x = -math.sin(angle) * 0.016
            perp_y =  math.cos(angle) * 0.016
            sx2 += perp_x; sy2 += perp_y
            dx2 += perp_x; dy2 += perp_y

        ax.annotate("",
                    xy=(dx2, dy2), xytext=(sx2, sy2),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color=THEME.ACCENT,
                                    lw=1.5, mutation_scale=13),
                    zorder=2)
        lbl = e.get("label", "")
        if lbl:
            # Place label near the SOURCE end of the arrow (30% along),
            # offset perpendicular to arrow direction.
            t = 0.28   # 28% from source
            lx_base = sx2 + t * (dx2 - sx2)
            ly_base = sy2 + t * (dy2 - sy2)
            perp_x = -math.sin(angle)
            perp_y =  math.cos(angle)
            # Flip perpendicular so label always goes to the RIGHT of the
            # arrow's travel direction (consistent for TB and LR layouts).
            # For TB (mostly vertical, angle ≈ -π/2): perp_x=-1 or +1
            # We want the label on the right side of downward arrows.
            if math.cos(angle) < 0:   # arrow going left → flip perp
                perp_x, perp_y = -perp_x, -perp_y
            offset = 0.055
            lx = lx_base + perp_x * offset
            ly = ly_base + perp_y * offset
            # ha: left if label is to the right of arrow, right if to the left
            ha = "left" if perp_x > 0 else ("right" if perp_x < 0 else "center")
            ax.text(lx, ly, lbl,
                    ha=ha, va="center",
                    fontsize=THEME.FS_SMALL, color=THEME.INK, style="italic",
                    fontweight="bold",
                    transform=ax.transAxes,
                    bbox=dict(facecolor=THEME.BG, edgecolor="none", pad=1.5),
                    zorder=6)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    for n in nodes:
        nid = n["id"]
        if nid not in pos:
            continue
        x, y = pos[nid]
        _draw_node(ax, x, y, n.get("label", nid),
                   n.get("shape", "rect"), n.get("color", THEME.SURFACE),
                   bw, bh, fontsize=node_fs)

    if title:
        fig.text(0.5, 0.98, title, ha="center", va="top",
                 fontsize=THEME.FS_H1, color=THEME.INK, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=THEME.DPI, bbox_inches="tight",
                facecolor=THEME.BG)
    # Also export as PDF (vector, lossless) alongside the PNG
    pdf_path = str(output_path).replace(".png", ".pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor=THEME.BG)
    plt.close(fig)
    return f"Flowchart saved: {output_path}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Render directed flowchart PNG")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    src = args.spec.strip()
    spec = json.loads(src) if src.startswith("{") else json.load(open(src, encoding="utf-8"))
    print(render_flowchart(spec, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
