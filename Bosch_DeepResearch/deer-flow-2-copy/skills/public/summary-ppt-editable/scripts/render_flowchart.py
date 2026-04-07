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
    {"id": "s",    "label": "Perception\nFusion",       "shape": "rect",    "color": "#dbeafe"},
    {"id": "odd",  "label": "ODD Boundary\nCheck",      "shape": "diamond", "color": "#fef9c3"},
    {"id": "plan", "label": "Path Planning",             "shape": "rect",    "color": "#dcfce7"},
    {"id": "tor",  "label": "Takeover Request\n(TOR)",  "shape": "rect",    "color": "#fee2e2"},
    {"id": "mrc",  "label": "Minimal Risk\nCondition",  "shape": "rounded", "color": "#fce7f3"},
    {"id": "ctrl", "label": "Execution\nControl",        "shape": "rect",    "color": "#ede9fe"}
  ],
  "edges": [
    {"from": "s",    "to": "odd"},
    {"from": "odd",  "to": "plan", "label": "within ODD"},
    {"from": "odd",  "to": "tor",  "label": "ODD exit"},
    {"from": "tor",  "to": "mrc",  "label": "timeout"},
    {"from": "plan", "to": "ctrl"},
    {"from": "mrc",  "to": "ctrl"}
  ],
  "fig_width": 11,
  "fig_height": 6
}

Shapes: "rect" | "diamond" | "rounded"
Layout: "LR" (left-to-right) | "TB" (top-to-bottom)
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

from viz_theme import THEME, setup_matplotlib

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
    # Shadow
    if shape == "diamond":
        pts = np.array([
            [x + 0.003, y + bh * 0.65],
            [x + bw * 0.58, y],
            [x + 0.003, y - bh * 0.65],
            [x - bw * 0.55, y],
        ])
        ax.add_patch(plt.Polygon(pts, closed=True, facecolor="#00000015",
                                 edgecolor="none", transform=ax.transAxes, zorder=2))
        pts2 = np.array([
            [x, y + bh * 0.65],
            [x + bw * 0.58, y],
            [x, y - bh * 0.65],
            [x - bw * 0.58, y],
        ])
        ax.add_patch(plt.Polygon(pts2, closed=True, facecolor=bg,
                                 edgecolor=THEME.BORDER, linewidth=1.2,
                                 transform=ax.transAxes, zorder=3))
    elif shape == "rounded":
        ax.add_patch(FancyBboxPatch(
            (x - bw / 2 + 0.003, y - bh / 2 - 0.003), bw, bh,
            boxstyle="round,pad=0.025", facecolor="#00000015",
            edgecolor="none", transform=ax.transAxes, zorder=2))
        ax.add_patch(FancyBboxPatch(
            (x - bw / 2, y - bh / 2), bw, bh,
            boxstyle="round,pad=0.025", facecolor=bg,
            edgecolor=THEME.BORDER, linewidth=1.2,
            transform=ax.transAxes, zorder=3))
    else:
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
            color=THEME.INK, fontweight="bold",
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
    # Font size scales with box size
    node_fs = max(THEME.FS_MICRO + 0.5,
                  min(THEME.FS_BODY, bh * 55 - max(0, max_lines - 1) * 1.5))

    fig, ax = plt.subplots(figsize=(fw, fh))
    fig.patch.set_facecolor(THEME.BG)
    ax.set_facecolor(THEME.BG)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # ── Edges (draw BEFORE nodes so arrows go under node shapes) ─────────────
    for e in edges:
        sx, sy = pos.get(e.get("from", ""), (0, 0))
        dx, dy = pos.get(e.get("to", ""), (0, 0))
        if (sx, sy) == (0, 0) or (dx, dy) == (0, 0):
            continue
        angle = math.atan2(dy - sy, dx - sx)
        # Add extra padding so arrow starts/ends clearly outside the node boundary
        pad = 0.012
        sx2 = sx + math.cos(angle) * (bw / 2 + pad)
        sy2 = sy + math.sin(angle) * (bh / 2 + pad)
        dx2 = dx - math.cos(angle) * (bw / 2 + pad)
        dy2 = dy - math.sin(angle) * (bh / 2 + pad)

        ax.annotate("",
                    xy=(dx2, dy2), xytext=(sx2, sy2),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color=THEME.ACCENT,
                                    lw=1.5, mutation_scale=13),
                    zorder=2)
        lbl = e.get("label", "")
        if lbl:
            mx, my = (sx2 + dx2) / 2, (sy2 + dy2) / 2
            ax.text(mx, my + 0.025, lbl,
                    ha="center", va="bottom",
                    fontsize=THEME.FS_SMALL, color=THEME.MUTED, style="italic",
                    transform=ax.transAxes,
                    bbox=dict(facecolor=THEME.BG, edgecolor="none", pad=1),
                    zorder=3)

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
