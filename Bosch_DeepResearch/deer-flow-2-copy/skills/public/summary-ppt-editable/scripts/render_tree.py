#!/usr/bin/env python3
"""
Render a hierarchical tree diagram PNG (vertical or horizontal).
Standalone — no DeerFlow dependency required.

Spec format:
{
  "type": "tree",
  "title": "ADAS Software Module Hierarchy",
  "direction": "TB",   // "TB" top-to-bottom | "LR" left-to-right
  "root": {
    "label": "ADAS Stack",
    "color": "#dbeafe",
    "children": [
      {
        "label": "Perception",
        "color": "#dcfce7",
        "children": [
          {"label": "Camera"},
          {"label": "Radar"},
          {"label": "LiDAR"}
        ]
      },
      {
        "label": "Planning",
        "color": "#fef9c3",
        "children": [
          {"label": "Prediction"},
          {"label": "Path Plan"}
        ]
      },
      {
        "label": "Control",
        "color": "#f3e8ff",
        "children": [
          {"label": "Lateral"},
          {"label": "Longitudinal"}
        ]
      }
    ]
  },
  "fig_width": 11,
  "fig_height": 6
}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from viz_theme import THEME, setup_matplotlib

setup_matplotlib()


def _assign_positions(node: Dict, depth: int, pos_counter: List[float],
                      positions: Dict, parent_id: Optional[str] = None,
                      id_prefix: str = "0") -> None:
    """Recursively assign x (leaf index) and y (depth) to each node."""
    node_id = id_prefix
    children = node.get("children", [])
    if not children:
        # Leaf: assign next horizontal position
        x = pos_counter[0]
        pos_counter[0] += 1.0
    else:
        for ci, child in enumerate(children):
            _assign_positions(child, depth + 1, pos_counter, positions,
                              node_id, f"{id_prefix}_{ci}")
        # Parent x = average of children
        child_xs = [positions[f"{id_prefix}_{ci}"][0] for ci in range(len(children))]
        x = sum(child_xs) / len(child_xs)

    positions[node_id] = (x, depth)
    node["_id"] = node_id


def render_tree(spec: Dict[str, Any], output_path: str) -> str:
    title: Optional[str] = spec.get("title")
    root: Dict[str, Any] = spec.get("root", {})
    direction: str = spec.get("direction", "TB").upper()
    fw = float(spec.get("fig_width", 11.0))
    fh = float(spec.get("fig_height", 6.0))

    if not root:
        raise ValueError("tree requires a root node")

    positions: Dict[str, Tuple[float, float]] = {}
    pos_counter = [0.0]
    _assign_positions(root, 0, pos_counter, positions)

    n_leaves = pos_counter[0]
    max_depth = max(v[1] for v in positions.values())

    fig, ax = plt.subplots(figsize=(fw, fh))
    fig.patch.set_facecolor(THEME.BG)
    ax.set_facecolor(THEME.BG)
    ax.axis("off")

    # Normalize positions to [0,1]
    def norm(x, y):
        nx = x / max(n_leaves - 1, 1)
        ny = 1.0 - y / max(max_depth, 1)
        if direction == "LR":
            return ny, nx
        return nx, ny

    # Node sizes
    node_w = min(0.16, 0.85 / max(n_leaves, 1))
    node_h = min(0.10, 0.75 / max(max_depth + 1, 1))

    def _draw_node(node: Dict, ax) -> None:
        nid = node["_id"]
        raw_x, raw_y = positions[nid]
        nx, ny = norm(raw_x, raw_y)

        color = node.get("color", THEME.SURFACE)
        label = node.get("label", "")
        is_root = nid == "0"

        border_color = THEME.ACCENT if is_root else THEME.BORDER
        lw = 1.5 if is_root else 0.9
        rect = mpatches.FancyBboxPatch(
            (nx - node_w / 2, ny - node_h / 2), node_w, node_h,
            boxstyle="round,pad=0.012",
            facecolor=color, edgecolor=border_color, linewidth=lw,
            transform=ax.transAxes, zorder=4
        )
        ax.add_patch(rect)
        fs = THEME.FS_BODY if is_root else THEME.FS_SMALL
        ax.text(nx, ny, label, ha="center", va="center",
                fontsize=fs, color=THEME.INK,
                fontweight="bold" if is_root else "normal",
                transform=ax.transAxes, zorder=5,
                multialignment="center")

        for ci, child in enumerate(node.get("children", [])):
            cid = f"{nid}_{ci}"
            crx, cry = positions[cid]
            cnx, cny = norm(crx, cry)
            # Draw edge
            ax.plot([nx, cnx], [ny - node_h / 2, cny + node_h / 2],
                    color=THEME.BORDER, lw=0.9, zorder=2,
                    transform=ax.transAxes)
            _draw_node(child, ax)

    _draw_node(root, ax)

    if title:
        fig.text(0.5, 0.99, title, ha="center", va="top",
                 fontsize=THEME.FS_H1, color=THEME.INK, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95 if title else 1.0])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=THEME.DPI, bbox_inches="tight", facecolor=THEME.BG)
    pdf_path = str(output_path).replace(".png", ".pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor=THEME.BG)
    plt.close(fig)
    return f"Tree diagram saved: {output_path}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Render tree diagram PNG")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    src = args.spec.strip()
    spec = json.loads(src) if src.startswith("{") else json.load(open(src, encoding="utf-8"))
    print(render_tree(spec, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
