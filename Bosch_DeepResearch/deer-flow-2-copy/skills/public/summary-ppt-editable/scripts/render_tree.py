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


def _all_nodes(node: Dict) -> List[Dict]:
    """Flatten all descendants."""
    result = []
    for child in node.get("children", []):
        result.append(child)
        result.extend(_all_nodes(child))
    return result


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

    # Use data coordinates for symmetric layout
    pad_x = 0.6
    pad_y = 0.7
    fig, ax = plt.subplots(figsize=(fw, fh))
    fig.patch.set_facecolor(THEME.BG)
    ax.set_facecolor(THEME.BG)
    ax.axis("off")
    ax.set_xlim(-pad_x, max(n_leaves - 1, 0) + pad_x)
    ax.set_ylim(-pad_y, max_depth + pad_y)
    ax.set_aspect("auto")

    def get_data_pos(raw_x, raw_y):
        if direction == "LR":
            return float(max_depth - raw_y), float(n_leaves - 1 - raw_x)
        return float(raw_x), float(max_depth - raw_y)

    # Node box size (data units) — adaptive to figure and tree size
    all_nd = [root] + _all_nodes(root)
    max_lbl_len = max((len(str(n.get("label", "x"))) for n in all_nd), default=4)
    x_span = max(n_leaves - 1, 1) + 2 * pad_x
    y_span = max(max_depth, 1) + 2 * pad_y
    # physical size per data unit (approx)
    px_per_xu = fw / x_span
    py_per_yu = fh / y_span
    nw = min(px_per_xu * 0.80, 1.6) / px_per_xu * (fw / fw)   # in data units
    nw = min(0.80 / (x_span / max(n_leaves,1)), x_span * 0.12)
    nh = min(0.45 / (y_span / max(max_depth+1,1)), y_span * 0.09)
    nw = max(nw, 0.20)
    nh = max(nh, 0.10)

    # Adaptive font
    fs_root = max(THEME.FS_SMALL, min(THEME.FS_H2, 11.0 - max_lbl_len * 0.15))
    fs_node = max(THEME.FS_MICRO + 1, fs_root - 1.5)

    def _draw_node(node: Dict, ax) -> None:
        nid = node["_id"]
        raw_x, raw_y = positions[nid]
        dx, dy = get_data_pos(raw_x, raw_y)

        color = node.get("color", THEME.SURFACE)
        label = node.get("label", "")
        is_root = (nid == "0")
        border_color = THEME.ACCENT if is_root else THEME.BORDER
        lw = 2.0 if is_root else 1.2

        rect = mpatches.FancyBboxPatch(
            (dx - nw / 2, dy - nh / 2), nw, nh,
            boxstyle="round,pad=0.025",
            facecolor=color, edgecolor=border_color, linewidth=lw,
            zorder=4
        )
        ax.add_patch(rect)
        ax.text(dx, dy, label, ha="center", va="center",
                fontsize=fs_root if is_root else fs_node,
                color=THEME.INK, fontweight="bold",
                zorder=5, multialignment="center")

        for ci, child in enumerate(node.get("children", [])):
            cid = f"{nid}_{ci}"
            crx, cry = positions[cid]
            cdx, cdy = get_data_pos(crx, cry)
            ax.plot([dx, cdx], [dy - nh / 2, cdy + nh / 2],
                    color=THEME.BORDER, lw=1.0, zorder=2)
            _draw_node(child, ax)

    _draw_node(root, ax)

    if title:
        fig.text(0.5, 0.99, title, ha="center", va="top",
                 fontsize=THEME.FS_TITLE, color=THEME.INK, fontweight="bold")

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
