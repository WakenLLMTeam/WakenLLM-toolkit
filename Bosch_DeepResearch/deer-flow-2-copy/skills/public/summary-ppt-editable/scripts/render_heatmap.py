#!/usr/bin/env python3
"""
Render a heatmap (color matrix) PNG.
Standalone — no DeerFlow dependency required.

Spec format:
{
  "type": "heatmap",
  "title": "Feature Coverage Matrix",
  "rows": ["Camera", "Radar", "LiDAR", "Ultrasonic"],
  "cols": ["Detection", "Tracking", "Depth", "Night", "Rain", "Cost"],
  "values": [
    [9, 8, 7, 6, 5, 9],
    [7, 8, 8, 8, 9, 7],
    [9, 8, 9, 7, 6, 3],
    [4, 4, 3, 6, 7, 9]
  ],
  "color_scheme": "blue",   // "blue" | "red" | "green" | "diverging"
  "show_values": true,
  "value_format": ".0f",    // Python format string for cell labels
  "fig_width": 10,
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

from viz_theme import THEME, setup_matplotlib, get_morandi_cmap

setup_matplotlib()


def render_heatmap(spec: Dict[str, Any], output_path: str) -> str:
    title: Optional[str] = spec.get("title")
    rows: List[str] = spec.get("rows", [])
    cols: List[str] = spec.get("cols", [])
    values: List[List[float]] = spec.get("values", [])
    color_scheme: str = spec.get("color_scheme", "blue")
    # Cell values are always rendered — show_values in spec is ignored intentionally.
    val_fmt: str = spec.get("value_format", ".0f")

    if not rows or not cols or not values:
        raise ValueError("heatmap requires rows, cols, and values")

    data = np.array(values, dtype=float)
    fw = float(spec.get("fig_width", max(8.0, len(cols) * 1.2)))
    fh = float(spec.get("fig_height", max(3.5, len(rows) * 0.85)))

    fig, ax = plt.subplots(figsize=(fw, fh))
    fig.patch.set_facecolor(THEME.BG)
    ax.set_facecolor(THEME.BG)

    cmap_name = get_morandi_cmap(color_scheme)
    vmin, vmax = float(data.min()), float(data.max())
    # Guard against all-identical values — imshow with vmin==vmax produces
    # a silent uniform (invisible) image; give a ±1 margin to force rendering.
    if vmax <= vmin:
        vmin, vmax = vmin - 1.0, vmax + 1.0
    im = ax.imshow(data, cmap=cmap_name, aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(rows)))
    ax.set_xticklabels(cols, fontsize=THEME.FS_SMALL, color=THEME.INK, rotation=30, ha="right", fontweight="bold")
    ax.set_yticklabels(rows, fontsize=THEME.FS_SMALL, color=THEME.INK, fontweight="bold")
    ax.tick_params(length=0)

    # Cell borders
    for edge in ["top", "bottom", "left", "right"]:
        ax.spines[edge].set_visible(False)

    # Always draw cell value labels
    _mid = (data.min() + data.max()) / 2
    for i in range(len(rows)):
        for j in range(len(cols)):
            val = data[i, j]
            # High-value cells use the dark end of the palette → white text is legible.
            # Low-value cells use the light end → dark ink is legible.
            text_color = "white" if val > _mid else THEME.INK
            ax.text(j, i, format(val, val_fmt),
                    ha="center", va="center",
                    fontsize=THEME.FS_BODY, color=text_color, fontweight="bold")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(labelsize=THEME.FS_MICRO, colors=THEME.MUTED)
    cbar.outline.set_visible(False)

    if title:
        fig.text(0.5, 0.99, title, ha="center", va="top",
                 fontsize=THEME.FS_TITLE, color=THEME.INK, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.94 if title else 1.0], pad=0.5)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=THEME.DPI, bbox_inches="tight", facecolor=THEME.BG)
    pdf_path = str(output_path).replace(".png", ".pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor=THEME.BG)
    plt.close(fig)
    return f"Heatmap saved: {output_path}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Render heatmap PNG")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    src = args.spec.strip()
    spec = json.loads(src) if src.startswith("{") else json.load(open(src, encoding="utf-8"))
    print(render_heatmap(spec, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
