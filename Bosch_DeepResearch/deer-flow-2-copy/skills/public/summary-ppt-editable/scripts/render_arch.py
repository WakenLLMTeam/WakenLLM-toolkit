#!/usr/bin/env python3
"""
Render a layered system architecture diagram PNG.
Standalone — no DeerFlow dependency required.

Designed for Bosch ADAS context: visualising ECU/SoC/software stack
layered architectures, sensor-to-actuator chains, etc.

Usage:
  python render_arch.py --spec arch.json --output out.png

Spec format:
{
  "type": "arch",
  "title": "ADAS System Architecture",
  "direction": "TB",
  "layers": [
    {
      "name": "Application Layer",
      "color": "#dbeafe",
      "blocks": [
        {"label": "Path Planning", "sublabel": "Behavior + Route", "badge": "SW"},
        {"label": "Scene Understanding", "sublabel": "Semantic Fusion"}
      ]
    },
    {
      "name": "Middleware / OS",
      "color": "#fef9c3",
      "blocks": [
        {"label": "AUTOSAR AP", "sublabel": "ARA::COM", "badge": "ASIL-B"},
        {"label": "ROS2 Bridge"}
      ]
    },
    {
      "name": "Hardware Abstraction",
      "color": "#dcfce7",
      "blocks": [
        {"label": "SoC (EyeQ6)", "sublabel": "12 TOPS", "badge": "HW"},
        {"label": "MCU Safety", "sublabel": "Lockstep", "badge": "ASIL-D"}
      ]
    },
    {
      "name": "Sensor Input",
      "color": "#f3e8ff",
      "blocks": [
        {"label": "Camera ×8", "sublabel": "1MP · 30fps"},
        {"label": "Radar ×5",  "sublabel": "300m range"},
        {"label": "LiDAR ×1",  "sublabel": "128-line"}
      ]
    }
  ],
  "connections": [
    {"from_layer": 3, "to_layer": 2, "label": "raw data"}
  ],
  "fig_width": 13,
  "fig_height": 5.5
}

Fields:
  direction       "TB" (top-to-bottom, default) or "BT" (bottom-to-top, sensor→app)
  layers[]        Each layer is a horizontal band containing blocks
  layers[].name   Layer name shown on left strip
  layers[].color  Background fill for the band
  layers[].blocks Blocks within the band
  blocks[].label     Primary text (≤15 chars)
  blocks[].sublabel  Secondary text (optional, ≤20 chars)
  blocks[].badge     Top-right small badge (optional, ≤8 chars)
  connections[]   Optional inter-layer arrows (by 0-based layer index, top=0)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

from viz_theme import THEME, setup_matplotlib

setup_matplotlib()


_LAYER_COLORS = [
    "#dbeafe", "#dcfce7", "#fef9c3", "#f3e8ff",
    "#fce7f3", "#ffedd5", "#ecfdf5", "#ede9fe",
]


def _is_dark(hex_color: str) -> bool:
    h = hex_color.lstrip("#")
    if len(h) < 6:
        return False
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (0.299 * r + 0.587 * g + 0.114 * b) < 160


def render_arch(spec: Dict[str, Any], output_path: str) -> str:
    layers: List[Dict[str, Any]] = spec.get("layers", [])
    title: Optional[str] = spec.get("title")
    direction = spec.get("direction", "TB").upper()
    connections: List[Dict[str, Any]] = spec.get("connections", [])

    if not layers:
        raise ValueError("arch spec requires at least one layer")

    n_layers = len(layers)
    fw = float(spec.get("fig_width", 13.0))
    fh = float(spec.get("fig_height", max(4.5, n_layers * 1.3)))

    # If BT (bottom-to-top): reverse layers for rendering, adjust connections
    if direction == "BT":
        layers = list(reversed(layers))
        connections = [
            {**c,
             "from_layer": n_layers - 1 - c.get("from_layer", 0),
             "to_layer": n_layers - 1 - c.get("to_layer", 0)}
            for c in connections
        ]

    fig, ax = plt.subplots(figsize=(fw, fh))
    fig.patch.set_facecolor(THEME.BG)
    ax.set_facecolor(THEME.BG)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    layer_h = 1.0 / n_layers          # height of each band in axes coords
    gap = 0.008                        # gap between bands
    label_strip_w = 0.11               # left strip for layer name
    content_left = label_strip_w + 0.01
    content_w = 1.0 - content_left - 0.01

    # Store band centre-y for drawing connections later
    band_centers: List[float] = []

    for li, layer in enumerate(layers):
        band_top = 1.0 - li * layer_h
        band_bot = band_top - layer_h + gap
        band_mid = (band_top + band_bot) / 2
        band_centers.append(band_mid)

        # Band background
        bg_color = layer.get("color", _LAYER_COLORS[li % len(_LAYER_COLORS)])
        if _is_dark(bg_color):
            bg_color = _LAYER_COLORS[li % len(_LAYER_COLORS)]

        ax.add_patch(FancyBboxPatch(
            (label_strip_w, band_bot), 1.0 - label_strip_w, band_top - band_bot - gap,
            boxstyle="round,pad=0.005",
            facecolor=bg_color, edgecolor=THEME.BORDER, linewidth=0.8,
            transform=ax.transAxes, zorder=1
        ))

        # Left name strip
        ax.add_patch(FancyBboxPatch(
            (0.0, band_bot), label_strip_w - 0.005, band_top - band_bot - gap,
            boxstyle="round,pad=0.003",
            facecolor=THEME.ACCENT + "22", edgecolor=THEME.ACCENT + "66", linewidth=0.8,
            transform=ax.transAxes, zorder=2
        ))
        ax.text(label_strip_w / 2, band_mid,
                (layer.get("name") or "").replace(" ", "\n"),
                ha="center", va="center",
                fontsize=THEME.FS_SMALL, color=THEME.INK,
                fontweight="bold", rotation=0,
                multialignment="center", linespacing=1.2,
                transform=ax.transAxes, zorder=3)

        # Blocks within band
        blocks: List[Dict[str, Any]] = layer.get("blocks", [])
        k = max(1, len(blocks))
        block_gap = 0.012
        block_w = (content_w - block_gap * (k - 1)) / k
        block_h_frac = (band_top - band_bot - gap) * 0.72
        block_y = band_mid - block_h_frac / 2

        label_fs = max(THEME.FS_MICRO + 0.5, THEME.FS_BODY - max(0, k - 4) * 0.4)
        sub_fs = max(THEME.FS_MICRO, THEME.FS_SMALL - max(0, k - 4) * 0.3)
        badge_fs = THEME.FS_MICRO

        for bi, block in enumerate(blocks):
            bx = content_left + bi * (block_w + block_gap)

            # Shadow
            ax.add_patch(FancyBboxPatch(
                (bx + 0.003, block_y - 0.004), block_w, block_h_frac,
                boxstyle="round,pad=0.01",
                facecolor="#00000012", edgecolor="none",
                transform=ax.transAxes, zorder=2
            ))
            # Card
            ax.add_patch(FancyBboxPatch(
                (bx, block_y), block_w, block_h_frac,
                boxstyle="round,pad=0.01",
                facecolor=THEME.BG, edgecolor=THEME.BORDER, linewidth=0.9,
                transform=ax.transAxes, zorder=3
            ))

            # Accent top bar
            bar_h = block_h_frac * 0.10
            ax.add_patch(FancyBboxPatch(
                (bx, block_y + block_h_frac - bar_h), block_w, bar_h,
                boxstyle="square,pad=0",
                facecolor=THEME.ACCENT + "44", edgecolor="none",
                transform=ax.transAxes, zorder=4
            ))

            label = block.get("label", "")
            sublabel = block.get("sublabel", "")
            badge = block.get("badge", "")

            label_y = band_mid + (0.012 if sublabel else 0)
            ax.text(bx + block_w / 2, label_y, label,
                    ha="center", va="center",
                    fontsize=label_fs, color=THEME.INK, fontweight="bold",
                    transform=ax.transAxes, zorder=5)

            if sublabel:
                ax.text(bx + block_w / 2, band_mid - 0.022, sublabel,
                        ha="center", va="center",
                        fontsize=sub_fs, color=THEME.MUTED,
                        transform=ax.transAxes, zorder=5)

            if badge:
                ax.text(bx + block_w - 0.006, block_y + block_h_frac - 0.010,
                        badge,
                        ha="right", va="top",
                        fontsize=badge_fs, color=THEME.ACCENT, fontweight="bold",
                        transform=ax.transAxes, zorder=6)

    # ── Inter-layer connections ───────────────────────────────────────────────
    for conn in connections:
        fl = conn.get("from_layer", 0)
        tl = conn.get("to_layer", 0)
        lbl = conn.get("label", "")
        if fl >= n_layers or tl >= n_layers:
            continue
        y_from = band_centers[fl]
        y_to = band_centers[tl]
        x_center = 0.5 + content_left / 2
        ax.annotate("",
                    xy=(x_center, y_to + 0.02 * np.sign(y_from - y_to)),
                    xytext=(x_center, y_from - 0.02 * np.sign(y_from - y_to)),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color=THEME.ACCENT,
                                   lw=1.6, mutation_scale=12),
                    zorder=8)
        if lbl:
            my = (y_from + y_to) / 2
            ax.text(x_center + 0.015, my, lbl,
                    ha="left", va="center",
                    fontsize=THEME.FS_MICRO, color=THEME.MUTED, style="italic",
                    transform=ax.transAxes, zorder=9)

    # ── Title ─────────────────────────────────────────────────────────────────
    if title:
        fig.text(0.5, 0.99, title,
                 ha="center", va="top",
                 fontsize=THEME.FS_H1, color=THEME.INK, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.96], pad=0.3)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=THEME.DPI, bbox_inches="tight",
                facecolor=THEME.BG)
    plt.close(fig)
    return f"Architecture diagram saved: {output_path}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Render layered architecture diagram PNG")
    parser.add_argument("--spec", required=True,
                        help="JSON spec file path or inline JSON string")
    parser.add_argument("--output", required=True, help="Output PNG path")
    args = parser.parse_args()
    src = args.spec.strip()
    spec = json.loads(src) if src.startswith("{") else json.load(open(src, encoding="utf-8"))
    print(render_arch(spec, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
