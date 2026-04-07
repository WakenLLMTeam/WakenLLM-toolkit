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

from viz_theme import THEME, setup_matplotlib, fit_fontsize

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

    if not layers:
        raise ValueError("arch spec requires at least one layer")

    n_layers = len(layers)
    fw = float(spec.get("fig_width", 13.0))
    fh = float(spec.get("fig_height", max(4.5, n_layers * 1.3)))

    # If BT (bottom-to-top): reverse layers for rendering
    if direction == "BT":
        layers = list(reversed(layers))

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

    # Store band centre-y (kept for potential future use)
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
        block_gap = 0.010
        block_w = (content_w - block_gap * (k - 1)) / k
        block_h_frac = (band_top - band_bot - gap) * 0.80
        block_y = band_mid - block_h_frac / 2

        # Adaptive font: shrink when many blocks per row
        # Dynamic font: shrink if too many blocks per row
        label_fs  = max(THEME.FS_MICRO + 0.5, THEME.FS_BODY  - max(0, k - 3) * 0.5)
        sub_fs    = max(THEME.FS_MICRO - 0.5, THEME.FS_SMALL - max(0, k - 3) * 0.4)
        # Further shrink using fit_fontsize based on actual block width in inches
        _block_w_in = block_w * fw
        _block_h_in = block_h_frac * fh
        label_fs = min(label_fs, fit_fontsize("WWWWWWWWWWWWWWW", _block_w_in, _block_h_in * 0.5,
                                              start_pt=label_fs))
        sub_fs   = min(sub_fs,   fit_fontsize("WWWWWWWWWWWWWWWWWWWWWWWWW", _block_w_in, _block_h_in * 0.35,
                                              start_pt=sub_fs))
        # Max chars for sublabel before wrapping (proportional to block width)
        # block_w in axes units; rough mapping: 0.1 ≈ 10 chars at sub_fs
        max_sub_chars = max(10, int(block_w * 95))

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

            label   = block.get("label", "")
            sublabel = block.get("sublabel", "")

            # Wrap sublabel at max_sub_chars using · as a break hint
            def _wrap(text: str, max_chars: int) -> str:
                if len(text) <= max_chars:
                    return text
                # try splitting on · first
                for sep in ("·", "·", ",", " "):
                    mid = len(text) // 2
                    idx = text.find(sep, mid - 4)
                    if 0 < idx < len(text) - 1:
                        return text[:idx + 1].rstrip() + "\n" + text[idx + 1:].lstrip()
                return text[:max_chars] + "\n" + text[max_chars:]

            sub_wrapped = _wrap(sublabel, max_sub_chars) if sublabel else ""
            sub_lines = sub_wrapped.count("\n") + 1 if sub_wrapped else 0

            # Vertical layout inside block:
            # total usable = block_h_frac (in axes coords)
            # label at top-ish, sublabel below, badge top-right
            has_sub = bool(sub_wrapped)
            total_text_h = (label_fs + (sub_fs * sub_lines if has_sub else 0)) / 72 / fh
            label_offset = total_text_h / 2 - (label_fs / 72 / fh / 2)
            label_y = band_mid + label_offset

            ax.text(bx + block_w / 2, label_y, label,
                    ha="center", va="center",
                    fontsize=label_fs, color=THEME.INK, fontweight="bold",
                    transform=ax.transAxes, zorder=5,
                    clip_on=True)

            if sub_wrapped:
                sub_y = label_y - (label_fs / 72 / fh) - (sub_fs * 0.6 / 72 / fh)
                ax.text(bx + block_w / 2, sub_y, sub_wrapped,
                        ha="center", va="top",
                        fontsize=sub_fs, color=THEME.MUTED,
                        transform=ax.transAxes, zorder=5,
                        multialignment="center", linespacing=1.1,
                        clip_on=True)

    # ── Title ─────────────────────────────────────────────────────────────────
    if title:
        fig.text(0.5, 0.99, title,
                 ha="center", va="top",
                 fontsize=THEME.FS_H1, color=THEME.INK, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.96], pad=0.3)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=THEME.DPI, bbox_inches="tight",
                facecolor=THEME.BG)
    # Also export as PDF (vector, lossless) alongside the PNG
    pdf_path = str(output_path).replace(".png", ".pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor=THEME.BG)
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
