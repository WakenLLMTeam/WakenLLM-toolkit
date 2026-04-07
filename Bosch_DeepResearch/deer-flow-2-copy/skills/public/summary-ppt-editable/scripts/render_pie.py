#!/usr/bin/env python3
"""
Render a nested (multi-ring) pie / donut chart PNG.

Supports 1, 2, or 3 concentric rings.  The visual style is preserved from
the original design: bold rotated labels, auto-sizing fonts, arrow callouts
for thin slices, and an optional outer ring that inherits colors from the
middle ring to show hierarchy.

Standalone — no DeerFlow dependency required.

Spec format:
{
  "type": "pie",
  "title": "Dataset Topic Distribution",
  "font": "Arial",             // optional, default Arial (Comic Sans not universal)
  "start_angle": -78.2,       // optional, default -90 (12 o'clock)
  "ring_width": 0.26,         // optional, default 0.26 (fraction of radius)
  "rings": [
    {
      "name": "inner",
      "slices": [
        {"label": "Fact-based",  "value": 43.66, "color": "#ABCBDF"},
        {"label": "Story-based", "value": 56.34, "color": "#F0C284"}
      ]
    },
    {
      "name": "middle",
      "slices": [
        {"label": "FLD",     "value": 21.13, "color": "#D6EFF4"},
        {"label": "FOLIO",   "value": 22.53, "color": "#B8E0EA"},
        {"label": "Science", "value": 28.17, "color": "#F5EBAE"},
        {"label": "Arts",    "value": 28.17, "color": "#F7FBC9"}
      ]
    },
    {
      "name": "outer",
      "inherit_colors": true,   // if true, each slice inherits color from parent middle slice
      "slices": [
        {"label": "Logic",       "value": 8.2,  "parent": "FLD"},
        {"label": "Ethics",      "value": 12.9, "parent": "FLD"},
        {"label": "Law",         "value": 9.1,  "parent": "FOLIO"},
        {"label": "Philosophy",  "value": 13.4, "parent": "FOLIO"},
        {"label": "Physics",     "value": 14.0, "parent": "Science"},
        {"label": "Biology",     "value": 14.2, "parent": "Arts"}
      ]
    }
  ],
  "min_slice_pct": 3.0,       // slices below this % are collapsed into "Other"
  "fig_size": 12,              // single number → square figure
  "dpi": 300
}

Simple single-ring example:
{
  "type": "pie",
  "title": "Market Share 2024",
  "rings": [
    {"name": "market", "slices": [
      {"label": "Tesla",   "value": 18.0, "color": "#E20015"},
      {"label": "BYD",     "value": 35.0, "color": "#16a34a"},
      {"label": "Others",  "value": 47.0, "color": "#dbeafe"}
    ]}
  ]
}
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from viz_theme import THEME, setup_matplotlib

setup_matplotlib()

# ── Default pastel palette (cycles if no color specified) ─────────────────────
_PALETTE = [
    "#ABCBDF", "#F0C284", "#D6EFF4", "#F5EBAE",
    "#F7FBC9", "#B8E0EA", "#fce7f3", "#dcfce7",
    "#ede9fe", "#ffedd5", "#dbeafe", "#f3e8ff",
    "#fef9c3", "#fee2e2", "#ecfdf5", "#fdf4ff",
]


def _auto_color(idx: int, user_color: Optional[str]) -> str:
    if user_color:
        return user_color
    return _PALETTE[idx % len(_PALETTE)]


def _collapse_small(slices: List[Dict], min_pct: float, total: float,
                    other_color: str = "#e5e7eb") -> List[Dict]:
    """Merge slices below min_pct of total into 'Other'."""
    kept, other_val = [], 0.0
    for s in slices:
        pct = s["value"] / total * 100 if total else 0
        if pct < min_pct:
            other_val += s["value"]
        else:
            kept.append(s)
    if other_val > 0:
        kept.append({"label": "Other", "value": other_val, "color": other_color})
    return kept


def _label_text(label: str, value: float, total: float) -> str:
    pct = value / total * 100 if total else 0
    # Auto line-break long labels
    words = label.split()
    if len(label) > 12 and len(words) > 1:
        mid = len(words) // 2
        label = " ".join(words[:mid]) + "\n" + " ".join(words[mid:])
    return f"{label}\n{pct:.1f}%"


def _adaptive_fontsize(arc_deg: float, radius: float,
                       n_chars: int, base_pt: float,
                       min_pt: float = 6.5) -> float:
    """Estimate readable font size given arc length and label width."""
    arc_len = abs(math.radians(arc_deg)) * radius
    # Each char ≈ 0.4 × (pt/72) inches; arc_len in axes units ≈ inches at fw=12
    estimated_pt = arc_len / (max(n_chars, 1) * 0.012)
    return max(min(estimated_pt, base_pt), min_pt)


def render_pie(spec: Dict[str, Any], output_path: str) -> str:
    title: Optional[str] = spec.get("title")
    rings_spec: List[Dict] = spec.get("rings", [])
    font: str = spec.get("font", "Comic Sans MS")
    start_angle: float = float(spec.get("start_angle", -90.0))
    ring_width: float = float(spec.get("ring_width", 0.26))
    min_slice_pct: float = float(spec.get("min_slice_pct", 2.5))
    fig_sz = spec.get("fig_size", 12)
    fw = fh = float(fig_sz) if not isinstance(fig_sz, (list, tuple)) else float(fig_sz[0])
    dpi: int = int(spec.get("dpi", THEME.DPI))

    if not rings_spec:
        raise ValueError("pie spec requires at least one ring")

    n_rings = len(rings_spec)

    # ── Pre-process each ring ─────────────────────────────────────────────────
    processed_rings: List[Dict] = []
    for ri, rspec in enumerate(rings_spec):
        raw_slices = rspec.get("slices", [])
        total = sum(s.get("value", 0) for s in raw_slices)
        if total <= 0:
            total = 1

        # Assign default colors
        for si, s in enumerate(raw_slices):
            s["_color"] = _auto_color(si + ri * 4, s.get("color"))

        # Collapse small slices
        collapsed = _collapse_small(raw_slices, min_slice_pct, total)
        total_collapsed = sum(s.get("value", 0) for s in collapsed)

        processed_rings.append({
            "name": rspec.get("name", f"ring{ri}"),
            "slices": collapsed,
            "total": total_collapsed,
            "inherit_colors": rspec.get("inherit_colors", False),
        })

    # ── Build outer ring color map from parent ─────────────────────────────────
    # If outer ring uses inherit_colors, map parent label → middle ring color
    parent_color_map: Dict[str, str] = {}
    if n_rings >= 2:
        middle = processed_rings[-2]  # second-to-last = middle
        for s in middle["slices"]:
            parent_color_map[s["label"]] = s["_color"]

    if n_rings >= 2:
        outer = processed_rings[-1]
        if outer["inherit_colors"]:
            for s in outer["slices"]:
                parent = s.get("parent", "")
                if parent in parent_color_map:
                    s["_color"] = parent_color_map[parent]

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(fw, fh))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_aspect("equal")

    outer_radius = 1.0
    inner_radius = outer_radius - n_rings * ring_width

    # ── Draw rings from outer to inner ────────────────────────────────────────
    for ri, ring in enumerate(reversed(processed_rings)):
        r_outer = outer_radius - ri * ring_width
        r_inner = r_outer - ring_width
        radius_mid = (r_outer + r_inner) / 2

        slices = ring["slices"]
        values = [s.get("value", 0) for s in slices]
        colors = [s["_color"] for s in slices]
        total = ring["total"]

        # Single ring: draw solid pie (no hole); multi-ring: draw donut ring
        if n_rings == 1:
            wedges, _ = ax.pie(
                values,
                radius=r_outer,
                colors=colors,
                labels=None,
                startangle=start_angle,
                wedgeprops=dict(edgecolor="white", linewidth=1.5),
                counterclock=False,
            )
        else:
            wedges, _ = ax.pie(
                values,
                radius=r_outer,
                colors=colors,
                labels=None,
                startangle=start_angle,
                wedgeprops=dict(width=ring_width, edgecolor="white", linewidth=1.5),
                counterclock=False,
            )

        # ── Labels ────────────────────────────────────────────────────────────
        for w, s, val in zip(wedges, slices, values):
            theta = (w.theta2 + w.theta1) / 2.0
            theta_rad = math.radians(theta)
            x = radius_mid * math.cos(theta_rad)
            y = radius_mid * math.sin(theta_rad)

            arc_deg = abs(w.theta2 - w.theta1)
            label_str = _label_text(s["label"], val, total)
            n_chars = max(len(line) for line in label_str.split("\n"))

            # Base font size: innermost ring larger, adapt to ring width
            # ring_width in axes units ≈ inches at fw=12; 1 inch ≈ 72pt
            ring_w_pt = ring_width * fw * 72 / 2.0   # rough pt per ring height
            base_pt = min(ring_w_pt * 0.28, 18 - ri * 2)
            base_pt = max(base_pt, 10.0)
            fs = _adaptive_fontsize(arc_deg, radius_mid, n_chars, base_pt, min_pt=6.5)

            # Rotation: align with slice tangent
            rot = theta - 90
            if ri > 0:  # outer rings: flip if on left half
                pass
            if 90 < rot % 360 < 270:
                rot += 180

            # Large slice: label inside
            if arc_deg >= 8:
                ax.text(x, y, label_str,
                        ha="center", va="center",
                        fontsize=fs, fontweight="bold",
                        color="black",
                        rotation=rot, rotation_mode="anchor",
                        fontfamily=font,
                        zorder=10)
            else:
                # Small slice: callout arrow
                scale = 1.05 + ri * 0.12
                x_out = scale * r_outer * math.cos(theta_rad)
                y_out = scale * r_outer * math.sin(theta_rad)
                ha_align = "left" if x_out >= 0 else "right"
                ax.annotate(
                    label_str,
                    xy=(x, y),
                    xytext=(x_out, y_out),
                    arrowprops=dict(arrowstyle="->", lw=0.7, color="gray"),
                    ha=ha_align, va="center",
                    fontsize=max(fs, 7.0), fontweight="bold",
                    fontfamily=font,
                    zorder=10,
                )

    # ── Fill center hole for multi-ring charts ───────────────────────────────
    # The innermost donut ring leaves a white hole; fill it with a white disk
    # so there is no visible gap at the center.
    if n_rings > 1:
        inner_r = outer_radius - n_rings * ring_width
        if inner_r > 0:
            fill = plt.Circle((0, 0), inner_r, color="white", zorder=8)
            ax.add_patch(fill)

    # ── Title ─────────────────────────────────────────────────────────────────
    if title:
        fig.text(0.5, 0.97, title,
                 ha="center", va="top",
                 fontsize=THEME.FS_TITLE + 2, color=THEME.INK,
                 fontweight="bold", fontfamily=font)

    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95 if title else 1.0])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    pdf_path = str(output_path).replace(".png", ".pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    return f"Pie chart saved: {output_path}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Render nested pie/donut chart PNG")
    parser.add_argument("--spec", required=True,
                        help="JSON spec file path or inline JSON string")
    parser.add_argument("--output", required=True, help="Output PNG path")
    args = parser.parse_args()
    src = args.spec.strip()
    spec = json.loads(src) if src.startswith("{") else json.load(open(src, encoding="utf-8"))
    print(render_pie(spec, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
