#!/usr/bin/env python3
"""
Render a high-quality comparison table / feature matrix PNG.
Standalone — no DeerFlow dependency required.

Usage:
  python render_comparison.py --spec comp.json --output out.png
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

from viz_theme import THEME, setup_matplotlib

setup_matplotlib()

_HEADER_BG = "#dbeafe"
_HEADER_FG = THEME.INK


def _draw_cell(ax, left, bottom, width, height, text, *,
               bg, fg, bold, fontsize):
    """Draw a cell with background fill and word-wrapped centered text."""
    ax.add_patch(plt.Rectangle(
        (left, bottom), width, height,
        facecolor=bg, edgecolor="none", zorder=2, transform=ax.transAxes))

    if not text:
        return

    # No manual wrapping — rely on the figure being wide enough.
    # matplotlib clip_on=False lets text breathe; the fig width is
    # computed from content so each cell has room for its longest text.
    ax.text(
        left + width / 2,
        bottom + height / 2,
        text,
        ha="center", va="center",
        fontsize=fontsize,
        color=fg,
        fontweight="bold" if bold else "normal",
        transform=ax.transAxes,
        zorder=3,
        multialignment="center",
        linespacing=1.2,
        clip_on=False,
    )


def render_comparison(spec: Dict[str, Any], output_path: str) -> str:
    rows: List[str] = spec.get("rows", [])
    cols: List[str] = spec.get("cols", [])
    cells: List[List[str]] = spec.get("cells", [])
    title: Optional[str] = spec.get("title")
    highlight_col: Optional[int] = spec.get("highlight_col")
    row_notes: List[str] = spec.get("row_notes", [""] * len(rows))
    accent = spec.get("accent_color", THEME.ACCENT)

    if not rows or not cols or not cells:
        raise ValueError("spec must have non-empty rows, cols, and cells")

    nr, nc = len(rows), len(cols)
    # Only show notes column if there are substantive notes (non-empty, non-whitespace)
    # AND the user hasn't explicitly disabled it via show_notes: false
    show_notes_flag = spec.get("show_notes", True)
    has_notes = show_notes_flag and any(n.strip() for n in row_notes)

    # ── Content-aware column widths ───────────────────────────────────────────
    # Measure the widest text in each column (header + all cells)
    # row-header col: max of all row labels
    row_header_max = max(len(r) for r in rows)

    col_max_chars = []
    for ci, col_header in enumerate(cols):
        col_cells = [cells[ri][ci] if ri < len(cells) and ci < len(cells[ri]) else ""
                     for ri in range(nr)]
        col_max_chars.append(max(len(col_header), max((len(c) for c in col_cells), default=0)))

    # Assign proportional width units: row-header + data cols
    # Base unit ≈ chars; row header gets its own proportion
    row_hdr_units = max(8, row_header_max)
    data_units    = [max(8, m) for m in col_max_chars]
    if has_notes:
        data_units.append(12)  # fixed note col width

    all_units = [row_hdr_units] + data_units
    total_units = sum(all_units)

    # Figure width: allocate ~0.13 inches per char-unit, clamp 10–20 inches
    fw = max(10.0, min(total_units * 0.145, 20.0))
    # Figure height: adaptive per row count
    fh = 0.65 + nr * 0.62 + (0.45 if title else 0.1)
    fh = max(3.0, min(fh, 13.0))

    # Adaptive font size: reduce when many rows or many columns
    body_fs = max(THEME.FS_MICRO + 0.5, THEME.FS_BODY - max(0, nr - 5) * 0.4
                                                       - max(0, nc - 2) * 0.3)
    small_fs = max(THEME.FS_MICRO, THEME.FS_SMALL - max(0, nr - 5) * 0.3)

    # Normalised column/row proportions
    col_w_units = [float(u) for u in all_units]
    total_w = sum(col_w_units)
    row_h_units = [1.1] + [1.0] * nr
    total_h = sum(row_h_units)

    def cx(ci): return sum(col_w_units[:ci]) / total_w
    def cy(ri): return 1.0 - sum(row_h_units[:ri + 1]) / total_h
    def cw(ci): return col_w_units[ci] / total_w
    def ch(ri): return row_h_units[ri] / total_h

    fig, ax = plt.subplots(figsize=(fw, fh))
    fig.patch.set_facecolor(THEME.BG)
    ax.set_facecolor(THEME.BG)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # ── Header row ────────────────────────────────────────────────────────────
    _draw_cell(ax, cx(0), cy(0), cw(0), ch(0),
               "", bg=_HEADER_BG, fg=_HEADER_FG, bold=True, fontsize=body_fs)
    for ci, col in enumerate(cols):
        bg = "#bfdbfe" if ci == highlight_col else _HEADER_BG
        _draw_cell(ax, cx(ci + 1), cy(0), cw(ci + 1), ch(0),
                   col, bg=bg, fg=_HEADER_FG, bold=True, fontsize=body_fs)
    if has_notes:
        _draw_cell(ax, cx(nc + 1), cy(0), cw(nc + 1), ch(0),
                   "Note", bg=_HEADER_BG, fg=_HEADER_FG, bold=True,
                   fontsize=small_fs)

    # ── Data rows ─────────────────────────────────────────────────────────────
    for ri, row_label in enumerate(rows):
        row_bg = THEME.ALT_ROW if ri % 2 == 0 else THEME.BG
        _draw_cell(ax, cx(0), cy(ri + 1), cw(0), ch(ri + 1),
                   row_label, bg=row_bg, fg=THEME.INK, bold=True,
                   fontsize=body_fs)
        row_cells = cells[ri] if ri < len(cells) else []
        for ci in range(nc):
            cell_text = row_cells[ci] if ci < len(row_cells) else "—"
            bg = THEME.ACCENT_LIGHT if ci == highlight_col else row_bg
            _draw_cell(ax, cx(ci + 1), cy(ri + 1), cw(ci + 1), ch(ri + 1),
                       cell_text, bg=bg, fg=THEME.INK,
                       bold=(ci == highlight_col), fontsize=body_fs)
        if has_notes:
            note = (row_notes[ri] if ri < len(row_notes) else "").strip()
            # Truncate very long notes to prevent overflow
            if len(note) > 60:
                note = note[:57] + "..."
            _draw_cell(ax, cx(nc + 1), cy(ri + 1), cw(nc + 1), ch(ri + 1),
                       note, bg=row_bg, fg=THEME.MUTED, bold=False,
                       fontsize=small_fs)

    # ── Grid lines ────────────────────────────────────────────────────────────
    total_cols = nc + 2 if has_notes else nc + 1
    for ci in range(total_cols + 1):
        lx = sum(col_w_units[:ci]) / total_w
        ax.axvline(lx, color=THEME.BORDER, linewidth=0.6, zorder=5)
    for ri in range(nr + 2):
        ly = 1.0 - sum(row_h_units[:ri]) / total_h
        ax.axhline(ly, color=THEME.BORDER, linewidth=0.6, zorder=5)

    # ── Highlight column border ───────────────────────────────────────────────
    if highlight_col is not None:
        hx = cx(highlight_col + 1)
        hw = cw(highlight_col + 1)
        ax.add_patch(FancyBboxPatch(
            (hx, 0.0), hw, 1.0,
            boxstyle="square,pad=0",
            facecolor="none", edgecolor=accent,
            linewidth=2.0, transform=ax.transAxes, zorder=6, clip_on=True))

    if title:
        fig.text(0.5, 0.99, title, ha="center", va="top",
                 fontsize=THEME.FS_H1, color=THEME.INK, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95 if title else 1.0], pad=0.2)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=THEME.DPI, bbox_inches="tight",
                facecolor=THEME.BG)
    plt.close(fig)
    return f"Comparison table saved: {output_path}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Render comparison table PNG")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    src = args.spec.strip()
    spec = json.loads(src) if src.startswith("{") else json.load(open(src, encoding="utf-8"))
    print(render_comparison(spec, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
