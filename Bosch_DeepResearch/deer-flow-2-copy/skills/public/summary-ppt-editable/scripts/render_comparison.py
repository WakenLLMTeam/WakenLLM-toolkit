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

_HEADER_BG        = "#AAB7C4"   # Morandi grey-blue header
_HEADER_BG_HI     = "#8EACCF"   # Morandi slate-blue for highlighted column header
_CELL_HI_BG       = "#dde4ea"   # very light Morandi tint for highlighted data cells
_HEADER_FG = THEME.INK


# ── CJK-aware text utilities ──────────────────────────────────────────────────

def _visual_width(text: str) -> float:
    """Visual character width: CJK/fullwidth chars count as 2, ASCII as 1."""
    w = 0.0
    for c in text:
        cp = ord(c)
        # CJK Unified Ideographs, Katakana, Hiragana, fullwidth punctuation, etc.
        if (0x1100 <= cp <= 0x11FF or   # Hangul Jamo
            0x2E80 <= cp <= 0x9FFF or   # CJK & radicals
            0xA000 <= cp <= 0xA4CF or   # Yi
            0xAC00 <= cp <= 0xD7AF or   # Hangul
            0xF900 <= cp <= 0xFAFF or   # CJK Compatibility
            0xFE30 <= cp <= 0xFE4F or   # CJK Compatibility Forms
            0xFF00 <= cp <= 0xFFEF):    # Fullwidth / Halfwidth
            w += 2.0
        else:
            w += 1.0
    return w


def _wrap_cell_text(text: str, max_visual_w: float) -> str:
    """
    Wrap text so each line's visual width ≤ max_visual_w.
    Works for both CJK (no spaces) and Latin (space-delimited) text.
    Returns newline-joined string.
    """
    text = (text or "").strip()
    if not text or _visual_width(text) <= max_visual_w:
        return text

    lines: List[str] = []
    line = ""
    line_w = 0.0

    # First try space-split (works well for Latin / mixed text)
    tokens = text.split(" ")
    use_char_split = len(tokens) <= 1 and len(text) > 4

    if not use_char_split:
        for tok in tokens:
            tok_w = _visual_width(tok)
            spacer_w = 1.0 if line else 0.0
            if line and line_w + spacer_w + tok_w > max_visual_w:
                lines.append(line)
                line = tok
                line_w = tok_w
            else:
                line = (line + " " + tok).lstrip() if line else tok
                line_w += spacer_w + tok_w
        if line:
            lines.append(line)
        # If any line is still too long, split those character-by-character
        final: List[str] = []
        for l in lines:
            if _visual_width(l) <= max_visual_w:
                final.append(l)
            else:
                sub, sub_w = "", 0.0
                for ch in l:
                    cw = 2.0 if _visual_width(ch) == 2 else 1.0
                    if sub_w + cw > max_visual_w:
                        final.append(sub)
                        sub, sub_w = ch, cw
                    else:
                        sub += ch
                        sub_w += cw
                if sub:
                    final.append(sub)
        return "\n".join(final)
    else:
        # Pure CJK or single long token — split character by character
        for ch in text:
            ch_w = 2.0 if _visual_width(ch) >= 2 else 1.0
            if line_w + ch_w > max_visual_w:
                lines.append(line)
                line = ch
                line_w = ch_w
            else:
                line += ch
                line_w += ch_w
        if line:
            lines.append(line)
        return "\n".join(lines)


def _draw_cell(ax, left, bottom, width, height, text, *,
               bg, fg, bold, fontsize, fig_w: float = 10.0):
    """Draw a cell with background fill and word-wrapped centered text."""
    ax.add_patch(plt.Rectangle(
        (left, bottom), width, height,
        facecolor=bg, edgecolor="none", zorder=2, transform=ax.transAxes))

    if not text:
        return

    # Compute how many visual-width units fit in this column
    col_w_inches   = width * fig_w
    char_w_inches  = fontsize * 0.58 / 72          # approx width of one ASCII char
    max_visual_w   = max(4.0, col_w_inches / char_w_inches * 0.88)

    wrapped = _wrap_cell_text(str(text), max_visual_w)

    ax.text(
        left + width / 2,
        bottom + height / 2,
        wrapped,
        ha="center", va="center",
        fontsize=fontsize,
        color=fg,
        fontweight="bold" if bold else "normal",
        transform=ax.transAxes,
        zorder=3,
        multialignment="center",
        linespacing=1.25,
        clip_on=True,          # hard clip — nothing escapes the cell
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

    # ── Content-aware column widths (CJK-aware) ──────────────────────────────
    # Use visual width (CJK = 2 units, ASCII = 1 unit) for sizing
    row_header_max = max(_visual_width(r) for r in rows)

    col_max_vw = []
    for ci, col_header in enumerate(cols):
        col_cells = [cells[ri][ci] if ri < len(cells) and ci < len(cells[ri]) else ""
                     for ri in range(nr)]
        col_max_vw.append(max(
            _visual_width(col_header),
            max((_visual_width(c) for c in col_cells), default=0),
        ))

    # Width units: row header + data cols (cap each column at 28 visual units
    # to prevent one verbose column from dominating the whole table)
    _CAP = 28.0
    row_hdr_units = max(8.0, min(row_header_max, _CAP))
    data_units    = [max(8.0, min(vw, _CAP)) for vw in col_max_vw]
    if has_notes:
        data_units.append(14.0)

    all_units = [row_hdr_units] + data_units
    total_units = sum(all_units)

    # Figure width: ~0.145 in per visual-unit, clamp 10–22 inches
    fw = float(spec.get("fig_width", 0)) or max(10.0, min(total_units * 0.145, 22.0))

    # Adaptive font size: reduce when many rows or many columns
    body_fs = max(THEME.FS_MICRO + 0.5, THEME.FS_BODY - max(0, nr - 5) * 0.4
                                                       - max(0, nc - 2) * 0.3)
    small_fs = max(THEME.FS_MICRO, THEME.FS_SMALL - max(0, nr - 5) * 0.3)

    # ── Compute per-cell wrap + font-size fitting loop ────────────────────────
    # Strategy:
    #   1. Wrap all cells at the current body_fs.
    #   2. If any row wraps to more than MAX_LINES lines, shrink body_fs by 0.5 pt
    #      and redo — text can fit in fewer lines at larger visual width per char.
    #   3. Stop when every row is within MAX_LINES, or body_fs hits MIN_FS.
    # This guarantees no cell silently overflows its row.

    MAX_LINES: int = 3          # max wrapped lines tolerated per row
    MIN_FS:   float = THEME.FS_MICRO   # hard floor (typically 6.5 pt)

    col_w_units = [float(u) for u in all_units]
    total_w     = sum(col_w_units)

    def _compute_wrapping(fs: float):
        """Wrap all cells at font size fs. Returns (wrapped_cells, wrapped_notes, row_line_counts)."""
        cw_in = fs * 0.58 / 72   # approx inch-width of one ASCII char at this font size
        _wc: List[List[str]] = []
        _wn: List[str]       = []
        _lc: List[int]       = []
        for _ri, _row_label in enumerate(rows):
            _max = 1
            _rw  = []
            for _ci in range(nc):
                _ct   = (cells[_ri][_ci] if _ri < len(cells) and _ci < len(cells[_ri]) else "—") or "—"
                _cwin = (col_w_units[_ci + 1] / total_w) * fw
                _mvw  = max(4.0, _cwin / cw_in * 0.85)
                _wt   = _wrap_cell_text(_ct, _mvw)
                _rw.append(_wt)
                _max  = max(_max, len(_wt.split("\n")))
            _wc.append(_rw)
            # Row label
            _rhin = (col_w_units[0] / total_w) * fw
            _rmvw = max(4.0, _rhin / cw_in * 0.85)
            _rlw  = _wrap_cell_text(_row_label, _rmvw)
            _max  = max(_max, len(_rlw.split("\n")))
            # Note
            if has_notes:
                _note   = (row_notes[_ri] if _ri < len(row_notes) else "").strip()
                _ncwin  = (col_w_units[nc + 1] / total_w) * fw
                _nmvw   = max(4.0, _ncwin / (small_fs * 0.58 / 72) * 0.85)
                _nw     = _wrap_cell_text(_note, _nmvw)
                _wn.append(_nw)
                _max = max(_max, len(_nw.split("\n")))
            else:
                _wn.append("")
            _lc.append(_max)
        return _wc, _wn, _lc

    # Iteratively shrink body_fs until all rows ≤ MAX_LINES (or hit MIN_FS)
    wrapped_cells, wrapped_notes, row_line_counts = _compute_wrapping(body_fs)
    while max(row_line_counts) > MAX_LINES and body_fs > MIN_FS:
        body_fs = max(MIN_FS, body_fs - 0.5)
        wrapped_cells, wrapped_notes, row_line_counts = _compute_wrapping(body_fs)

    # Dynamic row heights: taller rows for multi-line content
    row_h_units = [1.1] + [max(1.0, lc * 0.9) for lc in row_line_counts]
    total_h     = sum(row_h_units)

    # Figure height: scale by total row height units
    fh = float(spec.get("fig_height", 0)) or max(
        3.0, min(0.5 + total_h * 0.62 + (0.45 if title else 0.1), 16.0)
    )

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

    def _dc(ci_ax, ri_ax, text, *, bg, fg, bold, fs):
        _draw_cell(ax, cx(ci_ax), cy(ri_ax), cw(ci_ax), ch(ri_ax),
                   text, bg=bg, fg=fg, bold=bold, fontsize=fs, fig_w=fw)

    # ── Header row ────────────────────────────────────────────────────────────
    _dc(0, 0, "", bg=_HEADER_BG, fg=_HEADER_FG, bold=True, fs=body_fs)
    _final_cw_in = body_fs * 0.58 / 72   # char width at final (possibly reduced) body_fs
    for ci, col in enumerate(cols):
        bg = _HEADER_BG_HI if ci == highlight_col else _HEADER_BG
        col_w_in  = (col_w_units[ci + 1] / total_w) * fw
        col_max_vw = max(4.0, col_w_in / _final_cw_in * 0.85)
        col_wrapped = _wrap_cell_text(col, col_max_vw)
        _dc(ci + 1, 0, col_wrapped, bg=bg, fg=_HEADER_FG, bold=True, fs=body_fs)
    if has_notes:
        _dc(nc + 1, 0, "Note", bg=_HEADER_BG, fg=_HEADER_FG, bold=True, fs=small_fs)

    # ── Data rows (use pre-wrapped text) ──────────────────────────────────────
    for ri, row_label in enumerate(rows):
        row_bg = THEME.ALT_ROW if ri % 2 == 0 else THEME.BG
        # Row label — wrap it too (use final body_fs char width)
        rh_col_in  = (col_w_units[0] / total_w) * fw
        rh_max_vw  = max(4.0, rh_col_in / _final_cw_in * 0.85)
        rl_wrapped = _wrap_cell_text(row_label, rh_max_vw)
        _dc(0, ri + 1, rl_wrapped, bg=row_bg, fg=THEME.INK, bold=True, fs=body_fs)
        for ci in range(nc):
            wt = wrapped_cells[ri][ci] if ri < len(wrapped_cells) and ci < len(wrapped_cells[ri]) else "—"
            bg = _CELL_HI_BG if ci == highlight_col else row_bg
            _dc(ci + 1, ri + 1, wt, bg=bg, fg=THEME.INK,
                bold=(ci == highlight_col), fs=body_fs)
        if has_notes:
            _dc(nc + 1, ri + 1, wrapped_notes[ri], bg=row_bg,
                fg=THEME.MUTED, bold=False, fs=small_fs)

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
    # Also export as PDF (vector, lossless) alongside the PNG
    pdf_path = str(output_path).replace(".png", ".pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor=THEME.BG)
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
