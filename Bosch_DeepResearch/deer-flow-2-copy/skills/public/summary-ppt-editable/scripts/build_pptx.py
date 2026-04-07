#!/usr/bin/env python3
"""
Build editable-text PPTX from a JSON plan (summary-ppt-editable skill).

Usage:
  python build_pptx.py --plan-file plan.json --output-file out.pptx
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


def _add_picture_fit(slide, img_path: str, box_left, box_top, box_w, box_h):
    """
    Insert an image preserving its original aspect ratio, fitted (letterboxed)
    inside the given bounding box and centred within it.
    Avoids the distortion caused by passing both width and height to add_picture.
    """
    from PIL import Image as _PILImage
    with _PILImage.open(img_path) as im:
        img_w, img_h = im.size

    img_aspect = img_w / img_h
    box_aspect = box_w / box_h

    if img_aspect >= box_aspect:
        # Constrained by width
        fit_w = box_w
        fit_h = box_w / img_aspect
    else:
        # Constrained by height
        fit_h = box_h
        fit_w = box_h * img_aspect

    # Centre within box
    offset_left = (box_w - fit_w) / 2
    offset_top  = (box_h - fit_h) / 2

    slide.shapes.add_picture(
        img_path,
        box_left + offset_left,
        box_top  + offset_top,
        width=fit_w,
        height=fit_h,
    )


def _rgb(t: Optional[List[int]], default: Tuple[int, int, int]) -> RGBColor:
    if t and len(t) == 3:
        return RGBColor(int(t[0]), int(t[1]), int(t[2]))
    return RGBColor(default[0], default[1], default[2])


def _slide_size(aspect: str) -> Tuple[Any, Any]:
    if aspect == "4:3":
        return Inches(10), Inches(7.5)
    return Inches(13.333), Inches(7.5)


def _add_modules(
    text_frame,
    modules: List[Dict[str, Any]],
    *,
    font_name: str,
    body_rgb: RGBColor,
    heading_rgb: RGBColor,
    heading_pt: int,
    body_pt: int,
) -> None:
    """Multi-block text: each module has heading + bullets (editable)."""
    first = True
    for mod in modules:
        if not isinstance(mod, dict):
            continue
        h = (mod.get("heading") or "").strip()
        bs = mod.get("bullets") or []
        if h:
            p = text_frame.paragraphs[0] if first else text_frame.add_paragraph()
            first = False
            p.text = h
            p.font.bold = True
            p.font.size = Pt(heading_pt)
            p.font.color.rgb = heading_rgb
            p.space_after = Pt(4)
            try:
                p.font.name = font_name
            except Exception:
                pass
        for line in bs:
            line = (line or "").strip()
            if not line:
                continue
            p = text_frame.paragraphs[0] if first else text_frame.add_paragraph()
            first = False
            p.text = line
            p.level = 0
            p.font.size = Pt(body_pt)
            p.font.color.rgb = body_rgb
            p.space_after = Pt(6)
            try:
                p.font.name = font_name
            except Exception:
                pass


def _render_slide_body(text_frame, slide: Dict[str, Any], *, font_name: str, body_rgb: RGBColor, title_rgb: RGBColor) -> None:
    modules = slide.get("modules")
    if modules and isinstance(modules, list):
        _add_modules(
            text_frame,
            modules,
            font_name=font_name,
            body_rgb=body_rgb,
            heading_rgb=title_rgb,
            heading_pt=15,
            body_pt=14,
        )
    else:
        _add_bullets(text_frame, slide.get("bullets") or [], font_name=font_name, body_rgb=body_rgb, size_pt=16)


def _add_cards(
    sld,
    cards: List[Dict[str, Any]],
    *,
    left: Any,
    top: Any,
    total_w: Any,
    total_h: Any,
    font_name: str,
    body_rgb: RGBColor,
    heading_rgb: RGBColor,
    accent_rgb: RGBColor,
    card_bg_rgb: Optional[RGBColor] = None,
    cols: int = 0,
) -> None:
    """Render a list of cards as a grid of bordered text boxes.

    Each card dict supports:
      heading   – str, bold header shown at top of card (optional)
      bullets   – list[str], bullet lines (optional)
      icon      – str, single emoji / symbol prepended to heading (optional)
      bg_rgb    – list[int, int, int], per-card background override (optional)

    Layout is auto-computed:
      cols=0  → auto: 1→1 col, 2→2, 3→3, 4→2×2, 5-6→3 cols, 7+→3 cols
    """
    n = len(cards)
    if n == 0:
        return

    if cols <= 0:
        if n <= 1:
            cols = 1
        elif n == 2:
            cols = 2
        elif n <= 3:
            cols = 3
        elif n == 4:
            cols = 2
        else:
            cols = 3
    rows = (n + cols - 1) // cols

    gap = Inches(0.12)
    card_w = (total_w - gap * (cols - 1)) / cols
    card_h = (total_h - gap * (rows - 1)) / rows

    default_bg = card_bg_rgb or RGBColor(245, 247, 250)

    for idx, card in enumerate(cards):
        col_i = idx % cols
        row_i = idx // cols
        cx = left + col_i * (card_w + gap)
        cy = top + row_i * (card_h + gap)

        bg = card.get("bg_rgb")
        bg_color = RGBColor(int(bg[0]), int(bg[1]), int(bg[2])) if bg and len(bg) == 3 else default_bg
        rect = sld.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, cx, cy, card_w, card_h)
        rect.fill.solid()
        rect.fill.fore_color.rgb = bg_color
        rect.line.color.rgb = accent_rgb
        rect.line.width = Pt(1.0)
        rect.adjustments[0] = 0.05

        bar_h = Inches(0.055)
        bar = sld.shapes.add_shape(MSO_SHAPE.RECTANGLE, cx, cy, card_w, bar_h)
        bar.fill.solid()
        bar.fill.fore_color.rgb = accent_rgb
        bar.line.fill.background()

        pad_x = Inches(0.16)
        pad_y = Inches(0.18)
        tb = sld.shapes.add_textbox(cx + pad_x, cy + bar_h + pad_y, card_w - pad_x * 2, card_h - bar_h - pad_y * 2)
        tf = tb.text_frame
        tf.word_wrap = True

        first = True
        heading = (card.get("heading") or "").strip()
        icon = (card.get("icon") or "").strip()
        if heading:
            label = f"{icon}  {heading}" if icon else heading
            p = tf.paragraphs[0] if first else tf.add_paragraph()
            first = False
            p.text = label
            p.font.bold = True
            p.font.size = Pt(13)
            p.font.color.rgb = heading_rgb
            p.space_after = Pt(5)
            try:
                p.font.name = font_name
            except Exception:
                pass

        for line in (card.get("bullets") or []):
            line = (line or "").strip()
            if not line:
                continue
            p = tf.paragraphs[0] if first else tf.add_paragraph()
            first = False
            p.text = f"• {line}"
            p.font.size = Pt(11)
            p.font.color.rgb = body_rgb
            p.space_after = Pt(4)
            try:
                p.font.name = font_name
            except Exception:
                pass


def _add_bullets(text_frame, bullets: List[str], *, font_name: str, body_rgb: RGBColor, size_pt: int) -> None:
    lines = [((x or "").strip()) for x in bullets if (x or "").strip()]
    if not lines:
        text_frame.paragraphs[0].text = ""
        return
    first = True
    for line in lines:
        p = text_frame.paragraphs[0] if first else text_frame.add_paragraph()
        first = False
        p.text = line
        p.level = 0
        p.font.size = Pt(size_pt)
        p.font.color.rgb = body_rgb
        p.space_after = Pt(8)
        try:
            p.font.name = font_name
        except Exception:
            pass


def build_pptx(plan: Dict[str, Any], output_file: str) -> str:
    aspect = plan.get("aspect_ratio", "16:9")
    slide_w, slide_h = _slide_size(aspect)
    theme = plan.get("theme") or {}
    accent = _rgb(theme.get("accent_rgb"), (0, 71, 227))
    body_rgb = _rgb(theme.get("body_rgb"), (51, 65, 85))
    title_rgb = _rgb(theme.get("title_rgb"), (15, 23, 42))

    font_title = theme.get("font_title", "PingFang SC")
    font_body = theme.get("font_body", "PingFang SC")

    prs = Presentation()
    prs.slide_width = slide_w
    prs.slide_height = slide_h
    try:
        prs.core_properties.title = plan.get("title", "Presentation")
        prs.core_properties.author = theme.get("author", "DeerFlow summary-ppt-editable")
    except Exception:
        pass

    blank = prs.slide_layouts[6]
    slides_in = sorted(plan.get("slides", []), key=lambda s: int(s.get("slide_number", 0)))

    for slide in slides_in:
        stype = (slide.get("type") or "content").lower()
        sld = prs.slides.add_slide(blank)

        if stype == "title":
            box = sld.shapes.add_textbox(Inches(0.8), Inches(2.4), slide_w - Inches(1.6), Inches(1.2))
            tf = box.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.text = slide.get("title", "")
            p.font.bold = True
            p.font.size = Pt(40)
            p.font.color.rgb = title_rgb
            p.alignment = PP_ALIGN.CENTER
            try:
                p.font.name = font_title
            except Exception:
                pass
            sub = slide.get("subtitle") or ""
            if sub:
                p2 = tf.add_paragraph()
                p2.text = sub
                p2.font.size = Pt(22)
                p2.font.color.rgb = body_rgb
                p2.alignment = PP_ALIGN.CENTER
                try:
                    p2.font.name = font_body
                except Exception:
                    pass
        elif stype == "section":
            bar = sld.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(0.25), slide_h)
            bar.fill.solid()
            bar.fill.fore_color.rgb = accent
            bar.line.fill.background()
            box = sld.shapes.add_textbox(Inches(0.6), Inches(2.8), slide_w - Inches(1.2), Inches(1.0))
            tf = box.text_frame
            p = tf.paragraphs[0]
            p.text = slide.get("title", "")
            p.font.bold = True
            p.font.size = Pt(32)
            p.font.color.rgb = title_rgb
            try:
                p.font.name = font_title
            except Exception:
                pass
        else:
            # content | summary | cards
            title_h = Inches(0.95)
            tb = sld.shapes.add_textbox(Inches(0.55), Inches(0.35), slide_w - Inches(1.1), title_h)
            tfp = tb.text_frame
            tfp.word_wrap = True
            pp = tfp.paragraphs[0]
            pp.text = slide.get("title", "")
            pp.font.bold = True
            pp.font.size = Pt(26)
            pp.font.color.rgb = title_rgb
            try:
                pp.font.name = font_title
            except Exception:
                pass

            # ── Cards layout ──────────────────────────────────────────────
            cards = slide.get("cards")
            if cards and isinstance(cards, list):
                left_margin = Inches(0.55)
                top_body = Inches(1.35)
                _add_cards(
                    sld,
                    cards,
                    left=left_margin,
                    top=top_body,
                    total_w=slide_w - Inches(1.1),
                    total_h=slide_h - top_body - Inches(0.25),
                    font_name=font_body,
                    body_rgb=body_rgb,
                    heading_rgb=title_rgb,
                    accent_rgb=accent,
                    cols=int(slide.get("cards_cols", 0)),
                )
                # Notes
                notes = slide.get("notes")
                if notes:
                    try:
                        ns = sld.notes_slide
                        ns.notes_text_frame.text = str(notes)[:300]
                    except Exception:
                        pass
                continue  # skip figure logic below for cards slides

            fig = slide.get("figure") or {}
            img_path = fig.get("image_path") or ""
            has_figure = bool(img_path) or bool(fig.get("caption"))
            pos = (fig.get("position") or "right").lower()
            # Comparison tables and flowcharts always get full-slide layout
            viz_type = (fig.get("viz") or {}).get("type", "")
            if viz_type in ("comparison", "flowchart"):
                pos = "full"
            if pos not in ("right", "bottom", "full"):
                pos = "right"

            left_margin = Inches(0.55)
            top_body = Inches(1.35)
            full_text_w = slide_w - Inches(1.1)

            # ----- Full-slide: figure spans the entire content area (comparison tables) -----
            if has_figure and pos == "full":
                # Title area already drawn above; use remaining height for the figure
                fig_top = Inches(1.20)
                fig_h = slide_h - Inches(1.55)
                fig_left = left_margin
                fig_w = full_text_w
                if img_path and os.path.isfile(img_path):
                    _add_picture_fit(sld, img_path, fig_left, fig_top, fig_w, fig_h)
                cap = fig.get("caption")
                if cap:
                    cb = sld.shapes.add_textbox(fig_left, slide_h - Inches(0.38), fig_w, Inches(0.32))
                    ctf = cb.text_frame
                    cp = ctf.paragraphs[0]
                    cp.text = str(cap)
                    cp.font.size = Pt(9)
                    cp.font.color.rgb = body_rgb
                    cp.alignment = PP_ALIGN.CENTER

            # ----- Bottom band: timeline / pipeline (text above, figure full width below) -----
            elif has_figure and pos == "bottom":
                # 略增高底栏以容纳「高密度时间线」PNG（仍随图片拉伸至该矩形）
                body_h = Inches(2.55)
                fig_top = Inches(3.95)
                fig_h = Inches(3.15)
                body_box = sld.shapes.add_textbox(left_margin, top_body, full_text_w, body_h)
                bf = body_box.text_frame
                bf.word_wrap = True
                bf.vertical_anchor = MSO_ANCHOR.TOP
                _render_slide_body(bf, slide, font_name=font_body, body_rgb=body_rgb, title_rgb=title_rgb)

                pic_w = full_text_w
                pic_left = left_margin
                if img_path and os.path.isfile(img_path):
                    _add_picture_fit(sld, img_path, pic_left, fig_top, pic_w, fig_h)
                else:
                    ph = sld.shapes.add_shape(MSO_SHAPE.RECTANGLE, pic_left, fig_top, pic_w, fig_h)
                    ph.fill.background()
                    ph.line.color.rgb = accent
                    ph.line.width = Pt(1.5)
                    tbox = sld.shapes.add_textbox(pic_left, fig_top + fig_h * 0.38, pic_w, Inches(0.85))
                    tfph = tbox.text_frame
                    tfp0 = tfph.paragraphs[0]
                    tfp0.text = fig.get("placeholder_text") or "路线 / 时间线配图区（export PNG 与上文术语一致）"
                    tfp0.font.size = Pt(11)
                    tfp0.font.color.rgb = body_rgb
                    tfp0.alignment = PP_ALIGN.CENTER
                cap = fig.get("caption")
                if cap:
                    cb = sld.shapes.add_textbox(pic_left, fig_top + fig_h + Inches(0.06), pic_w, Inches(0.4))
                    ctf = cb.text_frame
                    cp = ctf.paragraphs[0]
                    cp.text = str(cap)
                    cp.font.size = Pt(10)
                    cp.font.color.rgb = body_rgb
                    cp.alignment = PP_ALIGN.CENTER
            elif has_figure and img_path and os.path.isfile(img_path):
                text_w = slide_w - Inches(6.1)
                pic_left = slide_w - Inches(5.35)
                pic_top = Inches(1.25)
                pic_w = Inches(4.75)
                pic_h = Inches(4.85)
                body_box = sld.shapes.add_textbox(left_margin, top_body, text_w, slide_h - Inches(1.65))
                bf = body_box.text_frame
                bf.word_wrap = True
                bf.vertical_anchor = MSO_ANCHOR.TOP
                _render_slide_body(bf, slide, font_name=font_body, body_rgb=body_rgb, title_rgb=title_rgb)
                _add_picture_fit(sld, img_path, pic_left, pic_top, pic_w, pic_h)
                cap = fig.get("caption")
                if cap:
                    cb = sld.shapes.add_textbox(pic_left, pic_top + pic_h + Inches(0.08), pic_w, Inches(0.45))
                    ctf = cb.text_frame
                    cp = ctf.paragraphs[0]
                    cp.text = str(cap)
                    cp.font.size = Pt(10)
                    cp.font.color.rgb = body_rgb
                    cp.alignment = PP_ALIGN.CENTER
            elif has_figure:
                text_w = slide_w - Inches(6.1)
                ph_left = slide_w - Inches(5.35)
                ph_top = Inches(1.25)
                ph_w = Inches(4.75)
                ph_h = Inches(4.85)
                body_box = sld.shapes.add_textbox(left_margin, top_body, text_w, slide_h - Inches(1.65))
                bf = body_box.text_frame
                bf.word_wrap = True
                bf.vertical_anchor = MSO_ANCHOR.TOP
                _render_slide_body(bf, slide, font_name=font_body, body_rgb=body_rgb, title_rgb=title_rgb)
                ph = sld.shapes.add_shape(MSO_SHAPE.RECTANGLE, ph_left, ph_top, ph_w, ph_h)
                ph.fill.background()
                ph.line.color.rgb = accent
                ph.line.width = Pt(1.5)
                tbox = sld.shapes.add_textbox(ph_left, ph_top + ph_h * 0.4, ph_w, Inches(0.8))
                tfph = tbox.text_frame
                tfp0 = tfph.paragraphs[0]
                tfp0.text = fig.get("placeholder_text") or "Figure / chart placeholder\n(export PNG and set figure.image_path)"
                tfp0.font.size = Pt(11)
                tfp0.font.color.rgb = body_rgb
                tfp0.alignment = PP_ALIGN.CENTER
                cap = fig.get("caption")
                if cap:
                    cb = sld.shapes.add_textbox(ph_left, ph_top + ph_h + Inches(0.08), ph_w, Inches(0.45))
                    ctf = cb.text_frame
                    cp = ctf.paragraphs[0]
                    cp.text = str(cap)
                    cp.font.size = Pt(10)
                    cp.font.color.rgb = body_rgb
                    cp.alignment = PP_ALIGN.CENTER
            else:
                body_box = sld.shapes.add_textbox(left_margin, top_body, full_text_w, slide_h - Inches(1.65))
                bf = body_box.text_frame
                bf.word_wrap = True
                bf.vertical_anchor = MSO_ANCHOR.TOP
                _render_slide_body(bf, slide, font_name=font_body, body_rgb=body_rgb, title_rgb=title_rgb)

        notes = slide.get("notes")
        if notes:
            try:
                ns = sld.notes_slide
                # Truncate notes to ~300 chars to stay within the notes pane
                notes_text = str(notes)
                if len(notes_text) > 300:
                    notes_text = notes_text[:297] + "..."
                ns.notes_text_frame.text = notes_text
            except Exception:
                pass

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    prs.save(output_file)
    return f"Saved {len(slides_in)} slides to {output_file}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build editable summary PPTX from JSON plan")
    parser.add_argument("--plan-file", required=True, help="Path to presentation plan JSON")
    parser.add_argument("--output-file", required=True, help="Output .pptx path")
    args = parser.parse_args()

    with open(args.plan_file, "r", encoding="utf-8") as f:
        plan = json.load(f)

    msg = build_pptx(plan, args.output_file)
    print(msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
