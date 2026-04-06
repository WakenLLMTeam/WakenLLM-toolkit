#!/usr/bin/env python3
"""
横向时间线 PNG（高密度）。默认高分辨率导出，避免插入 PPT 全宽后被放大发糊。

模糊原因：幻灯片底栏约 12～13 英寸宽；若 PNG 仅 1600px，等效 ~130dpi，高分屏会糊。
对策：默认 width≈2880（约 220～240dpi），字体/线宽随 scale 放大；PNG 写入 dpi 元数据。

JSON 示例见 templates/example_deck.spec.json。
"""

from __future__ import annotations

import argparse
import json
import textwrap
import sys
from pathlib import Path
from typing import Any, Dict, List


# 布局参考宽度；实际 w 更大时整体按比例放大（字体、间距、线宽）
BASE_W = 1600
# 默认输出宽度：约对应 16:9 幻灯片底栏全宽 @220dpi 量级（可按需在 spec 里改 width）
DEFAULT_EXPORT_WIDTH = 2880


def _load_font(size: int):
    from PIL import ImageFont

    for p in (
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap_detail(line: str, max_chars: int) -> List[str]:
    line = (line or "").strip()
    if not line:
        return []
    if max_chars < 8:
        max_chars = 8
    return textwrap.wrap(line, width=max_chars, break_long_words=True, break_on_hyphens=False)


def render_strip(spec: dict, output: Path) -> None:
    from PIL import Image, ImageDraw

    stages: List[Dict[str, Any]] = spec.get("stages") or []
    if len(stages) < 1:
        raise ValueError("stages must have at least 1 item")

    w = int(spec.get("width", DEFAULT_EXPORT_WIDTH))
    scale = w / float(BASE_W)

    n = len(stages)
    margin_x = int(28 * scale)
    gap = int(18 * scale)
    avail = w - 2 * margin_x - (n - 1) * gap
    box_w = max(int(140 * scale), avail // n)
    pad_x = int(10 * scale)
    # 每字约占用像素随字号放大
    char_px = max(9, int(11 * scale))
    max_chars = max(12, (box_w - 2 * pad_x) // char_px)

    detail_line_counts: List[int] = []
    for st in stages:
        lc = 0
        for d in st.get("details") or []:
            for part in _wrap_detail(str(d), max_chars):
                lc += 1
        detail_line_counts.append(lc)

    max_detail_lines = max(detail_line_counts) if detail_line_counts else 0
    has_kw = any((st.get("keywords") or "").strip() for st in stages)

    max_sub_lines = 0
    for st in stages:
        sub = (st.get("subtitle") or "").strip()
        max_sub_lines = max(max_sub_lines, len(_wrap_detail(sub, max_chars + 4)) if sub else 0)

    strip_title_h = int((28 if spec.get("strip_title") else 12) * scale)
    footer_note_h = int((22 if spec.get("footer_note") else 0) * scale)

    lh_sub = int(14 * scale)
    lh_det = int(13 * scale)
    inner_h = int(
        30 * scale
        + max_sub_lines * lh_sub
        + (int(18 * scale) if has_kw else 0)
        + max_detail_lines * lh_det
        + int(28 * scale)
    )
    auto_h = strip_title_h + inner_h + footer_note_h + int(20 * scale)
    h = int(spec.get("height") or max(int(340 * scale), min(int(620 * scale), auto_h)))

    bg = spec.get("background", "#f8fafc")
    img = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)

    ft = max(12, int(16 * scale))
    fs = max(10, int(11 * scale))
    f_title = _load_font(ft)
    f_sub = _load_font(fs)
    f_kw = _load_font(max(9, int(10 * scale)))
    f_det = _load_font(max(9, int(10 * scale)))
    f_strip = _load_font(max(9, int(10 * scale)))
    f_foot = _load_font(max(8, int(9 * scale)))

    y0 = int(8 * scale)
    hint = spec.get("strip_title", "")
    if hint:
        draw.text((w // 2, y0 + int(10 * scale)), hint, fill="#475569", font=f_strip, anchor="mm")

    margin_y = strip_title_h + int(8 * scale)
    box_h = h - margin_y - footer_note_h - int(16 * scale)
    if box_h < int(100 * scale):
        box_h = int(100 * scale)

    y1 = margin_y
    x = margin_x
    rad = max(6, int(10 * scale))
    border_w = max(2, int(2 * scale))

    for i, st in enumerate(stages):
        title = (st.get("title") or "").strip()
        sub = (st.get("subtitle") or "").strip()
        kw = (st.get("keywords") or "").strip()
        fill = st.get("fill") or "#e2e8f0"
        details: List[str] = []
        for d in st.get("details") or []:
            details.extend(_wrap_detail(str(d), max_chars))

        x2 = x + box_w
        y2 = y1 + box_h
        draw.rounded_rectangle([x, y1, x2, y2], radius=rad, outline="#334155", width=border_w, fill=fill)

        cx = (x + x2) // 2
        ty = y1 + int(14 * scale)
        draw.text((cx, ty), title, fill="#0f172a", font=f_title, anchor="mm")
        ty += int(22 * scale)
        if sub:
            for line in _wrap_detail(sub, max_chars + 4):
                draw.text((cx, ty), line, fill="#1e3a5f", font=f_sub, anchor="mm")
                ty += lh_sub
            ty += int(4 * scale)
        if kw:
            draw.text((cx, ty), f"[{kw}]", fill="#b45309", font=f_kw, anchor="mm")
            ty += int(15 * scale)

        lx = x + pad_x
        ty += int(2 * scale)
        for line in details:
            if ty > y2 - int(10 * scale):
                draw.text((lx, y2 - int(14 * scale)), "…", fill="#64748b", font=f_det, anchor="lt")
                break
            draw.text((lx, ty), f"· {line}", fill="#334155", font=f_det, anchor="lt")
            ty += lh_det

        mid_y = (y1 + y2) // 2
        if i < n - 1:
            arr_x1 = x2 + int(3 * scale)
            arr_x2 = x2 + gap - int(3 * scale)
            aw = max(2, int(2 * scale))
            draw.line([arr_x1, mid_y, arr_x2, mid_y], fill="#64748b", width=aw)
            ah = max(3, int(4 * scale))
            draw.polygon(
                [(arr_x2 + int(5 * scale), mid_y), (arr_x2 - 1, mid_y - ah), (arr_x2 - 1, mid_y + ah)],
                fill="#64748b",
            )
        x += box_w + gap

    fn = spec.get("footer_note", "")
    if fn:
        draw.text((w // 2, h - int(12 * scale)), fn, fill="#94a3b8", font=f_foot, anchor="mm")

    output.parent.mkdir(parents=True, exist_ok=True)
    # 写入 dpi 便于部分查看器/导出按物理尺寸理解（对 PPT 内显示亦有参考意义）
    png_dpi = int(spec.get("png_dpi", 220))
    img.save(output, format="PNG", dpi=(png_dpi, png_dpi))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stages-json", required=True, help="JSON with stages[] etc.")
    p.add_argument("-o", "--output", required=True, type=Path, help="Output PNG")
    args = p.parse_args()
    with open(args.stages_json, "r", encoding="utf-8") as f:
        spec = json.load(f)
    try:
        render_strip(spec, args.output)
    except Exception as e:
        print(e, file=sys.stderr)
        return 1
    print("Wrote", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
