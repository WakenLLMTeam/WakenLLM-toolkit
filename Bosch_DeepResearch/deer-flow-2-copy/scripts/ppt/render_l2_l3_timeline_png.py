#!/usr/bin/env python3
"""
生成与 L2→L3 叙事对齐的简易时间线/路线 PNG（纯 Pillow），供 summary-ppt-editable 底栏图使用。

用法:
  python scripts/render_l2_l3_timeline_png.py -o /path/to/timeline_l2_l3.png
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output PNG path")
    args = parser.parse_args()

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as e:
        print("需要 Pillow: pip install Pillow", e, file=__import__("sys").stderr)
        return 1

    w, h = 1280, 290
    img = Image.new("RGB", (w, h), "#f8fafc")
    draw = ImageDraw.Draw(img)

    def load_cn(size: int):
        for p in (
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        ):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
        return ImageFont.load_default()

    font_title = load_cn(20)
    font_sub = load_cn(13)
    font_hint = load_cn(12)

    boxes = [
        (40, 65, 340, 235, "L2 组合辅助驾驶", "驾驶员监督 · 感知 / 规划 / HMI", "#dbeafe"),
        (380, 65, 680, 235, "过渡：ODD / 接管 / 冗余", "能力与法规成熟", "#fef3c7"),
        (720, 65, 1020, 235, "L3 条件自动驾驶", "ODD 内系统驾驶 · 责任转移", "#d1fae5"),
    ]

    for x1, y1, x2, y2, title, sub, fill in boxes:
        draw.rounded_rectangle([x1, y1, x2, y2], radius=12, outline="#334155", width=2, fill=fill)
        cx = (x1 + x2) // 2
        draw.text((cx, y1 + 42), title, fill="#0f172a", font=font_title, anchor="mm")
        draw.text((cx, y1 + 118), sub, fill="#475569", font=font_sub, anchor="mm")

    draw.line([350, 150, 368, 150], fill="#64748b", width=3)
    draw.polygon([(375, 150), (365, 145), (365, 155)], fill="#64748b")
    draw.line([690, 150, 708, 150], fill="#64748b", width=3)
    draw.polygon([(715, 150), (705, 145), (705, 155)], fill="#64748b")

    draw.text((640, 16), "演变轴线（与幻灯片正文术语一致）", fill="#64748b", font=font_hint, anchor="mm")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    img.save(args.output, format="PNG")
    print("Wrote", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
