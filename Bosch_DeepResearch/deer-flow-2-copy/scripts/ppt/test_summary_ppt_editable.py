#!/usr/bin/env python3
"""
测试 summary-ppt-editable Skill：从 JSON 计划生成「可编辑文本 + 可选配图」的 PPTX。

不依赖 SDXL；仅需 python-pptx、Pillow（用于生成示例 PNG）。

用法:
  backend/.venv/bin/python scripts/test_summary_ppt_editable.py
  # 或指定输出目录:
  python scripts/test_summary_ppt_editable.py --out-dir ~/Downloads/summary_ppt_test
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _make_demo_png(path: Path) -> None:
    from PIL import Image, ImageDraw

    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = 640, 480
    img = Image.new("RGB", (w, h), (240, 249, 255))
    d = ImageDraw.Draw(img)
    d.rectangle([40, 40, w - 40, h - 40], outline=(0, 113, 227), width=4)
    d.text((80, 200), "Demo logic figure (PNG)", fill=(15, 23, 42))
    d.text((80, 240), "Replace with chart / Mermaid export", fill=(71, 85, 105))
    img.save(path, format="PNG")


def main() -> int:
    parser = argparse.ArgumentParser(description="Test summary-ppt-editable build_pptx")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: ~/.cache/deerflow/outputs/summary_ppt_test)",
    )
    args = parser.parse_args()

    root = _project_root()
    build_script = root / "skills/public/summary-ppt-editable/scripts/build_pptx.py"
    template = root / "skills/public/summary-ppt-editable/templates/sample_plan.json"

    if not build_script.is_file():
        print("Missing:", build_script, file=sys.stderr)
        return 1
    if not template.is_file():
        print("Missing:", template, file=sys.stderr)
        return 1

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = Path.home() / ".cache/deerflow/outputs/summary_ppt_test"
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    demo_png = out_dir / "demo_figure.png"
    _make_demo_png(demo_png)

    with open(template, "r", encoding="utf-8") as f:
        plan = json.load(f)

    for s in plan.get("slides", []):
        if s.get("title") == "逻辑结构（配图区）":
            fig = s.setdefault("figure", {})
            fig["image_path"] = str(demo_png)
            fig["caption"] = "示例 PNG（测试脚本生成，可替换为真实图表）"

    plan_path = out_dir / "test_summary_plan.json"
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)

    pptx_path = out_dir / "Summary_Editable_Text_Demo.pptx"

    venv_py = root / "backend/.venv/bin/python"
    py = str(venv_py) if venv_py.is_file() else sys.executable

    cmd = [
        py,
        str(build_script),
        "--plan-file",
        str(plan_path),
        "--output-file",
        str(pptx_path),
    ]
    print("Running:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(root))
    if r.returncode != 0:
        print("build_pptx failed with code", r.returncode, file=sys.stderr)
        return r.returncode

    print()
    print("OK — PPTX:", pptx_path)
    print("Plan JSON:", plan_path)
    print("Demo PNG:", demo_png)
    print()
    print("请用 PowerPoint / Keynote / LibreOffice 打开 PPTX，检查：标题与要点是否可直接编辑；右侧是否为示例配图。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
