#!/usr/bin/env python3
"""
一键演示：L2→L3 示例计划 + 时间线 PNG + 可编辑 PPTX（图文术语对齐）。

依赖: python-pptx, matplotlib, Pillow

  backend/.venv/bin/python scripts/test_l2_l3_summary_ppt.py
  python scripts/test_l2_l3_summary_ppt.py --out-dir ~/Downloads/l2_l3_demo
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _root() -> Path:
    return Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    root = _root()
    plan_src = root / "skills/public/summary-ppt-editable/templates/example_l2_to_l3_plan.json"
    build = root / "skills/public/summary-ppt-editable/scripts/build_pptx.py"
    timeline = root / "scripts/render_l2_l3_timeline_png.py"

    out = args.out_dir or (Path.home() / ".cache/deerflow/outputs/l2_l3_summary_demo")
    out = out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    png = out / "timeline_l2_l3.png"
    plan_out = out / "l2_l3_plan.json"
    pptx = out / "L2_to_L3_Evolution_Editable.pptx"

    vpy = root / "backend/.venv/bin/python"
    py = str(vpy) if vpy.is_file() else sys.executable

    r = subprocess.run([py, str(timeline), "-o", str(png)], cwd=str(root))
    if r.returncode != 0:
        return r.returncode

    with open(plan_src, "r", encoding="utf-8") as f:
        plan = json.load(f)

    for s in plan.get("slides", []):
        if s.get("title") == "能力路线：从 L2 到 L3 的递进关系":
            fig = s.setdefault("figure", {})
            fig["image_path"] = str(png)
            fig["position"] = "bottom"
            fig["caption"] = "与正文模块：L2 → 过渡（ODD/接管/冗余）→ L3 一致"

    with open(plan_out, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)

    r = subprocess.run(
        [py, str(build), "--plan-file", str(plan_out), "--output-file", str(pptx)],
        cwd=str(root),
    )
    if r.returncode != 0:
        return r.returncode

    print("OK")
    print("  PPTX:", pptx)
    print("  Plan:", plan_out)
    print("  Timeline PNG:", png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
