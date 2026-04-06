#!/usr/bin/env python3
"""一键生成 timeline-evolution-ppt 示例（图文并茂时间线）。"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ORCH = ROOT / "skills/public/timeline-evolution-ppt/scripts/orchestrate_timeline_pptx.py"
SPEC = ROOT / "skills/public/timeline-evolution-ppt/templates/example_deck.spec.json"
# 默认输出到「下载」目录，便于直接打开
OUT = Path.home() / "Downloads/timeline_evolution_agent_v1"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    py = ROOT / "backend/.venv/bin/python"
    exe = str(py) if py.is_file() else sys.executable
    cmd = [
        exe,
        str(ORCH),
        "--deck-spec",
        str(SPEC),
        "--output-dir",
        str(OUT),
        "--pptx-name",
        "Timeline_Evolution_Agent_v1.pptx",
    ]
    print(" ", " ".join(cmd))
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


if __name__ == "__main__":
    raise SystemExit(main())
