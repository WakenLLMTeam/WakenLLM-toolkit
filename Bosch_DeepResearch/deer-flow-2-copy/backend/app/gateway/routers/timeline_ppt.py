"""HTTP API：时间线演变 PPT（L2→L3 等），调用仓库内 timeline-evolution-ppt 编排脚本。"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ppt/timeline", tags=["ppt-timeline"])


def _repo_root() -> Path:
    """backend/app/gateway/routers/timeline_ppt.py → 仓库根（含 skills/public/...）。"""
    here = Path(__file__).resolve()
    for i in range(3, 10):
        if len(here.parents) <= i:
            break
        root = here.parents[i]
        if (root / "skills" / "public" / "timeline-evolution-ppt" / "scripts" / "orchestrate_timeline_pptx.py").is_file():
            return root
    raise RuntimeError("Cannot locate deer-flow repo root (skills/public/timeline-evolution-ppt not found)")


def _run_orchestrate_sync(deck_spec: Path, out_dir: Path, pptx_name: str) -> Path:
    root = _repo_root()
    script = root / "skills/public/timeline-evolution-ppt/scripts/orchestrate_timeline_pptx.py"
    if not script.is_file():
        raise FileNotFoundError(str(script))
    cmd = [
        sys.executable,
        str(script),
        "--deck-spec",
        str(deck_spec),
        "--output-dir",
        str(out_dir),
        "--pptx-name",
        pptx_name,
    ]
    r = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        logger.error("orchestrate stderr: %s", r.stderr)
        raise RuntimeError(r.stderr or r.stdout or "orchestrate_timeline_pptx failed")
    out = out_dir / pptx_name
    if not out.is_file():
        raise FileNotFoundError(f"Expected PPTX not found: {out}")
    return out


@router.post(
    "/l2-l3",
    summary="生成 L2→L3 自动驾驶演变时间线 PPTX",
    description=(
        "使用内置 `example_deck_v2.spec.json`（schema v2），"
        "输出可编辑文本 + 底栏高密度时间线 PNG 的合成演示文稿。"
    ),
    responses={
        200: {
            "content": {
                "application/vnd.openxmlformats-officedocument.presentationml.presentation": {},
            },
            "description": "PowerPoint 文件",
        }
    },
)
async def generate_l2_l3_timeline_pptx():
    root = _repo_root()
    spec = root / "skills/public/timeline-evolution-ppt/templates/example_deck_v2.spec.json"
    if not spec.is_file():
        raise HTTPException(status_code=500, detail=f"Missing spec: {spec}")

    out_dir = Path(tempfile.mkdtemp(prefix="timeline_ppt_l2l3_"))
    pptx_name = "L2_L3_ADAS_Evolution.pptx"

    try:
        pptx_path = await asyncio.to_thread(_run_orchestrate_sync, spec, out_dir, pptx_name)
    except Exception as e:
        logger.exception("timeline ppt generation failed")
        raise HTTPException(status_code=500, detail=str(e)) from e

    return FileResponse(
        path=str(pptx_path),
        filename=pptx_name,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
    )


class CustomGenerateBody(BaseModel):
    """可选：传入完整 deck spec JSON（与 orchestrate_timeline_pptx 一致）。"""

    deck_spec: dict | None = Field(None, description="若为 null 则使用内置 L2→L3 v2 模板")


@router.post("/generate")
async def generate_timeline_ppt_custom(body: CustomGenerateBody | None = None):
    """使用自定义 spec 生成；body 为空或 deck_spec 为空时等价于 `/l2-l3`。"""
    root = _repo_root()
    out_dir = Path(tempfile.mkdtemp(prefix="timeline_ppt_custom_"))
    pptx_name = "Timeline_Deck.pptx"

    if body and body.deck_spec:
        import json

        spec_path = out_dir / "deck_in.json"
        spec_path.write_text(json.dumps(body.deck_spec, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        spec_path = root / "skills/public/timeline-evolution-ppt/templates/example_deck_v2.spec.json"

    try:
        pptx_path = await asyncio.to_thread(_run_orchestrate_sync, spec_path, out_dir, pptx_name)
    except Exception as e:
        logger.exception("timeline ppt custom generation failed")
        raise HTTPException(status_code=500, detail=str(e)) from e

    return FileResponse(
        path=str(pptx_path),
        filename=pptx_name,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
    )
