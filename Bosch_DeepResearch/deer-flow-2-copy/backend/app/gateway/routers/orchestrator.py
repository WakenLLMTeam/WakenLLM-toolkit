"""HTTP API for the three-phase report orchestrator (parallel to LangGraph lead agent)."""

from __future__ import annotations

import asyncio
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/orchestrator", tags=["orchestrator"])

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="orch")


def _ensure_orchestrator_importable() -> None:
    """Add repo root + harness to ``sys.path`` (dev layout and Docker ``/app`` layout)."""
    here = Path(__file__).resolve()
    candidates: list[Path] = []
    for depth in (4, 5):
        if len(here.parents) > depth:
            candidates.append(here.parents[depth])
    seen: set[str] = set()
    for root in candidates:
        orch = root / "orchestrator"
        harness = root / "backend" / "packages" / "harness"
        if orch.is_dir() and harness.is_dir():
            for p in (str(harness), str(root)):
                if p not in seen:
                    sys.path.insert(0, p)
                    seen.add(p)
            return
    logger.error(
        "Could not locate orchestrator/ and backend/packages/harness from %s; tried roots %s",
        here,
        candidates,
    )


class OrchestratorRunRequest(BaseModel):
    thread_id: str = Field(..., description="LangGraph thread id (sandbox / uploads isolation)")
    message: str = Field(..., description="User request / report brief")
    model_name: str | None = Field(None, description="Optional model id from config.yaml")
    include_bash_in_report: bool = Field(False, description="Expose bash tool in phase-3 report agent")


class OrchestratorRunResponse(BaseModel):
    ok: bool
    thread_id: str
    final_report: str | None = None
    framework_title: str | None = None
    state: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


def _run_pipeline_sync(
    thread_id: str,
    message: str,
    model_name: str | None,
    include_bash: bool,
) -> dict[str, Any]:
    _ensure_orchestrator_importable()
    from orchestrator.pipeline import ReportWorkflowOrchestrator

    orch = ReportWorkflowOrchestrator(
        thread_id,
        model_name=model_name,
        include_bash_in_report=include_bash,
    )
    orch.run_full(message)
    fw = orch.state.framework
    title = fw.title if fw else None
    alignment = orch.validate_alignment()
    return {
        "final_report": orch.state.final_report,
        "framework_title": title,
        "state": orch.state.to_dict(),
        "alignment": alignment,
    }


@router.post(
    "/run",
    response_model=OrchestratorRunResponse,
    summary="Run three-phase report orchestrator",
    description=(
        "Runs Framework → Collection → Report agents sequentially. "
        "Uses the same config.yaml and thread sandbox as LangGraph, but does not invoke the lead agent graph."
    ),
)
async def run_orchestrator(body: OrchestratorRunRequest) -> OrchestratorRunResponse:
    if not body.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            _executor,
            _run_pipeline_sync,
            body.thread_id,
            body.message.strip(),
            body.model_name,
            body.include_bash_in_report,
        )
    except Exception as e:
        logger.exception("Orchestrator run failed for thread %s", body.thread_id)
        return OrchestratorRunResponse(
            ok=False,
            thread_id=body.thread_id,
            error=str(e),
        )

    return OrchestratorRunResponse(
        ok=True,
        thread_id=body.thread_id,
        final_report=result.get("final_report"),
        framework_title=result.get("framework_title"),
        state={
            **result.get("state", {}),
            "alignment": result.get("alignment"),
        },
    )
