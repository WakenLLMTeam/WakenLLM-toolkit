"""Three-phase report orchestration with one LangChain agent per phase."""

from typing import Any

from orchestrator.pipeline import ReportWorkflowOrchestrator
from orchestrator.state import (
    Chapter,
    CollectionBundle,
    DataRequirement,
    PhaseCheckpoint,
    ReportFramework,
    ReportWorkflowState,
    WorkflowPhase,
)

__all__ = [
    "Chapter",
    "CollectionBundle",
    "DataRequirement",
    "PhaseCheckpoint",
    "ReportFramework",
    "ReportWorkflowOrchestrator",
    "ReportWorkflowState",
    "WorkflowPhase",
    "get_orchestrator",
    "reset_orchestrator_registry",
]

_orchestrators: dict[str, ReportWorkflowOrchestrator] = {}


def get_orchestrator(thread_id: str = "default", **kwargs: Any) -> ReportWorkflowOrchestrator:
    """Return a cached :class:`ReportWorkflowOrchestrator` per ``thread_id`` (optional helper).

    Only the first call's ``kwargs`` apply for that ``thread_id``; call
    :func:`reset_orchestrator_registry` if you need different constructor options.
    """
    if thread_id not in _orchestrators:
        _orchestrators[thread_id] = ReportWorkflowOrchestrator(thread_id, **kwargs)
    return _orchestrators[thread_id]


def reset_orchestrator_registry() -> None:
    _orchestrators.clear()
