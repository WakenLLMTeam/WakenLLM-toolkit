"""Structured state for the three-phase report workflow (no dependency on ``src.agents``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class WorkflowPhase(str, Enum):
    INIT = "init"
    FRAMEWORK = "framework"
    COLLECTION = "collection"
    REPORT = "report"
    COMPLETE = "complete"
    FAILED = "failed"


class Chapter(BaseModel):
    chapter_id: str
    title: str
    objective: str = ""


class DataRequirement(BaseModel):
    requirement_id: str
    section_id: str
    metric: str
    priority: str = "P1"
    search_keywords: list[str] = Field(default_factory=list)


class ReportFramework(BaseModel):
    """Phase 1 output: fixed schema for downstream phases."""

    title: str
    objectives: str = ""
    chapters: list[Chapter] = Field(default_factory=list)
    data_requirements: list[DataRequirement] = Field(default_factory=list)


class CollectionBundle(BaseModel):
    """Phase 2 output: evidence pointers keyed by requirement id."""

    sources_by_requirement: dict[str, list[str]] = Field(default_factory=dict)
    notes: str = ""


@dataclass
class PhaseCheckpoint:
    phase: WorkflowPhase
    started_at: str
    completed_at: str | None = None
    status: str = "in_progress"
    error_message: str | None = None


@dataclass
class ReportWorkflowState:
    """Mutable orchestration state for a single report run."""

    current_phase: WorkflowPhase = WorkflowPhase.INIT
    checkpoints: list[PhaseCheckpoint] = field(default_factory=list)
    framework: ReportFramework | None = None
    collection: CollectionBundle | None = None
    final_report: str | None = None
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_phase": self.current_phase.value,
            "framework": self.framework.model_dump() if self.framework else None,
            "collection": self.collection.model_dump() if self.collection else None,
            "final_report": (self.final_report[:2000] + "…") if self.final_report and len(self.final_report) > 2000 else self.final_report,
            "last_error": self.last_error,
            "checkpoints": [
                {
                    "phase": c.phase.value,
                    "started_at": c.started_at,
                    "completed_at": c.completed_at,
                    "status": c.status,
                    "error_message": c.error_message,
                }
                for c in self.checkpoints
            ],
        }

    def start_checkpoint(self, phase: WorkflowPhase) -> None:
        self.current_phase = phase
        self.checkpoints.append(
            PhaseCheckpoint(phase=phase, started_at=datetime.now().isoformat(), status="in_progress"),
        )

    def complete_checkpoint(self, *, ok: bool, error: str | None = None) -> None:
        if not self.checkpoints:
            return
        cp = self.checkpoints[-1]
        cp.completed_at = datetime.now().isoformat()
        cp.status = "completed" if ok else "failed"
        cp.error_message = error
