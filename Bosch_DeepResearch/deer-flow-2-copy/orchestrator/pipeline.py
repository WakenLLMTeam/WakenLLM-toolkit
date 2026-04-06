"""Finite-state report pipeline: Framework agent → Collection agent → Report agent."""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import ValidationError

from orchestrator.bootstrap import ensure_deerflow_importable
from orchestrator.json_utils import parse_json_loose
from orchestrator.phase_agents import (
    build_collection_agent,
    build_framework_agent,
    build_report_agent,
    run_agent_sync,
)
from orchestrator.state import CollectionBundle, ReportFramework, ReportWorkflowState, WorkflowPhase

logger = logging.getLogger(__name__)


class ReportWorkflowOrchestrator:
    """Runs the three-phase report workflow with **three separate agents** (narrow tools per phase)."""

    def __init__(
        self,
        thread_id: str,
        *,
        model_name: str | None = None,
        thinking_enabled: bool = False,
        include_bash_in_report: bool = False,
        recursion_limit: int = 50,
    ) -> None:
        ensure_deerflow_importable()
        self.thread_id = thread_id
        self.model_name = model_name
        self.thinking_enabled = thinking_enabled
        self.include_bash_in_report = include_bash_in_report
        self.recursion_limit = recursion_limit
        self.state = ReportWorkflowState()

    def reset(self) -> None:
        """Clear workflow state to run a new report on the same ``thread_id``."""
        self.state = ReportWorkflowState()

    def _resolve_model_name(self) -> str | None:
        if self.model_name:
            return self.model_name
        from deerflow.config.app_config import get_app_config

        cfg = get_app_config()
        return cfg.models[0].name if cfg.models else None

    def run_framework(self, user_request: str) -> ReportFramework:
        """Phase 1: structured plan only (no tools)."""
        self.state.last_error = None
        if self.state.current_phase not in (WorkflowPhase.INIT, WorkflowPhase.FAILED, WorkflowPhase.COMPLETE):
            raise RuntimeError(f"Cannot start framework from phase {self.state.current_phase}")
        if self.state.current_phase in (WorkflowPhase.FAILED, WorkflowPhase.COMPLETE):
            self.reset()

        self.state.start_checkpoint(WorkflowPhase.FRAMEWORK)
        model_name = self._resolve_model_name()
        if not model_name:
            err = "No model configured (config.yaml models list empty)."
            self.state.last_error = err
            self.state.current_phase = WorkflowPhase.FAILED
            self.state.complete_checkpoint(ok=False, error=err)
            raise RuntimeError(err)

        agent = build_framework_agent(model_name=model_name, thinking_enabled=self.thinking_enabled)
        raw = run_agent_sync(
            agent,
            user_request,
            thread_id=self.thread_id,
            recursion_limit=self.recursion_limit,
        )
        try:
            data = parse_json_loose(raw)
            framework = ReportFramework.model_validate(data)
        except (ValueError, ValidationError, json.JSONDecodeError) as e:
            err = f"Framework JSON parse/validate failed: {e}"
            logger.exception("%s — raw prefix: %s", err, raw[:500])
            self.state.last_error = err
            self.state.current_phase = WorkflowPhase.FAILED
            self.state.complete_checkpoint(ok=False, error=err)
            raise RuntimeError(err) from e

        self.state.framework = framework
        self.state.current_phase = WorkflowPhase.COLLECTION
        self.state.complete_checkpoint(ok=True)
        return framework

    def run_collection(self) -> CollectionBundle:
        """Phase 2: web tools only; consumes ``state.framework``."""
        self.state.last_error = None
        if not self.state.framework:
            raise RuntimeError("Framework missing; run run_framework() first.")
        if self.state.current_phase != WorkflowPhase.COLLECTION:
            raise RuntimeError(f"Expected phase COLLECTION, got {self.state.current_phase}")

        self.state.start_checkpoint(WorkflowPhase.COLLECTION)
        model_name = self._resolve_model_name()
        if not model_name:
            err = "No model configured."
            self.state.last_error = err
            self.state.current_phase = WorkflowPhase.FAILED
            self.state.complete_checkpoint(ok=False, error=err)
            raise RuntimeError(err)

        payload = self.state.framework.model_dump_json(indent=2)
        user_text = (
            "Here is the approved report framework (JSON). Collect sources for each requirement_id.\n\n"
            f"{payload}\n\n"
            "Use web_search / web_fetch as needed. Respond with the required JSON bundle only."
        )
        agent = build_collection_agent(model_name=model_name, thinking_enabled=self.thinking_enabled)
        raw = run_agent_sync(
            agent,
            user_text,
            thread_id=self.thread_id,
            recursion_limit=self.recursion_limit,
        )
        try:
            data = parse_json_loose(raw)
            bundle = CollectionBundle.model_validate(data)
        except (ValueError, ValidationError, json.JSONDecodeError) as e:
            err = f"Collection JSON parse/validate failed: {e}"
            logger.exception("%s — raw prefix: %s", err, raw[:500])
            self.state.last_error = err
            self.state.current_phase = WorkflowPhase.FAILED
            self.state.complete_checkpoint(ok=False, error=err)
            raise RuntimeError(err) from e

        self.state.collection = bundle
        self.state.current_phase = WorkflowPhase.REPORT
        self.state.complete_checkpoint(ok=True)
        return bundle

    def run_report(self) -> str:
        """Phase 3: file tools + final Markdown report."""
        self.state.last_error = None
        if not self.state.framework or not self.state.collection:
            raise RuntimeError("Framework and collection required before report phase.")

        if self.state.current_phase != WorkflowPhase.REPORT:
            raise RuntimeError(f"Expected phase REPORT, got {self.state.current_phase}")

        self.state.start_checkpoint(WorkflowPhase.REPORT)
        model_name = self._resolve_model_name()
        if not model_name:
            err = "No model configured."
            self.state.last_error = err
            self.state.current_phase = WorkflowPhase.FAILED
            self.state.complete_checkpoint(ok=False, error=err)
            raise RuntimeError(err)

        user_text = (
            "## Framework\n```json\n"
            + self.state.framework.model_dump_json(indent=2)
            + "\n```\n\n## Evidence (sources_by_requirement)\n```json\n"
            + self.state.collection.model_dump_json(indent=2)
            + "\n```\n\nWrite the complete final report in Markdown."
        )
        agent = build_report_agent(
            model_name=model_name,
            thinking_enabled=self.thinking_enabled,
            include_bash=self.include_bash_in_report,
        )
        raw = run_agent_sync(
            agent,
            user_text,
            thread_id=self.thread_id,
            recursion_limit=self.recursion_limit,
        )
        if not raw.strip():
            err = "Report agent returned empty content."
            self.state.last_error = err
            self.state.current_phase = WorkflowPhase.FAILED
            self.state.complete_checkpoint(ok=False, error=err)
            raise RuntimeError(err)

        from deerflow.agents.middlewares.citation_middleware import CitationMiddleware

        evidence_urls: list[str] = []
        if self.state.collection and self.state.collection.sources_by_requirement:
            for _req, lst in self.state.collection.sources_by_requirement.items():
                for u in lst or []:
                    if isinstance(u, str) and u.strip():
                        evidence_urls.append(u.strip())
        final_md = CitationMiddleware.apply_to_markdown_with_allowed_urls(raw.strip(), evidence_urls)

        self.state.final_report = final_md.strip()
        self.state.current_phase = WorkflowPhase.COMPLETE
        self.state.complete_checkpoint(ok=True)
        return self.state.final_report

    def run_full(self, user_request: str) -> ReportWorkflowState:
        """Run all three phases in order."""
        self.run_framework(user_request)
        self.run_collection()
        self.run_report()
        return self.state

    def validate_alignment(self) -> dict[str, Any]:
        """Lightweight consistency check after ``COMPLETE``."""
        if self.state.current_phase != WorkflowPhase.COMPLETE:
            return {"ok": False, "reason": "workflow_not_complete"}
        fw = self.state.framework
        col = self.state.collection
        if not fw or not col or not self.state.final_report:
            return {"ok": False, "reason": "missing_artifacts"}
        req_ids = {r.requirement_id for r in fw.data_requirements}
        covered = set(col.sources_by_requirement.keys())
        missing = sorted(req_ids - covered)
        empty = sorted(k for k in req_ids & covered if not col.sources_by_requirement.get(k))
        return {
            "ok": not missing and not empty,
            "requirements_total": len(req_ids),
            "requirements_with_sources": len(covered & req_ids),
            "missing_requirement_keys": missing,
            "empty_requirement_keys": empty,
        }
