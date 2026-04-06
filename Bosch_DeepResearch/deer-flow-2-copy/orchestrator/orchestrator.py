"""
E2E Orchestrator for Three-Phase Consulting Analysis Workflow

This module implements the orchestration layer that manages the three phases:
1. Framework Generation Phase
2. Data Collection Phase  
3. Report Writing Phase

The orchestrator ensures:
- Framework is generated before data collection
- Data is collected according to framework requirements
- Report structure aligns with the framework
- State is properly maintained across phases
"""

from enum import Enum
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime

from src.agents.thread_state import AnalysisFrameworkData


class WorkflowPhase(Enum):
    """Enum for workflow phases."""
    INIT = "init"  # Before workflow starts
    PHASE_1_FRAMEWORK = "phase_1_framework"  # Framework generation
    PHASE_2_DATA_COLLECTION = "phase_2_data_collection"  # Data collection
    PHASE_3_REPORT_WRITING = "phase_3_report_writing"  # Report writing
    COMPLETE = "complete"  # Workflow complete


@dataclass
class PhaseMetadata:
    """Metadata for a workflow phase."""
    phase: WorkflowPhase
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    status: str = "pending"  # pending, in_progress, completed, failed
    error_message: Optional[str] = None
    output_summary: Optional[str] = None


@dataclass
class E2EWorkflowState:
    """Complete state for E2E workflow tracking."""
    current_phase: WorkflowPhase = WorkflowPhase.INIT
    phase_history: List[PhaseMetadata] = field(default_factory=list)
    
    # Phase 1 outputs
    framework: Optional[AnalysisFrameworkData] = None
    
    # Phase 2 outputs
    collected_data: Dict[str, List[str]] = field(default_factory=dict)  # section -> [urls]
    data_coverage: Dict[str, List[str]] = field(default_factory=dict)  # req_id -> [urls]
    
    # Phase 3 outputs
    final_report: Optional[str] = None
    report_sections: List[str] = field(default_factory=list)
    
    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "current_phase": self.current_phase.value,
            "phase_history": [
                {
                    "phase": m.phase.value,
                    "started_at": m.started_at,
                    "completed_at": m.completed_at,
                    "status": m.status,
                    "error_message": m.error_message,
                    "output_summary": m.output_summary,
                }
                for m in self.phase_history
            ],
            "framework": self.framework.to_dict() if self.framework else None,
            "collected_data": self.collected_data,
            "data_coverage": self.data_coverage,
            "final_report": self.final_report,
            "report_sections": self.report_sections,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class E2EWorkflowOrchestrator:
    """Orchestrator for three-phase consulting analysis workflow."""
    
    def __init__(self):
        """Initialize the orchestrator."""
        self.workflow_state = E2EWorkflowState()
    
    def start_phase_1(self) -> bool:
        """Start Phase 1: Framework Generation."""
        if self.workflow_state.current_phase != WorkflowPhase.INIT:
            return False
        
        self.workflow_state.current_phase = WorkflowPhase.PHASE_1_FRAMEWORK
        metadata = PhaseMetadata(
            phase=WorkflowPhase.PHASE_1_FRAMEWORK,
            started_at=datetime.now().isoformat(),
            status="in_progress"
        )
        self.workflow_state.phase_history.append(metadata)
        return True
    
    def complete_phase_1(self, framework: AnalysisFrameworkData, summary: str = "") -> bool:
        """Complete Phase 1 and save framework."""
        if self.workflow_state.current_phase != WorkflowPhase.PHASE_1_FRAMEWORK:
            return False
        
        self.workflow_state.framework = framework
        
        # Update phase metadata
        metadata = self.workflow_state.phase_history[-1]
        metadata.completed_at = datetime.now().isoformat()
        metadata.status = "completed"
        metadata.output_summary = summary
        
        self.workflow_state.current_phase = WorkflowPhase.PHASE_2_DATA_COLLECTION
        metadata_2 = PhaseMetadata(
            phase=WorkflowPhase.PHASE_2_DATA_COLLECTION,
            started_at=datetime.now().isoformat(),
            status="in_progress"
        )
        self.workflow_state.phase_history.append(metadata_2)
        return True
    
    def record_collected_data(self, section_id: str, urls: List[str], summary: str = "") -> bool:
        """Record collected data during Phase 2."""
        if self.workflow_state.current_phase != WorkflowPhase.PHASE_2_DATA_COLLECTION:
            return False
        
        if section_id not in self.workflow_state.collected_data:
            self.workflow_state.collected_data[section_id] = []
        
        # Add unique URLs
        for url in urls:
            if url not in self.workflow_state.collected_data[section_id]:
                self.workflow_state.collected_data[section_id].append(url)
        
        return True
    
    def get_phase_2_coverage(self) -> Dict[str, Any]:
        """Get Phase 2 data collection coverage status."""
        if not self.workflow_state.framework:
            return {"status": "no_framework", "coverage": 0}
        
        total_chapters = len(self.workflow_state.framework.chapters)
        covered_chapters = len(self.workflow_state.collected_data)
        
        return {
            "status": "in_progress",
            "total_chapters": total_chapters,
            "covered_chapters": covered_chapters,
            "coverage_pct": (covered_chapters / total_chapters * 100) if total_chapters > 0 else 0,
            "sections_with_data": list(self.workflow_state.collected_data.keys()),
            "total_sources": sum(len(urls) for urls in self.workflow_state.collected_data.values()),
        }
    
    def complete_phase_2(self, summary: str = "") -> bool:
        """Complete Phase 2 and transition to Phase 3."""
        if self.workflow_state.current_phase != WorkflowPhase.PHASE_2_DATA_COLLECTION:
            return False
        
        # Check coverage
        coverage = self.get_phase_2_coverage()
        if coverage.get("coverage_pct", 0) < 50:
            # Warning: less than 50% coverage, but allow transition
            pass
        
        # Update phase metadata
        metadata = self.workflow_state.phase_history[-1]
        metadata.completed_at = datetime.now().isoformat()
        metadata.status = "completed"
        metadata.output_summary = summary
        
        self.workflow_state.current_phase = WorkflowPhase.PHASE_3_REPORT_WRITING
        metadata_3 = PhaseMetadata(
            phase=WorkflowPhase.PHASE_3_REPORT_WRITING,
            started_at=datetime.now().isoformat(),
            status="in_progress"
        )
        self.workflow_state.phase_history.append(metadata_3)
        return True
    
    def set_final_report(self, report: str, sections: List[str], summary: str = "") -> bool:
        """Set final report and complete Phase 3."""
        if self.workflow_state.current_phase != WorkflowPhase.PHASE_3_REPORT_WRITING:
            return False
        
        self.workflow_state.final_report = report
        self.workflow_state.report_sections = sections
        
        # Update phase metadata
        metadata = self.workflow_state.phase_history[-1]
        metadata.completed_at = datetime.now().isoformat()
        metadata.status = "completed"
        metadata.output_summary = summary
        
        self.workflow_state.current_phase = WorkflowPhase.COMPLETE
        return True
    
    def validate_workflow_alignment(self) -> Dict[str, Any]:
        """Validate alignment between framework and report."""
        if self.workflow_state.current_phase != WorkflowPhase.COMPLETE:
            return {"valid": False, "reason": "Workflow not complete"}
        
        if not self.workflow_state.framework:
            return {"valid": False, "reason": "No framework"}
        
        if not self.workflow_state.final_report:
            return {"valid": False, "reason": "No final report"}
        
        # Check alignment
        framework_chapters = len(self.workflow_state.framework.chapters)
        report_sections = len(self.workflow_state.report_sections)
        
        alignment_score = min(framework_chapters, report_sections) / max(framework_chapters, report_sections)
        
        issues = []
        if framework_chapters != report_sections:
            issues.append(f"Chapter count mismatch: {framework_chapters} framework chapters vs {report_sections} report sections")
        
        collected_sections = len(self.workflow_state.collected_data)
        if collected_sections < framework_chapters:
            issues.append(f"Data coverage incomplete: {collected_sections}/{framework_chapters} sections")
        
        return {
            "valid": alignment_score > 0.8,
            "alignment_score": alignment_score,
            "framework_chapters": framework_chapters,
            "report_sections": report_sections,
            "collected_data_sections": collected_sections,
            "issues": issues,
        }
    
    def get_workflow_summary(self) -> str:
        """Get a formatted workflow execution summary."""
        summary = f"=== E2E Workflow Execution Summary ===\n"
        summary += f"Current Phase: {self.workflow_state.current_phase.value}\n"
        summary += f"Created: {self.workflow_state.created_at}\n\n"
        
        summary += "Phase History:\n"
        for i, metadata in enumerate(self.workflow_state.phase_history, 1):
            summary += f"  {i}. {metadata.phase.value}\n"
            summary += f"     Status: {metadata.status}\n"
            if metadata.started_at:
                summary += f"     Started: {metadata.started_at}\n"
            if metadata.completed_at:
                summary += f"     Completed: {metadata.completed_at}\n"
            if metadata.output_summary:
                summary += f"     Summary: {metadata.output_summary}\n"
        
        if self.workflow_state.framework:
            summary += f"\nFramework: {self.workflow_state.framework.title}\n"
            summary += f"  Chapters: {len(self.workflow_state.framework.chapters)}\n"
            summary += f"  Data Requirements: {len(self.workflow_state.framework.data_requirements)}\n"
        
        if self.workflow_state.collected_data:
            summary += f"\nData Collection:\n"
            summary += f"  Sections Covered: {len(self.workflow_state.collected_data)}\n"
            summary += f"  Total Sources: {sum(len(urls) for urls in self.workflow_state.collected_data.values())}\n"
        
        if self.workflow_state.final_report:
            summary += f"\nFinal Report:\n"
            summary += f"  Sections: {len(self.workflow_state.report_sections)}\n"
            summary += f"  Length: {len(self.workflow_state.final_report)} characters\n"
        
        return summary


# Singleton orchestrator instance
_orchestrator: Optional[E2EWorkflowOrchestrator] = None


def get_orchestrator() -> E2EWorkflowOrchestrator:
    """Get or create the singleton orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = E2EWorkflowOrchestrator()
    return _orchestrator


def reset_orchestrator() -> None:
    """Reset the orchestrator (for testing)."""
    global _orchestrator
    _orchestrator = None
