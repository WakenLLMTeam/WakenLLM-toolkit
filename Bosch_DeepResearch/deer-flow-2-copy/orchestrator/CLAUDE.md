# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Standalone three-phase consulting report pipeline. Runs **outside** the DeerFlow chat UI — invoked directly as a Python module. Each phase uses a separate LangChain agent with narrowly scoped tools to prevent phase-skipping.

## Phase Flow

```
user_request
    │
    ▼
[Phase 1] Framework Agent  →  ReportFramework (JSON, no tools)
    │
    ▼
[Phase 2] Collection Agent →  CollectionBundle (web tools only)
    │
    ▼
[Phase 3] Report Agent     →  final Markdown + IEEE citations (file tools only)
```

State machine is enforced: phases must run in order and transition is gated by `WorkflowPhase` enum. Calling a phase out of order raises `RuntimeError`.

## Key Files

| File | Role |
|------|------|
| `pipeline.py` | `ReportWorkflowOrchestrator` — the main entry point; run all phases via `run_full()` or individually |
| `phase_agents.py` | Builds each agent with its tool scope and system prompt; `run_agent_sync()` extracts last AI text |
| `state.py` | Pydantic schemas (`ReportFramework`, `CollectionBundle`) + `ReportWorkflowState` dataclass with checkpoint tracking |
| `bootstrap.py` | `ensure_deerflow_importable()` — inserts `backend/packages/harness` into `sys.path` at runtime |
| `json_utils.py` | `parse_json_loose()` — strips markdown fences before JSON parsing (models often wrap JSON in ```json) |

## Usage

```python
from orchestrator.pipeline import ReportWorkflowOrchestrator

orch = ReportWorkflowOrchestrator(
    thread_id="my-thread-001",
    model_name=None,          # None → uses first model in config.yaml
    thinking_enabled=False,
    include_bash_in_report=False,
)
state = orch.run_full("Research the ADAS L2→L3 transition landscape")
print(state.final_report)

# Check coverage alignment after completion
result = orch.validate_alignment()
# {"ok": True, "requirements_total": 6, "requirements_with_sources": 6, ...}
```

Must run from **repo root** (not `orchestrator/`) so `bootstrap.py` can locate `backend/packages/harness`.

## Tool Scopes Per Phase

- **Framework**: no tools — forces pure structured JSON output
- **Collection**: `group="web"` only (`web_search`, `web_fetch`); MCP disabled, subagents disabled
- **Report**: `group=["file:read", "file:write"]` + optionally `"bash"` if `include_bash_in_report=True`

Tools fetched via `deerflow.tools.get_available_tools(groups=..., include_mcp=False, subagent_enabled=False)`.

## Citation Injection (Phase 3)

After the Report Agent produces its Markdown, `CitationMiddleware.apply_to_markdown_with_allowed_urls()` is called statically with only URLs from the `CollectionBundle`. This validates citations are grounded in actual evidence — any URL the model invented that wasn't collected is dropped.

The agent should write inline markers as `[citation:Title](FULL_URL)`; the post-processor converts them to `[1][2]` numeric format and appends `## 参考文献`.

## Middlewares Used

Each phase agent gets `_phase_middlewares()`:
- `LoopDetectionMiddleware` — from `deerflow.agents.middlewares` (harness version, not repo-root)
- `ClarificationMiddleware` — same
- `build_lead_runtime_middlewares(lazy_init=True)` — standard error handling chain

The repo-root `middlewares/` versions are **not** used here (those are for the full DeerFlow lead agent).

## Error Handling

- JSON parse failures in Phase 1/2 set `state.current_phase = FAILED` and raise `RuntimeError`
- Empty report output in Phase 3 also fails the state
- `state.last_error` always holds the last failure message
- `reset()` clears state to re-run on the same `thread_id`
- Checkpoints (`state.checkpoints`) record start/end timestamps and status per phase
