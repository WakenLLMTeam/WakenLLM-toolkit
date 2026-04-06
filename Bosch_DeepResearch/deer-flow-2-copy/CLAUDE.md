# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a customized fork of DeerFlow 2.0 (LangGraph-based AI super agent harness) tailored for **Bosch deep-research and PPT generation workflows**. Key additions on top of the base system: a three-phase consulting report orchestrator, custom agent middlewares, ComfyUI-based image generation, and domain-specific skills (PPT, image, consulting, research).

**Services** (all running through nginx at `localhost:2026`):
- LangGraph Server (port 2024) — agent runtime
- Gateway API (port 8001) — REST API for models, MCP, skills, memory
- Frontend (port 3000) — Next.js UI
- Nginx (port 2026) — unified reverse proxy

## Commands

From the **project root**:
```bash
make config          # Generate config.yaml from config.example.yaml
make config-upgrade  # Non-destructively merge new fields into existing config.yaml
make check           # Verify system requirements (scripts/check.py)
make install         # Install all deps: uv sync (backend) + pnpm install (frontend)
make dev             # Start all services with hot-reload
make dev-daemon      # Same but daemonized in background
make start           # Production mode (no hot-reload)
make stop            # Kill all services (langgraph, uvicorn, next, nginx, containers)
make clean           # stop + delete .deer-flow/, .langgraph_api/, logs
make up / make down  # Docker Compose production build/teardown
```

From the **backend/** directory:
```bash
make dev             # LangGraph server only (port 2024)
make gateway         # Gateway API only (port 8001)
make test            # Run all backend tests
make lint            # Lint with ruff
make format          # Format with ruff

# Run a single test file
PYTHONPATH=. uv run pytest tests/test_<feature>.py -v
```

## Architecture

### Two-Layer Backend Split

**Strict dependency rule: `deerflow.*` (harness) never imports from `app.*`. Enforced by `tests/test_harness_boundary.py` in CI.**

- `backend/packages/harness/deerflow/` — publishable harness package (`deerflow.*`): agent orchestration, tools, sandbox, models, MCP, skills, config
- `backend/app/` — unpublished application (`app.*`): FastAPI Gateway, IM channels (Feishu/Slack/Telegram)

### Custom Additions (this fork's non-standard components)

#### `orchestrator/` — Three-Phase Consulting Report Pipeline
Standalone pipeline, separate from the DeerFlow chat UI. Runs three distinct LangChain agents in sequence with narrow tool scopes to prevent phase-skipping:
1. **Framework Agent** — no tools, JSON-only output; defines the research structure (`ReportFramework`)
2. **Collection Agent** — web tools only; gathers data per the framework
3. **Report Agent** — file read/write only; writes the final report

Key files: `pipeline.py` (`ReportWorkflowOrchestrator`), `phase_agents.py`, `state.py`, `bootstrap.py` (adds harness to `sys.path`).

#### `middlewares/` — Custom Agent Middlewares (repo-root level)
These live at the **repo root** (not inside the harness) and **shadow/extend** the harness built-ins (`deerflow.agents.middlewares.*`). The orchestrator imports from both. When the same middleware name exists in both places, the repo-root version takes precedence for the orchestrator.

Key custom middlewares:
- `citation_middleware.py` — IEEE numeric citations injected post-model; validates URLs from web tool results (anti-hallucination), appends `## 参考文献`. Uses `publication_date` utils.
- `loop_detection_middleware.py` — hashes tool call sets; warns at 3 identical calls, force-strips `tool_calls` at 5. Uses `HumanMessage` (not `SystemMessage`) to avoid Anthropic multi-system-message crash.
- `clarification_middleware.py` — intercepts `ask_clarification` and emits a LangGraph `Command(goto=END)` to interrupt graph execution.
- `subagent_limit_middleware.py` — clamps parallel `task` tool calls to 2–4 (more reliable than prompt-only limits).
- `token_usage_middleware.py`, `empty_table_guard_middleware.py` — additional guardrails.

#### `comfy_ppt_agent/` — ComfyUI Image Generation
`comfy_ppt_agent.py` drives a local ComfyUI server (`localhost:8188`) via REST API to generate slide images, then assembles PPTX. Start server with `comfy_ppt_agent/run_comfy_server.sh`. Default model: SD 1.5 (`v1-5-pruned-emaonly.safetensors`), override via `COMFY_CHECKPOINT`.

#### `scripts/ppt/` — PPT Dev & Experiment Scripts
All PPT-related experimentation scripts consolidated here (not production entry points):

| Script | Purpose |
|--------|---------|
| `ppt_generation_runner.py` | Primary runner; `--mode` (tesla/illustrated/logic/bosch), `--backend` (diffusers/comfy); handles container↔local path mapping |
| `flux_image_runner.py` | Standalone FLUX text-to-image via HuggingFace Diffusers |
| `render_l2_l3_timeline_png.py` | Generates L2→L3 timeline PNG via Pillow (used as slide footer image) |
| `run_timeline_evolution_demo.py` | One-click timeline-evolution-ppt demo |
| `run_agent_skills_ppt.py` | Drives DeerFlow agent via `ppt-generation` + `image-generation` skills |
| `run_agent_skills_ppt_direct.py` | Same but runs in-process (avoids subprocess env issues) |
| `standalone_fsd_ppt_agent.py` | Imports `make_lead_agent` directly, bypasses the UI |
| `generate_3slides.py` / `generate_fsd_ppt.py` / `generate_ppt_direct.py` / `generate_tesla_ppt.py` | One-off generation scripts for specific demos |
| `setup_fsd_ppt.py` | Prepares full PPT generation prompt for the agent |
| `test_l2_l3_summary_ppt.py` / `test_summary_ppt_editable.py` | Local tests for summary-ppt-editable skill |
| `bosch_batch_12_prompts.json` | Batch SDXL prompts for Bosch demo decks |

#### `scripts/dev/` — Developer Utilities
- `export_claude_code_oauth.py` — exports Claude Code OAuth credentials from macOS Keychain

#### `scripts/` (root) — System & Infra Scripts
Shell scripts called by `Makefile`: `check.sh`, `configure.py`, `deploy.sh`, `docker.sh`, `serve.sh`, `start-daemon.sh`, `config-upgrade.sh`, `wait-for-port.sh`, `cleanup-containers.sh`.

#### Timeline PPT API (`backend/app/gateway/routers/timeline_ppt.py`)
Gateway router at `/api/ppt/timeline`. Calls `skills/public/timeline-evolution-ppt/scripts/orchestrate_timeline_pptx.py` as a subprocess to build editable PPTX (L2→L3 ADAS evolution decks). Two endpoints:
- `POST /api/ppt/timeline/l2-l3` — uses built-in `example_deck_v2.spec.json`
- `POST /api/ppt/timeline/generate` — accepts custom `deck_spec` JSON body

### Skills System
Skills are SKILL.md files the agent fetches via `read_file()` at runtime — they are not compiled into the system prompt. The agent receives path pointers (`/mnt/skills/public/{name}/SKILL.md`) and reads them on demand. All skills live in `skills/public/`.

Key domain skills in this fork: `ppt-generation/`, `image-generation/`, `consulting-analysis/`, `deep-research/`, `timeline-evolution-ppt/`, `summary-ppt-editable/`, `github-deep-research/`.

The `pptx/` skill sub-directory contains helper scripts (`scripts/generate.py`, `scripts/office/pack.py`) and schema references for building editable PPTX via pptxgenjs.

### Configuration
- **`config.yaml`** (project root, gitignored) — main config. Config values starting with `$` are resolved as env vars (e.g. `$OPENAI_API_KEY`).
- **`config.daily-scout.yaml`** — alternate config for automated daily research scouting mode.
- **`extensions_config.json`** — MCP servers + skill enable/disable state.

### Harness / App Import Rule
```python
# Allowed
from deerflow.agents import make_lead_agent   # harness internal
from app.gateway.app import app               # app internal
from deerflow.config import get_app_config    # app → harness

# FORBIDDEN (breaks CI)
# from app.gateway.routers.uploads import ... # harness → app
```

## Code Style
- `ruff` for linting/formatting, line length 240, Python 3.12+, double quotes
- Tests: `backend/tests/test_<feature>.py`; circular import issues → add `sys.modules` mock in `tests/conftest.py`
- Every new feature or bug fix must have unit tests (`make test` before and after)
- Update `CLAUDE.md` for architecture/workflow changes; `README.md` for user-facing changes
