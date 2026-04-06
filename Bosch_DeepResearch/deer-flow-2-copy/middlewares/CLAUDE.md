# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Repo-root middleware layer that **shadows and extends** the harness built-ins at `backend/packages/harness/deerflow/agents/middlewares/`. The full DeerFlow lead agent loads these instead of (or in addition to) the harness versions. All files implement `AgentMiddleware` from `langchain.agents.middleware`.

## Middleware Reference

| File | Hook | What it does |
|------|------|--------------|
| `loop_detection_middleware.py` | `after_model` | Hashes tool call sets per thread; warns at 3 identical, force-strips `tool_calls` at 5. Uses `HumanMessage` (not `SystemMessage`) for the warning — Anthropic crashes on mid-conversation SystemMessages. |
| `citation_middleware.py` | `after_model` | IEEE numeric citations from web tool results. Validates URLs against actual tool output (anti-hallucination). Appends `## 参考文献`. Also has a static method `apply_to_markdown_with_allowed_urls()` used by the orchestrator. |
| `clarification_middleware.py` | `before_tool` | Intercepts `ask_clarification` tool calls; returns `Command(goto=END)` to interrupt graph execution and surface the question to the user. Must be **last** in middleware chain. |
| `subagent_limit_middleware.py` | `after_model` | Truncates excess `task` tool calls to `[2, 4]` range. More reliable than prompt-only limits. Imports `MAX_CONCURRENT_SUBAGENTS` from `deerflow.subagents.executor`. |
| `token_usage_middleware.py` | `after_model` | Logs `usage_metadata` from model response. No state mutation. |
| `empty_table_guard_middleware.py` | `after_model` | Detects empty Markdown tables (header + separator, no rows) and inserts a placeholder row to prevent rendering collapse. |
| `thread_data_middleware.py` | lifecycle | Creates per-thread directories under `.deer-flow/threads/{thread_id}/`. |
| `memory_middleware.py` | `after_model` | Queues conversations for async memory extraction. |
| `uploads_middleware.py` | `before_model` | Injects newly uploaded file list into conversation context. |
| `title_middleware.py` | `after_model` | Auto-generates thread title after first exchange. |
| `todo_middleware.py` | lifecycle | Provides `write_todos` tool when plan_mode is active. |
| `deferred_tool_filter_middleware.py` | `before_model` | Filters deferred tool results. |
| `dangling_tool_call_middleware.py` | `before_model` | Injects placeholder ToolMessages for AIMessage tool_calls missing responses (e.g. interrupted runs). |
| `tool_error_handling_middleware.py` | `after_tool` | Standard error wrapping for tool failures. |
| `view_image_middleware.py` | `before_model` | Injects base64 image data for vision models. |
| `runtime_thread.py` | utility | Shared threading utilities for middleware runtime. |

## Key Gotchas

**`HumanMessage` not `SystemMessage` for injections** (`loop_detection_middleware.py:206`): Anthropic's `langchain_anthropic` crashes with `_format_messages()` if a SystemMessage appears mid-conversation. Any middleware that injects a message must use `HumanMessage`.

**`citation_middleware.py` has a static API**: `CitationMiddleware.apply_to_markdown_with_allowed_urls(markdown, allowed_urls)` is called by `orchestrator/pipeline.py` directly — it's not just a middleware hook. If you change the signature, update the orchestrator too.

**`clarification_middleware.py` must be last**: It emits `Command(goto=END)` which stops the graph. Any middleware after it in the chain won't execute when clarification is triggered.

**Shadowing vs. replacing**: These files shadow harness versions of the same name. When the lead agent builds its middleware chain, it imports from this directory. The `orchestrator/` pipeline uses `deerflow.agents.middlewares.*` (harness versions), not these.

## Adding a New Middleware

1. Subclass `AgentMiddleware[AgentState]` (or a custom state schema).
2. Override `after_model` / `before_model` / `before_tool` / `after_tool` as needed — both sync and async variants (`aafter_model`, etc.).
3. Return `dict` to update agent state, or `None` to pass through. Return `Command(...)` from `before_tool` to redirect graph flow.
4. Register it in the lead agent's middleware builder (in the harness or via fork override).
