"""Three dedicated LangChain agents: framework (no tools), collection (web), report (files)."""

from __future__ import annotations

import logging
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from orchestrator.bootstrap import ensure_deerflow_importable

logger = logging.getLogger(__name__)

TOOL_GROUP_WEB = "web"
TOOL_GROUP_FILE_READ = "file:read"
TOOL_GROUP_FILE_WRITE = "file:write"
TOOL_GROUP_BASH = "bash"


def _ensure() -> None:
    if ensure_deerflow_importable() is None:
        logger.warning(
            "DeerFlow harness not found at backend/packages/harness; import deerflow may fail. "
            "Run from repo root or set PYTHONPATH to the harness directory.",
        )


def _phase_middlewares() -> list[Any]:
    _ensure()
    from deerflow.agents.middlewares.clarification_middleware import ClarificationMiddleware
    from deerflow.agents.middlewares.loop_detection_middleware import LoopDetectionMiddleware
    from deerflow.agents.middlewares.tool_error_handling_middleware import build_lead_runtime_middlewares

    m = build_lead_runtime_middlewares(lazy_init=True)
    m.append(LoopDetectionMiddleware())
    m.append(ClarificationMiddleware())
    return m


FRAMEWORK_SYSTEM_PROMPT = """You are the **Framework Agent** (phase 1 of a fixed report pipeline).
Your only job is to design a structured analysis/report plan. You do NOT browse the web and you have no tools.

Output **only** one JSON object (no markdown prose outside JSON) with this exact shape:
{
  "title": "<short report title>",
  "objectives": "<1-3 sentences>",
  "chapters": [
    {"chapter_id": "ch1", "title": "...", "objective": "..."}
  ],
  "data_requirements": [
    {
      "requirement_id": "r1",
      "section_id": "ch1",
      "metric": "what to measure or find",
      "priority": "P0|P1|P2",
      "search_keywords": ["kw1", "kw2"]
    }
  ]
}

Rules:
- Include 4–8 chapters with stable `chapter_id` values (slug-like).
- Include at least one `data_requirement` per chapter that needs external facts; `requirement_id` must be unique.
- Keep `search_keywords` concrete and short.
"""


def build_framework_agent(*, model_name: str | None, thinking_enabled: bool = False) -> Any:
    _ensure()
    from deerflow.agents.thread_state import ThreadState
    from deerflow.models import create_chat_model

    model = create_chat_model(name=model_name, thinking_enabled=thinking_enabled)
    return create_agent(
        model=model,
        tools=[],
        middleware=_phase_middlewares(),
        system_prompt=FRAMEWORK_SYSTEM_PROMPT,
        state_schema=ThreadState,
    )


def build_collection_agent(*, model_name: str | None, thinking_enabled: bool = False) -> Any:
    _ensure()
    from deerflow.agents.thread_state import ThreadState
    from deerflow.models import create_chat_model
    from deerflow.tools import get_available_tools

    model = create_chat_model(name=model_name, thinking_enabled=thinking_enabled)
    tools = get_available_tools(
        groups=[TOOL_GROUP_WEB],
        include_mcp=False,
        model_name=model_name,
        subagent_enabled=False,
    )
    system = """You are the **Collection Agent** (phase 2). You may only use **web_search** and **web_fetch** (and other tools in the **web** group if present).

For each `requirement_id` in the user message, gather 2–6 authoritative URLs and optionally short notes.
Do not write the final report.

When finished, reply with **only** a JSON object (no extra text):
{
  "sources_by_requirement": { "r1": ["https://...", "..."], "r2": [] },
  "notes": "optional brief summary of coverage gaps"
}
"""
    return create_agent(
        model=model,
        tools=tools,
        middleware=_phase_middlewares(),
        system_prompt=system,
        state_schema=ThreadState,
    )


def build_report_agent(
    *,
    model_name: str | None,
    thinking_enabled: bool = False,
    include_bash: bool = False,
) -> Any:
    _ensure()
    from deerflow.agents.thread_state import ThreadState
    from deerflow.models import create_chat_model
    from deerflow.tools import get_available_tools

    model = create_chat_model(name=model_name, thinking_enabled=thinking_enabled)
    groups = [TOOL_GROUP_FILE_READ, TOOL_GROUP_FILE_WRITE]
    if include_bash:
        groups.append(TOOL_GROUP_BASH)
    tools = get_available_tools(
        groups=groups,
        include_mcp=False,
        model_name=model_name,
        subagent_enabled=False,
    )
    system = """You are the **Report Agent** (phase 3). You write the final report using the framework and evidence provided in the user message.

You may use sandbox file tools (read/write/ls/str_replace, bash if available) to draft sections under the thread workspace if helpful, but the user-visible result must be the **full final report in Markdown** in your last assistant message.

Rules:
- Follow the chapter structure from the framework.
- Cite sources using IEEE-style inline markers: ``[citation:Short title](FULL_URL)`` where **FULL_URL** must appear exactly in the evidence JSON (same URL as collected). Do not invent URLs.
- A post-processor will convert these to numeric ``[1][2]`` and append an IEEE-formatted **References** list; you do not need to write that list yourself unless you already added one (it may be normalized).
- Do not restart data collection; use only the evidence given.
"""
    return create_agent(
        model=model,
        tools=tools,
        middleware=_phase_middlewares(),
        system_prompt=system,
        state_schema=ThreadState,
    )


def run_agent_sync(
    agent: Any,
    user_text: str,
    *,
    thread_id: str,
    recursion_limit: int = 50,
) -> str:
    """Invoke agent and return the last assistant text (concatenate text blocks if needed)."""
    from langchain_core.messages import AIMessage

    state: dict[str, Any] = {"messages": [HumanMessage(content=user_text)]}
    run_config: RunnableConfig = {
        "recursion_limit": recursion_limit,
        "configurable": {"thread_id": thread_id},
    }
    context: dict[str, Any] = {"thread_id": thread_id}
    final = agent.invoke(state, config=run_config, context=context)
    messages = final.get("messages") or []
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            c = msg.content
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                parts: list[str] = []
                for block in c:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(str(block.get("text", "")))
                    elif isinstance(block, str):
                        parts.append(block)
                return "\n".join(parts)
            return str(c)
    return ""
