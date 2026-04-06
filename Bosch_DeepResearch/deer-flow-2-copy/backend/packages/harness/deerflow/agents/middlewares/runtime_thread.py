"""Resolve ``thread_id`` from LangGraph runtime (multiple execution paths)."""

from __future__ import annotations

from langgraph.runtime import Runtime


def resolve_thread_id(runtime: Runtime) -> str | None:
    """Best-effort thread id: ``runtime.context`` → ``runtime.config`` → ``get_config()``."""
    context = getattr(runtime, "context", None) or {}
    if hasattr(context, "get"):
        tid = context.get("thread_id")
        if tid:
            return str(tid)

    config = getattr(runtime, "config", None)
    if config is not None:
        if hasattr(config, "get"):
            configurable = config.get("configurable", {}) or {}
        else:
            configurable = getattr(config, "configurable", None) or {}
        if hasattr(configurable, "get"):
            tid = configurable.get("thread_id")
            if tid:
                return str(tid)

    try:
        from langgraph.config import get_config

        tid = (get_config().get("configurable") or {}).get("thread_id")
        if tid:
            return str(tid)
    except Exception:
        pass

    return None
