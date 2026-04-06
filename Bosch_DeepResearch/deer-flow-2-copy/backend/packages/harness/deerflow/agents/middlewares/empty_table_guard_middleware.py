"""Guard middleware for Markdown empty tables.

Some model outputs include Markdown tables that have *no data*:
- Only header + separator lines, with no subsequent rows
- Or rows where every cell is effectively empty

In GitHub-flavored Markdown / Streamdown rendering, such tables can collapse or
look like "empty grids". This middleware replaces detected empty tables with
one placeholder data row so every column has content.
"""

from __future__ import annotations

import re
from typing import override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")

# Markdown table separator line, e.g. |---|:---:|---:|
_TABLE_SEPARATOR_RE = re.compile(
    r"^\s*\|?\s*"
    r":?-{3,}:?"
    r"(?:\s*\|\s*:?-{3,}:?\s*)+"
    r"\|?\s*$"
)


def _parse_md_table_row_cells(line: str) -> list[str]:
    """Split a markdown table row into cells."""
    s = line.strip()
    # Drop leading/trailing pipes if present
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    parts = [p.strip() for p in s.split("|")]
    # Filter out degenerate splits
    return parts


def _is_separator_line(line: str) -> bool:
    return bool(_TABLE_SEPARATOR_RE.match(line or ""))


def _is_table_candidate_header_line(line: str) -> bool:
    # Header line should contain pipe delimiters
    s = line.strip()
    return "|" in s and not s.startswith(("#", "```", "~")) and not s.startswith("-")


def _is_code_fence_line(line: str) -> bool:
    return bool(_FENCE_RE.match(line or ""))


def _fix_empty_markdown_tables(md: str) -> str:
    if not md:
        return md

    lines = md.splitlines()
    out: list[str] = []

    in_fence = False
    placeholder = "待补充 - 需要进一步验证"

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]

        # Toggle fenced code blocks; skip table parsing inside them.
        if _is_code_fence_line(line):
            in_fence = not in_fence
            out.append(line)
            i += 1
            continue

        if in_fence:
            out.append(line)
            i += 1
            continue

        # Detect table header + separator
        if i + 1 < n and _is_table_candidate_header_line(line) and _is_separator_line(
            lines[i + 1]
        ):
            header_line = line
            sep_line = lines[i + 1]
            header_cells = _parse_md_table_row_cells(header_line)
            col_count = max(2, len(header_cells))

            # Collect subsequent table rows (must contain pipes)
            j = i + 2
            data_rows: list[str] = []
            while j < n:
                cur = lines[j]
                if not cur.strip():
                    break
                if "```" in cur or _is_code_fence_line(cur):
                    break
                if "|" not in cur:
                    break
                # Stop if we encounter a new separator-like line (start of another table)
                if _is_separator_line(cur):
                    break
                row_cells = _parse_md_table_row_cells(cur)
                if len(row_cells) < 2:
                    break
                data_rows.append(cur)
                j += 1

            # Decide whether this table is "empty"
            is_empty = False
            if not data_rows:
                is_empty = True
            else:
                non_empty_found = False
                for r in data_rows:
                    cells = _parse_md_table_row_cells(r)
                    for c in cells:
                        cs = c.strip()
                        if cs and cs not in {"-", "—", "–"}:
                            non_empty_found = True
                            break
                    if non_empty_found:
                        break
                is_empty = not non_empty_found

            if is_empty:
                indent = re.match(r"^\s*", header_line).group(0)
                row_line = indent + "| " + " | ".join([placeholder] * col_count) + " |"
                out.extend([header_line, sep_line, row_line])
            else:
                out.extend([header_line, sep_line] + data_rows)

            i = j
            continue

        out.append(line)
        i += 1

    return "\n".join(out)


class EmptyTableGuardMiddlewareState(AgentState):
    """Compatible with the `ThreadState` schema."""

    pass


class EmptyTableGuardMiddleware(AgentMiddleware[EmptyTableGuardMiddlewareState]):
    """Post-process the final AI message to fix empty markdown tables."""

    state_schema = EmptyTableGuardMiddlewareState

    @staticmethod
    def _last_ai_message(messages: list) -> AIMessage | None:
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                # Skip tool-calling intermediate messages if present
                if getattr(msg, "tool_calls", None):
                    continue
                return msg
        return None

    def _process_state(self, state: EmptyTableGuardMiddlewareState) -> dict | None:
        messages = state.get("messages", [])
        last_ai_message = self._last_ai_message(messages)
        if last_ai_message is None:
            return None

        content = (
            last_ai_message.content
            if isinstance(last_ai_message.content, str)
            else str(last_ai_message.content)
        )

        fixed = _fix_empty_markdown_tables(content)
        if fixed == content:
            return None

        updated_message = AIMessage(id=last_ai_message.id, content=fixed)
        return {"messages": [updated_message]}

    @override
    def after_model(self, state: EmptyTableGuardMiddlewareState, runtime: Runtime):
        return self._process_state(state)

    @override
    async def aafter_model(
        self, state: EmptyTableGuardMiddlewareState, runtime: Runtime
    ):
        return self._process_state(state)

