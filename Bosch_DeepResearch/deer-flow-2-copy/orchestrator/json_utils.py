from __future__ import annotations

import json
import re


def extract_json_object(text: str) -> str:
    """Return a JSON object/array substring from model output (handles ```json fences)."""
    s = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", s, re.IGNORECASE)
    if fence:
        s = fence.group(1).strip()
    # Prefer outermost { ... } or [ ... ]
    start_obj = s.find("{")
    start_arr = s.find("[")
    if start_obj == -1 and start_arr == -1:
        return s
    if start_obj == -1:
        start = start_arr
        open_c, close_c = "[", "]"
    elif start_arr == -1 or start_obj < start_arr:
        start = start_obj
        open_c, close_c = "{", "}"
    else:
        start = start_arr
        open_c, close_c = "[", "]"
    depth = 0
    for i in range(start, len(s)):
        c = s[i]
        if c == open_c:
            depth += 1
        elif c == close_c:
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return s[start:]


def parse_json_loose(text: str) -> dict | list:
    blob = extract_json_object(text)
    return json.loads(blob)
