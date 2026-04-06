#!/usr/bin/env python3
"""
LLM-driven deck content planner.
Standalone — no DeerFlow dependency required.

Given a topic string, calls an LLM to produce a fully-structured deck plan JSON
(including figure.viz specs for all charts), ready for build_deck.py.

Supported LLM backends (auto-detected from env vars):
  1. Anthropic Claude  — ANTHROPIC_API_KEY
  2. OpenAI / Azure    — OPENAI_API_KEY
  3. Custom OpenAI-compatible endpoint — CUSTOM_LLM_BASE_URL + CUSTOM_LLM_API_KEY + CUSTOM_LLM_MODEL
  4. Ollama (local)    — OLLAMA_HOST (default http://localhost:11434), OLLAMA_MODEL

Usage (library):
  from llm_planner import generate_plan
  plan = generate_plan("L2 to L3 autonomous driving evolution")

Usage (CLI):
  python llm_planner.py --topic "L2 to L3 autonomous driving evolution" --output plan.json
  python llm_planner.py --topic "Transformer architecture explained" --slides 8
  python llm_planner.py --topic "..." --backend anthropic --model claude-opus-4-5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from typing import Any, Dict, Optional

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = textwrap.dedent("""
You are an expert presentation architect. Your job is to produce a richly-structured,
information-dense PowerPoint deck plan as a single valid JSON object.

## Output contract

Return ONLY the JSON object — no markdown fences, no commentary, no extra text.
The JSON must conform to this schema:

{
  "title": "string",
  "subtitle": "string (optional)",
  "aspect_ratio": "16:9",
  "theme": {
    "accent_rgb": [R, G, B],
    "body_rgb": [R, G, B],
    "title_rgb": [R, G, B],
    "author": "string"
  },
  "slides": [ <slide_object>, ... ]
}

Each slide_object:
{
  "slide_number": 1,
  "type": "title" | "section" | "content" | "summary",
  "title": "string",
  "subtitle": "string — only for type:title",
  "bullets": ["string", ...],          // OR use modules below
  "modules": [                          // multi-block layout
    { "heading": "string", "bullets": ["string", ...] }
  ],
  "figure": {                           // optional visualization
    "position": "right" | "bottom",
    "caption": "string",
    "viz": { ... }                      // see Viz spec below
  },
  "notes": "speaker notes string"
}

## Viz spec (figure.viz)

Choose ONE type per slide. Use the type that best conveys the slide's logic.

### timeline — sequence of stages over time
{
  "type": "timeline",
  "title": "string",
  "stages": [
    { "label": "string", "year": "string", "annotation": "short tag", "detail": "one-liner" }
  ],
  "highlight": [0-based index of the most important stage]
}
Use for: historical evolution, roadmaps, project phases.

### flowchart — directed decision/process graph
{
  "type": "flowchart",
  "title": "string",
  "layout": "LR" | "TB",
  "nodes": [
    { "id": "n1", "label": "string", "shape": "rect"|"diamond"|"rounded", "color": "#hex" }
  ],
  "edges": [
    { "from": "n1", "to": "n2", "label": "optional edge label" }
  ]
}
Use for: decision logic, system architecture, cause-effect chains.
Diamond shape for decision nodes. Rounded for terminal/result nodes.

### comparison — feature matrix table
{
  "type": "comparison",
  "title": "string",
  "highlight_col": 0-based int,
  "rows": ["row header", ...],
  "cols": ["col header", ...],
  "cells": [["cell00", "cell01", ...], ["cell10", ...], ...],
  "row_notes": ["", "Key difference", ...]
}
Use for: side-by-side feature comparison, capability matrices.

### pipeline — horizontal block diagram
{
  "type": "pipeline",
  "title": "string",
  "arrow_label": "data flow",
  "stages": [
    { "label": "string", "sublabel": "string", "color": "#hex", "badge": "ASIL-D" }
  ]
}
Use for: system module chains, data pipelines, processing steps.

## Slide count and quality rules

- Produce 8–12 slides for a comprehensive topic; 5–7 for a focused topic.
- Structure: title → 2–3 section dividers → content slides → summary.
- Every content slide should have either:
  - A figure.viz (timeline/flowchart/comparison/pipeline) — preferred for logic topics
  - OR rich modules with 3+ substantive bullet points each
- At least 3 content slides must have a figure.viz.
- Speaker notes must be **under 200 characters**. One punchy sentence of context only — no long paragraphs.
- For pipeline viz: **do NOT include `arrow_label`** — arrows are self-explanatory from context.
- Bullets must be specific and informative — avoid vague filler like "more efficient".
- Speaker notes should add the "why" or regulatory/technical nuance not in the slide text.
- Use position:"bottom" for timelines/pipelines; position:"right" for flowcharts/comparisons.

## Color guidance

Pick a coherent accent color appropriate to the domain:
- Automotive / engineering: blue [26, 86, 219] or dark navy [15, 40, 100]
- Technology / AI: indigo [79, 70, 229] or teal [13, 148, 136]
- Finance / data: green [5, 150, 105] or blue-grey [71, 85, 105]
- Medical / safety: deep blue [30, 64, 175] or purple [109, 40, 217]
""").strip()


_USER_PROMPT_TEMPLATE = textwrap.dedent("""
Topic: {topic}

Additional constraints:
- Language: English (all text in the deck must be English)
- Slides requested: {slides}
- Extra instructions: {extra}

Generate the complete deck plan JSON now.
""").strip()


# ── Backend adapters ──────────────────────────────────────────────────────────

def _call_anthropic(prompt: str, model: str) -> str:
    import anthropic
    # Support both ANTHROPIC_API_KEY and ANTHROPIC_AUTH_TOKEN (custom gateways)
    api_key = (os.environ.get("ANTHROPIC_API_KEY")
               or os.environ.get("ANTHROPIC_AUTH_TOKEN")
               or "")
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    kwargs = dict(api_key=api_key)
    if base_url:
        kwargs["base_url"] = base_url
    client = anthropic.Anthropic(**kwargs)
    msg = client.messages.create(
        model=model or "claude-sonnet-4-6",
        max_tokens=8192,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def _call_openai(prompt: str, model: str, base_url: Optional[str] = None,
                 api_key: Optional[str] = None) -> str:
    from openai import OpenAI
    client = OpenAI(
        api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
        base_url=base_url,
    )
    resp = client.chat.completions.create(
        model=model or "gpt-4o",
        max_tokens=8192,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content


def _call_ollama(prompt: str, model: str, host: str) -> str:
    import urllib.request
    url = f"{host.rstrip('/')}/api/chat"
    payload = json.dumps({
        "model": model or "llama3",
        "stream": False,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }).encode()
    req = urllib.request.Request(url, data=payload,
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read())
    return data["message"]["content"]


def _detect_backend() -> str:
    """Auto-detect which backend to use based on environment variables."""
    if (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("CUSTOM_LLM_BASE_URL"):
        return "custom"
    return "ollama"


def _call_llm(prompt: str, backend: str, model: str) -> str:
    if backend == "anthropic":
        return _call_anthropic(prompt, model)
    if backend == "openai":
        return _call_openai(prompt, model)
    if backend == "custom":
        return _call_openai(
            prompt, model or os.environ.get("CUSTOM_LLM_MODEL", "gpt-4o"),
            base_url=os.environ["CUSTOM_LLM_BASE_URL"],
            api_key=os.environ.get("CUSTOM_LLM_API_KEY"),
        )
    if backend == "ollama":
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        mdl = model or os.environ.get("OLLAMA_MODEL", "llama3")
        return _call_ollama(prompt, mdl, host)
    raise ValueError(f"Unknown backend: {backend!r}")


# ── JSON extraction ────────────────────────────────────────────────────────────

def _extract_json(raw: str) -> Dict[str, Any]:
    """Strip markdown fences and extract the first valid JSON object."""
    # Remove ```json ... ``` or ``` ... ```
    text = raw.strip()
    for fence in ("```json", "```"):
        if text.startswith(fence):
            text = text[len(fence):]
            if text.endswith("```"):
                text = text[:-3]
            break

    # Find outermost { ... }
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in LLM response")
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start: i + 1])
    raise ValueError("Incomplete JSON object in LLM response")


# ── Public API ────────────────────────────────────────────────────────────────

def generate_plan(
    topic: str,
    slides: int = 10,
    extra: str = "",
    backend: Optional[str] = None,
    model: str = "",
) -> Dict[str, Any]:
    """
    Generate a deck plan dict for the given topic using an LLM.

    Args:
        topic:   Deck topic (free-form English string)
        slides:  Target number of slides
        extra:   Extra instructions appended to the user prompt
        backend: "anthropic" | "openai" | "custom" | "ollama" | None (auto-detect)
        model:   Model name override (optional)

    Returns:
        Parsed plan dict ready for build_deck.build_deck()
    """
    if backend is None:
        backend = _detect_backend()

    user_prompt = _USER_PROMPT_TEMPLATE.format(
        topic=topic,
        slides=slides,
        extra=extra or "None",
    )
    print(f"[llm_planner] Backend: {backend}  Topic: {topic[:60]}", file=sys.stderr)
    raw = _call_llm(user_prompt, backend, model)
    plan = _extract_json(raw)
    print(f"[llm_planner] Plan generated: {len(plan.get('slides', []))} slides",
          file=sys.stderr)
    return plan


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a deck plan JSON from a topic using an LLM"
    )
    parser.add_argument("--topic", required=True,
                        help="Deck topic (free-form English string)")
    parser.add_argument("--output", default=None,
                        help="Write plan JSON to this file (default: stdout)")
    parser.add_argument("--slides", type=int, default=10,
                        help="Target number of slides (default: 10)")
    parser.add_argument("--extra", default="",
                        help="Extra instructions for the LLM")
    parser.add_argument("--backend", default=None,
                        choices=["anthropic", "openai", "custom", "ollama"],
                        help="LLM backend (default: auto-detect from env)")
    parser.add_argument("--model", default="",
                        help="Model name override")
    args = parser.parse_args()

    plan = generate_plan(args.topic, args.slides, args.extra,
                         args.backend, args.model)
    out = json.dumps(plan, ensure_ascii=False, indent=2)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(out, encoding="utf-8")
        print(f"[llm_planner] Plan written to {args.output}", file=sys.stderr)
    else:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
