#!/usr/bin/env python3
"""
LLM-driven deck content planner.
Standalone — no DeerFlow dependency required.

Two-stage pipeline (borrowed from PPTAgent):
  Stage 1 — Key-point extraction: LLM distils the raw topic into a structured
             outline of concise key points per section (prevents bloated bullets).
  Stage 2 — Plan generation: LLM uses the key-point outline to produce the full
             deck plan JSON with viz specs, character-count constraints, and
             layout decisions already baked in.

Supported LLM backends (auto-detected from env vars):
  1. Anthropic Claude  — ANTHROPIC_API_KEY
  2. OpenAI / Azure    — OPENAI_API_KEY
  3. Custom OpenAI-compatible endpoint — CUSTOM_LLM_BASE_URL + CUSTOM_LLM_API_KEY + CUSTOM_LLM_MODEL
  4. Ollama (local)    — OLLAMA_HOST (default http://localhost:11434), OLLAMA_MODEL

Usage (library):
  from llm_planner import generate_plan
  plan = generate_plan("L2 to L3 autonomous driving evolution", lang="zh")

Usage (CLI):
  python llm_planner.py --topic "L2 to L3 autonomous driving evolution" --output plan.json
  python llm_planner.py --topic "Transformer architecture explained" --slides 8 --lang zh
  python llm_planner.py --topic "..." --backend custom --model qwen3-max
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── Stage 1: Key-point extraction prompt ──────────────────────────────────────

_EXTRACT_SYSTEM = textwrap.dedent("""
You are a research analyst. Given a presentation topic, produce a concise
structured outline as a JSON object.

Return ONLY the JSON — no fences, no commentary.

Schema:
{
  "topic_summary": "2-3 sentence overview of the topic",
  "sections": [
    {
      "title": "Section name",
      "key_points": [
        "Concise factual point (max 15 words)",
        "Another point"
      ],
      "has_timeline": true|false,
      "has_comparison": true|false,
      "has_process": true|false,
      "has_radar": true|false,
      "has_arch": true|false
    }
  ]
}

Rules:
- 4-7 sections covering the topic comprehensively
- Each section: 3-6 key_points, each ≤15 words, factual and specific
- has_timeline: true when section covers chronological evolution/roadmap
- has_comparison: true when section compares multiple options/players
- has_process: true when section describes a pipeline/workflow/architecture
- has_radar: true when section compares 3+ players across multiple capability dimensions
- has_arch: true when section describes a layered hardware/software/system stack
""").strip()

_EXTRACT_USER_TEMPLATE = "Topic: {topic}\n\nExtra context: {extra}\n\nProduce the structured outline now."


# ── Stage 2: Plan generation prompt ───────────────────────────────────────────

_PLAN_SYSTEM = textwrap.dedent("""
You are an expert presentation architect. Given a structured topic outline,
produce a complete PowerPoint deck plan as a single valid JSON object.

Return ONLY the JSON — no markdown fences, no commentary.

## Output schema

{
  "title": "string",
  "subtitle": "string (optional)",
  "aspect_ratio": "16:9",
  "theme": {
    "accent_rgb": [R, G, B],
    "body_rgb": [51, 65, 85],
    "title_rgb": [15, 23, 42],
    "author": "string"
  },
  "slides": [ <slide_object>, ... ]
}

Each slide_object:
{
  "slide_number": 1,
  "type": "title" | "section" | "content" | "summary",
  "title": "string (≤40 chars)",

  // For type:title only:
  "subtitle": "string (≤60 chars)",

  // Content — choose ONE of: cards | modules | bullets
  // cards: use when slide has 2-6 parallel independent topics
  "cards": [
    {
      "heading": "string (≤20 chars)",
      "icon": "single emoji (optional)",
      "bullets": ["string (≤35 chars each)", ...]   // 2-4 bullets per card
    }
  ],
  "cards_cols": 0,   // 0=auto, or 2/3

  // modules: use when slide has 2-3 distinct named sub-topics
  "modules": [
    { "heading": "string (≤25 chars)", "bullets": ["string (≤50 chars)", ...] }
  ],

  // bullets: use for simple single-topic slides
  "bullets": ["string (≤60 chars)", ...],   // 3-6 bullets

  // Optional visualization
  "figure": {
    "position": "right" | "bottom",    // right=flowchart/comparison, bottom=timeline/pipeline
    "caption": "string (≤50 chars, optional)",
    "viz": { ... }                     // see Viz spec
  },

  "notes": "string (≤150 chars)"
}

## CHARACTER LIMITS — CRITICAL

These limits prevent text overflow in the final slides. You MUST respect them:
- Slide title: ≤40 characters
- Card heading: ≤20 characters
- Card bullet: ≤35 characters (2-4 bullets per card)
- Module heading: ≤25 characters
- Module bullet: ≤50 characters
- Plain bullet: ≤60 characters (3-6 bullets)
- Speaker notes: ≤150 characters

If a point cannot fit within the limit, split it into two bullets or abbreviate.

## Viz spec

### timeline
{
  "type": "timeline",
  "title": "string",
  "stages": [
    { "label": "≤20 chars", "year": "string", "annotation": "≤15 chars", "detail": "≤40 chars" }
  ],
  "highlight": [0-based index]
}
Use for: chronological evolution, roadmaps. Position: "bottom".

### flowchart
{
  "type": "flowchart",
  "layout": "LR" | "TB",
  "nodes": [ { "id": "n1", "label": "≤20 chars", "shape": "rect"|"diamond"|"rounded", "color": "#hex" } ],
  "edges": [ { "from": "n1", "to": "n2", "label": "≤12 chars" } ]
}
Use for: decision logic, system architecture. Position: "right".

### comparison
{
  "type": "comparison",
  "highlight_col": 0,
  "rows": ["≤20 chars", ...],
  "cols": ["≤15 chars", ...],
  "cells": [["≤18 chars", ...], ...],
  "row_notes": ["≤30 chars", ...]
}
Use for: side-by-side capability matrix. Position: "right".

### pipeline
{
  "type": "pipeline",
  "stages": [
    { "label": "≤15 chars", "sublabel": "≤25 chars", "color": "#hex", "badge": "≤8 chars" }
  ]
}
Use for: system module chains, processing steps. Position: "bottom".
Do NOT include arrow_label.

### radar  (Bosch specialty — competitor capability comparison)
{
  "type": "radar",
  "title": "string",
  "dimensions": ["≤15 chars each", ...],   // 4-7 capability axes
  "players": [
    {
      "name": "≤12 chars",
      "scores": [score_per_dimension],      // numbers in score_range
      "color": "#hex",
      "highlight": true                     // true for OUR player (Bosch/client)
    }
  ],
  "score_range": [0, 10],
  // Optional: use dimension_ranges instead of score_range when each axis has a different unit
  // "dimension_ranges": [[0,180],[0,100],[0,500],[0,10],[0,50]]  // [min,max] per dimension
}
Use for: comparing 3-5 competitors across 4-7 capability dimensions.
When dimensions have different natural units (FOV°, Latency ms, Range m, Power W …),
include "dimension_ranges" to normalise each axis independently; omit score_range in that case.
Position: "right".

### arch  (Bosch specialty — layered system architecture)
{
  "type": "arch",
  "title": "string",
  "direction": "BT",                        // BT=sensor-to-app (bottom=sensor), TB=top-to-app
  "layers": [
    {
      "name": "≤20 chars",
      "color": "#hex (light pastel)",
      "blocks": [
        { "label": "≤15 chars", "sublabel": "≤20 chars", "badge": "≤8 chars" }
      ]
    }
  ]
}
Use for: ECU/SoC/software stack, sensor-fusion architecture, AUTOSAR layering.
Position: "right" (full-height).

## Layout selection rules

- Use "cards" when: 3-6 parallel topics that are best compared side-by-side
- Use "modules" when: 2-3 named sub-topics, each with a few bullets
- Use "bullets" when: single continuous topic with simple list
- Add figure.viz when the section outline flags has_timeline/has_comparison/has_process/has_radar/has_arch
- Use "radar" viz when has_radar=true and there are ≥3 identifiable competitors
- Use "arch" viz when has_arch=true and the architecture has ≥2 identifiable layers
- Structure: title slide → section dividers → content slides → summary

## Bosch / ADAS domain guidance

When the topic is automotive, ADAS, autonomous driving, or Bosch products:
- Use accent_rgb: [226, 0, 21] (Bosch Red) for the theme
- Prefer "arch" viz for: sensor fusion stack, AUTOSAR layers, SoC/ECU architecture,
  functional safety decomposition
- Prefer "radar" viz for: competitor benchmarking (Mobileye, Waymo, Huawei, Continental,
  Nvidia, Qualcomm, Bosch), capability assessment, technology readiness comparison
- Use "pipeline" for: perception→prediction→planning→control chain, data processing flow
- Use "timeline" for: L2→L2+→L3→L4 evolution, regulation milestones (UN R79/R157, ISO 26262)
- Use "comparison" for: ASIL level matrix, regulation comparison (EU/US/China),
  sensor modality tradeoffs (camera/radar/lidar/ultrasonic)
- Key ADAS terms to use accurately: ODD, TOR, MRC, ASIL, ISO 26262, SOTIF ISO 21448,
  NOA, HWP, AEB, TJA, LKA, ACC, V2X, NCAP, FuSa, cybersecurity (UN R155/R156)
- For Chinese market: include NIO, Xpeng, Huawei ADS, Li Auto, BYD as relevant competitors

## Color guidance

- Bosch / automotive safety: [226, 0, 21] (Bosch Red) — use as primary accent
- Automotive / engineering (generic): [26, 86, 219] or [15, 40, 100]
- Technology / AI: [79, 70, 229] or [13, 148, 136]
- Finance / data: [5, 150, 105] or [71, 85, 105]
- Safety / medical: [30, 64, 175] or [109, 40, 217]
""").strip()

_PLAN_USER_TEMPLATE = textwrap.dedent("""
Topic outline:
{outline}

Additional constraints:
- Output language: {lang_instruction}
- Target slides: {slides}
- Extra instructions: {extra}

Generate the complete deck plan JSON now.
""").strip()

_LANG_INSTRUCTIONS = {
    "zh": "Chinese (Simplified) — all slide text, titles, bullets, and notes must be in Chinese",
    "en": "English — all slide text must be in English",
}


# ── Backend adapters ───────────────────────────────────────────────────────────

def _call_anthropic(system: str, prompt: str, model: str) -> str:
    import anthropic
    api_key   = os.environ.get("ANTHROPIC_API_KEY") or ""
    auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN") or ""
    base_url  = os.environ.get("ANTHROPIC_BASE_URL")
    kwargs: Dict[str, Any] = {}
    if api_key:
        kwargs["api_key"] = api_key
    elif auth_token:
        kwargs["auth_token"] = auth_token
    if base_url:
        kwargs["base_url"] = base_url
    client = anthropic.Anthropic(**kwargs)
    msg = client.messages.create(
        model=model or "claude-sonnet-4-6",
        max_tokens=8192,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    # Skip ThinkingBlocks (extended thinking) and find the first TextBlock
    for block in msg.content:
        if block.type == "text":
            return block.text
    raise ValueError(f"No text block in Anthropic response: {msg.content}")


def _call_openai(system: str, prompt: str, model: str,
                 base_url: Optional[str] = None,
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
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content


def _call_ollama(system: str, prompt: str, model: str, host: str) -> str:
    import urllib.request
    url = f"{host.rstrip('/')}/api/chat"
    payload = json.dumps({
        "model": model or "llama3",
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }).encode()
    req = urllib.request.Request(url, data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read())
    return data["message"]["content"]


def _detect_backend() -> str:
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("CUSTOM_LLM_BASE_URL"):
        return "custom"
    return "ollama"


def _call_llm(system: str, prompt: str, backend: str, model: str) -> str:
    if backend == "anthropic":
        return _call_anthropic(system, prompt, model)
    if backend == "openai":
        return _call_openai(system, prompt, model)
    if backend == "custom":
        return _call_openai(
            system, prompt,
            model or os.environ.get("CUSTOM_LLM_MODEL", "gpt-5.4"),
            base_url=os.environ["CUSTOM_LLM_BASE_URL"],
            api_key=os.environ.get("CUSTOM_LLM_API_KEY"),
        )
    if backend == "ollama":
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        mdl = model or os.environ.get("OLLAMA_MODEL", "llama3")
        return _call_ollama(system, prompt, mdl, host)
    raise ValueError(f"Unknown backend: {backend!r}")


# ── JSON extraction ────────────────────────────────────────────────────────────

def _extract_json(raw: str) -> Dict[str, Any]:
    """Robustly extract first valid JSON object from LLM response."""
    text = raw.strip()

    # Strip markdown fences (``` json, ```json, ~~~, etc.)
    text = re.sub(r"^```+\w*\s*", "", text)
    text = re.sub(r"\s*```+$", "", text)
    text = re.sub(r"^~~~+\w*\s*", "", text)
    text = re.sub(r"\s*~~~+$", "", text)
    text = text.strip()

    # Support both JSON objects ({...}) and arrays ([...])
    obj_start = text.find("{")
    arr_start = text.find("[")
    if arr_start != -1 and (obj_start == -1 or arr_start < obj_start):
        start = arr_start
        end_char = "]"
    elif obj_start != -1:
        start = obj_start
        end_char = "}"
    else:
        raise ValueError("No JSON object or array found in LLM response")

    # Walk chars tracking depth, but skip content inside strings
    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ("{", "["):
            depth += 1
        elif ch in ("}", "]"):
            depth -= 1
            if depth == 0 and ch == end_char:
                return json.loads(text[start: i + 1])

    raise ValueError("Incomplete JSON object in LLM response")


# ── Stage 1: key-point extraction ─────────────────────────────────────────────

def _extract_key_points(topic: str, extra: str,
                        backend: str, model: str) -> Dict[str, Any]:
    """Stage 1: distil topic into structured key-point outline."""
    user = _EXTRACT_USER_TEMPLATE.format(topic=topic, extra=extra or "None")
    print("[llm_planner] Stage 1: extracting key points…", file=sys.stderr)
    raw = _call_llm(_EXTRACT_SYSTEM, user, backend, model)
    outline = _extract_json(raw)
    n_sections = len(outline.get("sections", []))
    print(f"[llm_planner] Stage 1 done: {n_sections} sections", file=sys.stderr)
    return outline


def _outline_to_text(outline: Dict[str, Any]) -> str:
    """Serialise the key-point outline as readable text for Stage 2."""
    lines: List[str] = []
    lines.append(f"Summary: {outline.get('topic_summary', '')}")
    lines.append("")
    for sec in outline.get("sections", []):
        flags = []
        if sec.get("has_timeline"):
            flags.append("HAS_TIMELINE")
        if sec.get("has_comparison"):
            flags.append("HAS_COMPARISON")
        if sec.get("has_process"):
            flags.append("HAS_PROCESS")
        flag_str = f"  [{', '.join(flags)}]" if flags else ""
        lines.append(f"## {sec['title']}{flag_str}")
        for kp in sec.get("key_points", []):
            lines.append(f"  - {kp}")
        lines.append("")
    return "\n".join(lines)


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_plan(
    topic: str,
    slides: int = 10,
    extra: str = "",
    backend: Optional[str] = None,
    model: str = "",
    lang: str = "en",
    two_stage: bool = True,
) -> Dict[str, Any]:
    """
    Generate a deck plan dict for the given topic using an LLM.

    Args:
        topic:     Deck topic (free-form string)
        slides:    Target number of slides
        extra:     Extra instructions appended to the user prompt
        backend:   "anthropic" | "openai" | "custom" | "ollama" | None (auto-detect)
        model:     Model name override (optional)
        lang:      Output language: "en" (default) or "zh"
        two_stage: If True (default), run key-point extraction first (Stage 1)
                   before plan generation (Stage 2). Better quality, one extra
                   LLM call. Set False to skip Stage 1 (faster, single call).

    Returns:
        Parsed plan dict ready for build_deck.build_deck()
    """
    if backend is None:
        backend = _detect_backend()

    lang_instruction = _LANG_INSTRUCTIONS.get(lang, _LANG_INSTRUCTIONS["en"])
    print(f"[llm_planner] Backend: {backend}  Topic: {topic[:60]}", file=sys.stderr)

    if two_stage:
        # Stage 1: extract structured key points
        outline = _extract_key_points(topic, extra, backend, model)
        outline_text = _outline_to_text(outline)
    else:
        # Single-stage fallback: use topic directly as outline
        outline_text = f"Topic: {topic}\n\nExtra: {extra or 'None'}"

    # Stage 2: generate full plan from outline
    user = _PLAN_USER_TEMPLATE.format(
        outline=outline_text,
        lang_instruction=lang_instruction,
        slides=slides,
        extra=extra or "None",
    )
    print("[llm_planner] Stage 2: generating deck plan…", file=sys.stderr)
    raw = _call_llm(_PLAN_SYSTEM, user, backend, model)
    plan = _extract_json(raw)
    print(f"[llm_planner] Plan generated: {len(plan.get('slides', []))} slides",
          file=sys.stderr)
    return plan


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a deck plan JSON from a topic using an LLM"
    )
    parser.add_argument("--topic", required=True,
                        help="Deck topic (free-form string)")
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
    parser.add_argument("--lang", default="en", choices=["en", "zh"],
                        help="Output language: en (default) or zh")
    parser.add_argument("--no-two-stage", action="store_true",
                        help="Skip Stage 1 key-point extraction (faster, lower quality)")
    args = parser.parse_args()

    plan = generate_plan(
        args.topic, args.slides, args.extra,
        args.backend, args.model,
        lang=args.lang,
        two_stage=not args.no_two_stage,
    )
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
