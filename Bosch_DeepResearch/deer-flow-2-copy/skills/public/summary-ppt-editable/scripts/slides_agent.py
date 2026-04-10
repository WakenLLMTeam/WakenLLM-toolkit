#!/usr/bin/env python3
"""
slides_agent.py — Per-slide AI presentation generator.

Unlike agent_ppt.py (which generates the full deck plan in one LLM shot,
then batch-renders all viz), this agent decides viz type PER SLIDE:
  Stage 1: LLM generates a slide outline with goals (no viz specs)
  Stage 2: For each slide, LLM decides whether to add a viz and which type;
           renderer is called immediately; slide is accumulated
  Final:   Single call to build_pptx for assembly

Usage:
  # Full pipeline
  python slides_agent.py --topic "L2 to L3 autonomous driving evolution" \
      --output deck.pptx --slides 8 --theme bosch

  # Decisions only (inspect per-slide viz choices without rendering)
  python slides_agent.py --topic "Bosch ADAS competitive analysis" \
      --slides 6 --decisions-only --output decisions.json

  # Use existing outline JSON
  python slides_agent.py --outline '[{"slide_number":1,"type":"title",...}]' \
      --output deck.pptx

Environment variables (LLM backend — same as llm_planner.py):
  ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN  → Claude
  OPENAI_API_KEY                              → OpenAI
  CUSTOM_LLM_BASE_URL + CUSTOM_LLM_API_KEY    → custom OpenAI-compatible
  OLLAMA_HOST / OLLAMA_MODEL                  → Ollama local
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow importing sibling scripts without installing as a package
_SCRIPTS_DIR = Path(__file__).parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import llm_planner
import build_deck as _bd
import build_pptx as _bpx
import viz_theme

_RENDERERS: Dict[str, Any] = _bd._RENDERERS


# ── Stage 1: slide-list prompt ─────────────────────────────────────────────────

_SLIDE_LIST_SYSTEM = textwrap.dedent("""
You are an expert presentation architect. Given a structured topic outline,
produce ONLY a JSON array of slide outlines — no markdown fences, no commentary.

Schema (one element per slide):
{
  "slide_number": 1,
  "type": "title" | "section" | "content" | "summary",
  "title": "string (≤40 chars)",
  "subtitle": "string (≤60 chars, title slide only)",
  "bullets": ["string (≤60 chars)", ...],  // for content/summary slides, 2-5 items
  "goal": "string   // 1-sentence goal describing the informational purpose of this slide
}

Rules:
- title slide → type:"title", no bullets
- section dividers → type:"section", no bullets, 1-2 words title
- content slides → type:"content", 2-5 bullets, each ≤60 chars
- summary/closing → type:"summary"
- DO NOT include viz specs, figure objects, or image_path — those are added later
- All text must be in the specified output language
- Total slide count must match the target specified by the user
""").strip()

_SLIDE_LIST_USER_TEMPLATE = """\
Topic outline:
{outline}

Output language: {lang_instruction}
Target slide count: {slides}

Generate the slide list JSON array now."""


# ── Stage 2: per-slide viz decision prompt ────────────────────────────────────

_SLIDE_VIZ_DECISION_SYSTEM = textwrap.dedent("""
You are an expert in data visualization and presentation design.

Given a single slide outline, decide whether a visualization would genuinely help
the audience understand the content. Add a viz ONLY when it meaningfully enhances
comprehension — prefer text-only slides when the content is simple or conceptual.

AVAILABLE VIZ TYPES:

1.  timeline    — Horizontal stage timeline: chronological evolution, roadmaps, phase transitions. Schema: {"stages": [{"label": "Stage1", "year": "2020", "annotation": "Note", "detail": "Description"}]}
2.  flowchart   — Directed node-arrow graph: decision logic, system flows, state machines. Schema: {"nodes": [{"id": "n1", "label": "Node", "shape": "rect", "color": "#dbeafe"}], "edges": [{"from": "n1", "to": "n2"}]}
3.  comparison  — Feature matrix table: side-by-side capability comparison, pros/cons
4.  pipeline    — Horizontal block diagram: processing chains, module architectures. Schema: {"stages": [{"label": "Step1", "sublabel": "Details", "color": "#dbeafe"}]}
5.  arch        — Layered horizontal bands: software/hardware stack, AUTOSAR, sensor-fusion layers. Schema: {"layers": [{"name": "App", "color": "#dbeafe", "blocks": [{"label": "Planner", "sublabel": "Behavior", "badge": "L3"}]}]}
6.  bar_chart   — Grouped or stacked bars: market share, survey results, quantities over time
7.  line_chart  — Multi-series line chart: trends, time series, scaling laws
8.  scatter     — Scatter or bubble chart: maturity vs. impact, correlation analysis
9.  heatmap     — Color-encoded matrix: coverage grids, performance by category
10. waterfall  — Incremental bridge chart: cost/value bridges, variance analysis
11. funnel     — Top-to-bottom conversion funnel: funnel/conversion metrics
12. radar      — Multi-axis radar/spider chart: competitor benchmarking across dimensions
13. mindmap    — Radial mind map: concept breakdown, topic branches. Schema: {"center": "Topic", "branches": [{"label": "Branch", "color": "#dbeafe", "children": [{"label": "Child1"}, {"label": "Child2"}]}]}
14. tree       — Hierarchical tree: org charts, product taxonomies, decision trees. Schema: {"root": {"label": "Root", "color": "#dbeafe", "children": [{"label": "Child", "children": []}]}}
15. matrix_2x2 — 2x2 quadrant: prioritization, technology positioning
16. venn       — 2- or 3-circle Venn: capability overlap, set intersection
17. onion      — Concentric rings: layered systems, maturity stages
18. gantt      — Horizontal Gantt: project timelines, roadmap with milestones
19. swot       — 4-quadrant SWOT: competitive analysis, strategic positioning

DECISION RULES:
- timeline: chronological evolution, phases, roadmaps, milestones
- comparison: 2-4 items compared across 3+ features (capability matrix, pros/cons)
- flowchart: process, decision tree, state machine, conditional logic
- pipeline: sequential processing chain, module architecture
- arch: layered system (software stack, hardware, AUTOSAR)
- bar_chart / line_chart: numerical data, quantities, trends over time
- scatter / heatmap: 2D data grids, correlation, coverage
- waterfall: incremental bridge (cost, revenue, variance)
- funnel: conversion/adoption funnel
- swot: competitive/strategic 4-quadrant analysis
- mindmap / tree: hierarchical/taxonomic breakdown
- gantt: project timeline with start/end/milestones
- matrix_2x2: 2-axis prioritization or positioning
- venn: set intersection / overlap analysis
- onion: layered decomposition
- radar: 3+ players compared across multiple dimensions

Return a single JSON object:

If a viz is needed:
{
  "slide_number": <N>,
  "needs_viz": true,
  "figure": {
    "position": "right" | "bottom",
    "caption": "string (≤50 chars, optional)",
    "viz": { <complete viz spec> }
  }
}

If no viz is needed:
{
  "slide_number": <N>,
  "needs_viz": false
}

VIZ SPEC CONSTRAINTS:
- title: ≤40 chars
- arch layers: must have name + color + blocks (not "label"/"nodes")
- mindmap branches: must have label + color + children list
- flowchart nodes: must have id + label + shape + color
- timeline stages: must have label + year
- pipeline stages: must have label + sublabel + color
- comparison: must have rows + cols + cells (all non-empty)
- Be concise — abbreviate if needed
- Fill in realistic sample data relevant to the topic
""").strip()

_SLIDE_VIZ_DECISION_USER_TEMPLATE = """\
Slide to decide on:
  Slide number: {slide_number}
  Type: {slide_type}
  Title: {title}
  Content: {content}
  Deck title: {deck_title}
  Theme: {theme_instruction}

Viz diversity tracker (already used in this deck):
{diversity_hint}

DIVERSITY RULE: Strongly prefer viz types NOT in the "already used" list above.
Types marked [EXHAUSTED] must NOT be used — pick a different type.
Aim to introduce a new viz type on every slide.

Output a single JSON decision object now."""


# ── Theme config builder ───────────────────────────────────────────────────────

def _build_theme_config(theme: str) -> Dict[str, Any]:
    if theme == "bosch":
        return {
            "accent_rgb": [226, 0, 21],
            "body_rgb": [51, 65, 85],
            "title_rgb": [15, 23, 42],
        }
    return {
        "accent_rgb": [37, 99, 235],
        "body_rgb": [51, 65, 85],
        "title_rgb": [15, 23, 42],
    }


# ── Stage 1: deck outline ──────────────────────────────────────────────────────

# ── Text normalization: shorten text to fit PPTX character limits ─────────────

_TEXT_NORM_SYSTEM = textwrap.dedent("""
You are a text compression assistant. Given a list of slides with text that may exceed
PowerPoint character limits, rewrite each text field to be more concise while
preserving the essential meaning. Return the same JSON array with shortened fields.

CHARACTER LIMITS (strict — do not exceed):
- slide title (any type): ≤40 characters
- subtitle (title slide only): ≤60 characters
- section title: ≤40 characters
- bullet point: ≤60 characters
- card heading: ≤20 characters
- card bullet: ≤35 characters
- module heading: ≤25 characters
- module bullet: ≤50 characters
- figure caption: ≤60 characters
- notes: ≤150 characters

Rules:
- Preserve the semantic meaning and key information
- Remove filler words, articles, and redundant phrases
- Use abbreviations where standard (e.g. "Autonomous Driving" → "AD", "adas" → "ADAS")
- Do NOT add ellipsis ("...") — if text fits within the limit, return it as-is
- Return the SAME JSON structure with the same keys, only modifying text values
- Output language must match the original text language
""").strip()

_TEXT_NORM_USER_TEMPLATE = """\
Slide list to compress:
{slides_json}

Rewrite any text exceeding the character limits above. Return the full slide list JSON."""


def _normalize_text_lengths(
    slide_list: List[Dict[str, Any]],
    backend: str,
    model: str,
) -> List[Dict[str, Any]]:
    """Use an LLM to shorten text fields that exceed PPTX character limits."""
    import copy
    # Quick check: if no text exceeds limits, skip LLM call
    limits = {
        "title": 40, "subtitle": 60, "bullets": 60,
        "heading": 20, "notes": 150,
    }
    needs_llm = False
    for slide in slide_list:
        stype = slide.get("type", "")
        for field, limit in limits.items():
            val = slide.get(field)
            if isinstance(val, str) and len(val) > limit:
                needs_llm = True
                break
        for b in slide.get("bullets", []):
            if len(b) > 60:
                needs_llm = True
        for c in slide.get("cards", []):
            if len(c.get("heading", "")) > 20:
                needs_llm = True
            for b in c.get("bullets", []):
                if len(b) > 35:
                    needs_llm = True
        for m in slide.get("modules", []):
            if len(m.get("heading", "")) > 25:
                needs_llm = True
        fig = slide.get("figure", {})
        cap = fig.get("caption", "")
        if isinstance(cap, str) and len(cap) > 60:
            needs_llm = True

    if not needs_llm:
        return slide_list

    # One LLM call to normalize all slides
    slides_json = json.dumps(slide_list, ensure_ascii=False, indent=2)
    user = _TEXT_NORM_USER_TEMPLATE.format(slides_json=slides_json)
    try:
        raw = llm_planner._call_llm(_TEXT_NORM_SYSTEM, user, backend, model)
        result = llm_planner._extract_json(raw)
        if isinstance(result, list) and len(result) == len(slide_list):
            print(f"[slides_agent] Text normalization: shortened {sum(1 for a,b in zip(slide_list,result) if a!=b)} slides", file=sys.stderr)
            return result
    except Exception as exc:
        print(f"[slides_agent] WARNING text normalization failed: {exc}", file=sys.stderr)
    return slide_list


def _generate_slide_list(
    topic: str,
    slides: int,
    extra: str,
    backend: str,
    model: str,
    lang: str,
    two_stage: bool,
) -> List[Dict[str, Any]]:
    """Produce a list of slide outlines (no viz specs)."""
    lang_instruction = llm_planner._LANG_INSTRUCTIONS.get(
        lang, llm_planner._LANG_INSTRUCTIONS["en"]
    )

    if two_stage:
        outline = llm_planner._extract_key_points(topic, extra, backend, model)
        outline_text = llm_planner._outline_to_text(outline)
    else:
        outline_text = f"Topic: {topic}\n\nExtra: {extra or 'None'}"

    user = _SLIDE_LIST_USER_TEMPLATE.format(
        outline=outline_text,
        lang_instruction=lang_instruction,
        slides=slides,
    )
    raw = llm_planner._call_llm(_SLIDE_LIST_SYSTEM, user, backend, model)
    slide_list = llm_planner._extract_json(raw)

    if isinstance(slide_list, dict) and "slides" in slide_list:
        # LLM wrapped the array in a "slides" key
        slide_list = slide_list["slides"]
    if not isinstance(slide_list, list):
        raise ValueError(f"Expected JSON array from slide-list LLM, got {type(slide_list)}")
    return slide_list


# ── Stage 2: per-slide viz decision ──────────────────────────────────────────

def _validate_viz_spec(viz: Dict[str, Any]) -> Optional[str]:
    """Check if a viz spec has the minimum required fields for rendering.
    Returns an error string if invalid, or None if valid."""
    vt = (viz.get("type") or "").lower()
    if not vt or vt == "none":
        return "viz type is empty or missing"
    if vt not in _RENDERERS:
        return f"unknown viz type: {vt!r}"
    if vt == "timeline":
        if not viz.get("stages") or not isinstance(viz["stages"], list) or len(viz["stages"]) == 0:
            return "timeline requires a non-empty 'stages' list"
    elif vt == "flowchart":
        if not viz.get("nodes") or not isinstance(viz["nodes"], list) or len(viz["nodes"]) == 0:
            return "flowchart requires a non-empty 'nodes' list"
        if not viz.get("edges") or not isinstance(viz["edges"], list) or len(viz["edges"]) == 0:
            return "flowchart requires a non-empty 'edges' list"
    elif vt == "comparison":
        if not viz.get("rows") or not isinstance(viz["rows"], list) or len(viz["rows"]) == 0:
            return "comparison requires non-empty 'rows'"
        if not viz.get("cols") or not isinstance(viz["cols"], list) or len(viz["cols"]) == 0:
            return "comparison requires non-empty 'cols'"
        if not viz.get("cells") or not isinstance(viz["cells"], list) or len(viz["cells"]) == 0:
            return "comparison requires non-empty 'cells'"
    elif vt == "pipeline":
        if not viz.get("stages") or not isinstance(viz["stages"], list) or len(viz["stages"]) == 0:
            return "pipeline requires a non-empty 'stages' list"
    elif vt == "arch":
        if not viz.get("layers") or not isinstance(viz["layers"], list) or len(viz["layers"]) == 0:
            return "arch requires a non-empty 'layers' list"
    elif vt == "bar_chart":
        if not viz.get("categories") or len(viz["categories"]) == 0:
            return "bar_chart requires non-empty 'categories'"
        if not viz.get("series") or not isinstance(viz["series"], list) or len(viz["series"]) == 0:
            return "bar_chart requires at least one series"
    elif vt == "line_chart":
        if not viz.get("series") or not isinstance(viz["series"], list) or len(viz["series"]) == 0:
            return "line_chart requires at least one series"
    elif vt == "scatter":
        if not viz.get("series") or not isinstance(viz["series"], list) or len(viz["series"]) == 0:
            return "scatter requires at least one series"
    elif vt == "heatmap":
        if not viz.get("rows") or len(viz["rows"]) == 0:
            return "heatmap requires non-empty 'rows'"
        if not viz.get("cols") or len(viz["cols"]) == 0:
            return "heatmap requires non-empty 'cols'"
        if not viz.get("values"):
            return "heatmap requires 'values'"
    elif vt == "waterfall":
        if not viz.get("items") or not isinstance(viz["items"], list) or len(viz["items"]) == 0:
            return "waterfall requires a non-empty 'items' list"
    elif vt == "funnel":
        if not viz.get("stages") or not isinstance(viz["stages"], list) or len(viz["stages"]) == 0:
            return "funnel requires a non-empty 'stages' list"
    elif vt == "radar":
        if not viz.get("dimensions") or len(viz["dimensions"]) < 3:
            return "radar requires at least 3 'dimensions'"
        if not viz.get("players") or not isinstance(viz["players"], list) or len(viz["players"]) == 0:
            return "radar requires at least one player"
    elif vt == "mindmap":
        if not viz.get("branches") or not isinstance(viz["branches"], list) or len(viz["branches"]) == 0:
            return "mindmap requires a non-empty 'branches' list"
    elif vt == "tree":
        if not viz.get("root"):
            return "tree requires a 'root' object"
    elif vt == "matrix_2x2":
        if not viz.get("items") or not isinstance(viz["items"], list) or len(viz["items"]) == 0:
            return "matrix_2x2 requires at least one item"
    elif vt == "venn":
        if not viz.get("circles") or not isinstance(viz["circles"], list) or len(viz["circles"]) < 2:
            return "venn requires at least 2 circles"
    elif vt == "onion":
        if not viz.get("layers") or not isinstance(viz["layers"], list) or len(viz["layers"]) == 0:
            return "onion requires a non-empty 'layers' list"
    elif vt == "gantt":
        if not viz.get("tasks") or not isinstance(viz["tasks"], list) or len(viz["tasks"]) == 0:
            return "gantt requires a non-empty 'tasks' list"
    elif vt == "swot":
        qs = viz.get("quadrants") or {}
        if not any(qs.get(k) for k in ("strengths", "weaknesses", "opportunities", "threats")):
            return "swot requires at least one quadrant"
    return None


def _normalize_viz_spec(viz: Dict[str, Any]) -> Dict[str, Any]:
    """
    Auto-fix common schema mismatches in viz specs generated by LLM.
    LLM often uses wrong field names (e.g. 'label'+'nodes' instead of 'name'+'blocks' for arch).
    """
    import copy
    viz = copy.deepcopy(viz)
    vt = (viz.get("type") or "").lower()

    if vt == "arch":
        layers = viz.get("layers") or []
        fixed_layers = []
        for layer in layers:
            layer = dict(layer)
            # 'label' → 'name'
            if "label" in layer and "name" not in layer:
                layer["name"] = layer.pop("label")
            # 'nodes' (list of strings) → 'blocks' (list of {label})
            if "nodes" in layer and "blocks" not in layer:
                nodes = layer.pop("nodes")
                layer["blocks"] = [{"label": n} for n in nodes] if isinstance(nodes, list) else []
            # ensure blocks exists
            if "blocks" not in layer:
                layer["blocks"] = []
            # ensure color
            if "color" not in layer:
                layer["color"] = "#dbeafe"
            fixed_layers.append(layer)
        viz["layers"] = fixed_layers

    elif vt == "mindmap":
        branches = viz.get("branches") or []
        for branch in branches:
            # 'children' might be missing or be strings instead of dicts
            if "children" in branch:
                children = branch["children"]
                if children and isinstance(children[0], str):
                    branch["children"] = [{"label": c} for c in children]
            if "color" not in branch:
                branch["color"] = "#dbeafe"

    elif vt == "tree":
        def fix_node(node):
            node = dict(node)
            if "children" in node:
                children = node["children"]
                if children and isinstance(children[0], str):
                    node["children"] = [{"label": c} for c in children]
                else:
                    node["children"] = [fix_node(c) for c in children]
            if "color" not in node:
                node["color"] = "#dbeafe"
            return node
        if viz.get("root"):
            viz["root"] = fix_node(viz["root"])

    elif vt == "flowchart":
        nodes = viz.get("nodes") or []
        for node in nodes:
            if "label" in node and isinstance(node["label"], list):
                node["label"] = "\n".join(node["label"])
            if "color" not in node:
                node["color"] = "#dbeafe"

    elif vt == "timeline":
        stages = viz.get("stages") or []
        for stage in stages:
            if "detail" not in stage and "annotation" in stage:
                stage["detail"] = stage["annotation"]

    return viz


_RETRY_SYSTEM = textwrap.dedent("""
You are a data visualization spec validator. A previous attempt to generate a viz spec had the following error:

  {error}

The original slide context was:
  Title: {title}
  Type: {slide_type}
  Bullets: {bullets}
  Deck title: {deck_title}

Your task: produce a COMPLETE and VALID viz spec that fixes this error.

Requirements:
- Fill in ALL required fields with realistic, non-empty data
- For comparison: provide rows (≥2), cols (≥2), and cells matching rows×cols
- For pipeline: provide stages (≥2) with label and sublabel
- For timeline: provide stages (≥2) with label and year
- For radar: provide dimensions (≥3) and players (≥1) with score arrays
- For arch: each layer needs name + color + blocks (each block: label + optional sublabel + optional badge). Example: {"name": "Application", "color": "#dbeafe", "blocks": [{"label": "Planner", "sublabel": "Behavior"}, {"label": "AEB"}]}
- For mindmap: branches = [{"label": "BranchName", "color": "#dbeafe", "children": [{"label": "Child1"}, {"label": "Child2"}]}]
- For tree: root = {"label": "RootName", "color": "#dbeafe", "children": [{"label": "Child", "children": []}]}
- For flowchart: nodes = [{"id": "n1", "label": "NodeLabel", "shape": "rect", "color": "#dbeafe"}], edges = [{"from": "n1", "to": "n2", "label": "edge label"}]
- All labels must be ≤20 chars unless otherwise specified in the type constraints

Return ONLY a JSON object:
{{
  "slide_number": <N>,
  "needs_viz": true,
  "figure": {{
    "position": "right" | "bottom",
    "caption": "string (≤50 chars, optional)",
    "viz": {{ <complete fixed viz spec> }}
  }}
}}
""").strip()


def _build_diversity_hint(used_viz_types: Dict[str, int], max_per_type: int) -> str:
    """
    Build a diversity hint with soft penalty tiers based on use count.

    Tiers:
      0 uses  → STRONGLY PREFERRED  (fresh)
      1 use   → DEPRIORITIZED       (soft avoid — only pick if clearly best fit)
      2 uses  → STRONGLY AVOID      (heavy penalty — pick fresh type instead)
      3+ uses → DO NOT USE          (hard block)
    """
    ALL_TYPES = [
        "timeline", "flowchart", "comparison", "pipeline", "arch",
        "bar_chart", "line_chart", "scatter", "heatmap", "waterfall",
        "funnel", "radar", "mindmap", "tree", "matrix_2x2",
        "venn", "onion", "gantt", "swot",
    ]
    unused = [t for t in ALL_TYPES if t not in used_viz_types]
    used_once  = [t for t, n in used_viz_types.items() if n == 1]
    used_twice = [t for t, n in used_viz_types.items() if n == 2]
    used_heavy = [t for t, n in used_viz_types.items() if n >= 3]

    lines = []
    if unused:
        lines.append(f"  [STRONGLY PREFERRED — not yet used]: {', '.join(unused)}")
    if used_once:
        lines.append(f"  [DEPRIORITIZED — used once, avoid if a fresh type fits]: {', '.join(used_once)}")
    if used_twice:
        lines.append(f"  [STRONGLY AVOID — used twice, pick a fresh type instead]: {', '.join(used_twice)}")
    if used_heavy:
        lines.append(f"  [DO NOT USE — used 3+ times]: {', '.join(used_heavy)}")
    if not lines:
        lines.append("  (none used yet — all 19 types are available)")
    return "\n".join(lines)


def _decide_slide_viz(
    slide: Dict[str, Any],
    slide_index: int,
    total_slides: int,
    deck_title: str,
    backend: str,
    model: str,
    theme: str,
    used_viz_types: Optional[Dict[str, int]] = None,
    max_per_type: int = 2,
) -> Dict[str, Any]:
    """Ask LLM whether this slide needs a viz and which type. Retries once on invalid spec."""
    theme_instruction = "Bosch Red accent (#E20015)" if theme == "bosch" else "blue accent (#2563eb)"

    content = slide.get("bullets") or []
    content_str = "\n  ".join(f"- {b}" for b in content) if content else "(none)"

    sn = slide.get("slide_number", slide_index + 1)
    diversity_hint = _build_diversity_hint(used_viz_types or {}, max_per_type)

    user = _SLIDE_VIZ_DECISION_USER_TEMPLATE.format(
        slide_number=sn,
        slide_type=slide.get("type", "content"),
        title=slide.get("title", ""),
        content=content_str,
        deck_title=deck_title,
        theme_instruction=theme_instruction,
        diversity_hint=diversity_hint,
    )

    raw = llm_planner._call_llm(_SLIDE_VIZ_DECISION_SYSTEM, user, backend, model)
    decision = llm_planner._extract_json(raw)

    # Normalise: always return a dict with slide_number
    if not isinstance(decision, dict):
        return {"slide_number": sn, "needs_viz": False}
    decision.setdefault("slide_number", sn)

    # Validate spec; retry once with correction if invalid
    fig = decision.get("figure") or {}
    viz = fig.get("viz")
    if viz and isinstance(viz, dict):
        viz = _normalize_viz_spec(viz)  # auto-fix common schema mismatches first
        fig["viz"] = viz
        error = _validate_viz_spec(viz)
        if error:
            print(f"[slides_agent]   spec invalid ({error}), retrying…", file=sys.stderr)
            retry_user = _RETRY_SYSTEM.format(
                error=error,
                title=slide.get("title", ""),
                slide_type=slide.get("type", "content"),
                bullets=content_str,
                deck_title=deck_title,
            )
            try:
                raw_retry = llm_planner._call_llm(_RETRY_SYSTEM, retry_user, backend, model)
                decision = llm_planner._extract_json(raw_retry)
                if isinstance(decision, dict):
                    decision.setdefault("slide_number", sn)
                    # Normalize retry result too
                    fig_retry = decision.get("figure") or {}
                    viz_retry = fig_retry.get("viz")
                    if viz_retry and isinstance(viz_retry, dict):
                        fig_retry["viz"] = _normalize_viz_spec(viz_retry)
            except Exception as exc:
                print(f"[slides_agent]   retry failed: {exc}", file=sys.stderr)

    return decision


# ── Per-slide render ────────────────────────────────────────────────────────────

def _render_slide_viz(
    slide: Dict[str, Any],
    assets_dir: str,
) -> Dict[str, Any]:
    """Call the appropriate renderer for slide.figure.viz. Returns updated slide."""
    fig = slide.get("figure") or {}
    viz = fig.get("viz")
    if not isinstance(viz, dict):
        return slide

    viz_type = (viz.get("type") or "").lower()
    renderer = _RENDERERS.get(viz_type)
    if renderer is None:
        raise ValueError(f"Unknown viz type: {viz_type!r}")

    slide_num = slide.get("slide_number", "?")
    out_png = os.path.join(assets_dir, f"slide_{slide_num}_{viz_type}.png")

    msg = renderer(viz, out_png)
    print(f"[slides_agent]   rendered {viz_type} → {out_png}", file=sys.stderr)
    fig["image_path"] = out_png
    return slide


# ── Main pipeline ──────────────────────────────────────────────────────────────

def build_slides_pptx(
    topic: str,
    output_file: str,
    slides: int = 10,
    extra: str = "",
    backend: Optional[str] = None,
    model: str = "",
    lang: str = "en",
    theme: str = "default",
    assets_dir: Optional[str] = None,
    two_stage: bool = True,
    outline: Optional[List[Dict[str, Any]]] = None,
    max_per_type: int = 2,
) -> str:
    """
    Full pipeline: topic → per-slide outline → per-slide viz decision → render → PPTX.

    Args:
        topic:       Deck topic string
        output_file: Destination .pptx path
        slides:      Target slide count
        extra:       Extra instructions forwarded to the LLM
        backend:     LLM backend override (None = auto-detect)
        model:       Model name override
        lang:        Output language: "en" or "zh"
        theme:       "default" (blue) or "bosch" (red)
        assets_dir:  Directory for generated PNGs (auto-derived if None)
        two_stage:   Run Stage 1 key-point extraction (default True)
        outline:     Pre-defined slide list (skip Stage 1 outline generation)

    Returns:
        Absolute path to the generated .pptx file
    """
    if backend is None:
        backend = llm_planner._detect_backend()

    if assets_dir is None:
        p = Path(output_file)
        assets_dir = str(p.parent / (p.stem + "_assets"))
    Path(assets_dir).mkdir(parents=True, exist_ok=True)

    # ── Stage 1: deck outline ──────────────────────────────────────────────
    if outline is None:
        slide_list = _generate_slide_list(
            topic, slides, extra, backend, model, lang, two_stage
        )
    else:
        slide_list = copy.deepcopy(outline)

    # Ensure slide numbers are sequential
    for i, slide in enumerate(slide_list, 1):
        slide.setdefault("slide_number", i)

    # Normalize text lengths to avoid PPTX truncation with "…"
    slide_list = _normalize_text_lengths(slide_list, backend, model)

    # ── Stage 2: per-slide viz decision loop ───────────────────────────────
    viz_types_used: Dict[str, int] = {}

    for i, slide in enumerate(slide_list):
        sn = slide.get("slide_number", i + 1)
        print(f"[slides_agent] Slide {sn}: deciding viz…", file=sys.stderr)

        # If slide already has a valid viz spec from an outline, use it directly
        existing_fig = slide.get("figure") or {}
        existing_viz = existing_fig.get("viz")
        if existing_viz and isinstance(existing_viz, dict) and _validate_viz_spec(existing_viz) is None:
            # Outline already has a valid spec — skip LLM decision, use as-is
            fig = existing_fig
            viz = existing_viz
            viz_type = viz.get("type", "?")
            print(f"[slides_agent]   outline viz: {viz_type} (skipping decision)", file=sys.stderr)
        else:
            # LLM decides viz — pass current usage so it can diversify
            try:
                decision = _decide_slide_viz(
                    slide, i, len(slide_list),
                    deck_title=topic, backend=backend, model=model,
                    theme=theme,
                    used_viz_types=dict(viz_types_used),
                    max_per_type=max_per_type,
                )
            except Exception as exc:
                print(f"[slides_agent]   WARNING slide {sn} viz decision failed: {exc}", file=sys.stderr)
                decision = {"slide_number": sn, "needs_viz": False}

            # Merge decision into slide
            if decision.get("needs_viz"):
                fig = decision.get("figure") or {}
                slide["figure"] = fig
            else:
                slide.setdefault("figure", {})
            fig = slide.get("figure") or {}
            viz = fig.get("viz")

        # Render viz if present
        fig = slide.get("figure") or {}
        viz = fig.get("viz")
        if viz and isinstance(viz, dict):
            try:
                slide = _render_slide_viz(slide, assets_dir)
                vt = viz.get("type", "?")
                viz_types_used[vt] = viz_types_used.get(vt, 0) + 1
            except Exception as exc:
                import traceback
                print(f"[slides_agent]   WARNING slide {sn} render failed: {exc}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                fig["image_path"] = ""
        else:
            fig["image_path"] = ""

    # ── Final assembly ───────────────────────────────────────────────────────
    plan: Dict[str, Any] = {
        "title": topic,
        "aspect_ratio": "16:9",
        "theme": _build_theme_config(theme),
        "slides": slide_list,
    }

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    result = _bpx.build_pptx(plan, output_file)
    print(f"[slides_agent] {result}", file=sys.stderr)
    print(f"[slides_agent] Viz types used: {viz_types_used}", file=sys.stderr)
    return str(Path(output_file).resolve())


def generate_slides_plan(
    topic: str,
    slides: int = 10,
    extra: str = "",
    backend: Optional[str] = None,
    model: str = "",
    lang: str = "en",
    two_stage: bool = True,
) -> List[Dict[str, Any]]:
    """
    Stage 1 only: produce a slide outline list.
    Useful for inspecting the deck structure before building.
    """
    if backend is None:
        backend = llm_planner._detect_backend()
    return _generate_slide_list(topic, slides, extra, backend, model, lang, two_stage)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Per-slide AI presentation generator: topic → PPTX with per-slide viz decisions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--topic", metavar="TOPIC",
                     help="Free-form topic string (LLM generates the slide outline)")
    src.add_argument("--outline", metavar="JSON",
                     help="JSON array string or @filepath: use existing slide outline")

    parser.add_argument("--output", required=True, metavar="OUT",
                        help="Output .pptx path (or .json for --decisions-only)")
    parser.add_argument("--slides", type=int, default=10,
                        help="Target slide count (default: 10)")
    parser.add_argument("--extra", default="",
                        help="Extra instructions forwarded to the LLM")
    parser.add_argument("--backend", default=None,
                        choices=["anthropic", "openai", "custom", "ollama"],
                        help="LLM backend (default: auto-detect)")
    parser.add_argument("--model", default="",
                        help="Model name override")
    parser.add_argument("--lang", default="en", choices=["en", "zh"],
                        help="Output language: en (default) or zh")
    parser.add_argument("--theme", default="default", choices=["default", "bosch"],
                        help="Color theme: default=blue, bosch=Bosch Red (default: default)")
    parser.add_argument("--assets-dir", default=None, metavar="DIR",
                        help="Directory for generated PNG figures")
    parser.add_argument("--two-stage", dest="two_stage", action="store_true",
                        help="Run Stage 1 key-point extraction (default: True)")
    parser.add_argument("--no-two-stage", dest="two_stage", action="store_false",
                        help="Skip Stage 1 key-point extraction (faster, lower quality)")
    parser.add_argument("--decisions-only", action="store_true",
                        help="Output per-slide decisions JSON only, do not render or assemble")
    parser.set_defaults(two_stage=True)
    args = parser.parse_args()

    # Resolve outline
    if args.outline:
        if args.outline.startswith("@"):
            outline_path = args.outline[1:]
            with open(outline_path, encoding="utf-8") as f:
                outline = json.load(f)
        elif args.outline.startswith("["):
            outline = json.loads(args.outline)
        else:
            with open(args.outline, encoding="utf-8") as f:
                outline = json.load(f)
    else:
        outline = None

    # Decisions-only: Stage 1 outline + Stage 2 decisions, no rendering
    if args.decisions_only:
        slide_list = _generate_slide_list(
            args.topic, args.slides, args.extra,
            args.backend or llm_planner._detect_backend(),
            args.model, args.lang,
            args.two_stage,
        )
        for i, slide in enumerate(slide_list, 1):
            slide.setdefault("slide_number", i)
        decisions = []
        backend = args.backend or llm_planner._detect_backend()
        for i, slide in enumerate(slide_list):
            try:
                d = _decide_slide_viz(
                    slide, i, len(slide_list),
                    deck_title=args.topic,
                    backend=backend,
                    model=args.model,
                    theme=args.theme,
                )
            except Exception as exc:
                print(f"[slides_agent] WARNING slide {i+1} decision failed: {exc}", file=sys.stderr)
                d = {"slide_number": slide.get("slide_number", i+1), "needs_viz": False}
            decisions.append(d)
        out = json.dumps({"slides": slide_list, "decisions": decisions}, ensure_ascii=False, indent=2)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(out, encoding="utf-8")
        print(f"Decisions written to {args.output}")
        return 0

    # Full pipeline
    out_path = build_slides_pptx(
        topic=args.topic or "",
        output_file=args.output,
        slides=args.slides,
        extra=args.extra,
        backend=args.backend,
        model=args.model,
        lang=args.lang,
        theme=args.theme,
        assets_dir=args.assets_dir,
        two_stage=args.two_stage,
        outline=outline,
    )
    print(f"PPTX ready: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
