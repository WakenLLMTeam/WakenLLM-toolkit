#!/usr/bin/env python3
"""
canvas_slides_agent.py — LEGO-style per-slide composer.

Unlike the outline-first approach (slides_agent.py), this agent works
iteratively: for each slide, it sees the canvas state and places one
component at a time (text, viz, shape, image), then calls finish().

Typical usage:
  python canvas_slides_agent.py \
      --slide "L2 to L3 evolution timeline with bullet context" \
      --output /tmp/slide_demo.pptx \
      --theme bosch

The agent is driven by an LLM that has access to these tools:
  add_text   — place a text block
  add_viz    — place a visualization (renders PNG via the appropriate renderer)
  add_shape  — place a shape (accent bar, divider, background, etc.)
  add_image  — place an image
  add_badge  — place a small pill badge (convenience wrapper around add_shape)
  query      — inspect current canvas state
  finish     — render canvas to PPTX and close the slide

All coordinates are in INCHES. Standard 16:9 slide: 13.333" wide × 7.5" tall.
Origin is top-left; x increases rightward, y increases downward.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_SCRIPTS_DIR = Path(__file__).parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import llm_planner
import slide_canvas as _sc

# ── Agent system prompt ─────────────────────────────────────────────────────────

_CANVAS_SYSTEM = textwrap.dedent("""
You are a world-class PPT slide designer. You build slides by placing components
onto a canvas — like arranging objects on a digital whiteboard.

CANVAS CONVENTIONS (all units in INCHES):
  Standard 16:9 slide: 13.333" wide × 7.5" tall
  Origin: top-left corner (0, 0)
  x → increases rightward
  y → increases downward

TOOL REFERENCE:
  add_text(id, content, pos, size, font_size, bold, color_rgb, alignment, bullet, level, italic)
    id:         unique string, e.g. "title", "bullet_1", "c2"
    content:    text string. Use "\\n" for line breaks. Start with "• " for bullets.
    pos:       [x, y] — top-left corner of the text box
    size:      [width, height] in inches
    font_size: pt size (default 12). Title: 28-32, Subtitle: 18-22, Body: 12-14, Small: 9-11
    bold:      true for headings / emphasis
    color_rgb: [r, g, b] — RGB values 0-255. Common: accent=[226,0,21] (Bosch red), [37,99,235] (blue)
    alignment: "left" | "center" | "right"
    bullet:    true to prefix each line with "• "
    level:     0 = top-level bullet, 1 = sub-bullet
    italic:    true for captions / secondary text

  add_viz(id, viz_type, viz_data, pos, size, caption, theme_name, badge)
    viz_type:  one of: timeline | flowchart | comparison | pipeline | arch |
               bar_chart | line_chart | scatter | heatmap | waterfall | funnel |
               mindmap | tree | matrix_2x2 | venn | onion | gantt | swot
    viz_data:  full JSON spec for that viz type (schema per type — see viz reference)
    pos:       [x, y] of the top-left corner of the viz area
    size:      [width, height] — use height=4.5-5.5 for most viz, width fills available space
    caption:   optional short caption shown below the viz
    theme_name:"default" (blue) or "bosch" (red)
    badge:     optional small tag shown top-right (e.g. "L3 key", "ASIL-D")

  add_shape(id, shape_type, pos, size, fill_rgb, line_rgb, line_width, text, text_color_rgb, text_font_size)
    shape_type: "rect" | "rounded_rect" | "circle" | "accent_bar" | "divider"
    fill_rgb:   [r, g, b] for background fill
    line_rgb:   [r, g, b] for border (set to null for no border)
    text:       if set, shape displays this text centred inside

  add_badge(text, pos, color_rgb, text_color_rgb, font_size)
    Convenience: places a small pill (rounded rect) with centred text.
    pos:         [x, y] of top-left corner
    color_rgb:   pill background, default=[37,99,235]
    text_color_rgb: text color, default=[255,255,255]
    font_size:   pt, default=8

  add_image(id, image_path, pos, size, fit)
    image_path: absolute path to a PNG / JPG
    fit:        "contain" (letterbox) | "cover" | "fill" | "stretch"

  query()
    Returns: canvas size, list of all placed components with their ids/types/positions/sizes.
    Use this to check what's already placed and find gaps to fill.

  finish(output_path)
    Renders the canvas to PPTX at output_path. Must be called once per slide.
    The file will be saved; check the output path in the result message.

VIZ SCHEMA REFERENCE:
  timeline:    {stages:[{label,year,annotation?,detail?}]}
  flowchart:   {nodes:[{id,label,shape,color}],edges:[{from,to,label?}]}
  comparison:  {rows,cols,cells,highlight_col?,row_notes?}
  pipeline:    {stages:[{label,sublabel,color,badge?}]}
  arch:        {layers:[{name,color,blocks:[{label,sublabel?,badge?}]}]}
  bar_chart:   {categories,series:[{name,values,color}],unit?,mode?}
  line_chart:  {x_labels,series:[{name,values,color,marker?,linestyle?}]}
  scatter:     {x_label,y_label,quadrant_labels?,series:[{name,points:[{x,y,label?,size?,color?}]}]}
  heatmap:     {rows,cols,values,color_scheme?,show_values?}
  waterfall:   {unit?,items:[{label,value,type}]}
  funnel:      {stages:[{label,value,color}]}
  mindmap:     {center,branches:[{label,color,children:[{label}]}]}
  tree:        {root:{label,color,children:[{label,color,children}]}}
  matrix_2x2:  {x_label,y_label,quadrants,items:[{label,x,y,color}]}
  venn:        {circles:[{label,color,items}],overlaps:[{circles:[0,1],label}]}
  onion:       {center_label,layers:[{label,color,description}]}
  gantt:       {x_labels,tasks:[{label,start,end,color,milestone?}],groups?}
  swot:        {subject,quadrants:{strengths:{items},weaknesses:{items},opportunities:{items},threats:{items}}}

DESIGN PRINCIPLES:
  1. ALWAYS place a slide title text block first (font_size=26-30, bold, y≈0.3-0.5)
  2. NEVER leave empty whitespace on a finished slide — fill it or justify it
  3. Use accent_bar (fill_rgb=[226,0,21]) on the left edge (x≈0, width≈0.12) for brand consistency
  4. Viz goes right or bottom; text fills remaining space
  5. Call query() after 2-3 components if unsure about layout
  6. When mixing viz + bullets: viz takes ~55-65% width (right side), bullets fill left ~35%
  7. Common slide patterns:
     - Text + viz right:  text (x=0.5, w≈5.5), viz (x=6.2, w≈6.8)
     - Full-width viz:   viz (x=0.5, w≈12.3) at bottom, title+bullets at top
     - Bullet-heavy:     bullets (x=0.5, w≈12), add a small viz (w≈4) at bottom-right
     - Two-column:       left text (x=0.5, w=5.8), right text (x=6.8, w=5.8)

Return your response as a JSON tool-call block:
{
  "tool_calls": [
    {"name": "add_text", "args": {...}},
    {"name": "add_viz",  "args": {...}},
    ...
    {"name": "finish",  "args": {"output_path": "/tmp/slide.pptx"}}
  ]
}
""").strip()

# Fallback system (used when LLM doesn't return valid JSON tool calls)
_FALLBACK_SYSTEM = textwrap.dedent("""
You are a PPT slide designer. Given a slide description, place components
onto the canvas. Return a JSON array of tool calls.

Canvas: 13.333" × 7.5", origin top-left.
Tools: add_text | add_viz | add_shape | add_badge | query | finish

Always include:
  1. A bold title text block
  2. An accent bar shape on the left edge (shape_type="accent_bar", x=0, w=0.12, h=7.5)
  3. Body content (text or viz)
  4. finish() call

Return JSON: {"tool_calls": [{"name": "...", "args": {...}}, ...]}
""").strip()


def _parse_tool_calls(raw: str) -> List[Dict[str, Any]]:
    """Extract JSON tool_calls array from LLM response."""
    raw = raw.strip()
    # Try direct JSON parse first
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "tool_calls" in data:
            return data["tool_calls"]
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown fences
    for pattern in [r"```json\s*(\{.*?\})\s*```", r"```\s*(\{.*?\})\s*```",
                    r"\{[^{}]*\"tool_calls\"[^{}]*\}"]:
        m = re.search(pattern, raw, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(1))
                if "tool_calls" in data:
                    return data["tool_calls"]
            except json.JSONDecodeError:
                pass

    # Last resort: find any JSON array of objects with "name" field
    m = re.search(r'\[\s*\{[^]]*\"name\"[^]]*\}\s*\]', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse tool_calls from response: {raw[:200]}")


def _execute_tool_call(
    cv: _sc.Canvas,
    tool_name: str,
    args: dict[str, Any],
) -> str:
    """Execute a single tool call on the canvas and return a status message."""
    try:
        if tool_name == "add_text":
            cv.add_text(
                id=args["id"],
                content=args["content"],
                pos=tuple(args["pos"]),
                size=tuple(args["size"]),
                font_size=args.get("font_size", 12.0),
                bold=args.get("bold", False),
                color_rgb=args.get("color_rgb"),
                alignment=args.get("alignment", "left"),
                bullet=args.get("bullet", False),
                level=args.get("level", 0),
                italic=args.get("italic", False),
            )
            return f"OK: added text '{args['id']}'"

        elif tool_name == "add_viz":
            cv.add_viz(
                id=args["id"],
                viz_type=args["viz_type"],
                viz_data=args["viz_data"],
                pos=tuple(args["pos"]),
                size=tuple(args["size"]),
                caption=args.get("caption"),
                theme_name=args.get("theme_name", "default"),
                badge=args.get("badge"),
            )
            return f"OK: added viz '{args['id']}' ({args['viz_type']})"

        elif tool_name == "add_shape":
            cv.add_shape(
                id=args["id"],
                shape_type=args["shape_type"],
                pos=tuple(args["pos"]),
                size=tuple(args["size"]),
                fill_rgb=args.get("fill_rgb"),
                line_rgb=args.get("line_rgb"),
                line_width=args.get("line_width", 1.0),
                text=args.get("text"),
                text_color_rgb=args.get("text_color_rgb"),
                text_font_size=args.get("text_font_size", 10.0),
            )
            return f"OK: added shape '{args['id']}'"

        elif tool_name == "add_badge":
            cv.add_badge(
                text=args["text"],
                pos=tuple(args["pos"]),
                color_rgb=args.get("color_rgb"),
                text_color_rgb=args.get("text_color_rgb"),
                font_size=args.get("font_size", 8.0),
            )
            return f"OK: added badge '{args['text']}'"

        elif tool_name == "add_image":
            cv.add_image(
                id=args["id"],
                image_path=args["image_path"],
                pos=tuple(args["pos"]),
                size=tuple(args["size"]),
                fit=args.get("fit", "contain"),
            )
            return f"OK: added image '{args['id']}'"

        elif tool_name == "query":
            state = cv.query()
            return json.dumps(state, indent=2)

        elif tool_name == "finish":
            return "FINISH_SIGNAL"

        else:
            return f"ERROR: unknown tool '{tool_name}'"

    except Exception as exc:
        return f"ERROR in {tool_name}({args}): {exc}"


def _build_slide(
    slide_description: str,
    output_path: str,
    theme: str = "default",
    backend: Optional[str] = None,
    model: str = "",
    max_iterations: int = 12,
) -> str:
    """
    Build a single PPTX slide by having an LLM iteratively place components.

    Args:
        slide_description: natural language description of the slide content
        output_path:        .pptx output path
        theme:              "default" or "bosch"
        backend:            LLM backend override
        model:              model name override
        max_iterations:     max tool-call rounds before giving up

    Returns:
        Output .pptx path on success
    """
    backend = backend or llm_planner._detect_backend()

    # Build theme config
    accent_rgb = [226, 0, 21] if theme == "bosch" else [37, 99, 235]
    title_rgb  = [15, 23, 42]
    body_rgb   = [51, 65, 85]

    cv = _sc.Canvas(
        width_in=13.333,
        height_in=7.5,
        theme_name=theme,
        accent_rgb=accent_rgb,
        title_rgb=title_rgb,
        body_rgb=body_rgb,
    )

    # Theme instruction for LLM
    theme_note = (
        "Use Bosch Red accent (#E20015) for all shapes and highlights. "
        if theme == "bosch"
        else "Use blue accent (#2563eb) for shapes and highlights. "
    )

    user_msg = (
        f"Build the following slide. Place components using the tools.\n\n"
        f"Slide description:\n{slide_description}\n\n"
        f"{theme_note}"
        f"Canvas is 13.333\" wide × 7.5\" tall. Place components, then call finish()."
    )

    state_log: List[str] = []
    canvas_state = ""

    for iteration in range(max_iterations):
        # Check if previous calls left the canvas finished
        if canvas_state == "FINISH_SIGNAL":
            break

        try:
            raw = llm_planner._call_llm(_CANVAS_SYSTEM, user_msg, backend, model)
        except Exception as exc:
            # Fallback to simpler prompt
            fallback_user = f"Build this slide: {slide_description}\n{theme_note}"
            try:
                raw = llm_planner._call_llm(_FALLBACK_SYSTEM, fallback_user, backend, model)
            except Exception as exc2:
                raise RuntimeError(f"LLM call failed: {exc2}") from exc2

        try:
            tool_calls = _parse_tool_calls(raw)
        except ValueError as exc:
            state_log.append(f"[iter {iteration}] parse error: {exc}")
            # Retry with state
            user_msg = (
                f"Previous attempt could not be parsed:\n{raw[:400]}\n\n"
                f"Canvas state: {canvas_state[:300] if canvas_state else 'empty'}\n"
                f"Please try again with a valid JSON tool_calls array."
            )
            continue

        if not tool_calls:
            state_log.append(f"[iter {iteration}] no tool calls returned")
            break

        # Execute each tool call
        finish_called = False
        for tc in tool_calls:
            name = tc.get("name", "")
            args = tc.get("args") or {}
            result = _execute_tool_call(cv, name, args)
            state_log.append(f"[iter {iteration}] {name}: {result}")

            if result == "FINISH_SIGNAL":
                finish_called = True
                output_path = args.get("output_path", output_path)
                break
            elif result.startswith("ERROR"):
                # Report back to LLM so it can correct
                user_msg = (
                    f"Canvas error on last call: {result}\n"
                    f"Current state: {cv.query()}\n"
                    f"Please fix and continue."
                )
                break

        if finish_called:
            break

        # Provide updated context to LLM for next iteration
        state = cv.query()
        canvas_state = json.dumps(state, indent=2)
        n = state["component_count"]
        used = state["used_ids"]
        user_msg = (
            f"Current canvas ({n} component{'s' if n != 1 else ''} placed):\n"
            f"IDs placed: {used}\n"
            f"State:\n{canvas_state}\n\n"
            f"Continue placing components or call finish() when done."
        )

    # Execute finish if not already called
    if not finish_called:
        # Try one final pass with finish
        try:
            final_state = cv.query()
            state_log.append(f"Auto-finishing with {final_state['component_count']} components")
        except Exception:
            pass

    # Always try to render
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out = cv.finish(output_path)

    return out, "\n".join(state_log)


# ── Main pipeline: multi-slide deck ────────────────────────────────────────────

def build_deck(
    slide_descriptions: List[str],
    output_file: str,
    theme: str = "default",
    backend: Optional[str] = None,
    model: str = "",
    assets_dir: Optional[str] = None,
) -> List[str]:
    """
    Build a multi-slide deck by composing each slide with the canvas agent.

    Args:
        slide_descriptions: list of natural-language slide descriptions (one per slide)
        output_file:        base output path; slides are numbered automatically
        theme:               "default" or "bosch"
        backend:             LLM backend override
        model:               model name override
        assets_dir:          directory for viz PNG assets

    Returns:
        List of output .pptx paths (one per slide)
    """
    output_paths: List[str] = []
    deck_log: List[str] = []

    for i, desc in enumerate(slide_descriptions, 1):
        stem = Path(output_file).stem
        ext  = Path(output_file).suffix
        parent = Path(output_file).parent
        slide_out = str(parent / f"{stem}_slide_{i:02d}{ext}")

        if assets_dir:
            slide_assets = str(Path(assets_dir) / f"slide_{i:02d}_assets")
        else:
            slide_assets = str(parent / f"{stem}_slide_{i:02d}_assets")

        slide_path, log = _build_slide(
            slide_description=desc,
            output_path=slide_out,
            theme=theme,
            backend=backend,
            model=model,
        )
        output_paths.append(slide_path)
        deck_log.append(f"\n=== Slide {i} ===\n{log}")
        print(f"[canvas_slides_agent] Slide {i}/{len(slide_descriptions)}: {slide_path}")

    # Optionally merge into single deck
    # (Future: call build_pptx with a merged plan)
    return output_paths, "\n".join(deck_log)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Canvas-based slide composer: build slides component by component",
    )
    parser.add_argument("--slide", required=True,
                        help="Natural-language description of the slide")
    parser.add_argument("--output", required=True,
                        help="Output .pptx path")
    parser.add_argument("--theme", default="default", choices=["default", "bosch"],
                        help="Color theme")
    parser.add_argument("--backend", default=None,
                        choices=["anthropic", "openai", "custom", "ollama"])
    parser.add_argument("--model", default="")
    parser.add_argument("--iterations", type=int, default=12,
                        help="Max agent iterations per slide (default: 12)")
    args = parser.parse_args()

    try:
        out, log = _build_slide(
            slide_description=args.slide,
            output_path=args.output,
            theme=args.theme,
            backend=args.backend,
            model=args.model,
            max_iterations=args.iterations,
        )
        print(f"\nSlide saved: {out}")
        print(f"\nAgent log:\n{log}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
