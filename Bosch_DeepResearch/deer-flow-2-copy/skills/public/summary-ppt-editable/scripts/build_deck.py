#!/usr/bin/env python3
"""
Agent one-shot entry point: generate all viz figures from a deck plan JSON,
then build the PPTX — no manual image_path management required.

Usage:
  python build_deck.py --plan-file plan.json --output-file out.pptx [--assets-dir /tmp/deck_assets]

How it works:
  For each slide that has a figure.viz dict, this script:
    1. Calls the appropriate renderer (render_timeline / render_flowchart /
       render_comparison / render_pipeline) with the viz spec.
    2. Saves the PNG to <assets_dir>/<slide_N>_<type>.png.
    3. Injects the path back into slide.figure.image_path.
  Then calls build_pptx.build_pptx() on the resolved plan.

Supported viz types:
  "timeline"    -> render_timeline.render_timeline(spec, path)
  "flowchart"   -> render_flowchart.render_flowchart(spec, path)
  "comparison"  -> render_comparison.render_comparison(spec, path)
  "pipeline"    -> render_pipeline.render_pipeline(spec, path)
  "radar"       -> render_radar.render_radar(spec, path)
  "arch"        -> render_arch.render_arch(spec, path)

Viz spec is the same dict you would pass directly to each renderer's JSON format,
embedded under slide.figure.viz in the plan JSON.

Example slide with viz:
  {
    "slide_number": 3,
    "type": "content",
    "title": "L2 → L3 演变路线",
    "bullets": ["..."],
    "figure": {
      "position": "bottom",
      "caption": "L2 → L3 路线时间线",
      "viz": {
        "type": "timeline",
        "title": "L2 → L3 演变路线",
        "stages": [
          {"label": "L2 辅助", "year": "2018–2021", "annotation": "ACC + LKA"},
          {"label": "L2+ 扩展", "year": "2021–2024", "annotation": "高速 NOA"},
          {"label": "L3 条件自动", "year": "2024+", "annotation": "ODD 内免监视"}
        ],
        "highlight": [2]
      }
    }
  }
"""
from __future__ import annotations

import argparse
import base64
import copy
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Allow importing sibling scripts without installing as a package
_SCRIPTS_DIR = Path(__file__).parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import render_timeline
import render_flowchart
import render_comparison
import render_pipeline
import render_radar
import render_arch
import render_bar_chart
import render_line_chart
import render_scatter
import render_heatmap
import render_waterfall
import render_funnel
import render_mindmap
import render_tree
import render_matrix_2x2
import render_venn
import render_onion
import render_gantt
import render_swot
import render_pie
import build_pptx as _build_pptx_mod


_RENDERERS = {
    # Original 6
    "timeline":   render_timeline.render_timeline,
    "flowchart":  render_flowchart.render_flowchart,
    "comparison": render_comparison.render_comparison,
    "pipeline":   render_pipeline.render_pipeline,
    "radar":      render_radar.render_radar,
    "arch":       render_arch.render_arch,
    # Data viz
    "bar_chart":  render_bar_chart.render_bar_chart,
    "line_chart": render_line_chart.render_line_chart,
    "scatter":    render_scatter.render_scatter,
    "heatmap":    render_heatmap.render_heatmap,
    "waterfall":  render_waterfall.render_waterfall,
    "funnel":     render_funnel.render_funnel,
    "pie":        render_pie.render_pie,
    # Structure
    # "mindmap": render_mindmap.render_mindmap,  # disabled — use tree instead
    "tree":       render_tree.render_tree,
    "matrix_2x2":render_matrix_2x2.render_matrix_2x2,
    "venn":       render_venn.render_venn,
    "onion":      render_onion.render_onion,
    "gantt":      render_gantt.render_gantt,
    "swot":       render_swot.render_swot,
}


def _repair_spec(viz_spec: dict, error_msg: str) -> Optional[dict]:
    """
    Ask the same LLM backend used by llm_planner to fix a broken viz spec.
    Returns the repaired spec dict, or None if the LLM call fails.
    """
    # Import here to avoid circular dependency at module load time
    try:
        import llm_planner as _lp
    except ImportError:
        return None

    backend = _lp._detect_backend()
    system = (
        "You are a JSON repair assistant. "
        "A visualization spec caused a Python rendering error. "
        "Fix the JSON so it will render without errors. "
        "Return ONLY the corrected JSON object — no markdown fences, no commentary."
    )
    user = (
        f"ERROR:\n{error_msg}\n\n"
        f"ORIGINAL SPEC:\n{json.dumps(viz_spec, ensure_ascii=False, indent=2)}\n\n"
        "Return the corrected spec now."
    )
    try:
        raw = _lp._call_llm(system, user, backend, model="")
        return _lp._extract_json(raw)
    except Exception as e:
        print(f"[build_deck] spec repair LLM call failed: {e}", file=sys.stderr)
        return None


def _inspect_png(png_path: str) -> Tuple[bool, str]:
    """
    Tier-2: send the rendered PNG to a vision-capable LLM and ask whether
    the chart is clear and readable.  Returns (ok, feedback_text).

    Requires ANTHROPIC_API_KEY.  If unavailable, returns (True, "") so the
    caller can skip the visual-feedback loop gracefully.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return True, ""
    try:
        import anthropic
        data = Path(png_path).read_bytes()
        b64 = base64.standard_b64encode(data).decode()
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": b64},
                    },
                    {
                        "type": "text",
                        "text": (
                            "Inspect this presentation chart image. "
                            "Reply with a JSON object: "
                            "{\"ok\": true/false, \"issues\": \"one-sentence summary or empty string\"}. "
                            "ok=false if: labels overlap badly, text is cut off, bars/lines are missing, "
                            "or the chart is clearly unreadable. ok=true otherwise. "
                            "Return ONLY the JSON."
                        ),
                    },
                ],
            }],
        )
        for block in msg.content:
            if block.type == "text":
                result = json.loads(block.text.strip())
                return bool(result.get("ok", True)), result.get("issues", "")
    except Exception as e:
        print(f"[build_deck] visual inspection failed (non-fatal): {e}", file=sys.stderr)
    return True, ""


def _render_with_retry(
    renderer,
    viz_spec: dict,
    out_png: str,
    slide_num,
    viz_type: str,
    max_retries: int = 2,
    visual_inspect: bool = False,
) -> Optional[dict]:
    """
    Run renderer(viz_spec, out_png).  On failure, ask the LLM to repair the
    spec and retry up to max_retries times.  Optionally run a vision-model
    inspection pass after a successful render (Tier-2).

    Returns the (possibly repaired) spec on success, or None on total failure.
    """
    current_spec = viz_spec
    for attempt in range(1 + max_retries):
        try:
            msg = renderer(current_spec, out_png)
            print(f"[build_deck] {msg}")

            # Tier-2: optional visual inspection
            if visual_inspect and Path(out_png).exists():
                ok, issues = _inspect_png(out_png)
                if not ok:
                    print(
                        f"[build_deck] Visual inspection flagged slide {slide_num} "
                        f"{viz_type}: {issues} — attempting spec repair…",
                        file=sys.stderr,
                    )
                    repaired = _repair_spec(
                        current_spec,
                        f"Visual quality issue: {issues}",
                    )
                    if repaired and attempt < max_retries:
                        current_spec = repaired
                        continue   # re-render with repaired spec
                    # If no repair or no retries left, keep the imperfect render
                    print(
                        f"[build_deck] Keeping imperfect render for slide {slide_num}.",
                        file=sys.stderr,
                    )

            return current_spec  # success

        except Exception as exc:
            tb = traceback.format_exc()
            print(
                f"[build_deck] Attempt {attempt + 1} failed for {viz_type} "
                f"slide {slide_num}: {exc}",
                file=sys.stderr,
            )
            if attempt < max_retries:
                print("[build_deck] Asking LLM to repair spec…", file=sys.stderr)
                repaired = _repair_spec(current_spec, f"{exc}\n\n{tb}")
                if repaired is None:
                    print("[build_deck] LLM repair unavailable — giving up.", file=sys.stderr)
                    return None
                current_spec = repaired
            else:
                print(
                    f"[build_deck] All {1 + max_retries} attempts exhausted for "
                    f"slide {slide_num} {viz_type} — skipping figure.",
                    file=sys.stderr,
                )
                return None

    return None  # unreachable but satisfies type checker


def _resolve_viz(plan: Dict[str, Any], assets_dir: str, visual_inspect: bool = False) -> Dict[str, Any]:
    """Return a deep copy of plan with all figure.viz resolved to figure.image_path."""
    resolved = copy.deepcopy(plan)
    Path(assets_dir).mkdir(parents=True, exist_ok=True)

    for slide in resolved.get("slides", []):
        fig = slide.get("figure")
        if not isinstance(fig, dict):
            continue
        viz = fig.get("viz")
        if not isinstance(viz, dict):
            continue

        viz_type = (viz.get("type") or "").lower()
        renderer = _RENDERERS.get(viz_type)
        if renderer is None:
            print(
                f"[build_deck] WARNING: Unknown viz type '{viz_type}' on slide "
                f"{slide.get('slide_number')} — skipping.",
                file=sys.stderr,
            )
            continue

        slide_num = slide.get("slide_number", "?")
        out_png = os.path.join(assets_dir, f"slide_{slide_num}_{viz_type}.png")

        final_spec = _render_with_retry(
            renderer, viz, out_png, slide_num, viz_type,
            visual_inspect=visual_inspect,
        )
        if final_spec is not None:
            fig["image_path"] = out_png
            fig["viz"] = final_spec  # store repaired spec for inspection
        # Keep viz key so the JSON is inspectable; build_pptx ignores unknown keys

    return resolved


def build_deck_from_plan(
    plan: dict,
    output_file: str,
    assets_dir: str | None = None,
    visual_inspect: bool = False,
) -> str:
    """Build PPTX from an already-loaded plan dict. Used by agent_ppt.py."""
    if assets_dir is None:
        assets_dir = str(Path(output_file).parent / (Path(output_file).stem + "_assets"))
    resolved_plan = _resolve_viz(plan, assets_dir, visual_inspect=visual_inspect)
    return _build_pptx_mod.build_pptx(resolved_plan, output_file)


def build_deck(
    plan_file: str,
    output_file: str,
    assets_dir: str | None = None,
    visual_inspect: bool = False,
) -> str:
    with open(plan_file, "r", encoding="utf-8") as f:
        plan = json.load(f)
    return build_deck_from_plan(plan, output_file, assets_dir, visual_inspect=visual_inspect)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate viz figures from deck plan JSON, then build PPTX"
    )
    parser.add_argument("--plan-file", required=True, help="Path to deck plan JSON")
    parser.add_argument("--output-file", required=True, help="Output .pptx path")
    parser.add_argument(
        "--assets-dir",
        default=None,
        help="Directory to store generated PNG assets (default: <output_stem>_assets/ next to output)",
    )
    parser.add_argument(
        "--visual-inspect",
        action="store_true",
        help="After each successful render, send the PNG to a vision LLM for quality check "
             "and re-render if issues are found (requires ANTHROPIC_API_KEY).",
    )
    args = parser.parse_args()

    msg = build_deck(args.plan_file, args.output_file, args.assets_dir,
                     visual_inspect=args.visual_inspect)
    print(msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
