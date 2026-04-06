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
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

# Allow importing sibling scripts without installing as a package
_SCRIPTS_DIR = Path(__file__).parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import render_timeline
import render_flowchart
import render_comparison
import render_pipeline
import build_pptx as _build_pptx_mod


_RENDERERS = {
    "timeline": render_timeline.render_timeline,
    "flowchart": render_flowchart.render_flowchart,
    "comparison": render_comparison.render_comparison,
    "pipeline": render_pipeline.render_pipeline,
}


def _resolve_viz(plan: Dict[str, Any], assets_dir: str) -> Dict[str, Any]:
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

        try:
            msg = renderer(viz, out_png)
            print(f"[build_deck] {msg}")
            fig["image_path"] = out_png
        except Exception as exc:
            print(
                f"[build_deck] ERROR rendering {viz_type} for slide {slide_num}: {exc}",
                file=sys.stderr,
            )
        # Keep viz key so the JSON is inspectable; build_pptx ignores unknown keys

    return resolved


def build_deck_from_plan(plan: dict, output_file: str, assets_dir: str | None = None) -> str:
    """Build PPTX from an already-loaded plan dict. Used by agent_ppt.py."""
    if assets_dir is None:
        assets_dir = str(Path(output_file).parent / (Path(output_file).stem + "_assets"))
    resolved_plan = _resolve_viz(plan, assets_dir)
    return _build_pptx_mod.build_pptx(resolved_plan, output_file)


def build_deck(plan_file: str, output_file: str, assets_dir: str | None = None) -> str:
    with open(plan_file, "r", encoding="utf-8") as f:
        plan = json.load(f)
    return build_deck_from_plan(plan, output_file, assets_dir)


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
    args = parser.parse_args()

    msg = build_deck(args.plan_file, args.output_file, args.assets_dir)
    print(msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
