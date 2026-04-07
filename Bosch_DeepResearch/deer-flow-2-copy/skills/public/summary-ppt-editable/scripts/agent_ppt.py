#!/usr/bin/env python3
"""
agent_ppt.py — One-command AI presentation generator.
Standalone — works with any LLM backend, no DeerFlow required.
Can also be called as a library from DeerFlow agents.

Usage:
  # Full pipeline: topic → LLM plan → figures → PPTX
  python agent_ppt.py --topic "L2 to L3 autonomous driving evolution" --output deck.pptx

  # Skip LLM, use an existing plan JSON
  python agent_ppt.py --plan existing_plan.json --output deck.pptx

  # Generate plan only (inspect before building)
  python agent_ppt.py --topic "..." --plan-only --output plan.json

  # Custom LLM backend
  python agent_ppt.py --topic "..." --backend anthropic --model claude-opus-4-5 --output deck.pptx

  # Specify slide count and extra instructions
  python agent_ppt.py --topic "Transformer architecture" --slides 8 \\
      --extra "Focus on attention mechanism and scaling laws" --output transformer.pptx

Environment variables (LLM backend, at least one required unless using --plan):
  ANTHROPIC_API_KEY       → uses Claude (recommended)
  OPENAI_API_KEY          → uses OpenAI
  CUSTOM_LLM_BASE_URL     → OpenAI-compatible custom endpoint
  CUSTOM_LLM_API_KEY      → API key for custom endpoint
  CUSTOM_LLM_MODEL        → model name for custom endpoint
  OLLAMA_HOST             → Ollama server URL (default: http://localhost:11434)
  OLLAMA_MODEL            → Ollama model name (default: llama3)

DeerFlow integration:
  from agent_ppt import generate_pptx
  output_path = generate_pptx(
      topic="L2 to L3 evolution",
      output_file="/mnt/user-data/outputs/deck.pptx",
      assets_dir="/mnt/user-data/outputs/deck_assets",
  )
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Allow importing sibling scripts without installing
_SCRIPTS_DIR = Path(__file__).parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import llm_planner
import build_deck as _build_deck_mod


# ── Public API (for DeerFlow or library use) ──────────────────────────────────

def generate_pptx(
    topic: str,
    output_file: str,
    assets_dir: Optional[str] = None,
    slides: int = 10,
    extra: str = "",
    backend: Optional[str] = None,
    model: str = "",
    plan_file: Optional[str] = None,
    lang: str = "en",
    two_stage: bool = True,
) -> str:
    """
    Full pipeline: topic → LLM plan → figures → PPTX.

    Args:
        topic:       Free-form topic string (used if plan_file is None)
        output_file: Destination .pptx path
        assets_dir:  Where to save generated PNG figures (auto-derived if None)
        slides:      Target slide count
        extra:       Extra instructions forwarded to the LLM
        backend:     LLM backend override (None = auto-detect)
        model:       Model name override
        plan_file:   Path to an existing plan JSON — skips LLM if provided
        lang:        Output language: "en" (default) or "zh"
        two_stage:   Run two-stage key-point extraction (default True)

    Returns:
        Absolute path to the generated .pptx file
    """
    if plan_file:
        print(f"[agent_ppt] Loading plan from {plan_file}", file=sys.stderr)
        with open(plan_file, encoding="utf-8") as f:
            plan = json.load(f)
    else:
        plan = llm_planner.generate_plan(
            topic=topic, slides=slides, extra=extra,
            backend=backend, model=model,
            lang=lang, two_stage=two_stage,
        )

    if assets_dir is None:
        p = Path(output_file)
        assets_dir = str(p.parent / (p.stem + "_assets"))

    msg = _build_deck_mod.build_deck_from_plan(plan, output_file, assets_dir)
    print(f"[agent_ppt] {msg}", file=sys.stderr)
    return str(Path(output_file).resolve())


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="One-command AI presentation generator: topic → PPTX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--topic", metavar="TOPIC",
                     help="Free-form topic string (LLM generates the plan)")
    src.add_argument("--plan", metavar="PLAN_JSON",
                     help="Path to existing plan JSON (skips LLM)")

    parser.add_argument("--output", required=True, metavar="OUT.pptx",
                        help="Output .pptx file path")
    parser.add_argument("--assets-dir", default=None, metavar="DIR",
                        help="Directory to store generated PNG figures")
    parser.add_argument("--slides", type=int, default=10,
                        help="Target number of slides (default: 10, only with --topic)")
    parser.add_argument("--extra", default="",
                        help="Extra instructions forwarded to LLM (only with --topic)")
    parser.add_argument("--backend", default=None,
                        choices=["anthropic", "openai", "custom", "ollama"],
                        help="LLM backend (default: auto-detect from env)")
    parser.add_argument("--model", default="",
                        help="LLM model name override")
    parser.add_argument("--lang", default="en", choices=["en", "zh"],
                        help="Output language: en (default) or zh")
    parser.add_argument("--no-two-stage", action="store_true",
                        help="Skip Stage 1 key-point extraction (faster, lower quality)")
    parser.add_argument("--plan-only", action="store_true",
                        help="Generate plan JSON only, do not build PPTX. "
                             "--output is treated as the plan JSON path.")
    args = parser.parse_args()

    if args.plan_only:
        if not args.topic:
            parser.error("--plan-only requires --topic")
        plan = llm_planner.generate_plan(
            args.topic, args.slides, args.extra, args.backend, args.model,
            lang=args.lang, two_stage=not args.no_two_stage,
        )
        out = json.dumps(plan, ensure_ascii=False, indent=2)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(out, encoding="utf-8")
            print(f"Plan written to {args.output}")
        else:
            print(out)
        return 0

    out_path = generate_pptx(
        topic=args.topic or "",
        output_file=args.output,
        assets_dir=args.assets_dir,
        slides=args.slides,
        extra=args.extra,
        backend=args.backend,
        model=args.model,
        plan_file=args.plan,
        lang=args.lang,
        two_stage=not args.no_two_stage,
    )
    print(f"PPTX ready: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
