#!/usr/bin/env python3
"""
Deck spec (*.spec.json) → timeline PNG + summary-ppt-editable plan → PPTX.

Usage:
  python orchestrate_timeline_pptx.py --deck-spec deck.spec.json --output-dir ./out
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _here() -> Path:
    return Path(__file__).resolve().parent


def _skills_public() -> Path:
    # scripts/ -> timeline-evolution-ppt/ -> public/
    return _here().parent.parent


def _slide_sort_key(sl: Dict[str, Any]) -> int:
    return int(sl.get("slide_number", sl.get("order", 0)))


def normalize_deck_spec(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    支持两种形态：
    - v1：顶层 title / slides[]（与现有一致）
    - v2：schema_version + document{} + timeline_library{} + slides[]（可引用时间线定义）

    输出统一为 v1 形状，供后续 _to_build_pptx_slides 使用；另返回 deck_meta 供写入 plan。
    """
    sv = str(raw.get("schema_version", "1.0"))
    if sv.startswith("1") or "document" not in raw:
        meta = {
            "schema_version": sv,
            "format": "flat_v1",
        }
        return {"_normalized": raw, "_deck_meta": meta}

    doc = raw.get("document") or {}
    lib = raw.get("timeline_library") or {}

    slides_out: List[Dict[str, Any]] = []
    for sl in sorted(raw.get("slides") or [], key=_slide_sort_key):
        s2: Dict[str, Any] = {k: v for k, v in sl.items() if k not in ("id", "order", "meta", "timeline_ref", "figure_caption")}
        sn = sl.get("slide_number", sl.get("order"))
        if sn is not None:
            s2["slide_number"] = int(sn)

        if (sl.get("type") or "").lower() == "timeline" and sl.get("timeline_ref"):
            ref = sl["timeline_ref"]
            if ref not in lib:
                raise ValueError(f"timeline_ref {ref!r} not found in timeline_library")
            merged = {**dict(lib[ref])}
            if sl.get("figure_caption"):
                merged["caption"] = sl["figure_caption"]
            s2["timeline"] = merged

        slides_out.append(s2)

    normalized = {
        "title": doc.get("title") or raw.get("title", "Presentation"),
        "subtitle": doc.get("subtitle") or raw.get("subtitle", ""),
        "aspect_ratio": doc.get("aspect_ratio") or raw.get("aspect_ratio", "16:9"),
        "theme": doc.get("theme") or raw.get("theme") or {},
        "slides": slides_out,
    }

    deck_meta = {
        "schema_version": sv,
        "format": "document_v2",
        "language": doc.get("language"),
        "tags": doc.get("tags"),
        "source_notes": doc.get("source_notes"),
        "revision": doc.get("revision"),
        "audience": doc.get("audience"),
    }
    return {"_normalized": normalized, "_deck_meta": {k: v for k, v in deck_meta.items() if v is not None}}


def _to_build_pptx_slides(spec_slides: List[Dict[str, Any]], png_out: Path, tmp_dir: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for s in spec_slides:
        st = (s.get("type") or "content").lower()
        if st == "timeline":
            tl = s.get("timeline") or {}
            stages = tl.get("stages") or []
            tjson = {
                "strip_title": tl.get("strip_title", ""),
                "footer_note": tl.get("footer_note", ""),
                "stages": stages,
                "width": tl.get("width", 2880),
                "height": tl.get("height"),
                "background": tl.get("background", "#f8fafc"),
                "png_dpi": tl.get("png_dpi", 220),
            }
            if tjson["height"] is None:
                del tjson["height"]
            stages_path = tmp_dir / "timeline_stages_for_render.json"
            with open(stages_path, "w", encoding="utf-8") as f:
                json.dump(tjson, f, indent=2, ensure_ascii=False)
            render = _here() / "render_timeline_strip.py"
            subprocess.run(
                [sys.executable, str(render), "--stages-json", str(stages_path), "-o", str(png_out)],
                check=True,
            )
            slide: Dict[str, Any] = {
                "slide_number": s["slide_number"],
                "type": "content",
                "title": s.get("title", ""),
                "modules": s.get("modules"),
                "bullets": s.get("bullets"),
                "notes": s.get("notes"),
                "figure": {
                    "position": "bottom",
                    "image_path": str(png_out),
                    "caption": tl.get("caption", ""),
                },
            }
            out.append(slide)
        else:
            out.append(s)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck-spec", required=True, type=Path, help="Deck specification JSON")
    ap.add_argument("--output-dir", required=True, type=Path, help="Output directory")
    ap.add_argument("--pptx-name", default="Timeline_Evolution.pptx", help="Output pptx filename")
    args = ap.parse_args()

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.deck_spec, "r", encoding="utf-8") as f:
        raw_spec = json.load(f)

    tmp_dir = out_dir / "_work"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    packed = normalize_deck_spec(raw_spec)
    spec = packed["_normalized"]
    deck_meta = packed.get("_deck_meta") or {}

    slides_sorted = sorted(spec.get("slides") or [], key=_slide_sort_key)
    timeline_png = out_dir / "timeline_strip.png"

    plan: Dict[str, Any] = {
        "title": spec.get("title", "Presentation"),
        "subtitle": spec.get("subtitle", ""),
        "aspect_ratio": spec.get("aspect_ratio", "16:9"),
        "theme": spec.get("theme") or {},
        "deck_meta": deck_meta,
        "slides": _to_build_pptx_slides(slides_sorted, timeline_png, tmp_dir),
    }

    plan_path = out_dir / "presentation_plan.json"
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)

    build = _skills_public() / "summary-ppt-editable/scripts/build_pptx.py"
    pptx_path = out_dir / args.pptx_name
    subprocess.run(
        [sys.executable, str(build), "--plan-file", str(plan_path), "--output-file", str(pptx_path)],
        check=True,
    )
    print("Plan:", plan_path)
    print("PPTX:", pptx_path)
    if timeline_png.is_file():
        print("Timeline PNG:", timeline_png)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as e:
        print(e, file=sys.stderr)
        raise SystemExit(e.returncode)
