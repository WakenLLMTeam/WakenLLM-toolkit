---
name: summary-ppt-editable
description: Use this skill when the user wants information-rich, visually structured PowerPoint presentations with editable text and auto-generated figures (timeline, flowchart, comparison matrix, pipeline diagram). Works standalone or inside DeerFlow. Agent writes one topic string — the system calls an LLM to produce a structured plan, generates all figures, and assembles the PPTX.
---

# Summary PPT — Editable Text + Auto-Generated Figures

## Overview

A complete **agent toolkit** for generating information-dense `.pptx` presentations:

- **Titles and bullets are native editable PowerPoint text** — no locked images
- **Figures are auto-generated** from JSON specs (no external tools needed)
- **LLM-driven content planning** — Agent provides a topic; `llm_planner.py` produces a fully-structured deck plan with viz specs
- **Standalone or DeerFlow** — all scripts work independently; DeerFlow calls them via the skill interface

### Figure types available

| `viz.type` | What it renders | Best for |
|------------|-----------------|----------|
| `timeline` | Horizontal stage timeline with pills, dots, years, detail text | Evolution roadmaps, historical phases |
| `flowchart` | Directed node-arrow graph (auto-layout, shadows) | Decision flows, system logic |
| `comparison` | Feature matrix table with accent-highlighted column | Side-by-side capability diff |
| `pipeline` | Horizontal block diagram with step numbers, sublabels, badges | Module chains, data pipelines |

### Cards layout (text structure)

When a slide has a `cards` field, the content area is divided into **individual bordered boxes**, one per card, arranged in a grid. Each card has:
- An accent-colored top bar
- A bold heading (with optional icon/emoji prefix)
- Bullet lines below

This is the recommended layout when a slide has **multiple parallel topics** that should be visually separated (e.g. 3 technology pillars, 4 business dimensions, 6 key findings).

```json
{
  "slide_number": 3,
  "type": "content",
  "title": "三大核心能力",
  "cards": [
    {
      "heading": "感知层",
      "icon": "👁",
      "bullets": ["多模态融合", "实时目标检测", "语义分割"]
    },
    {
      "heading": "决策层",
      "icon": "🧠",
      "bullets": ["行为预测", "路径规划", "风险评估"]
    },
    {
      "heading": "执行层",
      "icon": "⚙️",
      "bullets": ["精准控制", "冗余制动", "故障自检"]
    }
  ],
  "cards_cols": 3
}
```

`cards_cols` is optional (0 or omitted = auto):
- 1 card → 1 column
- 2 cards → 2 columns
- 3 cards → 3 columns (default for 3)
- 4 cards → 2×2 grid
- 5–6 cards → 3 columns (2 rows)

Each card also supports `bg_rgb: [r, g, b]` to override the card background color per card.

---

## Usage modes

### Mode 1 — Fully automated (topic → PPTX in one command)

```bash
# Standalone — works anywhere Python + dependencies are installed
cd /path/to/summary-ppt-editable/scripts

python agent_ppt.py \
  --topic "L2 to L3 autonomous driving evolution" \
  --output /tmp/l2_l3_deck.pptx

# With options
python agent_ppt.py \
  --topic "Transformer architecture and scaling laws" \
  --slides 8 \
  --extra "Focus on attention mechanism internals" \
  --backend anthropic \
  --output /tmp/transformer.pptx
```

**LLM backend auto-detection** (set at least one):
```bash
export ANTHROPIC_API_KEY=sk-...        # → Claude (recommended)
export OPENAI_API_KEY=sk-...           # → OpenAI
export CUSTOM_LLM_BASE_URL=https://... # → any OpenAI-compatible API
export CUSTOM_LLM_API_KEY=...
export CUSTOM_LLM_MODEL=gpt-4o
export OLLAMA_HOST=http://localhost:11434  # → local Ollama
export OLLAMA_MODEL=llama3
```

### Mode 2 — Inspect plan before building

```bash
# Step 1: Generate plan only
python agent_ppt.py --topic "ADAS sensor fusion" --plan-only --output plan.json

# Step 2: Edit plan.json if needed, then build
python agent_ppt.py --plan plan.json --output deck.pptx
```

### Mode 3 — Agent writes plan manually (DeerFlow default)

Agent writes the plan JSON with `figure.viz` fields, then calls `build_deck.py`:

```bash
python /mnt/skills/public/summary-ppt-editable/scripts/build_deck.py \
  --plan-file /mnt/user-data/workspace/my-deck-plan.json \
  --output-file /mnt/user-data/outputs/my-deck.pptx
```

### Mode 4 — Bring your own images

Agent sets `figure.image_path` to pre-existing PNGs, calls `build_pptx.py` directly:

```bash
python /mnt/skills/public/summary-ppt-editable/scripts/build_pptx.py \
  --plan-file /mnt/user-data/workspace/plan.json \
  --output-file /mnt/user-data/outputs/deck.pptx
```

---

## DeerFlow integration

```python
# From a DeerFlow agent or subagent
from agent_ppt import generate_pptx

output_path = generate_pptx(
    topic="L2 to L3 autonomous driving evolution",
    output_file="/mnt/user-data/outputs/deck.pptx",
    assets_dir="/mnt/user-data/outputs/deck_assets",
    slides=10,
)
# → present_file(output_path)
```

---

## Deck plan JSON schema

Write the plan to `/mnt/user-data/workspace/<name>-deck-plan.json`.

### Top-level

| Field | Required | Description |
|-------|----------|-------------|
| `title` | yes | Deck title |
| `subtitle` | no | Subtitle on title slide |
| `aspect_ratio` | no | `"16:9"` (default) or `"4:3"` |
| `theme` | no | `accent_rgb [R,G,B]`, `body_rgb`, `title_rgb`, `bg_rgb` (slide + chart background, default `[255,255,255]`) |
| `slides` | yes | Ordered list of slide objects |

### Slide object

| Field | Required | Description |
|-------|----------|-------------|
| `slide_number` | yes | Unique integer |
| `type` | yes | `"title"` \| `"section"` \| `"content"` \| `"summary"` |
| `title` | yes | Slide title |
| `bullets` | no | Flat list of bullet strings |
| `modules` | no | Multi-block layout: `[{heading, bullets}]` — replaces flat bullets |
| `figure` | no | Visualization object (see below) |
| `notes` | no | Speaker notes |

### `figure` object

| Field | Required | Description |
|-------|----------|-------------|
| `viz` | no* | JSON spec for auto-generated figure (see Viz reference) |
| `image_path` | no* | Absolute path to existing PNG (used if `viz` absent) |
| `caption` | no | Caption below figure |
| `position` | no | `"right"` (default) or `"bottom"` |

*Provide either `viz` or `image_path`. If neither, a placeholder box is drawn.

---

## Viz field reference

### `timeline`

```json
"viz": {
  "type": "timeline",
  "title": "L2 to L3 Autonomous Driving Evolution",
  "highlight": [2],
  "stages": [
    { "label": "L2 Assisted",    "year": "2018-2021", "annotation": "ACC + LKA",        "detail": "Driver monitors at all times" },
    { "label": "L2+ Extended",   "year": "2021-2024", "annotation": "Highway NOA",       "detail": "Continuous assist in limited ODD" },
    { "label": "L3 Conditional", "year": "2024+",     "annotation": "Eyes-off in ODD",   "detail": "System owns driving task; TOR on exit" }
  ]
}
```

`highlight`: 0-based indices drawn with accent color (default: all). Use `position: "bottom"`.

### `flowchart`

```json
"viz": {
  "type": "flowchart",
  "title": "L3 ODD Decision Flow",
  "layout": "TB",
  "nodes": [
    { "id": "s",    "label": "Perception\nFusion",      "shape": "rect",    "color": "#dbeafe" },
    { "id": "odd",  "label": "ODD Boundary\nCheck",     "shape": "diamond", "color": "#fef9c3" },
    { "id": "plan", "label": "Path Planning",            "shape": "rect",    "color": "#dcfce7" },
    { "id": "tor",  "label": "Takeover Request",         "shape": "rect",    "color": "#fee2e2" },
    { "id": "mrc",  "label": "Minimal Risk\nCondition", "shape": "rounded", "color": "#fce7f3" }
  ],
  "edges": [
    { "from": "s",   "to": "odd" },
    { "from": "odd", "to": "plan", "label": "within ODD" },
    { "from": "odd", "to": "tor",  "label": "ODD exit" },
    { "from": "tor", "to": "mrc",  "label": "timeout" }
  ]
}
```

Shapes: `"rect"` | `"diamond"` | `"rounded"`. Layout: `"LR"` | `"TB"`. Use `position: "right"`.

### `comparison`

```json
"viz": {
  "type": "comparison",
  "title": "L2 / L2+ / L3 Capability Matrix",
  "highlight_col": 2,
  "rows": ["Responsibility", "Planning scope", "Takeover", "Redundancy", "Regulation"],
  "cols": ["L2 Assisted", "L2+ Extended", "L3 Conditional"],
  "cells": [
    ["Driver",        "Driver",          "System (within ODD)"],
    ["ACC + LKA",    "Highway NOA",     "Full longitudinal/lateral"],
    ["Anytime",      "Anytime",         "time-to-takeover + MRC"],
    ["Single fault", "Partial",         "Full ASIL-D redundancy"],
    ["No mandate",   "No mandate",      "UN R157 / regional law"]
  ],
  "row_notes": ["", "", "Key differentiator", "", ""]
}
```

Use `position: "bottom"` for wide tables.

### `pipeline`

```json
"viz": {
  "type": "pipeline",
  "title": "ADAS System Module Chain",
  "arrow_label": "data flow",
  "stages": [
    { "label": "Raw Sensors",  "sublabel": "Camera · Radar · LiDAR", "color": "#dbeafe" },
    { "label": "Perception",   "sublabel": "Detection · Lane",        "color": "#dcfce7" },
    { "label": "Prediction",   "sublabel": "Trajectory · Intent",     "color": "#fef9c3" },
    { "label": "Planning",     "sublabel": "Behavior · Path",         "color": "#fce7f3", "badge": "L3 key" },
    { "label": "Control",      "sublabel": "Longitudinal · Lateral",  "color": "#ede9fe" }
  ]
}
```

Use `position: "bottom"`. Optional `badge` field for ASIL levels, status tags, etc.

---

## Script reference

| Script | Role |
|--------|------|
| `agent_ppt.py` | **One-command entry point** — topic → PPTX |
| `llm_planner.py` | Calls LLM to generate structured deck plan JSON |
| `build_deck.py` | Resolves `viz` specs → PNGs, then calls `build_pptx.py` |
| `build_pptx.py` | Assembles final PPTX from plan with resolved image paths |
| `render_timeline.py` | Generates timeline PNG |
| `render_flowchart.py` | Generates flowchart PNG |
| `render_comparison.py` | Generates comparison table PNG |
| `render_pipeline.py` | Generates pipeline block diagram PNG |
| `viz_theme.py` | Shared design tokens + CJK font auto-detection |

## Templates

| File | Description |
|------|-------------|
| `templates/example_viz_plan.json` | Full 10-slide L2→L3 deck using all 4 viz types |
| `templates/example_l2_to_l3_plan.json` | Manual image_path approach (no auto-viz) |
| `templates/sample_plan.json` | Minimal 5-slide example |

## Dependencies

```
python-pptx
matplotlib
numpy
anthropic        # if using Claude backend
openai           # if using OpenAI or custom endpoint backend
```

Install: `pip install python-pptx matplotlib numpy anthropic openai`
