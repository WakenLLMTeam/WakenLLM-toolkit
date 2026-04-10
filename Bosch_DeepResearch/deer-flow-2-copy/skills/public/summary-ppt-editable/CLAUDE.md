# summary-ppt-editable — Developer Reference

This skill generates information-dense editable PowerPoint presentations: native text (not burned-in images) combined with auto-generated figure PNGs. This file documents every viz type, schema, and script for contributors and agent maintainers.

## Architecture

```
summary-ppt-editable/
├── SKILL.md                  # Agent-facing spec (LLM reads this at runtime)
├── CLAUDE.md                 # Developer reference (this file)
├── scripts/
│   ├── agent_ppt.py          # One-command entry point: topic → PPTX (batch viz)
│   ├── slides_agent.py        # Per-slide agent: topic → PPTX (per-slide viz decisions)
│   ├── llm_planner.py        # Calls LLM to generate deck plan JSON
│   ├── build_deck.py         # Resolves viz specs → PNGs, then calls build_pptx
│   ├── build_pptx.py         # Assembles final PPTX from plan + resolved images
│   ├── viz_theme.py          # Shared design tokens + CJK font detection
│   ├── render_timeline.py
│   ├── render_flowchart.py
│   ├── render_comparison.py
│   ├── render_pipeline.py
│   ├── render_arch.py
│   ├── render_bar_chart.py
│   ├── render_line_chart.py
│   ├── render_scatter.py
│   ├── render_heatmap.py
│   ├── render_waterfall.py
│   ├── render_funnel.py
│   ├── render_mindmap.py
│   ├── render_tree.py
│   ├── render_matrix_2x2.py
│   ├── render_venn.py
│   ├── render_onion.py
│   ├── render_gantt.py
│   └── render_swot.py
└── templates/
    ├── example_viz_plan.json       # Full 10-slide deck using all 4 original viz types
    ├── example_l2_to_l3_plan.json  # Manual image_path approach
    └── sample_plan.json            # Minimal 5-slide example
```

## How build_deck.py dispatches

`build_deck.py` reads the plan JSON, finds every `slide.figure.viz` object, and routes `viz.type` to the corresponding renderer. The full dispatch table:

| `"type"` value | Renderer |
|---|---|
| `"timeline"` | `render_timeline.render_timeline` |
| `"flowchart"` | `render_flowchart.render_flowchart` |
| `"comparison"` | `render_comparison.render_comparison` |
| `"pipeline"` | `render_pipeline.render_pipeline` |
| `"arch"` | `render_arch.render_arch` |
| `"bar_chart"` | `render_bar_chart.render_bar_chart` |
| `"line_chart"` | `render_line_chart.render_line_chart` |
| `"scatter"` | `render_scatter.render_scatter` |
| `"heatmap"` | `render_heatmap.render_heatmap` |
| `"waterfall"` | `render_waterfall.render_waterfall` |
| `"funnel"` | `render_funnel.render_funnel` |
| `"mindmap"` | `render_mindmap.render_mindmap` |
| `"tree"` | `render_tree.render_tree` |
| `"matrix_2x2"` | `render_matrix_2x2.render_matrix_2x2` |
| `"venn"` | `render_venn.render_venn` |
| `"onion"` | `render_onion.render_onion` |
| `"gantt"` | `render_gantt.render_gantt` |
| `"swot"` | `render_swot.render_swot` |

Generated PNG and PDF are saved side-by-side at `<assets_dir>/slide_{n}_{type}.png/.pdf`.

CLI for every renderer: `--spec <json-file-or-inline-json>  --output <path.png>`

---

## Design tokens — viz_theme.py

Two theme instances are available. Import the one you need:

```python
from viz_theme import THEME          # default blue
from viz_theme import BOSCH_THEME    # Bosch corporate red
```

### Default THEME

| Token | Value | Use |
|---|---|---|
| `ACCENT` | `#2563eb` | Lines, arrows, borders, dots — never background fill |
| `ACCENT_LIGHT` | `#eff6ff` | Lightest allowed bg tint |
| `ACCENT_MID` | `#93c5fd` | Medium blue for highlights |
| `INK` / `BODY` | `#111111` | All body/heading text |
| `MUTED` | `#444444` | Captions, sub-labels |
| `BG` | `#ffffff` | Figure background |
| `SURFACE` | `#f5f7fa` | Card/node fill |
| `ALT_ROW` | `#eef2f7` | Alternating table rows |
| `BORDER` | `#c8d3e0` | Border color |
| Node palette | GREEN, YELLOW, RED, PURPLE, PINK | Hex pairs: `*` (fill) + `*_BORDER` (stroke) |
| Font sizes | `FS_TITLE=14`, `FS_H1=12`, `FS_H2=9.5`, `FS_BODY=8.5`, `FS_SMALL=7.5`, `FS_MICRO=6.5` | pt |
| `DPI` | `300` | PNG output resolution |
| `FIG_W` / `FIG_H` | `13.0` / `4.0` | Default figure size in inches |

### BOSCH_THEME overrides

| Token | Value | Use |
|---|---|---|
| `ACCENT` | `#E20015` | Bosch red primary |
| `NAVY` | `#003366` | Secondary accent |
| `SAFE_GREEN` | `#00873D` | OK / L2+ operational |
| `WARN_AMBER` | `#F5A623` | Conditional / transitional |
| `ALERT_RED` | `#E20015` | System limit / ASIL violation |
| `INFO_BLUE` | `#0057A8` | Passive monitoring |
| `LAYER_SENSOR/HAL/MW/APP` | `#f3e8ff / #dcfce7 / #fef9c3 / #dbeafe` | Arch diagram layer colors |

`setup_matplotlib()` auto-detects the best available CJK font (PingFang SC → Noto Sans CJK SC → SimHei → fallback).

---

## Viz type reference

All renderers accept optional `fig_width` and `fig_height` (inches). Most auto-size based on data count if omitted.

---

### `timeline`

Horizontal stage timeline with pills, dots, years, and detail text.

```json
"viz": {
  "type": "timeline",
  "title": "L2 to L3 Evolution",
  "highlight": [2],
  "stages": [
    { "label": "L2 Assisted",    "year": "2018-2021", "annotation": "ACC + LKA",      "detail": "Driver monitors at all times" },
    { "label": "L2+ Extended",   "year": "2021-2024", "annotation": "Highway NOA",    "detail": "Continuous assist in limited ODD" },
    { "label": "L3 Conditional", "year": "2024+",     "annotation": "Eyes-off in ODD","detail": "System owns driving task; TOR on exit" }
  ]
}
```

`highlight`: 0-based stage indices drawn with accent color. Recommended `position: "bottom"`.

---

### `flowchart`

Directed node-arrow graph with auto-layout and shadow effects.

```json
"viz": {
  "type": "flowchart",
  "title": "ODD Decision Flow",
  "layout": "TB",
  "nodes": [
    { "id": "s",    "label": "Perception\nFusion",   "shape": "rect",    "color": "#dbeafe" },
    { "id": "odd",  "label": "ODD Check",             "shape": "diamond", "color": "#fef9c3" },
    { "id": "plan", "label": "Path Planning",          "shape": "rect",    "color": "#dcfce7" },
    { "id": "tor",  "label": "Takeover Request",       "shape": "rect",    "color": "#fee2e2" },
    { "id": "mrc",  "label": "Minimal Risk\nCond.",   "shape": "rounded", "color": "#fce7f3" }
  ],
  "edges": [
    { "from": "s",   "to": "odd" },
    { "from": "odd", "to": "plan", "label": "within ODD" },
    { "from": "odd", "to": "tor",  "label": "ODD exit" },
    { "from": "tor", "to": "mrc",  "label": "timeout" }
  ]
}
```

`layout`: `"LR"` or `"TB"`. `shape`: `"rect"` | `"diamond"` | `"rounded"`. Recommended `position: "right"`.

---

### `comparison`

Feature matrix table with accent-highlighted column.

```json
"viz": {
  "type": "comparison",
  "title": "L2 / L2+ / L3 Capability Matrix",
  "highlight_col": 2,
  "rows": ["Responsibility", "Planning scope", "Takeover", "Redundancy", "Regulation"],
  "cols": ["L2 Assisted", "L2+ Extended", "L3 Conditional"],
  "cells": [
    ["Driver",       "Driver",        "System (within ODD)"],
    ["ACC + LKA",    "Highway NOA",   "Full longitudinal/lateral"],
    ["Anytime",      "Anytime",       "time-to-takeover + MRC"],
    ["Single fault", "Partial",       "Full ASIL-D redundancy"],
    ["No mandate",   "No mandate",    "UN R157 / regional law"]
  ],
  "row_notes": ["", "", "Key differentiator", "", ""]
}
```

`highlight_col`: 0-based column index to accent. `row_notes`: optional right-margin annotation per row. Recommended `position: "bottom"` for wide tables.

---

### `pipeline`

Horizontal block diagram with step numbers, sublabels, and optional badges.

```json
"viz": {
  "type": "pipeline",
  "title": "ADAS System Module Chain",
  "arrow_label": "data flow",
  "stages": [
    { "label": "Raw Sensors", "sublabel": "Camera · Radar · LiDAR", "color": "#dbeafe" },
    { "label": "Perception",  "sublabel": "Detection · Lane",        "color": "#dcfce7" },
    { "label": "Prediction",  "sublabel": "Trajectory · Intent",     "color": "#fef9c3" },
    { "label": "Planning",    "sublabel": "Behavior · Path",         "color": "#fce7f3", "badge": "L3 key" },
    { "label": "Control",     "sublabel": "Longitudinal · Lateral",  "color": "#ede9fe" }
  ]
}
```

`badge`: small top-right tag (ASIL level, status, etc.). Recommended `position: "bottom"`.

---

### `arch`

Layered horizontal-band system architecture with blocks and optional sublabels/badges.

```json
"viz": {
  "type": "arch",
  "title": "ADAS Software Stack",
  "direction": "TB",
  "layers": [
    {
      "name": "Application",
      "color": "#dbeafe",
      "blocks": [
        { "label": "Path Planner", "sublabel": "Behavior · Route", "badge": "ASIL-D" },
        { "label": "AEB",          "sublabel": "Emergency braking" }
      ]
    },
    {
      "name": "Middleware",
      "color": "#fef9c3",
      "blocks": [
        { "label": "ROS2",    "sublabel": "DDS transport" },
        { "label": "Autosar", "sublabel": "COM stack" }
      ]
    },
    {
      "name": "Hardware",
      "color": "#f3e8ff",
      "blocks": [
        { "label": "SoC",  "sublabel": "Orin / EyeQ5" },
        { "label": "FPGA", "sublabel": "Pre-proc" }
      ]
    }
  ]
}
```

`direction`: `"TB"` (top = index 0) or `"BT"` (bottom = index 0). `block.label` ≤ 15 chars, `sublabel` ≤ 20 chars, `badge` ≤ 8 chars.

---

### `bar_chart`

Grouped or stacked bar chart, vertical or horizontal, with optional value labels.

```json
"viz": {
  "type": "bar_chart",
  "title": "Market Share 2022-2024",
  "mode": "grouped",
  "orientation": "vertical",
  "categories": ["2022", "2023", "2024"],
  "series": [
    { "name": "Bosch",   "values": [32, 35, 38], "color": "#E20015" },
    { "name": "Mobileye","values": [28, 27, 25], "color": "#2563eb" },
    { "name": "Waymo",   "values": [10, 14, 18], "color": "#16a34a" }
  ],
  "unit": "%",
  "show_values": true
}
```

`mode`: `"grouped"` | `"stacked"`. `orientation`: `"vertical"` | `"horizontal"`. `show_values` is suppressed automatically when `n_series > 3` in grouped mode. `bar_gap` and `group_gap` control spacing in grouped mode.

---

### `line_chart`

Single or multi-series line chart with optional confidence bands and log scale.

```json
"viz": {
  "type": "line_chart",
  "title": "Compute Scaling Law",
  "x_labels": ["1B", "10B", "100B", "1T"],
  "series": [
    {
      "name": "FLOP/token",
      "values": [1.2, 12.5, 130, 1400],
      "color": "#2563eb",
      "band": [0.9, 1.5, 10, 15, 110, 150, 1200, 1600],
      "marker": "o",
      "linestyle": "-"
    }
  ],
  "unit": "PF·s",
  "log_scale": true,
  "show_grid": true
}
```

`band`: interleaved `[lower0, upper0, lower1, upper1, ...]` — length must be `len(values) × 2`. Rendered at 12% alpha. `marker`: `"o"` | `"s"` | `"^"` | `"none"`. `linestyle`: `"-"` | `"--"` | `":"`.

---

### `scatter`

Scatter or bubble chart with individually sized/colored points and optional quadrant dividers.

```json
"viz": {
  "type": "scatter",
  "title": "Technology Maturity vs. Business Impact",
  "x_label": "Maturity →",
  "y_label": "↑ Business Impact",
  "quadrant_labels": ["High Impact\nLow Maturity", "High Impact\nHigh Maturity",
                      "Low Impact\nLow Maturity",  "Low Impact\nHigh Maturity"],
  "series": [
    {
      "name": "ADAS Features",
      "points": [
        { "x": 8.2, "y": 9.1, "label": "AEB",       "size": 300, "color": "#E20015" },
        { "x": 3.5, "y": 7.8, "label": "LiDAR Fusion","size": 200,"color": "#2563eb" }
      ]
    }
  ]
}
```

`size`: marker area in pts² (default 150); larger values produce bubble-chart style. Quadrant dividers are drawn at midpoint of data range. Up to 4 `quadrant_labels` in order: `[top-left, top-right, bottom-left, bottom-right]`.

---

### `heatmap`

Color-encoded 2D matrix with row/column labels and optional value annotations.

```json
"viz": {
  "type": "heatmap",
  "title": "Sensor Coverage by Scenario",
  "rows": ["Highway", "Urban", "Parking", "Adverse Weather"],
  "cols": ["Camera", "Radar", "LiDAR", "Ultrasonic"],
  "values": [
    [9, 7, 8, 3],
    [8, 8, 9, 6],
    [5, 6, 4, 9],
    [4, 9, 5, 7]
  ],
  "color_scheme": "blue",
  "show_values": true,
  "value_format": ".0f"
}
```

`color_scheme`: `"blue"` | `"red"` | `"green"` | `"diverging"` (RdYlGn) | `"purple"`. `value_format`: any Python format string (`".1f"`, `".2%"`, etc.). Cell text auto-flips white/black based on value vs. midpoint.

---

### `waterfall`

Incremental waterfall chart with dashed connectors, color-coded by type.

```json
"viz": {
  "type": "waterfall",
  "title": "Cost Bridge 2023 → 2024",
  "unit": "M€",
  "items": [
    { "label": "2023 Base",      "value": 100, "type": "start" },
    { "label": "Volume +",       "value": 25,  "type": "positive" },
    { "label": "Price Mix",      "value": -8,  "type": "negative" },
    { "label": "Cost Savings",   "value": 15,  "type": "positive" },
    { "label": "FX Headwind",    "value": -5,  "type": "negative" },
    { "label": "2024 Total",     "value": 127, "type": "total" }
  ]
}
```

`type`: `"start"` (slate), `"positive"` (green), `"negative"` (accent color), `"total"` (accent, semi-transparent). Value labels use `+g` / `-g` format for positive/negative and plain `g` for start/total.

---

### `funnel`

Top-to-bottom conversion funnel with proportionally-sized bars and optional conversion-rate annotations.

```json
"viz": {
  "type": "funnel",
  "title": "ADAS Adoption Funnel",
  "show_percent": true,
  "unit": "K",
  "stages": [
    { "label": "Awareness",  "value": 5000, "color": "#dbeafe" },
    { "label": "Evaluation", "value": 2100, "color": "#93c5fd" },
    { "label": "Trial",      "value": 850,  "color": "#2563eb" },
    { "label": "Purchase",   "value": 320,  "color": "#1d4ed8" }
  ]
}
```

`show_percent`: shows `↓ X%` conversion rate between stages. Values ≥ 1M auto-format as `X.XM`, ≥ 1K as `XK`. Bar width ranges from 20% to 80% of canvas.

---

### `mindmap`

Radial mind map with a central node and branches with child leaf nodes.

```json
"viz": {
  "type": "mindmap",
  "center": "ADAS Architecture",
  "branches": [
    { "label": "Perception",  "color": "#dbeafe", "children": ["Camera", "Radar", "LiDAR"] },
    { "label": "Prediction",  "color": "#dcfce7", "children": ["Trajectory", "Intent"] },
    { "label": "Planning",    "color": "#fef9c3", "children": ["Behavior", "Path Opt."] },
    { "label": "Control",     "color": "#fce7f3", "children": ["Longitudinal", "Lateral"] }
  ]
}
```

Branches distribute from top (π/2), spread evenly. Child spread is capped at 38% of inter-branch angle to avoid overlap. Center node uses `THEME.ACCENT` fill with white text.

---

### `tree`

Recursive hierarchical tree in top-to-bottom or left-to-right orientation.

```json
"viz": {
  "type": "tree",
  "title": "Bosch ADAS Product Line",
  "direction": "TB",
  "root": {
    "label": "ADAS Portfolio",
    "color": "#dbeafe",
    "children": [
      {
        "label": "Highway Assist",
        "color": "#dcfce7",
        "children": [
          { "label": "ACC",  "color": "#f0fdf4" },
          { "label": "LKA",  "color": "#f0fdf4" }
        ]
      },
      {
        "label": "Urban Assist",
        "color": "#fef9c3",
        "children": [
          { "label": "AEB",  "color": "#fefce8" },
          { "label": "BSD",  "color": "#fefce8" }
        ]
      }
    ]
  }
}
```

`direction`: `"TB"` | `"LR"`. Root node has accent-colored border and thicker line weight. Font sizes and node dimensions adapt to tree span and longest label length.

---

### `matrix_2x2`

2×2 quadrant positioning chart with colored backgrounds and scatter-dot items.

```json
"viz": {
  "type": "matrix_2x2",
  "title": "Technology Prioritization Matrix",
  "x_label": "Implementation Complexity →",
  "y_label": "↑ Strategic Value",
  "quadrants": {
    "top_left":     { "label": "Quick Wins",    "color": "#dcfce7" },
    "top_right":    { "label": "Major Projects", "color": "#dbeafe" },
    "bottom_left":  { "label": "Fill-ins",       "color": "#fef9c3" },
    "bottom_right": { "label": "Hard Slogs",     "color": "#fee2e2" }
  },
  "items": [
    { "label": "AEB 2.0",      "x": 0.3, "y": 0.85, "color": "#E20015" },
    { "label": "LiDAR Fusion", "x": 0.8, "y": 0.90, "color": "#2563eb" },
    { "label": "OTA Update",   "x": 0.2, "y": 0.40, "color": "#16a34a" }
  ]
}
```

`x` / `y` are normalized [0, 1] coordinates. Items with `x < 0.5, y > 0.5` land in top-left, etc.

---

### `venn`

2-circle or 3-circle Venn diagram with colored transparent fills and overlap labels.

```json
"viz": {
  "type": "venn",
  "title": "Sensor Capability Overlap",
  "circles": [
    { "label": "Camera",      "color": "#dbeafe", "items": ["Color", "Lane detect", "Sign recog."] },
    { "label": "Radar",       "color": "#fef9c3", "items": ["Velocity", "Long range", "All weather"] },
    { "label": "LiDAR",       "color": "#dcfce7", "items": ["3D point cloud", "Precise range"] }
  ],
  "overlaps": [
    { "circles": [0, 1],    "label": "Object detect" },
    { "circles": [0, 2],    "label": "Shape classify" },
    { "circles": [1, 2],    "label": "Ranging" },
    { "circles": [0, 1, 2], "label": "Fusion core" }
  ]
}
```

Exactly 2 or 3 circles (raises `ValueError` otherwise). Each circle: max 3 items shown in exclusive region. Fill is 35% alpha. Overlap label is placed at geometric center.

---

### `onion`

Concentric rings diagram — index 0 is innermost, last index is outermost.

```json
"viz": {
  "type": "onion",
  "title": "Autonomous Driving Stack Layers",
  "center_label": "Core\nLogic",
  "layers": [
    { "label": "Sensor",     "color": "#f3e8ff", "description": "Camera · Radar · LiDAR" },
    { "label": "Perception", "color": "#dbeafe", "description": "Detection · Segmentation" },
    { "label": "Planning",   "color": "#dcfce7", "description": "Behavior · Path" },
    { "label": "Control",    "color": "#fef9c3", "description": "Longitudinal · Lateral" }
  ]
}
```

Labels distribute at evenly spaced angles from π × 0.20. Text alignment auto-flips left/center/right based on x-position.

---

### `gantt`

Horizontal Gantt chart with task bars, diamond milestones, and optional group separators.

```json
"viz": {
  "type": "gantt",
  "title": "L3 Program Roadmap",
  "x_labels": ["Q1'24", "Q2'24", "Q3'24", "Q4'24", "Q1'25", "Q2'25"],
  "tasks": [
    { "label": "Sensor Integration", "start": 0, "end": 2, "color": "#dbeafe" },
    { "label": "Stack Validation",   "start": 1, "end": 4, "color": "#dcfce7" },
    { "label": "SOP",                "start": 5, "end": 5, "color": "#E20015", "milestone": true }
  ],
  "groups": [
    { "label": "Phase 1 — Development", "rows": [0, 1] },
    { "label": "Phase 2 — Launch",      "rows": [2] }
  ]
}
```

`milestone: true` renders a diamond `◇` marker instead of a bar. Group labels appear as italic italic text with a dashed separator rule. `start` and `end` are 0-based column indices (same value = single-column milestone).

---

### `swot`

4-quadrant SWOT analysis with colored cells and fixed green/red/blue/amber header bars.

```json
"viz": {
  "type": "swot",
  "title": "Bosch ADAS Competitive Position",
  "subject": "Bosch",
  "quadrants": {
    "strengths":     { "color": "#f0fdf4", "items": ["Tier-1 scale", "ASIL-D expertise", "OEM relationships"] },
    "weaknesses":    { "color": "#fef2f2", "items": ["Software velocity", "SW talent gap"] },
    "opportunities": { "color": "#eff6ff", "items": ["L3 regulation window", "China EV growth"] },
    "threats":       { "color": "#fffbeb", "items": ["Mobileye vertical int.", "Tesla in-house stack"] }
  }
}
```

Max 5 items per quadrant (excess silently truncated). `subject` renders as a rounded box at the center intersection. Fixed header colors: green (strengths), red (weaknesses), blue (opportunities), amber (threats). Fixed axis annotations: "Internal" (top row), "External" (bottom row).

---

## Deck plan JSON top-level schema

```json
{
  "title": "Deck Title",
  "subtitle": "Optional subtitle",
  "aspect_ratio": "16:9",
  "theme": {
    "accent_rgb": [226, 0, 21],
    "body_rgb": [17, 17, 17],
    "title_rgb": [17, 17, 17]
  },
  "slides": [
    {
      "slide_number": 1,
      "type": "title",
      "title": "...",
      "subtitle": "..."
    },
    {
      "slide_number": 2,
      "type": "content",
      "title": "...",
      "bullets": ["Point A", "Point B"],
      "modules": [
        { "heading": "Section 1", "bullets": ["A", "B"] },
        { "heading": "Section 2", "bullets": ["C", "D"] }
      ],
      "cards": [
        { "heading": "Card A", "icon": "🔵", "bullets": ["x", "y"], "bg_rgb": [219, 234, 254] }
      ],
      "cards_cols": 3,
      "figure": {
        "position": "right",
        "caption": "Optional caption",
        "viz": { "type": "<see above>", "...": "..." }
      },
      "notes": "Speaker notes text"
    }
  ]
}
```

### Slide types

| `type` | Purpose |
|---|---|
| `"title"` | Cover slide with title + subtitle |
| `"section"` | Section divider |
| `"content"` | Main content slide (bullets / modules / cards + optional figure) |
| `"summary"` | Closing / summary slide |

### `figure.position`

| Value | Layout |
|---|---|
| `"right"` (default) | Text left ~55%, figure right ~45% |
| `"bottom"` | Text top, figure full-width below — use for wide charts (comparison, pipeline, gantt) |

### `cards` layout

Replaces `bullets` when content has multiple parallel topics. Grid is auto-calculated:

| Card count | Default layout |
|---|---|
| 1 | 1 column |
| 2 | 2 columns |
| 3 | 3 columns |
| 4 | 2×2 grid |
| 5–6 | 3 columns, 2 rows |

Override with `"cards_cols": N`. Each card: `heading`, optional `icon` (emoji), `bullets` list, optional `bg_rgb: [r, g, b]`.

---

## Running the scripts

### Per-slide agent (slides_agent.py)

`slides_agent.py` differs from `agent_ppt.py` in one critical way:

| | `agent_ppt.py` | `slides_agent.py` |
|---|---|---|
| Viz decision | LLM decides all viz types **upfront** in one shot | LLM decides viz type **per slide**, one at a time |
| Rendering | `build_deck` batch-renders all viz after plan is complete | Renderer called **immediately** after each per-slide decision |
| Flexibility | Viz type locked in by the time plan is generated | Agent can adapt viz choice based on slide content context |

Use `slides_agent.py` when you want the viz selection to be responsive to each slide's specific content goal rather than pre-baked in a global plan.

```bash
cd skills/public/summary-ppt-editable/scripts

# Full pipeline: topic → per-slide decisions → PPTX
python slides_agent.py \
  --topic "L2 to L3 autonomous driving evolution" \
  --output /tmp/deck.pptx \
  --slides 8 \
  --theme bosch

# Inspect per-slide decisions without rendering
python slides_agent.py \
  --topic "Bosch ADAS competitive analysis" \
  --slides 6 \
  --decisions-only \
  --output /tmp/decisions.json

# Use existing outline JSON (skip Stage 1)
python slides_agent.py \
  --outline @my_outline.json \
  --output /tmp/deck.pptx \
  --theme default

# Use an existing outline inline (JSON string)
python slides_agent.py \
  --outline '[{"slide_number":1,"type":"title","title":"Deck Title"}]' \
  --output /tmp/deck.pptx
```

### One-command (topic → PPTX)

```bash
cd skills/public/summary-ppt-editable/scripts

python agent_ppt.py \
  --topic "L2 to L3 autonomous driving evolution" \
  --slides 8 \
  --output /tmp/deck.pptx

# Inspect plan before building
python agent_ppt.py --topic "ADAS sensor fusion" --plan-only --output plan.json
python agent_ppt.py --plan plan.json --output deck.pptx
```

LLM backend: auto-detected from env vars. Set one of:
- `ANTHROPIC_API_KEY` → Claude (recommended)
- `OPENAI_API_KEY` → OpenAI
- `CUSTOM_LLM_BASE_URL` + `CUSTOM_LLM_API_KEY` + `CUSTOM_LLM_MODEL` → any OpenAI-compatible API
- `OLLAMA_HOST` + `OLLAMA_MODEL` → local Ollama

### From a plan file

```bash
python scripts/build_deck.py \
  --plan-file /path/to/deck-plan.json \
  --output-file /path/to/output.pptx \
  --assets-dir /path/to/assets/   # optional; defaults to <output_stem>_assets/
```

### DeerFlow sandbox paths

```bash
python /mnt/skills/public/summary-ppt-editable/scripts/build_deck.py \
  --plan-file /mnt/user-data/workspace/my-deck-plan.json \
  --output-file /mnt/user-data/outputs/my-deck.pptx
```

### Individual renderer (standalone test)

```bash
python scripts/render_bar_chart.py \
  --spec '{"type":"bar_chart","categories":["A","B"],"series":[{"name":"S1","values":[3,5],"color":"#E20015"}]}' \
  --output /tmp/test_bar.png
```

---

## Adding a new viz type

1. Create `scripts/render_<name>.py` with a `render_<name>(spec: dict, output_path: str)` function.
2. Import `THEME` from `viz_theme` for all colors; call `setup_matplotlib()` at module level.
3. Save output as both PNG and PDF (`output_path.replace(".png", ".pdf")`).
4. Register in `build_deck.py`'s `_RENDERERS` dict: `"<name>": render_<name>.render_<name>`.
5. Document the JSON schema in this file and add a usage example to `SKILL.md`.
6. Add a template example to `templates/example_viz_plan.json`.

---

## Dependencies

```
python-pptx    # PPTX assembly
matplotlib     # All figure rendering
numpy          # Numeric ops in charts
Pillow         # Image handling
anthropic      # Claude backend (optional)
openai         # OpenAI / custom endpoint backend (optional)
```

Install: `pip install python-pptx matplotlib numpy Pillow anthropic openai`
