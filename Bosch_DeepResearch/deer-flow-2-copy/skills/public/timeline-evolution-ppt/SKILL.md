---
name: timeline-evolution-ppt
description: Use this skill when the user wants an editable PowerPoint that narrates evolution over time or capability stages—with a dedicated timeline strip figure aligned to the text. Combines structured modules (editable) with a generated timeline PNG (illustration).
---

# Timeline Evolution PPT Agent System

## 与「Agent」的关系（产品 vs 实现）

- **产品形态**：用户侧「说一句 → 出一份带时间线的可编辑 PPT」**应当**由 **LLM Agent**（理解意图、填 `document`/slides、改 `timeline_library`、调本脚本）完成。
- **本仓库实现**：提供 **Skill + 确定性脚本**（渲染 PNG、拼 pptx）；**不包含**内置大模型。DeerFlow / 其他 Agent 读本 SKILL 后，即构成完整 Agent 能力。

---

## Overview

End-to-end workflow for **图文并茂的时间线发展** decks:

1. **Text (editable)**: Titles, per-module headings, bullets—native PowerPoint text.
2. **Figure (timeline strip)**: One PNG generated from a **stage list** whose labels **must match** the narrative.
3. **Assembly**: `summary-ppt-editable` builds the `.pptx` (bottom-band figure layout).

This skill provides **scripts + JSON contracts**; the LLM agent fills content and invokes the orchestrator.

## Agent phases

| Phase | Action |
|-------|--------|
| 1. Scope | Topic (e.g. L2→L3), audience, slide count. |
| 2. Stages | Define ordered stages for the **timeline strip** (`title` + `subtitle` per stage)—single vocabulary. |
| 3. Body | Write `modules` (heading + bullets) per content slide; align terminology with stages. |
| 4. Render | Run `render_timeline_strip.py` with the same stages JSON → PNG. |
| 5. Build | Run `orchestrate_timeline_pptx.py` → `build_pptx.py` → deliver `.pptx`. |

## Files (this skill)

| Path | Role |
|------|------|
| `scripts/render_timeline_strip.py` | Stages JSON → horizontal timeline PNG (Pillow). |
| `scripts/orchestrate_timeline_pptx.py` | Deck spec JSON → `presentation_plan.json` for `summary-ppt-editable` + calls `build_pptx.py`. |
| `templates/example_deck.spec.json` | Full example (L2→L3). |

## Commands (container paths)

```bash
# 1) Timeline image only
python /mnt/skills/public/timeline-evolution-ppt/scripts/render_timeline_strip.py \
  --stages-json /mnt/user-data/workspace/timeline_stages.json \
  --output /mnt/user-data/outputs/timeline_strip.png

# 2) Full deck from one spec file
python /mnt/skills/public/timeline-evolution-ppt/scripts/orchestrate_timeline_pptx.py \
  --deck-spec /mnt/user-data/workspace/my_deck.spec.json \
  --output-dir /mnt/user-data/outputs/
```

Locally (repo root), use `skills/public/timeline-evolution-ppt/scripts/...`.

## Deck spec format (`*.spec.json`)

### v1（扁平，兼容旧稿）

顶层：`title`, `subtitle`, `aspect_ratio`, `theme`, `slides[]`。时间线可内联在某一页的 `timeline` 字段中。

### v2（推荐，信息更完整）

- `schema_version`: `"2.0"`
- `document`: 文档级元数据  
  - `title`, `subtitle`, `language`, `aspect_ratio`, `theme`  
  - `audience`, `revision`, `tags`, `source_notes`（便于 Agent / 检索 / 审计）
- `timeline_library`: **具名时间线资产**（可复用、可版本化）  
  - 键为 ID（如 `main_l2_l3`），值为与原先 `timeline` 内联相同的结构（`stages`, `width`, `strip_title`…）
- `slides[]`: 每页可有  
  - `id`, `order`（与 `slide_number` 二选一或并存）、`meta`  
  - `type: timeline` 时 **`timeline_ref`** 指向 `timeline_library` 的键，**`figure_caption`** 可覆盖库内 `caption`

编排器会将 v2 **归一化为内部 v1**，再调用 `build_pptx.py`。示例：`templates/example_deck_v2.spec.json`。

---

### 幻灯片对象（v1 / v2 共用字段）

Top-level（仅 v1）：

- `title`, `subtitle`, `aspect_ratio`, `theme` (same as summary-ppt-editable).
- `slides`: array of slide objects.

Slide types:

- `title` | `section` | `content` | `summary` — same as summary-ppt-editable.
- **`timeline`**: has `title`, `modules`, `timeline` object:
  - `timeline.stages[]`: 每格可包含：
  - `title` / `subtitle`：阶段名与一行定位说明。
  - `keywords`：短标签（显示为 `[关键词]`，便于扫读）。
  - `details[]`：多条要点字符串（自动按栏宽换行；**信息密度主要加在这里**）。
  - `fill`：背景色（hex）。
- `timeline.strip_title`：整条时间线上方标题。
- `timeline.footer_note`：整条时间线下方灰色脚注（可选）。
- `timeline.width` / `timeline.height`：PNG 像素尺寸。**建议 `width` ≥ 2400**（默认 2880），避免插入全宽幻灯片后被放大发糊；`height` 省略时按内容自动估算。
- `timeline.png_dpi`：写入 PNG 元数据（默认 220），便于查看器识别物理尺寸。
- `timeline.caption`：PPT 图下方说明（可选）。

The orchestrator renders PNG, injects `figure.position: bottom`, and writes paths.

## Alignment rules

- Stage **titles** in PNG must appear in slide text (or synonyms documented in notes)—no orphan labels.
- Numbers and legal claims stay in **text**, not in the strip image.

## Dependency

- `python-pptx`, `Pillow`.
- Reuses `../summary-ppt-editable/scripts/build_pptx.py` (no duplication).
