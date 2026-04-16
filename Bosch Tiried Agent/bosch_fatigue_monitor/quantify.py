"""
量化疲劳评分分析工具
展示不同传感器数值组合如何影响 composite_score 和 SeverityTier

用法:
    python quantify.py          # 全部分析
    python quantify.py --grid   # 仅 PERCLOS × 哈欠 热力表
    python quantify.py --sweep  # 单信号扫描
    python quantify.py --cases  # 真实案例对比
"""
import asyncio, sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich import box
from rich.console import Console
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.panel import Panel

from models.fatigue_context import (
    FatigueContext, EnrichedFatigueContext,
    TextSignals, ImageSignals, AudioSignals, MapContext, RoadType, DriverProfile,
)
from judge.judge_agent import FatigueJudgeAgent
from models.judge_verdict import SeverityTier
from llm.mock_llm_client import MockLLMClient
import time

console = Console()

# ──────────────────────────────────────────────────────────────────────────────
# 核心：直接计算 composite score（不走 LLM，用权重公式还原）
# 权重来自 config.py: text=0.30 image=0.50 audio=0.20
# image_score = min(1.0, perclos / 0.15 * 0.6)
# ──────────────────────────────────────────────────────────────────────────────

def compute_score(
    perclos: float,
    yawns_per_min: float = 0.0,
    verbal: bool = False,
    steering: float = 0.0,
    brake_delta_ms: float = 0.0,
    speed_variance: float = 10.0,
) -> tuple[float, float, float, float, SeverityTier]:
    """Returns (text_score, image_score, audio_score, composite, tier)"""

    # ── Image score (PERCLOS-driven, threshold=0.15) ──
    image_score = min(1.0, perclos / 0.15 * 0.6)

    # ── Text score: steering + brake + speed variance ──
    steering_score  = min(1.0, max(0.0, (steering - 15.0) / 20.0))     # 0→15°=0; 35°=1.0
    brake_score     = min(1.0, max(0.0, (brake_delta_ms - 150) / 250))  # 0→150ms=0; 400ms=1.0
    variance_score  = min(1.0, max(0.0, (0.5 - speed_variance) / 0.5)) if speed_variance < 0.5 else 0.0
    text_score = min(1.0, (steering_score + brake_score + variance_score) / 3 * 1.5 + 0.10)

    # ── Audio score: yawns + verbal ──
    yawn_score   = min(1.0, yawns_per_min / 2.0 * 0.6)
    verbal_boost = 0.30 if verbal else 0.0
    audio_score  = min(1.0, yawn_score + verbal_boost)

    # ── Composite (weighted sum) ──
    composite = 0.30 * text_score + 0.50 * image_score + 0.20 * audio_score

    # ── Tier ──
    if composite >= 0.75:   tier = SeverityTier.SEVERE
    elif composite >= 0.55: tier = SeverityTier.MODERATE
    elif composite >= 0.30: tier = SeverityTier.MILD
    else:                   tier = SeverityTier.NONE

    return round(text_score, 3), round(image_score, 3), round(audio_score, 3), round(composite, 3), tier


TIER_COLOR = {
    SeverityTier.NONE:     "green",
    SeverityTier.MILD:     "yellow",
    SeverityTier.MODERATE: "dark_orange",
    SeverityTier.SEVERE:   "red",
}
TIER_LABEL = {
    SeverityTier.NONE:     "NONE    ",
    SeverityTier.MILD:     "MILD    ",
    SeverityTier.MODERATE: "MODERATE",
    SeverityTier.SEVERE:   "SEVERE  ",
}


def tier_cell(tier: SeverityTier, score: float) -> Text:
    color = TIER_COLOR[tier]
    bar_len = int(score * 16)
    bar = "█" * bar_len + "░" * (16 - bar_len)
    t = Text()
    t.append(f"{bar} ", style=color)
    t.append(f"{score:.2f} ", style=f"bold {color}")
    t.append(TIER_LABEL[tier], style=f"dim {color}")
    return t


# ──────────────────────────────────────────────────────────────────────────────
# 1. 热力表：PERCLOS × 哈欠频率
# ──────────────────────────────────────────────────────────────────────────────

def print_heatmap():
    console.print(Rule("[bold cyan]热力表：PERCLOS × 哈欠频率 → Composite Score[/bold cyan]"))
    console.print("[dim]固定：方向盘=10°（正常），刹车Δ=80ms（正常），语音疲劳=否[/dim]\n")

    perclos_vals = [0.04, 0.08, 0.12, 0.15, 0.18, 0.22, 0.28, 0.35, 0.42]
    yawn_vals    = [0.0,  0.5,  1.0,  2.0,  3.0,  4.0,  5.0]

    tbl = Table(box=box.ROUNDED, show_lines=True)
    tbl.add_column("PERCLOS ↓  Yawns/min →", style="bold", width=20)
    for y in yawn_vals:
        tbl.add_column(f"{y:.1f}/min", justify="center", width=20)

    for p in perclos_vals:
        row_label = Text(f"PERCLOS = {p:.2f}")
        if p < 0.15:
            row_label.stylize("green")
        elif p < 0.25:
            row_label.stylize("yellow")
        elif p < 0.35:
            row_label.stylize("dark_orange")
        else:
            row_label.stylize("red")

        cells = [row_label]
        for y in yawn_vals:
            _, _, _, composite, tier = compute_score(perclos=p, yawns_per_min=y)
            color = TIER_COLOR[tier]
            bar = "█" * int(composite * 10) + "░" * (10 - int(composite * 10))
            cell = Text()
            cell.append(f"{bar}\n", style=color)
            cell.append(f"  {composite:.2f}  ", style=f"bold {color}")
            cell.append(TIER_LABEL[tier][:3], style=f"dim {color}")
            cells.append(cell)
        tbl.add_row(*cells)

    console.print(tbl)
    console.print()


# ──────────────────────────────────────────────────────────────────────────────
# 2. 单信号扫描（固定其他信号，只变一个）
# ──────────────────────────────────────────────────────────────────────────────

def print_sweep():
    console.print(Rule("[bold cyan]单信号扫描：每个维度对 Composite Score 的影响[/bold cyan]"))

    sweeps = [
        {
            "title": "① PERCLOS（眼闭合率）— 其他信号静止",
            "var": "perclos",
            "values": [0.04, 0.08, 0.12, 0.15, 0.18, 0.22, 0.28, 0.35, 0.42, 0.50],
            "fixed": dict(yawns_per_min=0.5, steering=8.0, brake_delta_ms=60.0, speed_variance=5.0),
        },
        {
            "title": "② 哈欠频率（Yawns/min）— PERCLOS=0.20 固定",
            "var": "yawns_per_min",
            "values": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
            "fixed": dict(perclos=0.20, steering=8.0, brake_delta_ms=60.0, speed_variance=5.0),
        },
        {
            "title": "③ 方向盘幅度（Steering °）— PERCLOS=0.20 固定",
            "var": "steering",
            "values": [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0],
            "fixed": dict(perclos=0.20, yawns_per_min=1.0, brake_delta_ms=60.0, speed_variance=5.0),
        },
        {
            "title": "④ 刹车反应延迟 Δms — PERCLOS=0.20 固定",
            "var": "brake_delta_ms",
            "values": [0, 50, 100, 150, 200, 250, 300, 400],
            "fixed": dict(perclos=0.20, yawns_per_min=1.0, steering=8.0, speed_variance=5.0),
        },
    ]

    for sw in sweeps:
        console.print(f"[bold]{sw['title']}[/bold]")
        tbl = Table(box=box.SIMPLE_HEAVY, expand=False)
        tbl.add_column("值",        justify="right", style="cyan", width=10)
        tbl.add_column("Text",      justify="right", width=7)
        tbl.add_column("Image",     justify="right", width=7)
        tbl.add_column("Audio",     justify="right", width=7)
        tbl.add_column("Composite", width=38)
        tbl.add_column("Tier",      width=10)

        for val in sw["values"]:
            kwargs = dict(sw["fixed"])
            kwargs[sw["var"]] = val
            ts, im, au, comp, tier = compute_score(**kwargs)
            color = TIER_COLOR[tier]
            bar_len = int(comp * 24)
            bar = Text("█" * bar_len + "░" * (24 - bar_len), style=color)
            bar.append(f" {comp:.3f}", style=f"bold {color}")
            tier_txt = Text(tier.name, style=f"bold {color}")
            tbl.add_row(str(val), f"{ts:.2f}", f"{im:.2f}", f"{au:.2f}", bar, tier_txt)

        console.print(tbl)
        console.print()


# ──────────────────────────────────────────────────────────────────────────────
# 3. 真实案例对比
# ──────────────────────────────────────────────────────────────────────────────

def print_cases():
    console.print(Rule("[bold cyan]真实驾驶场景对比[/bold cyan]"))

    cases = [
        # (label, perclos, yawns, verbal, steering, brake_ms, speed_var, profile_hint)
        ("✅ 正常驾驶",          0.05, 0.2, False, 8.0,  50.0,  8.0,  "基准"),
        ("✅ 正常（晨间眨眼）",   0.13, 0.5, False, 7.0,  60.0,  6.0,  "干眼/误报风险"),
        ("⚠️ 轻度疲劳",          0.18, 1.5, False, 11.0, 100.0, 3.0,  "通勤者早期"),
        ("⚠️ 干眼误报（仅图像）", 0.22, 0.3, False, 6.0,  55.0, 10.0, "PERCLOS高但无辅助"),
        ("🟠 中度（哈欠+图像）",  0.25, 2.8, False, 14.0, 140.0, 1.0,  "需语音提醒"),
        ("🟠 中度（方向盘失控）", 0.16, 1.2, False, 25.0, 200.0, 0.3,  "文本信号主导"),
        ("🚨 重度（全信号）",     0.35, 3.8, True,  22.0, 210.0, 0.1,  "长途司机"),
        ("🚨 极重（凌晨3点）",   0.42, 4.5, True,  28.0, 260.0, 0.05, "×1.6 时间乘数"),
    ]

    tbl = Table(box=box.ROUNDED, expand=True, show_lines=True)
    tbl.add_column("场景",         style="bold", width=20)
    tbl.add_column("PERCLOS",      justify="center", width=9)
    tbl.add_column("Yawn/min",     justify="center", width=9)
    tbl.add_column("Verbal",       justify="center", width=7)
    tbl.add_column("Steer°",       justify="center", width=7)
    tbl.add_column("Brake Δms",    justify="center", width=9)
    tbl.add_column("Text",         justify="center", width=6)
    tbl.add_column("Image",        justify="center", width=6)
    tbl.add_column("Audio",        justify="center", width=6)
    tbl.add_column("Composite",    width=26)
    tbl.add_column("判级",          width=10)
    tbl.add_column("备注",          style="dim", width=18)

    for label, p, y, v, st, br, sv, note in cases:
        ts, im, au, comp, tier = compute_score(p, y, v, st, br, sv)
        color = TIER_COLOR[tier]
        bar_len = int(comp * 14)
        bar = Text("█" * bar_len + "░" * (14 - bar_len) + f" {comp:.2f}", style=color)
        tier_txt = Text(tier.name, style=f"bold {color}")
        tbl.add_row(
            label,
            f"{p:.2f}", f"{y:.1f}", "是" if v else "否",
            f"{st:.0f}", f"{br:.0f}",
            f"{ts:.2f}", f"{im:.2f}", f"{au:.2f}",
            bar, tier_txt, note,
        )

    console.print(tbl)
    console.print()

    # Cross-validation highlight
    console.print(Panel(
        "[bold]关键设计：干眼误报防护[/bold]\n\n"
        "行第4条「干眼误报」：PERCLOS=0.22（超标）但哈欠0.3/min、方向盘正常 → [green]NONE/MILD[/green]\n"
        "系统不单靠 PERCLOS 触发，必须有辅助信号（哈欠 OR 方向盘 OR 语音）才升级。\n\n"
        "[bold]时间风险乘数影响（写入 LLM Prompt，由 gpt-5 加权）[/bold]\n"
        "极重场景（第8条）× 1.6 → LLM 会在相同 composite 下倾向判更高 tier。",
        border_style="dim",
    ))


# ──────────────────────────────────────────────────────────────────────────────
# 4. 权重拆解——同一 PERCLOS 下，辅助信号的增量贡献
# ──────────────────────────────────────────────────────────────────────────────

def print_contribution():
    console.print(Rule("[bold cyan]信号增量贡献：加一个信号，分数涨多少？[/bold cyan]"))
    console.print("[dim]基准：PERCLOS=0.20，无其他信号[/dim]\n")

    BASE = dict(perclos=0.20, yawns_per_min=0.0, verbal=False,
                steering=8.0, brake_delta_ms=60.0, speed_variance=5.0)

    additions = [
        ("基准（仅 PERCLOS=0.20）",              {}),
        ("+ 哈欠 1.0/min",                      dict(yawns_per_min=1.0)),
        ("+ 哈欠 2.5/min",                      dict(yawns_per_min=2.5)),
        ("+ 语音确认「有点困」",                  dict(verbal=True)),
        ("+ 方向盘 20°（大幅修正）",              dict(steering=20.0)),
        ("+ 刹车延迟 +200ms",                   dict(brake_delta_ms=200.0)),
        ("+ 哈欠2.5 + 语音 + 方向盘20°",         dict(yawns_per_min=2.5, verbal=True, steering=20.0)),
        ("全部叠加（PERCLOS=0.35）",              dict(perclos=0.35, yawns_per_min=3.8,
                                                      verbal=True, steering=22.0, brake_delta_ms=210.0)),
    ]

    tbl = Table(box=box.SIMPLE_HEAVY, expand=True)
    tbl.add_column("信号组合",      style="bold", width=30)
    tbl.add_column("Text",          justify="right", width=6)
    tbl.add_column("Image",         justify="right", width=6)
    tbl.add_column("Audio",         justify="right", width=6)
    tbl.add_column("Composite",     width=34)
    tbl.add_column("Δ vs 基准",     justify="right", width=10)
    tbl.add_column("Tier",          width=10)

    base_comp = None
    for label, overrides in additions:
        kwargs = dict(BASE, **overrides)
        ts, im, au, comp, tier = compute_score(**kwargs)
        color = TIER_COLOR[tier]
        bar_len = int(comp * 20)
        bar = Text("█" * bar_len + "░" * (20 - bar_len) + f" {comp:.3f}", style=color)

        if base_comp is None:
            base_comp = comp
            delta_txt = Text("—", style="dim")
        else:
            delta = comp - base_comp
            sign = "+" if delta >= 0 else ""
            delta_txt = Text(f"{sign}{delta:.3f}", style="green" if delta > 0 else "dim")

        tbl.add_row(label, f"{ts:.2f}", f"{im:.2f}", f"{au:.2f}",
                    bar, delta_txt, Text(tier.name, style=f"bold {color}"))

    console.print(tbl)
    console.print()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]

    console.print()
    console.print(Panel.fit(
        "[bold white]疲劳评分量化分析[/bold white]\n"
        "[dim]公式: composite = 0.30×text + 0.50×image + 0.20×audio\n"
        "MILD ≥0.30  |  MODERATE ≥0.55  |  SEVERE ≥0.75[/dim]",
        border_style="blue",
    ))
    console.print()

    if "--grid" in args:
        print_heatmap()
    elif "--sweep" in args:
        print_sweep()
    elif "--cases" in args:
        print_cases()
    elif "--delta" in args:
        print_contribution()
    else:
        print_heatmap()
        print_sweep()
        print_cases()
        print_contribution()


if __name__ == "__main__":
    main()
