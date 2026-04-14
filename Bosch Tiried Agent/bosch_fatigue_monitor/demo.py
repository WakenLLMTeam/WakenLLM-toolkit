"""
Bosch Fatigue Monitor — Interactive Terminal Demo
Simulates two real scenarios end-to-end and renders the agent pipeline live.

Usage:
    python demo.py               # run both scenarios
    python demo.py --scenario 1  # long-haul truck driver only
    python demo.py --scenario 2  # city commuter only
"""
import asyncio
import sys
import time
from typing import Optional

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.columns import Columns

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.driver_memory import COFFEE_LOVER, DriverMemory

from models.fatigue_context import (
    FatigueContext, EnrichedFatigueContext,
    TextSignals, ImageSignals, AudioSignals, MapContext, RoadType, DriverProfile,
)
from models.judge_verdict import JudgeVerdict, SeverityTier, ModalityScore
from judge.judge_agent import FatigueJudgeAgent
from actions.screen_display_agent import ScreenDisplayAgent
from actions.voice_broadcast_agent import VoiceBroadcastAgent
from actions.video_record_agent import VideoRecordAgent
from actions.phone_push_agent import PhonePushAgent
from actions.context_action_agent import ContextActionAgent
from llm.mock_llm_client import MockLLMClient
from config import config

console = Console()

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

TIER_COLOR = {
    SeverityTier.NONE:     "green",
    SeverityTier.MILD:     "yellow",
    SeverityTier.MODERATE: "dark_orange",
    SeverityTier.SEVERE:   "red",
}
TIER_EMOJI = {
    SeverityTier.NONE:     "✅",
    SeverityTier.MILD:     "⚠️",
    SeverityTier.MODERATE: "🟠",
    SeverityTier.SEVERE:   "🚨",
}
PROFILE_LABEL = {
    DriverProfile.LONG_HAUL: "[blue]LONG_HAUL[/blue]",
    DriverProfile.COMMUTER:  "[cyan]COMMUTER[/cyan]",
    DriverProfile.UNKNOWN:   "[dim]UNKNOWN[/dim]",
}


def _score_bar(score: float, width: int = 22) -> Text:
    filled = int(score * width)
    color = (
        "green"      if score < 0.30 else
        "yellow"     if score < 0.55 else
        "dark_orange" if score < 0.75 else
        "red"
    )
    t = Text("█" * filled + "░" * (width - filled), style=color)
    t.append(f"  {score:.0%}", style=f"bold {color}")
    return t


def _fmt_duration(minutes: float) -> str:
    if minutes < 60:
        return f"{int(minutes)} min"
    h, m = int(minutes // 60), int(minutes % 60)
    return f"{h}h {m:02d}m"


def _alert(val, hi=None, lo=None) -> str:
    if lo is not None:
        return "[red]ALERT[/red]" if val < lo else "[green] OK [/green]"
    return "[red]ALERT[/red]" if val > hi else "[green] OK [/green]"


# ──────────────────────────────────────────────────────────────────────────────
# Panels
# ──────────────────────────────────────────────────────────────────────────────

def make_sensor_table(ctx: EnrichedFatigueContext) -> Table:
    fc = ctx.fatigue
    ts, im, au = fc.text_signals, fc.image_signals, fc.audio_signals
    tbl = Table(title="[bold]Sensor Readings[/bold]", box=box.ROUNDED, expand=True)
    tbl.add_column("Modality", style="bold cyan", width=10)
    tbl.add_column("Signal",   style="dim", width=22)
    tbl.add_column("Value",    justify="right", width=10)
    tbl.add_column("Status",   justify="center", width=8)

    tbl.add_row("Text",  "Speed variance",        f"{ts.speed_change_variance:.2f}",
                _alert(ts.speed_change_variance, lo=0.5))
    tbl.add_row("",      "Steering magnitude",    f"{ts.steering_correction_magnitude:.1f}°",
                _alert(ts.steering_correction_magnitude, hi=15.0))
    tbl.add_row("",      "Brake reaction Δ",      f"{ts.brake_reaction_time_delta_ms:.0f} ms",
                _alert(ts.brake_reaction_time_delta_ms, hi=150.0))
    tbl.add_row("Image", "PERCLOS",               f"{im.perclos_score:.3f}",
                _alert(im.perclos_score, hi=0.15))
    tbl.add_row("",      "Lane deviations",       f"{im.lane_deviation_count}",
                _alert(im.lane_deviation_count, hi=3))
    tbl.add_row("Audio", "Yawns/min",             f"{au.yawn_count_per_minute:.1f}",
                _alert(au.yawn_count_per_minute, hi=2.0))
    tbl.add_row("",      "Verbal fatigue",        str(au.verbal_fatigue_detected),
                "[red]YES[/red]" if au.verbal_fatigue_detected else "[green] NO [/green]")
    tbl.add_row("Trip",  "Driving duration",      _fmt_duration(fc.driving_duration_min), "")
    return tbl


def make_context_panel(ctx: EnrichedFatigueContext) -> Panel:
    m = ctx.map
    lines = [
        f"  Road type   : [bold]{m.road_type.value.upper()}[/bold]",
        f"  Rest stop   : {m.rest_spot_name or '—'}   ({m.nearest_rest_km or '?'} km)",
        f"  Coffee shop : {m.coffee_shop_name or '—'}  ({m.nearest_coffee_km or '?'} km)",
        f"  Profile     : {PROFILE_LABEL[ctx.driver_profile]}",
        f"  Time risk × : [bold]{ctx.time_risk_multiplier:.1f}x[/bold]"
        + ("  [dim](2AM night risk)[/dim]" if ctx.time_risk_multiplier >= 1.6 else ""),
    ]
    return Panel("\n".join(lines), title="[bold]Enriched Context[/bold]", box=box.ROUNDED)


def make_verdict_panel(verdict: JudgeVerdict) -> Panel:
    color = TIER_COLOR[verdict.severity_tier]
    emoji = TIER_EMOJI[verdict.severity_tier]

    tbl = Table(box=None, padding=(0, 1), expand=True, show_header=False)
    tbl.add_column("Modality", style="dim", width=8)
    tbl.add_column("Bar", width=30)
    tbl.add_column("Rationale", style="dim italic")

    for key, ms in verdict.modality_scores.items():
        rat = ms.rationale
        if len(rat) > 52:
            rat = rat[:52] + "…"
        tbl.add_row(key.capitalize(), _score_bar(ms.score), rat)

    header = Text()
    header.append(f"{emoji}  Severity: ", style="bold")
    header.append(f"{verdict.severity_tier.name}", style=f"bold {color}")
    header.append("     Composite: ", style="bold")
    header.append_text(_score_bar(verdict.composite_score))
    header.append(f"\n   Reasoning: ", style="dim")
    header.append(verdict.reasoning[:80], style="dim italic")

    from rich.console import Group
    return Panel(Group(header, Text(""), tbl),
                 title="[bold]FatigueJudgeAgent Verdict[/bold]",
                 box=box.DOUBLE_EDGE, border_style=color)


def make_actions_table(tier: SeverityTier, results: list) -> Table:
    icons = {
        "screen_display":  "🖥  screen_display ",
        "voice_broadcast": "🔊 voice_broadcast",
        "video_record":    "📹 video_record   ",
        "phone_push":      "📱 phone_push     ",
        "context_action":  "🧭 context_action ",
    }
    # Pyramid rule annotation
    pyramid = {
        "screen_display":  "MILD+",
        "voice_broadcast": "MODERATE+",
        "context_action":  "MODERATE+",
        "video_record":    "SEVERE",
        "phone_push":      "SEVERE",
    }

    tbl = Table(title="[bold]OrchestratorAgent → Action Dispatch[/bold]",
                box=box.ROUNDED, expand=True)
    tbl.add_column("Agent",      style="bold", width=22)
    tbl.add_column("Threshold",  style="dim", width=11, justify="center")
    tbl.add_column("Fired?",     justify="center", width=7)
    tbl.add_column("Output")

    for name, result in results:
        label = icons.get(name, name)
        thresh = pyramid.get(name, "")
        if result is not None:
            fired = "[bold green]  ✓  [/bold green]"
            msg = result.message.replace("\n", " ")
            if len(msg) > 88:
                msg = msg[:88] + "…"
            out = Text(msg)
        else:
            fired = "[dim]  —  [/dim]"
            out = Text("(cooldown / tier too low)", style="dim")
        tbl.add_row(label, thresh, fired, out)
    return tbl


# ──────────────────────────────────────────────────────────────────────────────
# Capturing orchestrator
# ──────────────────────────────────────────────────────────────────────────────

class _Demo:
    def __init__(self):
        self.judge   = FatigueJudgeAgent(MockLLMClient())
        self.screen  = ScreenDisplayAgent()
        self.voice   = VoiceBroadcastAgent()
        self.video   = VideoRecordAgent()
        self.push    = PhonePushAgent("http://localhost:8080/push")
        self.context = ContextActionAgent()

    async def run_step(self, label: str, ctx: EnrichedFatigueContext,
                       injected_verdict: Optional[JudgeVerdict] = None):
        console.print(Rule(f"[bold]  {label}  [/bold]", style="cyan"))
        await asyncio.sleep(0.2)

        # ── 1. Sensors + Context side-by-side ──
        with Progress(SpinnerColumn(), TextColumn("[cyan]Aggregating sensor signals…"),
                      transient=True) as p:
            p.add_task("", total=None)
            await asyncio.sleep(0.7)
        console.print(Columns([make_sensor_table(ctx), make_context_panel(ctx)],
                               equal=False, expand=True))

        # ── 2. Judge ──
        with Progress(SpinnerColumn(),
                      TextColumn("[yellow]FatigueJudgeAgent  ·  gpt-5 multimodal reasoning…"),
                      transient=True) as p:
            p.add_task("", total=None)
            verdict = injected_verdict or await self.judge.evaluate(ctx)
            await asyncio.sleep(0.8)
        console.print(make_verdict_panel(verdict))

        # ── 3. Orchestrator dispatch ──
        with Progress(SpinnerColumn(),
                      TextColumn("[green]OrchestratorAgent  ·  pyramid dispatch…"),
                      transient=True) as p:
            p.add_task("", total=None)
            results = await self._dispatch(verdict, ctx)
            await asyncio.sleep(0.5)
        console.print(make_actions_table(verdict.severity_tier, results))
        console.print()

    async def _dispatch(self, verdict, ctx):
        results = []
        tier = verdict.severity_tier

        results.append(("screen_display",
                         await self.screen.execute(verdict, ctx) if tier != SeverityTier.NONE else None))

        results.append(("voice_broadcast",
                         await self.voice.execute(verdict, ctx)
                         if tier in (SeverityTier.MODERATE, SeverityTier.SEVERE) else None))

        # context_action fires at MODERATE+ (memory personalisation is most useful here)
        results.append(("context_action",
                         await self.context.execute(verdict, ctx)
                         if tier in (SeverityTier.MODERATE, SeverityTier.SEVERE) else None))

        if tier == SeverityTier.SEVERE:
            results.append(("video_record", await self.video.execute(verdict, ctx)))
            results.append(("phone_push",   await self.push.execute(verdict, ctx)))
        else:
            results.append(("video_record", None))
            results.append(("phone_push",   None))
        return results


# ──────────────────────────────────────────────────────────────────────────────
# Pre-built SEVERE verdict (what gpt-5 returns for extreme signals)
# ──────────────────────────────────────────────────────────────────────────────

SEVERE_VERDICT = JudgeVerdict(
    composite_score=0.87,
    severity_tier=SeverityTier.SEVERE,
    modality_scores={
        "text":  ModalityScore("text",  0.78,
                               "Near-zero speed variance; massive steering corrections; "
                               "brake delay 220ms above baseline.", []),
        "image": ModalityScore("image", 0.95,
                               "PERCLOS=0.42 far exceeds 0.12 long-haul threshold; "
                               "5 lane deviations/min.", []),
        "audio": ModalityScore("audio", 0.82,
                               "4.2 yawns/min + verbal 「好困啊撑不住了」 confirmed.", []),
    },
    reasoning=(
        "All three modalities indicate SEVERE fatigue. "
        "Night driving (02:00) × time-risk 1.6. "
        "200 min continuous — immediate intervention required."
    ),
    context_tags=["highway"],
)

MODERATE_COMMUTER_VERDICT = JudgeVerdict(
    composite_score=0.61,
    severity_tier=SeverityTier.MODERATE,
    modality_scores={
        "text":  ModalityScore("text",  0.55,
                               "Steering correction 18° and brake Δ 180ms — both above threshold.", []),
        "image": ModalityScore("image", 0.72,
                               "PERCLOS=0.28, above commuter threshold 0.20; "
                               "3 lane deviations.", []),
        "audio": ModalityScore("audio", 0.48,
                               "2.8 yawns/min + verbal 「有点困」 detected.", []),
    },
    reasoning=(
        "MODERATE fatigue for COMMUTER profile. "
        "Multiple modalities corroborate — not a dry-eye false positive. "
        "Gentle city-mode intervention."
    ),
    context_tags=["city"],
)


# ──────────────────────────────────────────────────────────────────────────────
# Scenarios
# ──────────────────────────────────────────────────────────────────────────────

def _make_highway_ctx(perclos, yawns, steering, brake, duration, verbal=False,
                      rest_km=12.5):
    return EnrichedFatigueContext(
        fatigue=FatigueContext(
            text_signals=TextSignals(
                speed_change_variance=0.08 if steering > 18 else 0.3,
                steering_correction_magnitude=steering,
                brake_reaction_time_delta_ms=brake,
            ),
            image_signals=ImageSignals(perclos_score=perclos, lane_deviation_count=5 if perclos > 0.35 else 2,
                                       face_frame_b64="MOCK_JPEG"),
            audio_signals=AudioSignals(yawn_count_per_minute=yawns, verbal_fatigue_detected=verbal,
                                       transcript_snippet="好困啊撑不住了" if verbal else ""),
            window_seconds=60.0, timestamp=time.time(), driving_duration_min=duration,
        ),
        map=MapContext(road_type=RoadType.HIGHWAY, nearest_rest_km=rest_km,
                       nearest_coffee_km=8.0, rest_spot_name="G2 高速服务区",
                       coffee_shop_name="星巴克"),
        driver_profile=DriverProfile.LONG_HAUL,
        time_risk_multiplier=1.6,
    )


def _make_city_ctx(perclos, yawns, steering, brake, duration, verbal=False,
                   coffee_km=1.2, traffic=0.5, memory=None):
    return EnrichedFatigueContext(
        fatigue=FatigueContext(
            text_signals=TextSignals(
                speed_change_variance=0.2 if steering > 15 else 5.0,
                steering_correction_magnitude=steering,
                brake_reaction_time_delta_ms=brake,
            ),
            image_signals=ImageSignals(perclos_score=perclos, lane_deviation_count=3 if perclos > 0.25 else 1),
            audio_signals=AudioSignals(yawn_count_per_minute=yawns, verbal_fatigue_detected=verbal,
                                       transcript_snippet="有点困" if verbal else ""),
            window_seconds=60.0, timestamp=time.time(), driving_duration_min=duration,
        ),
        map=MapContext(road_type=RoadType.CITY, nearest_rest_km=5.0,
                       nearest_coffee_km=coffee_km, rest_spot_name="停车场",
                       coffee_shop_name="星巴克", traffic_density=traffic),
        driver_profile=DriverProfile.COMMUTER,
        time_risk_multiplier=1.0,
        driver_memory=memory,
    )


SCENARIOS = [
    {
        "id": 1,
        "title": "Scenario 1 — 长途货车司机  |  G2 京沪高速  |  凌晨 02:00",
        "subtitle": (
            "货运驾驶员已连续行驶 3 小时以上，时间风险系数 1.6×。"
            "系统在疲劳初现时温和提醒，疲劳加重后全链路拦截。"
        ),
        "steps": [
            {
                "label": "Step 1 · MILD  —  初现迹象（PERCLOS 微升）",
                "ctx": _make_highway_ctx(perclos=0.18, yawns=1.5, steering=12.0,
                                          brake=100.0, duration=190.0, rest_km=25.0),
                "verdict": None,   # let mock judge run
            },
            {
                "label": "Step 2 · MODERATE  —  疲劳加重（PERCLOS 0.35 + 打哈欠）",
                "ctx": _make_highway_ctx(perclos=0.35, yawns=3.8, steering=20.0,
                                          brake=200.0, duration=200.0, verbal=True, rest_km=12.5),
                "verdict": None,
            },
            {
                "label": "Step 3 · SEVERE  —  极度疲劳  🚨  全链路触发",
                "ctx": _make_highway_ctx(perclos=0.42, yawns=4.2, steering=28.0,
                                          brake=240.0, duration=210.0, verbal=True, rest_km=7.0),
                "verdict": SEVERE_VERDICT,   # inject real-looking gpt-5 verdict
            },
        ],
    },
    {
        "id": 2,
        "title": "Scenario 2 — 城市通勤者  |  市区道路  |  早高峰",
        "subtitle": (
            "上班途中轻度眼疲劳，高误报风险（干眼 / 晨间眯眼）。"
            "COMMUTER 模式：更宽松阈值 + 温和语气，不打断驾驶节奏。"
        ),
        "steps": [
            {
                "label": "Step 1 · MILD  —  疑似干眼（PERCLOS 仅轻微超标）",
                "ctx": _make_city_ctx(perclos=0.22, yawns=1.0, steering=8.0,
                                       brake=80.0, duration=25.0),
                "verdict": None,
            },
            {
                "label": "Step 2 · MODERATE  —  多模态确认，非误报",
                "ctx": _make_city_ctx(perclos=0.28, yawns=2.8, steering=18.0,
                                       brake=180.0, duration=35.0, verbal=True),
                "verdict": MODERATE_COMMUTER_VERDICT,
            },
        ],
    },
    {
        "id": 3,
        "title": "Scenario 3 — 驾驶员记忆  |  个性化 Action  |  咖啡偏好 & 低流量停车",
        "subtitle": (
            "系统预存司机偏好（喜欢喝咖啡）。"
            "疲劳触发时不只是「建议」——直接自动下单最近的星巴克；"
            "路段车流稀少时提示安全停车位置，而非泛泛叫停。"
        ),
        "steps": [
            {
                "label": "Step 1 · 记忆偏好  —  附近星巴克 1.2km，自动下单美式",
                "ctx": _make_city_ctx(perclos=0.28, yawns=2.8, steering=18.0,
                                       brake=180.0, duration=35.0, verbal=True,
                                       coffee_km=1.2, traffic=0.5,
                                       memory=COFFEE_LOVER),
                "verdict": MODERATE_COMMUTER_VERDICT,
            },
            {
                "label": "Step 2 · 低车流  —  周边无车，直接建议路边停靠",
                "ctx": _make_city_ctx(perclos=0.28, yawns=2.8, steering=18.0,
                                       brake=180.0, duration=35.0, verbal=True,
                                       coffee_km=3.5,   # too far
                                       traffic=0.12,    # nearly empty road
                                       memory=DriverMemory(likes_coffee=False,
                                                           ok_to_pull_over_city=True)),
                "verdict": MODERATE_COMMUTER_VERDICT,
            },
        ],
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

async def run_demo(scenario_ids: list):
    console.print()
    console.print(Panel.fit(
        "[bold white]Bosch Fatigue Monitor[/bold white]   [dim]Agent Pipeline Demo[/dim]\n"
        "[dim]Sensors → StateAggregator → ContextEnricher"
        " → FatigueJudgeAgent (gpt-5) → OrchestratorAgent → 5× ActionAgents[/dim]",
        border_style="blue",
    ))
    console.print()

    demo = _Demo()

    for sid in scenario_ids:
        sc = next(s for s in SCENARIOS if s["id"] == sid)

        console.print(Panel(
            f"[bold]{sc['title']}[/bold]\n[dim]{sc['subtitle']}[/dim]",
            border_style="magenta", box=box.HEAVY, padding=(1, 2),
        ))
        console.print()

        for step in sc["steps"]:
            await demo.run_step(step["label"], step["ctx"], step.get("verdict"))
            await asyncio.sleep(0.3)

        console.print(Rule(style="magenta"))
        console.print()

    # ── Summary ──
    summary = Table(title="[bold]系统能力总结[/bold]", box=box.ROUNDED, expand=True)
    summary.add_column("模块",       style="bold cyan",   width=22)
    summary.add_column("实现要点",   style="",            width=38)
    summary.add_column("对应产品需求", style="dim italic", width=28)
    rows = [
        ("DriverProfile分类",    "行驶时长 + 路型 + 时段 → LONG_HAUL / COMMUTER",  "精准触达，不一刀切"),
        ("PERCLOS交叉验证",      "单纯高 PERCLOS 不触发，需辅助信号",             "避免干眼误报"),
        ("时间风险乘数",          "2-5AM × 1.6、13-15 × 1.3，写入 LLM 提示词",  "凌晨疲劳风险更高"),
        ("行驶时长里程碑",        "2h / 4h / 6h 独立触发 Judge（不依赖传感器）", "超时自动兜底"),
        ("金字塔式 Action 派发",  "MILD→屏显 MODERATE→+语音 SEVERE→全链路",    "渐进式干预"),
        ("自适应冷却时间",        "连续忽略 → 缩短间隔 → 升级等级",             "拒绝无效重复告警"),
        ("画像感知 Action 内容",  "长途：ETA+时长+收益框架；通勤：温和+咖啡",   "内容匹配使用场景"),
        ("Bosch gpt-5 多模态",   "文本+PERCLOS图像→一次 LLM 推理，结构化输出", "多模态判断"),
    ]
    for r in rows:
        summary.add_row(*r)
    console.print(summary)
    console.print()
    console.print(Panel.fit(
        "[bold green]Demo complete.[/bold green]\n\n"
        "[dim]  python demo.py --scenario 1[/dim]  仅跑长途场景\n"
        "[dim]  python main.py             [/dim]  实时 pipeline（30s 后疲劳触发）\n"
        "[dim]  python main.py --bosch     [/dim]  接 Bosch gpt-5 真实多模态推理",
        title="[bold]Run commands[/bold]", border_style="green",
    ))


# ──────────────────────────────────────────────────────────────────────────────
# Decision-flow visualization
# ──────────────────────────────────────────────────────────────────────────────

def print_decision_flow():
    from rich.tree import Tree

    console.print()
    console.print(Panel.fit(
        "[bold white]Agent 决策流程[/bold white]",
        border_style="blue",
    ))
    console.print()

    # ── Pipeline tree ──
    pipeline = Tree("🚗  [bold cyan]驾驶数据输入[/bold cyan]")
    sensors = pipeline.add("[bold]传感器层[/bold]  (每 0.5–2 s 采集)")
    sensors.add("📷  Image Sensor  →  PERCLOS 眼闭合率 + 车道偏移")
    sensors.add("📡  Text Sensor   →  速度方差 + 方向盘幅度 + 刹车反应")
    sensors.add("🎤  Audio Sensor  →  哈欠频率 + 语音疲劳词")
    sensors.add("⏱  Duration Sensor →  连续行驶时长（每 60 s 推送）")

    agg = pipeline.add("[bold]StateAggregator[/bold]  滚动窗口 60 s")
    agg.add("PERCLOS 交叉验证：单纯高 PERCLOS ≠ 疲劳，需辅助信号")
    agg.add("时长里程碑：2h / 4h / 6h 自动触发 Judge（独立于传感器）")

    enrich = pipeline.add("[bold]ContextEnricher[/bold]  地图 + 画像")
    enrich.add("🗺  MapClient → 最近服务区 / 星巴克距离")
    enrich.add("👤  DriverProfile → LONG_HAUL | COMMUTER")
    enrich.add("🌙  时间风险乘数 → 2-5AM × 1.6 | 13-15 × 1.3")
    enrich.add("🧠  DriverMemory → 咖啡偏好 / 停车偏好 / 历史记录")

    judge = pipeline.add("[bold yellow]FatigueJudgeAgent[/bold yellow]  gpt-5 多模态")
    judge.add("输入：文本信号 + PERCLOS 图像帧 + 音频转录")
    judge.add("输出：text/image/audio 各模态评分 + composite + SeverityTier")
    tiers = judge.add("判级：")
    tiers.add("[green]NONE[/green]  composite < 0.30")
    tiers.add("[yellow]MILD[/yellow]  0.30–0.55")
    tiers.add("[dark_orange]MODERATE[/dark_orange]  0.55–0.75")
    tiers.add("[red]SEVERE[/red]  ≥ 0.75")

    orch = pipeline.add("[bold green]OrchestratorAgent[/bold green]  自适应冷却 + 金字塔派发")
    orch.add("冷却检查：连续忽略 → 间隔缩短 80%/60%/40% → 等级升级")
    pyramid = orch.add("金字塔 Action：")
    pyramid.add("[yellow]MILD+[/yellow]   → 🖥  ScreenDisplay（中文疲劳等级 + 行驶时长）")
    pyramid.add("[dark_orange]MODERATE+[/dark_orange] → 🔊 VoiceBroadcast + 🧭 ContextAction")
    pyramid.add("[red]SEVERE[/red]    → 📹 VideoRecord + 📱 PhonePush")

    console.print(pipeline)
    console.print()

    # ── ContextAction decision table ──
    console.print(Rule("[bold]ContextAction 决策树[/bold]", style="cyan"))
    console.print()

    ctx_tree = Tree("🧭  [bold]ContextActionAgent[/bold]")
    lh = ctx_tree.add("[blue]LONG_HAUL[/blue] 画像")
    lh.add("高速 + 服务区已知  →  导航到服务区（含 ETA + 已行驶时长 + 收益框架）")
    lh.add("其他              →  通用「寻找安全停靠点」")

    cm = ctx_tree.add("[cyan]COMMUTER[/cyan] 画像")
    p1 = cm.add("Path 1 · [bold]DriverMemory.likes_coffee = True[/bold]")
    p1.add("星巴克 ≤ coffee_max_km  →  [bold green]☕ 自动下单[/bold green]（preferred_coffee_order）")
    p1.add("星巴克 > coffee_max_km  →  跳到 Path 3")
    p2 = cm.add("Path 2 · likes_coffee = False  &  [bold]traffic_density < 0.30[/bold]")
    p2.add("城市路 + 周边无车  →  [bold]🅿 建议路边停靠[/bold]（红灯深呼吸）")
    p3 = cm.add("Path 3 · 兜底")
    p3.add("有咖啡店  →  建议前往（不自动下单）")
    p3.add("无咖啡店  →  泛泛「安全地点停车」")

    console.print(ctx_tree)
    console.print()

    # ── Memory schema ──
    console.print(Rule("[bold]DriverMemory 字段[/bold]", style="cyan"))
    mem_tbl = Table(box=box.SIMPLE, expand=False)
    mem_tbl.add_column("字段", style="bold cyan", width=26)
    mem_tbl.add_column("类型", style="dim", width=8)
    mem_tbl.add_column("作用")
    rows = [
        ("likes_coffee",          "bool",  "触发自动点咖啡逻辑"),
        ("preferred_coffee_order","str",   "下单的具体品类，如「美式（大杯）」"),
        ("coffee_max_km",         "float", "超过此距离不自动下单，改为建议"),
        ("ok_to_pull_over_city",  "bool",  "是否允许「路边停靠」建议"),
        ("prefers_nap",           "bool",  "重度疲劳时优先推荐小憩而非咖啡"),
        ("voice_enabled",         "bool",  "是否允许语音播报"),
        ("push_enabled",          "bool",  "是否允许手机推送"),
        ("ignored_alert_streak",  "int",   "连续忽略告警次数（自适应冷却输入）"),
        ("last_rest_stop",        "str?",  "上次使用的服务区名（避免重复推荐）"),
    ]
    for r in rows:
        mem_tbl.add_row(*r)
    console.print(mem_tbl)
    console.print()


def _parse_args():
    args = sys.argv[1:]
    if "--flow" in args:
        return None   # signal: print flow only
    if "--scenario" in args:
        idx = args.index("--scenario")
        return [int(args[idx + 1])]
    return [1, 2, 3]


if __name__ == "__main__":
    ids = _parse_args()
    if ids is None:
        print_decision_flow()
    else:
        asyncio.run(run_demo(ids))
