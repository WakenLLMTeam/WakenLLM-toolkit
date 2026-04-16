"""
交互式疲劳检测输入工具
自己输入传感器数值 → 实时看 agent 决策

python interactive.py
"""
import asyncio, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich.console import Console
from rich.prompt import Prompt, Confirm, FloatPrompt, IntPrompt
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
from rich.rule import Rule

from models.fatigue_context import (
    FatigueContext, EnrichedFatigueContext,
    TextSignals, ImageSignals, AudioSignals, MapContext, RoadType, DriverProfile,
)
from models.judge_verdict import JudgeVerdict, SeverityTier, ModalityScore
from models.driver_memory import DriverMemory
from actions.screen_display_agent import ScreenDisplayAgent
from actions.voice_broadcast_agent import VoiceBroadcastAgent
from actions.video_record_agent import VideoRecordAgent
from actions.phone_push_agent import PhonePushAgent
from actions.context_action_agent import ContextActionAgent

console = Console()

TC = {
    SeverityTier.NONE:     "green",
    SeverityTier.MILD:     "yellow",
    SeverityTier.MODERATE: "dark_orange",
    SeverityTier.SEVERE:   "red",
}

def _tier(score):
    if score >= 0.75: return SeverityTier.SEVERE
    if score >= 0.55: return SeverityTier.MODERATE
    if score >= 0.30: return SeverityTier.MILD
    return SeverityTier.NONE

def _bar(score, w=20):
    n = int(score * w)
    color = TC[_tier(score)]
    t = Text("█" * n + "░" * (w - n), style=color)
    t.append(f"  {score:.2f}", style=f"bold {color}")
    return t

def _ask(prompt, default, lo=0.0, hi=None, type_=float):
    while True:
        try:
            raw = Prompt.ask(f"  [cyan]{prompt}[/cyan]", default=str(default))
            val = type_(raw)
            if hi is not None and val > hi:
                console.print(f"    [red]请输入 ≤ {hi}[/red]")
                continue
            if val < lo:
                console.print(f"    [red]请输入 ≥ {lo}[/red]")
                continue
            return val
        except ValueError:
            console.print("    [red]请输入数字[/red]")

def _ask_choice(prompt, options):
    """options: list of (key, label)"""
    for i, (k, label) in enumerate(options, 1):
        console.print(f"    [dim]{i}.[/dim] {label}")
    while True:
        raw = Prompt.ask(f"  [cyan]{prompt}[/cyan]", default="1")
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx][0]
        except ValueError:
            pass
        console.print("    [red]请输入序号[/red]")


def collect_inputs():
    console.print()
    console.print(Panel.fit(
        "[bold white]Bosch 疲劳检测 — 交互式输入[/bold white]\n"
        "[dim]输入传感器数值，实时查看 agent 决策[/dim]",
        border_style="blue"
    ))

    # ── 1. 驾驶员画像 ──
    console.print()
    console.print(Rule("[bold]1. 驾驶员类型[/bold]"))
    profile = _ask_choice("选择画像", [
        (DriverProfile.LONG_HAUL, "长途司机（货车/客运，高速为主）"),
        (DriverProfile.COMMUTER,  "城市通勤（上下班，市区为主）"),
    ])

    # ── 2. 路况 ──
    console.print()
    console.print(Rule("[bold]2. 路况[/bold]"))
    road = _ask_choice("选择路型", [
        (RoadType.HIGHWAY, "高速公路"),
        (RoadType.CITY,    "城市道路"),
    ])

    # ── 3. 行驶时长 ──
    console.print()
    console.print(Rule("[bold]3. 行驶时长[/bold]"))
    console.print("  [dim]提示：≥120min 触发长途里程碑，≥200min 夜间风险×1.6[/dim]")
    duration = _ask("连续行驶时长（分钟）", default=30, lo=0, hi=600)

    # ── 4. 图像信号 ──
    console.print()
    console.print(Rule("[bold]4. 图像信号（眼部）[/bold]"))
    console.print("  [dim]PERCLOS：0.05=正常，0.15=临界，0.30+=疲劳，0.42+=极重[/dim]")
    perclos = _ask("PERCLOS 眼闭合率（0.0–1.0）", default=0.10, lo=0.0, hi=1.0)
    lane_dev = _ask("车道偏移次数（次/分钟）", default=0, lo=0, hi=20, type_=int)

    # ── 5. 文本/驾驶行为信号 ──
    console.print()
    console.print(Rule("[bold]5. 驾驶行为信号[/bold]"))
    console.print("  [dim]方向盘：≤15°正常，20°=轻度，>25°=明显修正[/dim]")
    steering = _ask("方向盘修正幅度（度°）", default=8.0, lo=0.0, hi=60.0)
    console.print("  [dim]刹车延迟：≤150ms正常，200ms=轻度，>300ms=明显[/dim]")
    brake = _ask("刹车反应延迟 Δms", default=60.0, lo=0.0, hi=500.0)
    console.print("  [dim]速度方差：>1.0正常，0.3=偏低，<0.1=极单调（疲劳）[/dim]")
    speed_var = _ask("速度方差（0.0–10.0）", default=5.0, lo=0.0, hi=10.0)

    # ── 6. 音频信号 ──
    console.print()
    console.print(Rule("[bold]6. 音频信号[/bold]"))
    console.print("  [dim]哈欠：≤1.0正常，2.0=临界，>3.0=明显疲劳[/dim]")
    yawns = _ask("哈欠频率（次/分钟）", default=0.5, lo=0.0, hi=10.0)
    verbal = Confirm.ask("  [cyan]是否检测到语音疲劳词（「困了」「撑不住」等）？[/cyan]", default=False)

    # ── 7. 地图信息 ──
    console.print()
    console.print(Rule("[bold]7. 周边地图[/bold]"))
    rest_km   = _ask("最近服务区/停车场距离（km）", default=12.5, lo=0.0, hi=100.0)
    coffee_km = _ask("最近咖啡店距离（km）", default=2.0, lo=0.0, hi=20.0)
    traffic   = _ask("当前路段车流密度（0.0=空旷 ~ 1.0=拥堵）", default=0.5, lo=0.0, hi=1.0)

    # ── 8. 驾驶员记忆偏好 ──
    console.print()
    console.print(Rule("[bold]8. 驾驶员记忆偏好[/bold]"))
    likes_coffee = Confirm.ask("  [cyan]司机喜欢喝咖啡（附近有咖啡店时自动下单）？[/cyan]", default=False)
    coffee_order = "美式（大杯）"
    coffee_max   = 2.0
    if likes_coffee:
        console.print("  [dim]示例：美式（大杯）/ 拿铁 / 卡布奇诺[/dim]")
        coffee_order = Prompt.ask("  [cyan]惯常点什么？[/cyan]", default="美式（大杯）")
        coffee_max   = _ask("最远自动下单距离（km）", default=2.0, lo=0.1, hi=10.0)
    pull_over = Confirm.ask("  [cyan]允许「路边安全停车」建议？[/cyan]", default=True)

    return dict(
        profile=profile, road=road, duration=duration,
        perclos=perclos, lane_dev=lane_dev,
        steering=steering, brake=brake, speed_var=speed_var,
        yawns=yawns, verbal=verbal,
        rest_km=rest_km, coffee_km=coffee_km, traffic=traffic,
        likes_coffee=likes_coffee, coffee_order=coffee_order,
        coffee_max=coffee_max, pull_over=pull_over,
    )


def compute_score(p):
    # Image
    image_score = min(1.0, p["perclos"] / 0.15 * 0.6)

    # Text
    st = min(1.0, max(0.0, (p["steering"] - 15.0) / 20.0))
    br = min(1.0, max(0.0, (p["brake"] - 150.0) / 250.0))
    sv = min(1.0, max(0.0, (0.5 - p["speed_var"]) / 0.5)) if p["speed_var"] < 0.5 else 0.0
    text_score = min(1.0, (st + br + sv) / 3 * 1.5 + 0.10)

    # Audio
    yawn_score  = min(1.0, p["yawns"] / 2.0 * 0.6)
    audio_score = min(1.0, yawn_score + (0.30 if p["verbal"] else 0.0))

    composite = round(0.30 * text_score + 0.50 * image_score + 0.20 * audio_score, 3)
    return round(text_score, 3), round(image_score, 3), round(audio_score, 3), composite


def build_verdict_and_ctx(p, text_s, image_s, audio_s, composite):
    tier = _tier(composite)

    verdict = JudgeVerdict(
        composite_score=composite,
        severity_tier=tier,
        modality_scores={
            "text":  ModalityScore("text",  text_s,  f"方向盘{p['steering']:.0f}°, 刹车Δ{p['brake']:.0f}ms, 速度方差{p['speed_var']:.2f}", []),
            "image": ModalityScore("image", image_s, f"PERCLOS={p['perclos']:.3f}, 车道偏移{p['lane_dev']}次", []),
            "audio": ModalityScore("audio", audio_s, f"哈欠{p['yawns']:.1f}/min, 语音={'是' if p['verbal'] else '否'}", []),
        },
        reasoning=f"composite={composite:.3f} → {tier.name}",
        context_tags=[p["road"].value],
    )

    memory = DriverMemory(
        likes_coffee=p["likes_coffee"],
        preferred_coffee_order=p["coffee_order"],
        coffee_max_km=p["coffee_max"],
        ok_to_pull_over_city=p["pull_over"],
    ) if (p["likes_coffee"] or p["pull_over"]) else None

    ctx = EnrichedFatigueContext(
        fatigue=FatigueContext(
            text_signals=TextSignals(p["speed_var"], p["steering"], p["brake"]),
            image_signals=ImageSignals(p["perclos"], p["lane_dev"]),
            audio_signals=AudioSignals(p["yawns"], p["verbal"],
                                       "语音疲劳词已检测" if p["verbal"] else None),
            window_seconds=60.0,
            timestamp=time.time(),
            driving_duration_min=p["duration"],
        ),
        map=MapContext(
            road_type=p["road"],
            nearest_rest_km=p["rest_km"],
            nearest_coffee_km=p["coffee_km"],
            rest_spot_name="G2 高速服务区" if p["road"] == RoadType.HIGHWAY else "附近停车场",
            coffee_shop_name="星巴克",
            traffic_density=p["traffic"],
        ),
        driver_profile=p["profile"],
        time_risk_multiplier=1.6 if p["duration"] >= 120 else 1.0,
        driver_memory=memory,
    )
    return verdict, ctx


async def run_agents(verdict, ctx):
    import io
    from contextlib import redirect_stdout
    import logging

    screen  = ScreenDisplayAgent()
    voice   = VoiceBroadcastAgent()
    video   = VideoRecordAgent()
    push    = PhonePushAgent("http://localhost:8080/push")
    ctx_act = ContextActionAgent()

    tier = verdict.severity_tier
    with redirect_stdout(io.StringIO()):
        logging.disable(logging.CRITICAL)
        scr_r = await screen.execute(verdict, ctx)  if tier != SeverityTier.NONE else None
        voi_r = await voice.execute(verdict, ctx)   if tier in (SeverityTier.MODERATE, SeverityTier.SEVERE) else None
        ctx_r = await ctx_act.execute(verdict, ctx) if tier in (SeverityTier.MODERATE, SeverityTier.SEVERE) else None
        vid_r = await video.execute(verdict, ctx)   if tier == SeverityTier.SEVERE else None
        psh_r = await push.execute(verdict, ctx)    if tier == SeverityTier.SEVERE else None
        logging.disable(logging.NOTSET)

    return scr_r, voi_r, ctx_r, vid_r, psh_r


def render_result(p, text_s, image_s, audio_s, composite, verdict, ctx, agents):
    scr_r, voi_r, ctx_r, vid_r, psh_r = agents
    tier = verdict.severity_tier
    color = TC[tier]

    console.print()
    console.print(Rule("[bold]评分结果[/bold]", style=color))

    # ── Score breakdown ──
    score_tbl = Table(box=box.SIMPLE, expand=False, show_header=False)
    score_tbl.add_column("label", style="dim", width=12)
    score_tbl.add_column("bar",   width=28)
    score_tbl.add_column("note",  style="dim italic")

    score_tbl.add_row("Text ×0.30",  _bar(text_s,  16), f"方向盘 + 刹车 + 速度方差")
    score_tbl.add_row("Image ×0.50", _bar(image_s, 16), f"PERCLOS={p['perclos']:.3f}")
    score_tbl.add_row("Audio ×0.20", _bar(audio_s, 16), f"哈欠{p['yawns']:.1f}/min {'+ 语音' if p['verbal'] else ''}")
    score_tbl.add_row("", Text(""), "")

    comp_text = Text()
    comp_text.append("Composite  ", style="bold")
    comp_text.append_text(_bar(composite, 20))
    comp_text.append(f"   →  ", style="bold")
    comp_text.append(tier.name, style=f"bold {color}")

    console.print(score_tbl)
    console.print(comp_text)
    console.print()

    # ── Agent dispatch ──
    console.print(Rule("[bold]Agent 决策[/bold]", style=color))

    dispatch_tbl = Table(box=box.ROUNDED, expand=True)
    dispatch_tbl.add_column("Agent",       style="bold", width=20)
    dispatch_tbl.add_column("触发条件",     style="dim",  width=12, justify="center")
    dispatch_tbl.add_column("状态",         width=7,  justify="center")
    dispatch_tbl.add_column("输出")

    def row(icon, name, threshold, result):
        if result:
            status = Text("  ✓  ", style="bold green")
            msg = result.message.replace("\n", " ")
            out = Text(msg[:100] + ("…" if len(msg) > 100 else ""))
        else:
            status = Text("  —  ", style="dim")
            out = Text("未触发", style="dim")
        dispatch_tbl.add_row(f"{icon} {name}", threshold, status, out)

    row("🖥 ", "screen_display",  "MILD+",     scr_r)
    row("🔊", "voice_broadcast", "MODERATE+", voi_r)
    row("🧭", "context_action",  "MODERATE+", ctx_r)
    row("📹", "video_record",    "SEVERE",    vid_r)
    row("📱", "phone_push",      "SEVERE",    psh_r)

    console.print(dispatch_tbl)

    # ── 如果没有任何 agent 触发 ──
    if tier == SeverityTier.NONE:
        console.print()
        console.print(Panel(
            "[green bold]✓ 驾驶员状态正常，无需干预。[/green bold]\n"
            "[dim]composite < 0.30，所有 agent 静默。[/dim]",
            border_style="green"
        ))

    # ── 关键字段说明 ──
    if ctx_r and ctx_r.payload:
        console.print()
        import json
        payload_str = json.dumps(ctx_r.payload, ensure_ascii=False, indent=2)
        console.print(Panel(payload_str, title="[dim]context_action payload[/dim]",
                            border_style="dim", expand=False))


async def main():
    while True:
        try:
            p = collect_inputs()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]退出。[/dim]")
            break

        text_s, image_s, audio_s, composite = compute_score(p)
        verdict, ctx = build_verdict_and_ctx(p, text_s, image_s, audio_s, composite)
        agents = await run_agents(verdict, ctx)
        render_result(p, text_s, image_s, audio_s, composite, verdict, ctx, agents)

        console.print()
        again = Confirm.ask("[bold cyan]再试一组数据？[/bold cyan]", default=True)
        if not again:
            break

    console.print()
    console.print(Panel.fit(
        "其他命令：\n"
        "  [cyan]python demo.py[/cyan]          预设场景演示\n"
        "  [cyan]python quantify.py[/cyan]      量化分析热力图\n"
        "  [cyan]python decision_matrix.py[/cyan]  20场景决策矩阵",
        border_style="dim"
    ))


if __name__ == "__main__":
    asyncio.run(main())
