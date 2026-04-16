"""
决策矩阵：20 个不同疲劳分数 → 实际 agent 输出对比
验证：分数不同，决策真的不一样吗？

python decision_matrix.py
"""
import asyncio, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich import box
from rich.console import Console
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
from rich.columns import Columns

from models.fatigue_context import (
    FatigueContext, EnrichedFatigueContext,
    TextSignals, ImageSignals, AudioSignals, MapContext, RoadType, DriverProfile,
)
from models.judge_verdict import JudgeVerdict, SeverityTier, ModalityScore
from models.driver_memory import COFFEE_LOVER, DriverMemory
from actions.screen_display_agent import ScreenDisplayAgent
from actions.voice_broadcast_agent import VoiceBroadcastAgent
from actions.video_record_agent import VideoRecordAgent
from actions.phone_push_agent import PhonePushAgent
from actions.context_action_agent import ContextActionAgent

console = Console()

# ── Color helpers ──────────────────────────────────────────────────────────────
TC = {
    SeverityTier.NONE:     "green",
    SeverityTier.MILD:     "yellow",
    SeverityTier.MODERATE: "dark_orange",
    SeverityTier.SEVERE:   "red",
}

def _bar(score: float, w: int = 12) -> Text:
    n = int(score * w)
    color = TC[_tier(score)]
    t = Text("█" * n + "░" * (w - n), style=color)
    t.append(f" {score:.2f}", style=f"bold {color}")
    return t

def _tier(score: float) -> SeverityTier:
    if score >= 0.75: return SeverityTier.SEVERE
    if score >= 0.55: return SeverityTier.MODERATE
    if score >= 0.30: return SeverityTier.MILD
    return SeverityTier.NONE

def _tier_badge(tier: SeverityTier) -> Text:
    labels = {
        SeverityTier.NONE:     "  NONE  ",
        SeverityTier.MILD:     "  MILD  ",
        SeverityTier.MODERATE: "MODERATE",
        SeverityTier.SEVERE:   " SEVERE ",
    }
    return Text(labels[tier], style=f"bold {TC[tier]}")

def _tick(fired: bool) -> Text:
    return Text("  ✓  ", style="bold green") if fired else Text("  —  ", style="dim")


# ── Build synthetic verdict from a composite score ─────────────────────────────
def _make_verdict(composite: float, text_s: float, image_s: float, audio_s: float,
                  rationale: str = "") -> JudgeVerdict:
    tier = _tier(composite)
    return JudgeVerdict(
        composite_score=composite,
        severity_tier=tier,
        modality_scores={
            "text":  ModalityScore("text",  text_s,  f"Text={text_s:.2f}", []),
            "image": ModalityScore("image", image_s, f"PERCLOS→image={image_s:.2f}", []),
            "audio": ModalityScore("audio", audio_s, f"Audio={audio_s:.2f}", []),
        },
        reasoning=rationale or f"Composite={composite:.2f}; tier={tier.name}.",
        context_tags=["demo"],
    )


# ── Build context for each scenario ───────────────────────────────────────────
def _ctx(profile: DriverProfile, road: RoadType, duration_min: float,
         traffic: float = 0.5, coffee_km: float = 1.2,
         memory=None) -> EnrichedFatigueContext:
    return EnrichedFatigueContext(
        fatigue=FatigueContext(
            text_signals=TextSignals(0.1, 10.0, 80.0),
            image_signals=ImageSignals(0.2, 2),
            audio_signals=AudioSignals(1.0, False),
            window_seconds=60.0,
            timestamp=time.time(),
            driving_duration_min=duration_min,
        ),
        map=MapContext(
            road_type=road,
            nearest_rest_km=12.5,
            nearest_coffee_km=coffee_km,
            rest_spot_name="G2 高速服务区",
            coffee_shop_name="星巴克",
            traffic_density=traffic,
        ),
        driver_profile=profile,
        time_risk_multiplier=1.6 if duration_min > 120 else 1.0,
        driver_memory=memory,
    )


# ── 20 test cases ──────────────────────────────────────────────────────────────
CASES = [
    # id  label                           comp   txt   img   aud   profile          road          dur    traffic  coffee_km  memory
    ( 1, "完全清醒",                      0.05, 0.05, 0.08, 0.02, DriverProfile.COMMUTER,  RoadType.CITY,    10,  0.6, 1.2, None),
    ( 2, "轻微眨眼（干眼候选）",           0.18, 0.08, 0.28, 0.05, DriverProfile.COMMUTER,  RoadType.CITY,    15,  0.5, 1.2, None),
    ( 3, "临界 MILD（单纯PERCLOS）",       0.30, 0.10, 0.50, 0.05, DriverProfile.COMMUTER,  RoadType.CITY,    20,  0.5, 1.2, None),
    ( 4, "MILD—通勤无记忆",               0.38, 0.12, 0.62, 0.10, DriverProfile.COMMUTER,  RoadType.CITY,    25,  0.5, 1.2, None),
    ( 5, "MILD—通勤+咖啡偏好",            0.38, 0.12, 0.62, 0.10, DriverProfile.COMMUTER,  RoadType.CITY,    25,  0.5, 1.2, COFFEE_LOVER),
    ( 6, "MILD—低车流可停车",             0.38, 0.12, 0.62, 0.10, DriverProfile.COMMUTER,  RoadType.CITY,    25,  0.12,3.5, DriverMemory(likes_coffee=False, ok_to_pull_over_city=True)),
    ( 7, "MILD—长途高速早期",             0.42, 0.15, 0.68, 0.12, DriverProfile.LONG_HAUL, RoadType.HIGHWAY, 70,  0.5, 8.0, None),
    ( 8, "临界MODERATE（PERCLOS+哈欠）",  0.55, 0.18, 0.80, 0.35, DriverProfile.COMMUTER,  RoadType.CITY,    30,  0.5, 1.2, None),
    ( 9, "MODERATE—通勤无记忆",           0.62, 0.22, 0.88, 0.48, DriverProfile.COMMUTER,  RoadType.CITY,    35,  0.5, 1.2, None),
    (10, "MODERATE—通勤+自动点咖啡",      0.62, 0.22, 0.88, 0.48, DriverProfile.COMMUTER,  RoadType.CITY,    35,  0.5, 1.2, COFFEE_LOVER),
    (11, "MODERATE—低车流停车",           0.62, 0.22, 0.88, 0.48, DriverProfile.COMMUTER,  RoadType.CITY,    35,  0.12,3.5, DriverMemory(likes_coffee=False, ok_to_pull_over_city=True)),
    (12, "MODERATE—长途高速",             0.65, 0.25, 0.90, 0.50, DriverProfile.LONG_HAUL, RoadType.HIGHWAY, 130, 0.5, 8.0, None),
    (13, "MODERATE—长途+2h行驶",          0.68, 0.28, 0.92, 0.52, DriverProfile.LONG_HAUL, RoadType.HIGHWAY, 200, 0.5, 8.0, None),
    (14, "临界SEVERE",                    0.75, 0.35, 1.00, 0.60, DriverProfile.LONG_HAUL, RoadType.HIGHWAY, 200, 0.5, 8.0, None),
    (15, "SEVERE—长途全链路",             0.82, 0.70, 1.00, 0.72, DriverProfile.LONG_HAUL, RoadType.HIGHWAY, 210, 0.5, 8.0, None),
    (16, "SEVERE—凌晨2点×1.6",           0.87, 0.78, 1.00, 0.82, DriverProfile.LONG_HAUL, RoadType.HIGHWAY, 220, 0.5, 8.0, None),
    (17, "SEVERE—通勤极端（罕见）",        0.80, 0.65, 1.00, 0.68, DriverProfile.COMMUTER,  RoadType.CITY,    60,  0.5, 1.2, COFFEE_LOVER),
    (18, "SEVERE—极重（全信号顶满）",      0.95, 0.90, 1.00, 0.95, DriverProfile.LONG_HAUL, RoadType.HIGHWAY, 240, 0.5, 8.0, None),
    (19, "SEVERE—未知路型兜底",           0.82, 0.70, 1.00, 0.72, DriverProfile.LONG_HAUL, RoadType.UNKNOWN, 200, 0.5, None, None),
    (20, "临界NONE→MILD边界",             0.29, 0.09, 0.45, 0.04, DriverProfile.COMMUTER,  RoadType.CITY,    10,  0.5, 1.2, None),
]


async def run_all():
    screen  = ScreenDisplayAgent()
    voice   = VoiceBroadcastAgent()
    video   = VideoRecordAgent()
    push    = PhonePushAgent("http://localhost:8080/push")
    ctx_act = ContextActionAgent()

    results = []
    import io
    from contextlib import redirect_stdout

    for row in CASES:
        cid, label, comp, ts, im, au, profile, road, dur, traffic, coffee_km, memory = row
        verdict = _make_verdict(comp, ts, im, au, label)
        ctx     = _ctx(profile, road, dur, traffic, coffee_km or 99.0, memory)

        tier = verdict.severity_tier

        # suppress print output from agents
        with redirect_stdout(io.StringIO()):
            import logging
            logging.disable(logging.CRITICAL)

            scr_r = await screen.execute(verdict, ctx) if tier != SeverityTier.NONE else None
            voi_r = await voice.execute(verdict, ctx)  if tier in (SeverityTier.MODERATE, SeverityTier.SEVERE) else None
            ctx_r = await ctx_act.execute(verdict, ctx) if tier in (SeverityTier.MODERATE, SeverityTier.SEVERE) else None
            vid_r = await video.execute(verdict, ctx)  if tier == SeverityTier.SEVERE else None
            psh_r = await push.execute(verdict, ctx)   if tier == SeverityTier.SEVERE else None

            logging.disable(logging.NOTSET)

        results.append({
            "id": cid, "label": label,
            "comp": comp, "tier": tier,
            "text_s": ts, "image_s": im, "audio_s": au,
            "profile": profile, "road": road, "dur": dur,
            "traffic": traffic, "coffee_km": coffee_km,
            "memory": memory,
            "screen":  scr_r,
            "voice":   voi_r,
            "ctx":     ctx_r,
            "video":   vid_r,
            "push":    psh_r,
        })

    return results


# ── Render ─────────────────────────────────────────────────────────────────────

def render_matrix(results):
    console.print()
    console.print(Panel.fit(
        "[bold white]决策矩阵：20 个疲劳分数 → Agent 实际输出[/bold white]\n"
        "[dim]验证：不同分数是否真的导致不同决策？[/dim]",
        border_style="blue",
    ))
    console.print()

    # ── Table 1: 触发矩阵 ──────────────────────────────────────────────────────
    console.print(Rule("[bold cyan]表1  触发矩阵（哪些 Agent 被触发）[/bold cyan]"))
    console.print()

    t1 = Table(box=box.ROUNDED, show_lines=True, expand=True)
    t1.add_column("#",            width=3,  justify="right")
    t1.add_column("场景",          width=22)
    t1.add_column("Score",         width=18)
    t1.add_column("Tier",          width=10, justify="center")
    t1.add_column("画像",           width=10, justify="center")
    t1.add_column("🖥 屏显",        width=7,  justify="center")
    t1.add_column("🔊 语音",        width=7,  justify="center")
    t1.add_column("🧭 导航/咖啡",   width=10, justify="center")
    t1.add_column("📹 录像",        width=7,  justify="center")
    t1.add_column("📱 推送",        width=7,  justify="center")

    profile_label = {
        DriverProfile.LONG_HAUL: Text("长途", style="blue"),
        DriverProfile.COMMUTER:  Text("通勤", style="cyan"),
    }

    for r in results:
        t1.add_row(
            str(r["id"]),
            r["label"],
            _bar(r["comp"]),
            _tier_badge(r["tier"]),
            profile_label.get(r["profile"], Text("?")),
            _tick(r["screen"] is not None),
            _tick(r["voice"]  is not None),
            _tick(r["ctx"]    is not None),
            _tick(r["video"]  is not None),
            _tick(r["push"]   is not None),
        )

    console.print(t1)
    console.print()

    # ── Table 2: 屏显内容对比 ──────────────────────────────────────────────────
    console.print(Rule("[bold cyan]表2  屏幕显示内容（同样「显示」，文案是否不同？）[/bold cyan]"))
    console.print()
    t2 = Table(box=box.SIMPLE_HEAVY, expand=True)
    t2.add_column("#",      width=3,  justify="right")
    t2.add_column("Tier",   width=10, justify="center")
    t2.add_column("Score",  width=6,  justify="right")
    t2.add_column("屏幕显示文案",  style="dim")

    for r in results:
        if r["screen"]:
            msg = r["screen"].message.replace("\n", " ")
            if len(msg) > 80: msg = msg[:80] + "…"
            t2.add_row(str(r["id"]), _tier_badge(r["tier"]),
                       f"{r['comp']:.2f}", msg)

    console.print(t2)
    console.print()

    # ── Table 3: 语音/导航内容对比 ────────────────────────────────────────────
    console.print(Rule("[bold cyan]表3  语音播报 & 导航/咖啡 内容（MODERATE+ 才有）[/bold cyan]"))
    console.print()
    t3 = Table(box=box.SIMPLE_HEAVY, expand=True, show_lines=True)
    t3.add_column("#",         width=3,  justify="right")
    t3.add_column("场景",       width=20)
    t3.add_column("Tier",      width=10, justify="center")
    t3.add_column("🔊 语音文案",  width=42)
    t3.add_column("🧭 导航/咖啡文案", width=42)

    for r in results:
        if r["voice"] or r["ctx"]:
            vm = r["voice"].message.replace("\n", " ") if r["voice"] else "—"
            cm = r["ctx"].message.replace("\n", " ")   if r["ctx"]   else "—"
            if len(vm) > 40: vm = vm[:40] + "…"
            if len(cm) > 40: cm = cm[:40] + "…"
            t3.add_row(str(r["id"]), r["label"], _tier_badge(r["tier"]), vm, cm)

    console.print(t3)
    console.print()

    # ── Table 4: SEVERE 全链路 ────────────────────────────────────────────────
    console.print(Rule("[bold red]表4  SEVERE 全链路（录像 + 推送 + 完整导航）[/bold red]"))
    console.print()
    t4 = Table(box=box.ROUNDED, expand=True, show_lines=True)
    t4.add_column("#",        width=3,  justify="right")
    t4.add_column("场景",      width=22)
    t4.add_column("Score",    width=6,  justify="right")
    t4.add_column("📹 录像文件",     width=22)
    t4.add_column("📱 推送内容",     width=22)
    t4.add_column("🧭 完整导航/建议",  width=38)

    for r in results:
        if r["tier"] == SeverityTier.SEVERE:
            vf = r["video"].payload.get("filename", "—") if r["video"] else "—"
            pm = r["push"].message[:20] + "…" if r["push"] and len(r["push"].message) > 20 else (r["push"].message if r["push"] else "—")
            cm = r["ctx"].message.replace("\n", " ")[:36] + "…" if r["ctx"] and len(r["ctx"].message) > 36 else (r["ctx"].message.replace("\n", " ") if r["ctx"] else "—")
            t4.add_row(
                str(r["id"]), r["label"], f"{r['comp']:.2f}",
                Text(vf, style="dim"), Text(pm, style="dark_orange bold"), Text(cm, style="red"),
            )

    console.print(t4)
    console.print()

    # ── Summary insight ────────────────────────────────────────────────────────
    none_cnt = sum(1 for r in results if r["tier"] == SeverityTier.NONE)
    mild_cnt = sum(1 for r in results if r["tier"] == SeverityTier.MILD)
    mod_cnt  = sum(1 for r in results if r["tier"] == SeverityTier.MODERATE)
    sev_cnt  = sum(1 for r in results if r["tier"] == SeverityTier.SEVERE)

    # Unique screen messages
    screen_msgs = list({r["screen"].message for r in results if r["screen"]})
    voice_msgs  = list({r["voice"].message  for r in results if r["voice"]})
    ctx_msgs    = list({r["ctx"].message    for r in results if r["ctx"]})

    console.print(Panel(
        f"[bold]20 个场景分布：[/bold]\n"
        f"  [green]NONE[/green]     {none_cnt} 个 → 0 个 agent 触发\n"
        f"  [yellow]MILD[/yellow]     {mild_cnt} 个 → 仅屏显（1 个 agent）\n"
        f"  [dark_orange]MODERATE[/dark_orange] {mod_cnt} 个 → 屏显 + 语音 + 导航/咖啡（3 个 agent）\n"
        f"  [red]SEVERE[/red]   {sev_cnt} 个 → 全部 5 个 agent\n\n"
        f"[bold]内容差异化验证：[/bold]\n"
        f"  屏幕文案  → {len(screen_msgs)} 种不同版本\n"
        f"  语音文案  → {len(voice_msgs)} 种不同版本\n"
        f"  导航/咖啡 → {len(ctx_msgs)} 种不同版本\n\n"
        f"[bold]结论：[/bold] 同一分数等级内，画像（长途/通勤）、记忆（咖啡偏好）、车流密度\n"
        f"  → 导致[bold] 导航/咖啡文案进一步分叉[/bold]，不是「分数相同就输出相同」。",
        title="[bold]汇总洞察[/bold]", border_style="blue",
    ))


import json, datetime

def save_results(results):
    output = {
        "generated_at": datetime.datetime.now().isoformat(),
        "formula": {
            "composite": "0.30 × text_score + 0.50 × image_score + 0.20 × audio_score",
            "tiers": {"NONE": "< 0.30", "MILD": "0.30–0.54", "MODERATE": "0.55–0.74", "SEVERE": "≥ 0.75"},
            "weights": {"text": 0.30, "image": 0.50, "audio": 0.20},
        },
        "scenarios": [],
    }

    for r in results:
        entry = {
            "id": r["id"],
            "label": r["label"],
            "input": {
                "composite_score": r["comp"],
                "severity_tier": r["tier"].name,
                "driver_profile": r["profile"].value,
                "road_type": r["road"].value,
                "driving_duration_min": r["dur"],
                "has_driver_memory": r["memory"] is not None,
                "memory_likes_coffee": getattr(r["memory"], "likes_coffee", None),
                "memory_ok_pull_over": getattr(r["memory"], "ok_to_pull_over_city", None),
                "traffic_density": r["traffic"],
                "coffee_shop_km": r["coffee_km"],
                "modality_scores": {
                    "text":  r["text_s"],
                    "image": r["image_s"],
                    "audio": r["audio_s"],
                },
            },
            "output": {
                "agents_fired": [
                    name for name, fired in [
                        ("screen_display",  r["screen"] is not None),
                        ("voice_broadcast", r["voice"]  is not None),
                        ("context_action",  r["ctx"]    is not None),
                        ("video_record",    r["video"]  is not None),
                        ("phone_push",      r["push"]   is not None),
                    ] if fired
                ],
                "screen_display":  {"message": r["screen"].message} if r["screen"] else None,
                "voice_broadcast": {"message": r["voice"].message}  if r["voice"]  else None,
                "context_action":  {
                    "message": r["ctx"].message,
                    "action":  r["ctx"].payload.get("action"),
                    "auto_ordered": r["ctx"].payload.get("auto_ordered"),
                    "payload": r["ctx"].payload,
                } if r["ctx"] else None,
                "video_record":    {"filename": r["video"].payload.get("filename")} if r["video"] else None,
                "phone_push":      {"message": r["push"].message, "payload": r["push"].payload} if r["push"] else None,
            },
        }
        output["scenarios"].append(entry)

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenario_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    return path


async def main():
    args = sys.argv[1:]
    console.print("[dim]Running 20 scenarios…[/dim]")
    results = await run_all()

    if "--save" in args or True:   # always save
        path = save_results(results)
        console.print(f"[dim]Saved → {path}[/dim]\n")

    render_matrix(results)


if __name__ == "__main__":
    asyncio.run(main())
