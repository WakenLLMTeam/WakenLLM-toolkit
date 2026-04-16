"""
Bosch Fatigue Monitor — Proactive Agent  ·  Stakeholder Demo

Animated walkthrough of the three-layer "when & what to alert" logic:
  Layer 1  Problem Identification — is it worth interrupting the driver?
  Layer 2  Agent                  — which action fits this driver?
  Layer 3  Driver Reaction        — what happens when they respond?

Each round ends with a real car-HUD warning + live voice broadcast.
Press  Y  to acknowledge  /  N  to ignore — drives the escalation logic.

Usage:
    python proactive_demo.py
"""
from __future__ import annotations
import asyncio, os, sys, tty, termios, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from models.judge_verdict import SeverityTier
from models.driver_memory import DriverMemory

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# Theme
# ─────────────────────────────────────────────────────────────────────────────

TIER_COLOR = {
    SeverityTier.NONE:     "green",
    SeverityTier.MILD:     "yellow",
    SeverityTier.MODERATE: "dark_orange",
    SeverityTier.SEVERE:   "red",
}
TIER_ICON = {
    SeverityTier.NONE:     "✅",
    SeverityTier.MILD:     "⚠️ ",
    SeverityTier.MODERATE: "🟠",
    SeverityTier.SEVERE:   "🚨",
}
IGNORE_FACTOR = {0: 1.0, 1: 0.8, 2: 0.6}
BASE_COOLDOWN = {
    SeverityTier.MILD:     30.0,
    SeverityTier.MODERATE: 60.0,
    SeverityTier.SEVERE:   120.0,
}
PIPELINE_STAGES = [
    ("Sensors",  "📡"),
    ("StateAgg", "📊"),
    ("Context",  "🗺 "),
    ("Judge",    "🧠"),
    ("Orch",     "⚙️ "),
]

# ─────────────────────────────────────────────────────────────────────────────
# Interactive input — single keypress, no Enter needed
# ─────────────────────────────────────────────────────────────────────────────

def _getch() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

async def _wait_keypress() -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _getch)

# ─────────────────────────────────────────────────────────────────────────────
# Voice — Qwen3-TTS-Flash via Bosch gateway → afplay; fallback: macOS say
# ─────────────────────────────────────────────────────────────────────────────

_BOSCH_API_KEY  = "4a52b2bf90254d29bfb86919924c6d7d"
_BOSCH_BASE_URL = "https://aigc.bosch.com.cn/llmservice/api/v1"
_TTS_MODEL      = "qwen3-tts-flash"

_tts_proc: asyncio.subprocess.Process | None = None
_tts_tmp:  str | None = None          # temp file path to clean up

def _stop_audio() -> None:
    """Kill any in-progress afplay process."""
    global _tts_proc, _tts_tmp
    if _tts_proc is not None:
        try:
            _tts_proc.kill()
        except Exception:
            pass
        _tts_proc = None
    if _tts_tmp and os.path.exists(_tts_tmp):
        try:
            os.unlink(_tts_tmp)
        except Exception:
            pass
        _tts_tmp = None

_VOICE_AGENT  = "longxiaochun"   # agent  — Mandarin female
_VOICE_DRIVER = "longcheng"      # driver — Mandarin male

async def _fetch_audio(text: str, voice: str = _VOICE_AGENT) -> str | None:
    """
    Call Qwen3-TTS-Flash API, write bytes to a temp file.
    Returns the temp file path on success, None on failure.
    Does NOT start playback — call _play_audio() for that.
    """
    global _tts_tmp
    if not text:
        return None
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=_BOSCH_API_KEY, base_url=_BOSCH_BASE_URL)
        response = await client.audio.speech.create(
            model=_TTS_MODEL, voice=voice, input=text,
        )
        audio = response.content
        if not audio:
            raise ValueError("TTS API returned empty audio")
        ext = ".wav" if audio[:4] == b"RIFF" else ".mp3"
        tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        tmp.write(audio)
        tmp.close()
        _tts_tmp = tmp.name
        return tmp.name
    except Exception as e:
        console.print(f"[bold red]  ⚠ TTS fetch error: {e}[/bold red]")
        return None

async def _play_audio(path: str | None, fallback_text: str = "") -> None:
    """
    Play audio file via afplay.  If path is None (fetch failed), fall back to
    macOS `say` with the best available Chinese voice.
    Non-blocking: stores process in _tts_proc so callers can await or kill it.
    """
    global _tts_proc, _tts_tmp
    # Kill the previous playback process
    if _tts_proc is not None:
        try:
            _tts_proc.kill()
        except Exception:
            pass
        _tts_proc = None
    # Delete the OLD temp file only if it's different from the one we're about to play
    if _tts_tmp and _tts_tmp != path:
        try:
            if os.path.exists(_tts_tmp):
                os.unlink(_tts_tmp)
        except Exception:
            pass
        _tts_tmp = None

    if path and os.path.exists(path):
        try:
            _tts_proc = await asyncio.create_subprocess_exec(
                "afplay", path,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            _tts_tmp = path   # track so _stop_audio can clean it up later
            return
        except Exception as e:
            console.print(f"[bold red]  ⚠ afplay error: {e}[/bold red]")
    # Fallback: macOS built-in TTS
    if fallback_text:
        for voice in ("Mei-Jia", "Ting-Ting", "Sin-ji"):
            try:
                _tts_proc = await asyncio.create_subprocess_exec(
                    "say", "-v", voice, "-r", "175", fallback_text,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                break
            except Exception:
                continue

async def _speak(text: str) -> None:
    """Convenience: fetch + play in one call (blocks until fetch completes)."""
    path = await _fetch_audio(text)
    await _play_audio(path, fallback_text=text)

# ─────────────────────────────────────────────────────────────────────────────
# Primitive helpers
# ─────────────────────────────────────────────────────────────────────────────

def _score_bar(v: float, width: int = 14) -> Text:
    c = "green" if v < .30 else "yellow" if v < .55 else "dark_orange" if v < .75 else "red"
    t = Text("█" * int(v * width) + "░" * (width - int(v * width)), style=c)
    t.append(f" {v:.0%}", style=f"bold {c}")
    return t

def _sig_bar(v: float, hi: float, width: int = 10) -> Text:
    norm = min(1.0, v / (hi * 2))
    c = "red" if v > hi else "green"
    return Text("█" * int(norm * width) + "░" * (width - int(norm * width)), style=c)

def _sparkline(scores: list[float], pred_steps: int = 10) -> Text:
    """Colored block-character trend line + dashed prediction zone + threshold marker."""
    BLOCKS = " ▁▂▃▄▅▆▇█"
    t = Text()
    for s in scores:
        idx = min(8, int(s / 0.75 * 8))
        c = ("green" if s < 0.30 else "yellow" if s < 0.55
             else "dark_orange" if s < 0.75 else "red")
        t.append(BLOCKS[idx], style=f"bold {c}")
    t.append("  ", style="dim")
    for _ in range(min(pred_steps, 12)):
        t.append("· ", style="dim white")
    t.append(" ⚠ MILD", style="bold yellow")
    return t

def _build_dialogue_content(dialogue: dict) -> Group:
    """Render the driver↔agent voice exchange panel, including any tool calls."""
    listen_ln = Text()
    listen_ln.append("  🎤  ASR  qwen3-asr-flash  ", style="bold magenta")
    listen_ln.append("▌▋▍▌▋▌▍▋ ", style="magenta")
    listen_ln.append("转写完成", style="dim green")

    driver_ln = Text()
    driver_ln.append("  👤  司机   ", style="bold white")
    driver_ln.append(f'「{dialogue["driver_text"]}」', style="italic white")

    sep = Text("  " + "─" * 44, style="dim")

    lines: list = [Text(""), listen_ln, Text(""), driver_ln, Text(""), sep]

    # ── Tool calls (proactive actions the agent took) ──────────────────────
    for tc in dialogue.get("tool_calls", []):
        call_ln = Text()
        call_ln.append("  🔧  tool_call  ", style="bold yellow")
        call_ln.append(tc["call"], style="bold white")
        lines.append(Text(""))
        lines.append(call_ln)

        resp_ln = Text()
        resp_ln.append("             → ", style="dim yellow")
        resp_ln.append(tc["result"], style="green")
        lines.append(resp_ln)

    lines.append(Text(""))

    agent_ln = Text()
    agent_ln.append("  🤖  Agent  ", style="bold cyan")
    agent_ln.append(dialogue["agent_reply"], style="cyan")
    lines.append(agent_ln)

    if dialogue.get("memory_update"):
        mem_ln = Text()
        mem_ln.append("\n  📝  Memory  ", style="dim")
        mem_ln.append(dialogue["memory_update"], style="bold dim yellow")
        lines.append(mem_ln)

    lines.append(Text(""))
    return Group(*lines)

# ─────────────────────────────────────────────────────────────────────────────
# Panel builders
# ─────────────────────────────────────────────────────────────────────────────

def _panel_pipeline(active: int, done: set[int]) -> Panel:
    parts: list[Text] = []
    for i, (name, icon) in enumerate(PIPELINE_STAGES):
        if i in done:
            t = Text(f"{icon}{name}✓", style="bold green")
        elif i == active:
            t = Text(f"{icon}{name}…", style="bold yellow")
        else:
            t = Text(f"{icon}{name} ", style="dim")
        parts.append(t)
        if i < len(PIPELINE_STAGES) - 1:
            parts.append(Text(" → ", style="green" if i in done else "dim"))
    row = Text()
    for p in parts:
        row.append_text(p)
    return Panel(Align.center(row), title="Pipeline", box=box.ROUNDED,
                 border_style="blue", padding=(0, 1))


def _panel_problem_id(
    tier: SeverityTier, score: float, reasoning: str,
    ignore: int, elapsed: float, allowed: bool,
    perclos: float, yawns: float, steering: float, minutes: int, road: str,
) -> Panel:
    base   = BASE_COOLDOWN.get(tier, 0.0)
    factor = IGNORE_FACTOR.get(ignore, 0.4)
    eff_cd = base * factor
    eff_tier = (
        SeverityTier(int(tier) + 1)
        if ignore >= 2 and tier not in (SeverityTier.NONE, SeverityTier.SEVERE)
        else tier
    )
    ec = TIER_COLOR[eff_tier]

    sig = Text()
    sig.append(f"PERCLOS   {perclos:.2f}  ",
               style="red" if perclos > 0.15 else "white")
    sig.append("!\n" if perclos > 0.15 else "✓\n",
               style="bold red" if perclos > 0.15 else "green")
    sig.append(f"Yawns     {yawns:.1f}   ",
               style="red" if yawns > 2.0 else "white")
    sig.append("!\n" if yawns > 2.0 else "✓\n",
               style="bold red" if yawns > 2.0 else "green")
    sig.append(f"Steering  {steering:.0f}°   ",
               style="red" if steering > 15 else "white")
    sig.append("!\n" if steering > 15 else "✓\n",
               style="bold red" if steering > 15 else "green")
    sig.append(f"\n{road} · {minutes}min", style="dim")

    gate = Text()
    gate.append(f"Base CD   {base:.0f}s\n", style="white")
    gate.append(f"Ignore    {ignore}\n",
                style="bold red" if ignore else "green")
    gate.append(f"Eff CD    {eff_cd:.0f}s  ({int(factor*100)}%)\n",
                style="yellow" if ignore else "white")
    gate.append(f"Elapsed   {elapsed:.0f}s\n\n", style="white")
    if tier == SeverityTier.NONE:
        gate.append("✗  SILENT", style="bold dim")
    elif allowed:
        gate.append("✓  FIRE ALERT", style="bold green")
    else:
        gate.append(f"✗  WAIT {eff_cd - elapsed:.0f}s", style="bold red")

    verdict = Text()
    verdict.append(f"{TIER_ICON[eff_tier]} {eff_tier.name}",
                   style=f"bold {ec}")
    if eff_tier != tier:
        verdict.append("  ↑ escalated", style="bold red")
    verdict.append("\n")
    verdict.append_text(_score_bar(score, width=12))
    verdict.append(f"\n\n{reasoning[:22]}", style="dim italic")

    tbl = Table(box=box.SIMPLE_HEAD, expand=True, show_header=True,
                padding=(0, 2), show_edge=False)
    tbl.add_column("Signals Detected", ratio=1)
    tbl.add_column("Gate Check",       ratio=1)
    tbl.add_column("Verdict",          ratio=1)
    tbl.add_row(sig, gate, verdict)

    border = "green" if (allowed and tier != SeverityTier.NONE) else "dim"
    return Panel(tbl, title="[bold]Layer 1 · Problem Identification[/bold]",
                 box=box.DOUBLE_EDGE, border_style=border, padding=(0, 1))


def _panel_agent(memory: DriverMemory) -> Panel:
    """Layer 2: show the driver memory that informed the agent's decision."""
    def flag(v: bool) -> Text:
        return Text("yes", style="bold cyan") if v else Text("no", style="dim")

    tbl = Table(box=None, show_header=False, padding=(0, 1))
    tbl.add_column("", style="dim", width=20)
    tbl.add_column("", width=30)
    tbl.add_row("likes_coffee",  flag(memory.likes_coffee))
    tbl.add_row("prefers_nap",   flag(memory.prefers_nap))
    tbl.add_row("ok_pullover",   flag(memory.ok_to_pull_over_city))
    tbl.add_row("ignore streak",
                Text(str(memory.ignored_alert_streak),
                     style="bold red" if memory.ignored_alert_streak else "green"))
    if memory.notes:
        tbl.add_row("", Text(""))
        for note in memory.notes[-2:]:
            tbl.add_row("📌 promise", Text(note, style="bold yellow"))
    return Panel(tbl, title="[bold]Layer 2 · Agent[/bold]",
                 box=box.ROUNDED, border_style="cyan", padding=(0, 1))


def _panel_predictive(
    scores: list[float],
    current_score: float,
    trend_per_min: float,
    time_to_threshold_min: int,
    scenario_tags: list[str],
    risk_factors: list[str],
    suggestion: str,
) -> Panel:
    """Predictive trend panel — fires BEFORE the fatigue threshold is crossed."""
    t_hist = len(scores)

    chart = Text("  ")
    chart.append_text(_sparkline(scores, pred_steps=min(time_to_threshold_min // 2, 10)))

    axis = Text("  " + "━" * 52, style="dim")

    lbl = Text()
    lbl.append(f"  -{t_hist}m", style="dim")
    lbl.append(" " * 12)
    lbl.append(f"当前 {current_score:.2f}  (+{trend_per_min:.3f}/min)", style="bold yellow")
    lbl.append(" " * 6)
    lbl.append(f"预计 +{time_to_threshold_min}min 触发", style="bold red")

    scen = Text("\n  场景  ")
    for tag in scenario_tags:
        scen.append(f"{tag}  ", style="cyan")

    risk = Text("  风险  ")
    for i, rf in enumerate(risk_factors):
        if i:
            risk.append("  ·  ", style="dim")
        risk.append(rf, style="yellow")

    sug_lines: list = [Text("\n  💡 建议", style="bold green")]
    for line in suggestion.split("\n"):
        sug_lines.append(Text(f"     {line}", style="white"))
    sug_lines.append(Text(""))

    content = Group(Text(""), chart, axis, lbl, scen, risk, *sug_lines)
    return Panel(
        content,
        title="[bold yellow]⚡  Predictive Alert — 主动预测干预[/bold yellow]",
        box=box.DOUBLE_EDGE,
        border_style="yellow",
        padding=(0, 1),
    )


def _panel_route_map(
    road_name: str,
    direction: str,
    dest_name: str,
    dest_icon: str,
    dist_km: float,
    eta_min: int,
    via: list[str],
    layout: str = "highway",   # "highway" | "city"
) -> Panel:
    """Visual terminal road-map — rendered with box-drawing road graphics."""
    lines: list = []

    # ── header ──────────────────────────────────────────────────────────────
    hdr = Text()
    hdr.append(f"  {road_name}", style="bold white")
    hdr.append(f"   {direction}", style="dim cyan")
    lines += [Text(""), hdr, Text("")]

    if layout == "highway":
        # ── road shoulders (dashed centre line) ─────────────────────────────
        lines.append(Text("  " + "╌" * 48, style="dim white"))

        # ── upper edge ──────────────────────────────────────────────────────
        top = Text()
        top.append("  ╔" + "═" * 40 + "╦══", style="bold cyan")
        top.append("  出口", style="bold yellow")
        lines.append(top)

        # ── car lane ────────────────────────────────────────────────────────
        lane = Text()
        lane.append("  ║ ", style="bold cyan")
        lane.append("🚗", style="bold yellow")
        lane.append(" ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ", style="white")
        lane.append("║", style="bold cyan")
        lines.append(lane)

        # ── lower edge with exit stub ────────────────────────────────────────
        bot = Text()
        bot.append("  ╚" + "═" * 40 + "╝   ", style="bold cyan")
        bot.append("↘  ", style="bold yellow")
        bot.append(f"{dest_icon} {dest_name}", style="bold green")
        lines.append(bot)

        # ── road shoulders ───────────────────────────────────────────────────
        lines.append(Text("  " + "╌" * 48, style="dim white"))

        # ── distance bar ────────────────────────────────────────────────────
        dist = Text()
        dist.append("\n       ⟵── ", style="dim cyan")
        dist.append(f"{dist_km:.1f} km  ·  约 {eta_min} 分钟", style="bold cyan")
        dist.append(" ──⟶", style="dim cyan")
        lines.append(dist)

    else:  # city — straight then right-turn down
        # ── horizontal road ─────────────────────────────────────────────────
        horiz = Text()
        horiz.append("  ══", style="bold cyan")
        horiz.append("🚗", style="bold yellow")
        horiz.append("═" * 20 + "╗", style="bold cyan")
        horiz.append(f"   ← {via[0]}", style="bold yellow")
        lines.append(horiz)

        # ── vertical road ───────────────────────────────────────────────────
        vert_label = f"   {dist_km:.1f} km · 约 {eta_min} 分钟"
        for i in range(3):
            v = Text()
            v.append("  " + " " * 24 + "║", style="bold cyan")
            if i == 1:
                v.append(vert_label, style="dim cyan")
            lines.append(v)

        # ── destination ─────────────────────────────────────────────────────
        arr = Text()
        arr.append("  " + " " * 24 + "↓", style="bold yellow")
        lines.append(arr)

        dest_ln = Text()
        dest_ln.append("  " + " " * 22)
        dest_ln.append(f"{dest_icon}  {dest_name}", style="bold green")
        lines.append(dest_ln)

    # ── turn-by-turn steps ───────────────────────────────────────────────────
    steps = Text()
    steps.append("\n  📍 ", style="cyan")
    for i, step in enumerate(via):
        if i:
            steps.append("  →  ", style="dim")
        steps.append(step, style="dim cyan")
    lines += [steps, Text("")]

    return Panel(Group(*lines), title="[bold]🗺  Route Navigation[/bold]",
                 box=box.ROUNDED, border_style="blue", padding=(0, 1))


def _panel_car_hud(
    tier: SeverityTier,
    hud_lines: str,
    action_label: str,
) -> Panel:
    """Simulated in-car HMI warning screen."""
    color = TIER_COLOR[tier]
    icon  = TIER_ICON[tier]

    content = Group(
        Text(""),
        Align.center(Text(f"{icon}  {tier.name.upper()} ALERT",
                          style=f"bold {color}")),
        Text(""),
        Align.center(Text(hud_lines, style="bold white", justify="center")),
        Text(""),
        Align.center(Text(f"🔊  Voice broadcasting  ·  {action_label}",
                          style="dim cyan")),
        Text(""),
    )
    return Panel(
        content,
        title="[bold]🚗  BOSCH SafeDriver™[/bold]",
        box=box.HEAVY,
        border_style=color,
        padding=(0, 4),
    )


def _panel_driver_reaction(
    response: str,
    ig_before: int,
    ig_after: int,
    base_cd: float,
    eff_tier_after: SeverityTier | None,
) -> Panel:
    tbl = Table(box=box.SIMPLE_HEAD, expand=True, show_header=True,
                padding=(0, 2), show_edge=False)
    tbl.add_column("Response",       ratio=1)
    tbl.add_column("System Updates", ratio=2)

    if response == "IGNORED":
        resp_text = Text("✗  IGNORED", style="bold red")
        old = base_cd * IGNORE_FACTOR.get(ig_before, 0.4)
        new = base_cd * IGNORE_FACTOR.get(ig_after,  0.4)
        upd = Text()
        upd.append(f"streak        {ig_before} → {ig_after}\n", style="yellow")
        upd.append(f"next cooldown {old:.0f}s → {new:.0f}s  ↓", style="yellow")
        if eff_tier_after:
            upd.append(f"\neff. tier     MILD → {eff_tier_after.name}  ↑",
                       style="bold red")
        border = "red"
    else:
        resp_text = Text("✓  ACKNOWLEDGED", style="bold green")
        upd = Text()
        upd.append(f"streak        {ig_before} → 0  (reset)\n", style="green")
        upd.append("next cooldown restored to base", style="green")
        border = "green"

    tbl.add_row(resp_text, upd)
    return Panel(tbl, title="[bold]Layer 3 · Driver Reaction[/bold]",
                 box=box.ROUNDED, border_style=border, padding=(0, 1))

# ─────────────────────────────────────────────────────────────────────────────
# Round runner
# ─────────────────────────────────────────────────────────────────────────────

def _panel_proactive_actions(tool_calls: list[dict], question: str) -> Panel:
    """Shows agent tool calls already executed + a proactive question to driver."""
    lines: list = [Text("")]
    for tc in tool_calls:
        call_ln = Text()
        call_ln.append("  🔧  tool_call  ", style="bold yellow")
        call_ln.append(tc["call"], style="bold white")
        lines.append(call_ln)
        resp_ln = Text()
        resp_ln.append("             → ", style="dim yellow")
        resp_ln.append(tc["result"], style="green")
        lines.append(resp_ln)
        lines.append(Text(""))

    q_ln = Text()
    q_ln.append("  🤖  Agent  ", style="bold cyan")
    q_ln.append(question, style="bold cyan")
    lines.append(q_ln)
    lines.append(Text(""))
    return Panel(Group(*lines),
                 title="[bold yellow]⚡  Proactive Actions[/bold yellow]",
                 box=box.ROUNDED, border_style="yellow", padding=(0, 1))


async def run_predictive_round(
    *,
    scores: list[float],
    current_score: float,
    trend_per_min: float,
    time_to_threshold_min: int,
    scenario_tags: list[str],
    risk_factors: list[str],
    suggestion: str,
    voice_text: str,
    proactive: dict,          # tool_calls, question, route, voice_yes, voice_no
    memory: DriverMemory,
) -> None:
    """Proactive round — fires BEFORE threshold; already acts, then asks driver."""
    console.print()
    console.print(Rule(
        "[bold yellow]  Predictive Round  ·  主动预测干预  [/bold yellow]",
        style="yellow", characters="━"))
    console.print()

    # Pre-fetch trend narration + both Y/N replies concurrently
    audio_task     = asyncio.create_task(_fetch_audio(voice_text))
    audio_yes_task = asyncio.create_task(_fetch_audio(proactive["voice_yes"]))
    audio_no_task  = asyncio.create_task(_fetch_audio(proactive["voice_no"]))

    sections: list = []
    done: set[int] = set()

    with Live(console=console, refresh_per_second=16) as live:
        for i in range(len(PIPELINE_STAGES)):
            sections = [_panel_pipeline(active=i, done=done)]
            live.update(Group(*sections))
            await asyncio.sleep(0.22)
            done.add(i)
        sections = [_panel_pipeline(active=-1, done=done)]
        live.update(Group(*sections))
        await asyncio.sleep(0.15)

        sections.append(_panel_predictive(
            scores, current_score, trend_per_min,
            time_to_threshold_min, scenario_tags, risk_factors, suggestion,
        ))
        live.update(Group(*sections))
        await asyncio.sleep(0.90)

        sections.append(_panel_agent(memory))
        live.update(Group(*sections))
        await asyncio.sleep(0.60)

    # ── Phase 2: show proactive tool calls + ask driver ───────────────────────
    audio_path = await audio_task
    await _play_audio(audio_path, fallback_text=voice_text)

    console.print()
    console.print(_panel_proactive_actions(
        proactive["tool_calls"], proactive["question"]))

    # Show route map (navigation already planned)
    if proactive.get("route"):
        await asyncio.sleep(0.70)
        console.print(_panel_route_map(**proactive["route"]))

    console.print()
    console.print(Align.center(
        Text("  [ Y ]  好，帮我订咖啡      [ N ]  不用了  ",
             style="bold white on grey23")))
    console.print()

    # Wait for both the narration and the keypress
    global _tts_proc
    key_task   = asyncio.create_task(_wait_keypress())
    audio_done = asyncio.create_task(
        _tts_proc.wait() if _tts_proc else asyncio.sleep(0))
    key = await key_task
    audio_done.cancel()
    _stop_audio()

    if key.lower() == "y":
        reply_path = await audio_yes_task
        audio_no_task.cancel()
        await _play_audio(reply_path, fallback_text=proactive["voice_yes"])
        order_ln = Text()
        order_ln.append("  ✅  ", style="bold green")
        order_ln.append("已为您预订服务区美式咖啡（大杯），到达后取餐即可。",
                         style="green")
        console.print()
        console.print(Panel(order_ln, border_style="green",
                            box=box.ROUNDED, padding=(0, 1)))
    else:
        reply_path = await audio_no_task
        audio_yes_task.cancel()
        await _play_audio(reply_path, fallback_text=proactive["voice_no"])

    if _tts_proc is not None:
        try:
            await asyncio.wait_for(_tts_proc.wait(), timeout=20.0)
        except asyncio.TimeoutError:
            pass
    await asyncio.sleep(0.8)


async def run_round(
    *,
    num: int,
    label: str,
    tier: SeverityTier,
    score: float,
    reasoning: str,
    ignore: int,
    elapsed: float,
    hud_lines: str,
    action_label: str,
    voice_text: str,
    memory: DriverMemory,
    perclos: float,
    yawns: float,
    steering: float,
    minutes: int,
    road: str,
    route: dict | None = None,
    dialogue: dict | None = None,
) -> str | None:
    """Returns the driver's response: 'ACKNOWLEDGED', 'IGNORED', or None."""
    base_cd = BASE_COOLDOWN.get(tier, 0.0)
    allowed = (
        tier != SeverityTier.NONE
        and elapsed >= base_cd * IGNORE_FACTOR.get(ignore, 0.4)
    )

    console.print()
    console.print(Rule(f"[bold]  Round {num}/4  ·  {label}  [/bold]",
                       style="blue", characters="━"))
    console.print()

    sections: list = []
    done: set[int] = set()

    # Pre-fetch TTS audio in background while Phase 1 animation plays.
    # The API call (~2-3 s) runs concurrently with the ~3 s animation,
    # so audio is ready to start instantly when the HUD appears.
    audio_task: asyncio.Task | None = (
        asyncio.create_task(_fetch_audio(voice_text)) if voice_text else None
    )

    # ── Phase 1: Animated reveals (inside Live) ───────────────────────────────
    with Live(console=console, refresh_per_second=16) as live:

        # Pipeline animation
        for i in range(len(PIPELINE_STAGES)):
            sections = [_panel_pipeline(active=i, done=done)]
            live.update(Group(*sections))
            await asyncio.sleep(0.28)
            done.add(i)
        sections = [_panel_pipeline(active=-1, done=done)]
        live.update(Group(*sections))
        await asyncio.sleep(0.18)

        # Layer 1 — Problem Identification
        sections.append(_panel_problem_id(
            tier, score, reasoning, ignore, elapsed, allowed,
            perclos, yawns, steering, minutes, road,
        ))
        live.update(Group(*sections))
        await asyncio.sleep(0.90)

        if not allowed:
            sections.append(Align.center(
                Text("  System stays silent this cycle.  ", style="dim italic")))
            live.update(Group(*sections))
            await asyncio.sleep(1.5)
            if audio_task:
                audio_task.cancel()
            return None

        # Layer 2 — Agent memory
        sections.append(_panel_agent(memory))
        live.update(Group(*sections))
        await asyncio.sleep(0.90)

    # ── Phase 2: Car HUD + voice + keyboard (outside Live) ───────────────────

    # Audio was pre-fetched during Phase 1 — play it now (instant start)
    if audio_task:
        audio_path = await audio_task
        await _play_audio(audio_path, fallback_text=voice_text)

    # Show the car HMI warning, then reveal route map below it
    console.print()
    console.print(_panel_car_hud(tier, hud_lines, action_label))
    if route:
        await asyncio.sleep(0.80)
        console.print(_panel_route_map(**route))
    console.print()

    prompt_text = (
        "  [ Y ]  Acknowledge      [ N ]  Ignore      [ V ]  语音回应  "
        if dialogue else
        "  [ Y ]  Acknowledge      [ N ]  Ignore  "
    )
    console.print(Align.center(Text(prompt_text, style="bold white on grey23")))
    console.print()

    # Wait for actual keypress
    key = await _wait_keypress()
    _stop_audio()   # stop alert audio once driver responds

    if key.lower() == "v" and dialogue:
        # Pre-fetch BOTH voices concurrently while we show the "listening" indicator
        driver_task = asyncio.create_task(
            _fetch_audio(dialogue["driver_text"], voice=_VOICE_DRIVER))
        agent_task  = asyncio.create_task(
            _fetch_audio(dialogue["agent_reply"],  voice=_VOICE_AGENT))

        # ── Phase A: driver speaks ──────────────────────────────────────────
        console.print()
        listen_panel = Panel(
            Group(
                Text(""),
                Text("  🎤  ASR  qwen3-asr-flash  ▌▋▍▌▋▌▍▋  正在聆听…",
                     style="bold magenta"),
                Text(""),
                Text(f'  👤  司机   「{dialogue["driver_text"]}」',
                     style="italic white"),
                Text(""),
            ),
            title="[bold]🎤  Voice Dialogue[/bold]",
            box=box.ROUNDED, border_style="magenta", padding=(0, 1),
        )
        console.print(listen_panel)
        driver_path = await driver_task
        await _play_audio(driver_path, fallback_text=dialogue["driver_text"])
        global _tts_proc
        if _tts_proc is not None:
            try:
                await asyncio.wait_for(_tts_proc.wait(), timeout=15.0)
            except asyncio.TimeoutError:
                pass
        await asyncio.sleep(3.0)

        # ── Phase B: agent acts + replies ───────────────────────────────────
        console.print()
        console.print(Panel(
            _build_dialogue_content(dialogue),
            title="[bold]🎤  Voice Dialogue[/bold]",
            box=box.ROUNDED, border_style="magenta", padding=(0, 1),
        ))
        agent_path = await agent_task
        await _play_audio(agent_path, fallback_text=dialogue["agent_reply"])
        if dialogue.get("memory_update"):
            memory.notes.append(dialogue["memory_update"])
        if _tts_proc is not None:
            try:
                await asyncio.wait_for(_tts_proc.wait(), timeout=15.0)
            except asyncio.TimeoutError:
                pass
        response = dialogue.get("intent", "ACKNOWLEDGED")
    elif key.lower() == "y":
        response = "ACKNOWLEDGED"
    else:
        response = "IGNORED"

    # Layer 3 — Driver Reaction
    console.print()
    ig_after = ignore + 1 if response == "IGNORED" else 0
    eff_after: SeverityTier | None = None
    if (response == "IGNORED" and ig_after >= 2
            and tier not in (SeverityTier.NONE, SeverityTier.SEVERE)):
        eff_after = SeverityTier(int(tier) + 1)

    console.print(_panel_driver_reaction(response, ignore, ig_after, base_cd, eff_after))
    await asyncio.sleep(1.5)
    return response

# ─────────────────────────────────────────────────────────────────────────────
# Scenario data
# ─────────────────────────────────────────────────────────────────────────────

_MEM = DriverMemory(
    likes_coffee=True,
    preferred_coffee_order="美式（大杯）",
    coffee_max_km=2.0,
    ok_to_pull_over_city=True,
)

# ─── Predictive round (fires before threshold) ────────────────────────────────
PREDICTIVE_ROUND = dict(
    scores=[0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.25, 0.28],
    current_score=0.28,
    trend_per_min=0.022,
    time_to_threshold_min=18,
    scenario_tags=["🛣 长途高速", "⏱ 已行驶 3.2h", "🌤 下午困倦时段"],
    risk_factors=["下午困倦高峰 ×1.3", "前方 120km 无服务区", "连续未休息"],
    suggestion=(
        "当前得分 0.28，尚未触发警报，但 9 分钟内持续上升。\n"
        "已主动规划导航并查询周边服务，等待您确认。"
    ),
    voice_text=(
        "系统检测到疲劳趋势持续上升，预计十八分钟后触发警报。"
        "已为您规划导航至前方枫泾服务区，预计十分钟到达。"
        "检测到您偏好咖啡，是否为您提前预订？"
    ),
    proactive=dict(
        tool_calls=[
            dict(
                call='navigation_api.plan_route(dest="枫泾服务区", dist_km=22)',
                result='{"status": "ready", "route": "G2高速直行·出口32B", "eta_min": 10}',
            ),
            dict(
                call='driver_memory.get_preference(key="coffee")',
                result='{"likes_coffee": true, "order": "美式（大杯）", "max_km": 2.0}',
            ),
        ],
        question="导航已规划好，检测到您偏好咖啡 ☕ — 是否为您提前预订服务区咖啡？",
        route=dict(
            road_name="沪杭高速",
            direction="上海方向 →",
            dest_name="枫泾服务区",
            dest_icon="🅿",
            dist_km=22.0,
            eta_min=10,
            via=["G2高速直行", "出口32B右转", "服务区匝道入口"],
            layout="highway",
        ),
        voice_yes=(
            "好的，已为您预订枫泾服务区美式咖啡大杯，到达后直接取餐即可。"
            "导航已启动，请安心驾驶。"
        ),
        voice_no=(
            "好的，导航已为您规划好，请根据需要靠站休息。注意保持清醒。"
        ),
    ),
)

ROUNDS = [
    dict(
        num=1, label="MILD — first alert",
        tier=SeverityTier.MILD, score=0.44,
        reasoning="PERCLOS 0.19 above threshold.",
        ignore=0, elapsed=35.0,
        hud_lines="建议前往前方服务区休息\n沪杭高速服务区  ·  十二点五公里",
        action_label="🖥  Screen alert",
        voice_text=(
            "注意，系统检测到疲劳驾驶。"
            "已为您规划导航：沿高速直行，"
            "前方三十二号出口右转，约八分钟可到达服务区，"
            "建议立即前往稍作休息。"
        ),
        perclos=0.19, yawns=1.5, steering=12.0, minutes=190, road="Highway",
        route=dict(
            road_name="沪杭高速",
            direction="上海方向 →",
            dest_name="沪杭服务区",
            dest_icon="🅿",
            dist_km=12.5,
            eta_min=8,
            via=["高速直行", "三十二号出口右转", "服务区匝道入口"],
            layout="highway",
        ),
        dialogue=dict(
            driver_text="没事，我再开半个小时就好",
            tool_calls=[
                dict(
                    call='vehicle_api.open_window(position="front", level=30)',
                    result='{"status": "ok", "window": "前排车窗已开启 30%"}',
                ),
            ],
            agent_reply=(
                "好的，已记录您的承诺，三十分钟后重新检测。"
                "前方路段平直，已为您自动开启前排车窗30%以保持清醒。"
            ),
            intent="ACKNOWLEDGED",
            memory_update="承诺三十分钟后休息（已超时提醒）",
        ),
    ),
    dict(
        num=2, label="MILD — cooldown shortened to 80%",
        tier=SeverityTier.MILD, score=0.46,
        reasoning="PERCLOS 0.21 持续. No improvement.",
        ignore=1, elapsed=25.0,
        hud_lines="已连续行驶三小时，轻度疲劳\n服务区  ·  十点二公里  ·  约六分钟",
        action_label="🖥  Screen alert",
        voice_text=(
            "提醒您，已连续驾驶超过三小时，检测到轻度疲劳。"
            "导航已更新，前方服务区距您十点二公里，"
            "预计六分钟到达，请保持注意力，计划靠站休息。"
        ),
        perclos=0.21, yawns=1.8, steering=14.0, minutes=200, road="Highway",
        route=dict(
            road_name="沪杭高速",
            direction="上海方向 →",
            dest_name="沪杭服务区",
            dest_icon="🅿",
            dist_km=10.2,
            eta_min=6,
            via=["高速直行", "三十二号出口右转", "服务区匝道入口"],
            layout="highway",
        ),
    ),
    dict(
        num=3, label="MILD detected — escalated to MODERATE",
        tier=SeverityTier.MILD, score=0.45,
        reasoning="2 ignores → effective MODERATE.",
        ignore=2, elapsed=20.0,
        hud_lines="疲劳警报升级  ·  已自动预订咖啡\n星巴克来福士广场  ·  一点二公里",
        action_label="🔊  Voice  ·  ☕ Auto-order",
        voice_text=(
            "警告，疲劳警报升级。"
            "已为您自动预订附近一点二公里的星巴克美式咖啡，"
            "导航已启动：前方路口右转，沿滨江大道南行，"
            "约三分钟到达来福士广场，请前往取餐休息。"
        ),
        perclos=0.20, yawns=1.9, steering=13.0, minutes=210, road="Highway",
        route=dict(
            road_name="滨江大道",
            direction="市区方向 ↓",
            dest_name="星巴克（来福士广场）",
            dest_icon="☕",
            dist_km=1.2,
            eta_min=3,
            via=["前方路口右转", "滨江大道南行", "来福士广场停车场"],
            layout="city",
        ),
    ),
    dict(
        num=4, label="Score drops — system stays silent",
        tier=SeverityTier.NONE, score=0.21,
        reasoning="Driver rested. PERCLOS 0.08.",
        ignore=0, elapsed=35.0,
        hud_lines="", action_label="", voice_text="",
        perclos=0.08, yawns=0.3, steering=5.0, minutes=215, road="Highway",
        route=None,
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    console.print()
    console.print(Panel(
        Align.center(Group(
            Text(""),
            Text("Bosch Fatigue Monitor", style="bold white", justify="center"),
            Text("Proactive Agent  ·  Stakeholder Demo", style="dim", justify="center"),
            Text(""),
            Text("How does the system decide  when  and  what  to alert?",
                 style="italic", justify="center"),
            Text(""),
            Text("Layer 1  Problem Identification  —  when is it worth interrupting?",
                 style="bold cyan",   justify="center"),
            Text("Layer 2  Agent                   —  right action for this driver?",
                 style="bold green",  justify="center"),
            Text("Layer 3  Driver Reaction         —  what if the driver ignores?",
                 style="bold yellow", justify="center"),
            Text(""),
        )),
        border_style="blue", box=box.HEAVY,
    ))
    await asyncio.sleep(1.2)

    mem = _MEM
    responses: list[str | None] = []

    # ── Predictive round: proactive, fires before threshold ───────────────────
    await run_predictive_round(memory=mem, **PREDICTIVE_ROUND)

    for rd in ROUNDS:
        resp = await run_round(memory=mem, **rd)
        responses.append(resp)
        if resp == "IGNORED":
            mem.on_alert_ignored()
        elif resp == "ACKNOWLEDGED":
            mem.on_alert_acknowledged()
        await asyncio.sleep(3.0)

    # ── Summary timeline ──────────────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold]Decision Timeline[/bold]", style="blue", characters="━"))
    console.print()

    gates = [
        "✓  35s > 30s  (100%)",
        "✓  25s > 24s  (×0.8)",
        "✓  20s > 18s  (×0.6)",
        Text("✗  Silent (NONE tier)", style="dim"),
    ]
    actions = [
        "Screen display",
        "Screen display",
        "Voice + ☕ Auto-order",
        Text("—", style="dim"),
    ]

    tbl = Table(box=box.ROUNDED, expand=True, show_lines=True)
    tbl.add_column("Round",    style="bold", width=7,  justify="center")
    tbl.add_column("Signal",                width=14)
    tbl.add_column("Gate",                  width=22)
    tbl.add_column("Action",                width=22)
    tbl.add_column("Response",              width=15)

    signals = ["MILD 0.44", "MILD 0.46", "MILD→MOD 0.45", "NONE 0.21"]
    for i, (rd, resp) in enumerate(zip(ROUNDS, responses)):
        if resp == "ACKNOWLEDGED":
            resp_cell = Text("✓  Acknowledged", style="green")
        elif resp == "IGNORED":
            resp_cell = Text("✗  Ignored", style="red")
        else:
            resp_cell = Text("—", style="dim")
        tbl.add_row(str(i + 1), signals[i], gates[i], actions[i], resp_cell)

    console.print(tbl)
    console.print()
    console.print(Panel.fit(
        "  [dim]python proactive_demo.py[/dim]   replay\n"
        "  [dim]python demo.py          [/dim]   full 3-driver scenario\n"
        "  [dim]python main.py          [/dim]   live real-time pipeline",
        title="[bold]Run[/bold]", border_style="green",
    ))


if __name__ == "__main__":
    asyncio.run(main())
