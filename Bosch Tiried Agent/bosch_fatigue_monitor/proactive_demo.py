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
# Modal output animations
# ─────────────────────────────────────────────────────────────────────────────

async def _animate_window_open(level: int = 30) -> None:
    """
    Car window slides DOWN row-by-row (top → bottom).
    ▓ = solid glass  |  ▄ = glass leading edge (half-block)  |  ╌ = open air.
    Car shape: rectangular door outline + inner window frame (╔═╗).
    """
    WIN_ROWS = 5       # glass pane height inside frame
    WIN_COLS = 24      # glass width inside frame
    FRAMES   = 36      # ~1.5 s at 24 fps

    # Inner widths must stay constant every row for stable panel:
    #   left margin "      │    ║"  = 12 chars
    #   content                     = WIN_COLS chars
    #   right margin "║        │"   = 10 chars  → total = 22 + WIN_COLS
    # Outer box: "      ┌" + "─"*W + "┐" must equal 22+WIN_COLS chars
    #   → 7 + W + 1 = 22 + WIN_COLS  → W = 14 + WIN_COLS
    W = WIN_COLS + 14   # outer box ─ count

    with Live(console=console, refresh_per_second=24) as live:
        for f in range(FRAMES + 1):
            p          = f / FRAMES
            # half-block precision: each terminal row = 2 half-rows
            total_half = p * WIN_ROWS * 2 * level / 100
            full_rows  = int(total_half) // 2   # fully open rows (╌)
            has_tip    = bool(int(total_half) % 2)  # leading-edge row active

            # scrolling wind chars in open rows (cycles leftward)
            wo         = f % WIN_COLS
            wind_pat   = ("╌╌╌╌╌╌~╌╌╌╌╌╌╌╌╌╌~╌╌╌╌╌╌" * 3)
            wind_row   = wind_pat[wo: wo + WIN_COLS]

            car: list = [Text("")]

            # ── car outer top ────────────────────────────────────────────────
            car.append(Text("      ┌" + "─" * W + "┐", style="dim white"))

            # ── window frame top ─────────────────────────────────────────────
            car.append(Text(
                "      │    ╔" + "═" * WIN_COLS + "╗        │",
                style="white"))

            # ── window rows (glass slides downward) ──────────────────────────
            for row_i in range(WIN_ROWS):
                t = Text()
                t.append("      │    ║", style="white")

                if row_i < full_rows:
                    t.append(wind_row, style="dim white")      # open air + wind
                elif row_i == full_rows and has_tip:
                    t.append("▄" * WIN_COLS, style="bold blue")  # glass leading edge
                else:
                    t.append("▓" * WIN_COLS, style="bold blue")  # solid glass

                t.append("║        │", style="white")
                car.append(t)

            # ── window frame bottom ───────────────────────────────────────────
            car.append(Text(
                "      │    ╚" + "═" * WIN_COLS + "╝        │",
                style="white"))

            # ── door body ────────────────────────────────────────────────────
            # avoid emoji in fixed-width line (double-width glyphs break column math)
            door_text = "        SafeDriver™ Active"
            car.append(Text(
                "      │" + door_text + " " * (W - len(door_text)) + "│",
                style="dim white"))

            # ── car outer bottom ─────────────────────────────────────────────
            car.append(Text("      └" + "─" * W + "┘", style="dim white"))

            # ── wheels ───────────────────────────────────────────────────────
            car.append(Text(
                "        ●" + " " * (W - 4) + "●",
                style="bold white"))

            car.append(Text(""))

            # ── status (text only) ───────────────────────────────────────────
            pct = round(p * level)
            if f == FRAMES:
                car.append(Text(
                    "  ✓ 前排车窗已开启 30%  💨  新鲜空气注入，助您保持清醒",
                    style="bold green"))
            else:
                car.append(Text(f"  🪟  车窗开启中  {pct}%…", style="dim yellow"))
            car.append(Text(""))

            live.update(Panel(
                Group(*car),
                title=(
                    "[bold yellow]🪟  车窗控制  ·  "
                    f"vehicle_api.open_window(level={level}%)[/bold yellow]"
                ),
                box=box.ROUNDED, border_style="yellow", padding=(0, 2),
            ))
            await asyncio.sleep(0.042)
    await asyncio.sleep(0.45)


_EYE_SEQ = [
    ("◉", "◉"), ("◉", "◉"), ("◐", "◐"), ("▬", "▬"),
    ("─", "─"), ("─", "─"), ("▬", "▬"), ("◑", "◑"), ("◉", "◉"),
]


async def _animate_eye_scan(perclos: float) -> None:
    """Biometric face-camera scan — shows live PERCLOS eye-closure detection."""
    import math
    c      = "red" if perclos > 0.15 else "yellow" if perclos > 0.10 else "green"
    bw     = 14
    filled = min(bw, int(bw * perclos / 0.30))

    with Live(console=console, refresh_per_second=18, transient=True) as live:
        # Phase 1 — horizontal sweep scan (0.5 s)
        for i in range(10):
            col = int(22 * i / 9)
            row = "·" * col + "▌" + "·" * (22 - col)
            t   = Text()
            t.append("  📷  qwen-vision  扫描面部关键点\n\n", style="bold magenta")
            t.append(f"  ╔══{row}══╗\n", style="magenta")
            t.append(f"  ║  {'·' * 22}  ║\n", style="dim magenta")
            t.append("  ╚══════════════════════════╝\n\n", style="dim magenta")
            t.append("  检测中…", style="dim")
            live.update(Panel(
                t, title="[bold magenta]👁  Biometric Sensor  ·  PERCLOS[/bold magenta]",
                box=box.ROUNDED, border_style="magenta", padding=(0, 1),
            ))
            await asyncio.sleep(0.05)

        # Phase 2 — eye blink cycle (2 cycles, speed varies with drowsiness)
        speed = 0.038 if perclos > 0.17 else 0.075
        for _cycle in range(2):
            for le, re in _EYE_SEQ:
                t = Text()
                t.append("  📷  Face Camera\n\n", style="bold magenta")
                t.append("  ╔══════════════════════════╗\n", style="dim")
                t.append(f"  ║     {le}           {re}       ║\n",
                         style=f"bold {c}")
                t.append("  ║          👃             ║\n", style="dim white")
                t.append("  ║        ─────            ║\n", style="dim white")
                t.append("  ╚══════════════════════════╝\n\n", style="dim")
                t.append("  PERCLOS  [", style="white")
                t.append("█" * filled, style=f"bold {c}")
                t.append("░" * (bw - filled), style="dim")
                t.append(f"]  {perclos:.2f}  ", style=f"bold {c}")
                t.append(
                    "⚠ 超阈值 0.15" if perclos > 0.15 else "正常",
                    style="bold red" if perclos > 0.15 else "green",
                )
                live.update(Panel(
                    t, title="[bold magenta]👁  Biometric Sensor  ·  PERCLOS[/bold magenta]",
                    box=box.ROUNDED, border_style="magenta", padding=(0, 1),
                ))
                await asyncio.sleep(speed)
    await asyncio.sleep(0.25)


async def _animate_tts_waveform(
    label: str = "🔊  Voice Broadcasting",
    frames: int = 30,
) -> None:
    """Animated audio waveform — plays concurrently while TTS broadcast is active."""
    import math
    BAR   = "▁▂▃▄▅▆▇█"
    WIDTH = 30
    with Live(console=console, refresh_per_second=15, transient=True) as live:
        for f in range(frames):
            bars = Text("  ")
            for j in range(WIDTH):
                h = int(3.5 + 3.2 * math.sin(j * 0.55 + f * 0.45))
                h = max(0, min(7, h))
                bars.append(BAR[h], style="bold cyan" if j % 2 == 0 else "blue")
            live.update(Panel(
                Group(Text(""), bars, Text(f"\n  {label}", style="dim cyan"), Text("")),
                box=box.ROUNDED, border_style="cyan", padding=(0, 2),
            ))
            await asyncio.sleep(0.068)


_STEAM_ANIM = [
    "   ( )  ( )  ( )  ",
    "  ( )  ( )  ( )   ",
    " (  ) (  ) (  )   ",
    "  (  )(  ) (  )   ",
]


def _make_qr_lines() -> list[str]:
    """
    Build a visually authentic 21×21 QR-code ASCII art.
    Uses correct finder patterns in three corners + timing strips +
    LCG-derived pseudo-random data modules.  Not scannable but indistinguishable
    from a real QR code at a glance.
    """
    G = [[0] * 21 for _ in range(21)]

    def _finder(r0: int, c0: int) -> None:
        for dr in range(7):
            for dc in range(7):
                outer = dr in (0, 6) or dc in (0, 6)
                core  = 2 <= dr <= 4 and 2 <= dc <= 4
                G[r0 + dr][c0 + dc] = 1 if (outer or core) else 0

    _finder(0, 0)    # top-left
    _finder(0, 14)   # top-right
    _finder(14, 0)   # bottom-left

    # Timing strips (row 6 cols 8-12, col 6 rows 8-12)
    for i in range(8, 13):
        G[6][i] = i % 2
        G[i][6] = i % 2

    # Format info area (col 8 rows 0-8, row 8 cols 0-8 — leave as data)

    # LCG fill for data modules
    seed = 0xC0_DE_CA_FE
    for r in range(21):
        for c in range(21):
            if (r < 8 and c < 8) or (r < 8 and c >= 13) or (r >= 14 and c < 8):
                continue                 # finder + separator zones — already set
            if r == 6 or c == 6:
                continue                 # timing strips — already set
            seed = (seed * 6_364_136_223_846_793_005 + 1_442_695_040_888_963_407) \
                   & 0xFFFF_FFFF
            G[r][c] = (seed >> 16) & 1

    # Render: each module → "██" (dark) or "  " (light)
    return ["  " + "".join("██" if cell else "  " for cell in row) for row in G]


async def _animate_coffee_order(shop: str, order: str) -> None:
    """
    Two-phase coffee order animation:
      Phase 1 — steam rising from cup + connecting to merchant system
      Phase 2 — QR code revealed row-by-row, then shown with order details
    """
    qr_lines = _make_qr_lines()          # 21 rows, each 42 chars wide
    P1_FRAMES  = 26                       # steam / loading phase
    P2_FRAMES  = len(qr_lines) + 10      # QR reveal + hold

    with Live(console=console, refresh_per_second=12) as live:

        # ── Phase 1: steam + connecting ──────────────────────────────────────
        for f in range(P1_FRAMES):
            steam  = _STEAM_ANIM[f % len(_STEAM_ANIM)]
            pct    = int(f / (P1_FRAMES - 1) * 100)
            bw     = 18
            filled = int(bw * f / (P1_FRAMES - 1))

            t = Text()
            t.append(f"  {steam}\n", style="dim white")
            t.append("  ┌─────────────────┐\n", style="bold yellow")
            t.append("  │    ☕  COFFEE    │\n", style="bold yellow")
            t.append(f"  │  {order[:13]:<13}  │\n", style="yellow")
            t.append("  └───────┬─────────┘\n", style="bold yellow")
            t.append("          │\n\n", style="dim yellow")
            t.append(f"  📍  {shop}\n", style="bold green")
            t.append(f"  📦  {order}\n\n", style="green")
            t.append("  正在连接商家系统  [", style="white")
            t.append("█" * filled, style="bold green")
            t.append("░" * (bw - filled), style="dim")
            t.append(f"]  {pct}%", style="bold green")
            if f == P1_FRAMES - 1:
                t.append("\n\n  ✅  预订成功！正在生成取餐二维码…", style="bold green")
            else:
                t.append("\n\n  请稍候…", style="dim yellow")

            live.update(Panel(t,
                              title="[bold green]☕  Auto-Order  ·  智能咖啡预订[/bold green]",
                              box=box.ROUNDED, border_style="green", padding=(0, 2)))
            await asyncio.sleep(0.085)

        # ── Phase 2: QR code reveal (row by row, stacked layout) ────────────
        # Side-by-side is avoided entirely: emoji + CJK chars have variable
        # terminal widths, making column alignment impossible without wcwidth.
        # Instead: order details header → QR code → footer, all left-aligned.
        QR_BLANK = "  " + "  " * 21          # placeholder row same width as QR
        SCAN_BAR = "▌"

        for f in range(P2_FRAMES):
            visible = min(len(qr_lines), f + 1)
            done    = visible >= len(qr_lines)

            parts: list = [Text("")]

            # ── header ───────────────────────────────────────────────────────
            parts.append(Text(f"  📍  {shop}    📦  {order}", style="bold green"))
            parts.append(Text(
                "  有效期 24 小时  ·  到达后出示此码向店员扫取",
                style="dim"))
            parts.append(Text("  " + "─" * 44, style="dim"))
            parts.append(Text(""))

            # ── QR rows (revealed top-to-bottom) ─────────────────────────────
            for row in qr_lines[:visible]:
                parts.append(Text(row, style="white"))

            # Keep panel height stable while QR is still appearing
            if not done:
                # scanning cursor line
                parts.append(Text(
                    "  " + "  " * (visible % 21) + SCAN_BAR,
                    style="bold green"))
                # fill remaining blank rows
                for _ in range(len(qr_lines) - visible - 1):
                    parts.append(Text(QR_BLANK))

            # ── footer ────────────────────────────────────────────────────────
            parts.append(Text(""))
            if done:
                parts.append(Text(
                    "  📱  扫码取餐  ✅  二维码已生成  🎉",
                    style="bold green"))
            else:
                dots = "·" * (f % 4 + 1)
                parts.append(Text(f"  生成中 {dots}", style="dim yellow"))
            parts.append(Text(""))

            live.update(Panel(
                Group(*parts),
                title="[bold green]☕  取餐二维码  ·  Auto-Order Confirmed[/bold green]",
                box=box.ROUNDED, border_style="green", padding=(0, 2),
            ))
            await asyncio.sleep(0.065)

    await asyncio.sleep(0.6)


# Thermometer tick marks: bottom = 16°C, top = 30°C, every 2°C
_THERMO_TICKS = [30, 28, 26, 24, 22, 20, 18, 16]   # top → bottom


async def _animate_temperature_set(from_temp: float, to_temp: float) -> None:
    """
    Vertical thermometer animation: mercury drops from from_temp → to_temp.
    Left column shows °C labels; right side shows live current-temp pointer.
    """
    FRAMES = 32

    def _mercury_color(t: float) -> str:
        if t > 24:   return "bold red"
        if t > 21:   return "bold yellow"
        return "bold cyan"

    with Live(console=console, refresh_per_second=20) as live:
        for f in range(FRAMES + 1):
            # Ease-out: fast drop at start, slow settle at end
            p       = f / FRAMES
            eased   = 1.0 - (1.0 - p) ** 2.2
            cur     = from_temp - (from_temp - to_temp) * eased

            lines: list = [Text("")]

            for tick in _THERMO_TICKS:
                is_target  = (tick == int(round(to_temp)))
                filled     = (tick <= cur)           # mercury reaches this level?
                at_cur     = (abs(tick - cur) < 1.5) # closest tick to cur reading

                line = Text()

                # ── left label ────────────────────────────────────────────────
                if is_target:
                    line.append(f"  🎯 {tick:2d}°C  ", style="bold cyan")
                else:
                    line.append(f"     {tick:2d}°C  ", style="dim white")

                # ── thermometer wall + mercury ─────────────────────────────────
                line.append("║", style="bold white")
                if filled:
                    mc = _mercury_color(tick)
                    line.append("▓▓", style=mc)
                else:
                    line.append("  ", style="dim")
                line.append("║", style="bold white")

                # ── right: current-temp live pointer ──────────────────────────
                if at_cur:
                    line.append(f"  ◄── {cur:5.1f}°C", style="bold yellow")
                elif is_target and cur <= to_temp + 0.5:
                    line.append("  ✓ 目标达到", style="bold green")
                else:
                    line.append("")

                lines.append(line)

            # ── bulb ──────────────────────────────────────────────────────────
            bulb = Text()
            bulb.append("          ╰──", style="bold white")
            bulb.append("⬤", style=_mercury_color(cur))
            bulb.append("──╯", style="bold white")
            lines.append(bulb)
            lines.append(Text(""))

            # ── progress bar ──────────────────────────────────────────────────
            prog   = min(1.0, (from_temp - cur) / max(0.01, from_temp - to_temp))
            bw     = 24
            filled_b = int(bw * prog)
            bar = Text()
            bar.append(f"  ❄  {from_temp:.0f}°C → {to_temp:.0f}°C  [", style="cyan")
            bar.append("█" * filled_b,        style="bold cyan")
            bar.append("░" * (bw - filled_b), style="dim")
            bar.append(f"]  {cur:.1f}°C", style="bold white")
            if f == FRAMES:
                bar.append("  ✓ 降温完成", style="bold green")
            else:
                bar.append("  降温中…", style="dim cyan")
            lines.append(bar)
            lines.append(Text(""))

            live.update(Panel(
                Group(*lines),
                title=(
                    f"[bold cyan]❄  车厢温控  ·  "
                    f'vehicle_api.set_temperature(temp={int(to_temp)}, mode="cool")[/bold cyan]'
                ),
                box=box.ROUNDED, border_style="cyan", padding=(0, 2),
            ))
            await asyncio.sleep(0.052)
    await asyncio.sleep(0.50)


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
    await _animate_tts_waveform("🔊  系统语音播报")

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
        await _animate_tts_waveform("🔊  Agent 确认播报")
        console.print()
        await _animate_coffee_order("枫泾服务区", "美式（大杯）")
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
    coffee_order: tuple[str, str] | None = None,
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

    # ── Biometric eye scan before pipeline ───────────────────────────────────
    await _animate_eye_scan(perclos)

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
        await _animate_tts_waveform()   # ~2 s waveform while audio plays in background

    # Show the car HMI warning, then reveal route map below it
    console.print()
    console.print(_panel_car_hud(tier, hud_lines, action_label))
    if route:
        await asyncio.sleep(0.80)
        console.print(_panel_route_map(**route))
    if coffee_order:
        await asyncio.sleep(0.50)
        await _animate_coffee_order(*coffee_order)

    # Proactive vehicle actions fire regardless of how driver responds
    # (open_window is an agent-initiated action, not a reply to driver input)
    if dialogue:
        import re as _re
        for tc in dialogue.get("tool_calls", []):
            if "open_window" in tc.get("call", ""):
                m = _re.search(r"level=(\d+)", tc["call"])
                await _animate_window_open(level=int(m.group(1)) if m else 30)
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
        await _animate_tts_waveform("🔊  Agent Voice Reply")
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
# Randomised scenario generator
# ─────────────────────────────────────────────────────────────────────────────

import random as _random

_MEM = DriverMemory(
    likes_coffee=True,
    preferred_coffee_order="美式（大杯）",
    coffee_max_km=2.0,
    ok_to_pull_over_city=True,
)

# Signal ranges per severity tier
_SIG = {
    SeverityTier.NONE:     dict(perclos=(0.05, 0.12), yawns=(0.0, 0.8),  steering=(3.0,  8.0)),
    SeverityTier.MILD:     dict(perclos=(0.16, 0.25), yawns=(1.1, 2.9),  steering=(9.0,  18.0)),
    SeverityTier.MODERATE: dict(perclos=(0.25, 0.33), yawns=(2.9, 4.3),  steering=(18.0, 27.0)),
    SeverityTier.SEVERE:   dict(perclos=(0.33, 0.46), yawns=(4.3, 6.5),  steering=(27.0, 36.0)),
}

def _rand_signals(tier: SeverityTier, rng: _random.Random) -> tuple[float, float, float]:
    r = _SIG[tier]
    return (
        round(rng.uniform(*r["perclos"]), 2),
        round(rng.uniform(*r["yawns"]),   1),
        round(rng.uniform(*r["steering"]), 1),
    )

def _compute_score(perclos: float, yawns: float, steering: float) -> float:
    """Weighted composite: image 50%, text 30%, audio 20%."""
    img   = min(1.0, perclos / 0.40)
    audio = min(1.0, yawns   / 6.0)
    text  = min(1.0, steering / 30.0)
    return round(text * 0.30 + img * 0.50 + audio * 0.20, 2)

def _auto_reasoning(perclos: float, yawns: float, steering: float) -> str:
    parts: list[str] = []
    if perclos > 0.15:
        parts.append(f"PERCLOS {perclos:.2f} ↑")
    if yawns > 2.0:
        parts.append(f"Yawns {yawns:.1f}/min")
    if steering > 15:
        parts.append(f"Steer {steering:.0f}° drift")
    return "  ·  ".join(parts) if parts else "All signals normal"

def _cn_float(v: float, decimals: int = 1) -> str:
    """Format a float for natural Chinese TTS (e.g. 12.5 → '十二点五')."""
    CN = ["零","一","二","三","四","五","六","七","八","九"]
    s = f"{v:.{decimals}f}"
    out = ""
    for ch in s:
        if ch == ".":
            out += "点"
        elif ch.isdigit():
            out += CN[int(ch)]
        else:
            out += ch
    return out

# ── Predictive round ──────────────────────────────────────────────────────────

def _build_predictive_round(rng: _random.Random) -> dict:
    # Rising score history: 8-10 steps, ends just below 0.30
    n_steps = rng.randint(8, 11)
    start   = round(rng.uniform(0.08, 0.14), 2)
    steps   = [start]
    for _ in range(n_steps - 1):
        steps.append(round(steps[-1] + rng.uniform(0.012, 0.028), 2))
    current = min(steps[-1], 0.29)
    steps[-1] = current

    trend_per_min     = round(rng.uniform(0.016, 0.030), 3)
    time_to_threshold = max(4, int((0.30 - current) / trend_per_min))
    drive_h           = round(rng.uniform(2.8, 3.8), 1)
    gap_km            = rng.randint(18, 28)
    eta_min           = rng.randint(8, 14)

    suggestion = (
        f"当前得分 {current:.2f}，尚未触发警报，但预计 {time_to_threshold} 分钟内持续上升。\n"
        "已主动规划导航并查询周边服务，等待您确认。"
    )
    voice_text = (
        f"系统检测到疲劳趋势持续上升，预计{_cn_float(time_to_threshold, 0)}分钟后触发警报。"
        f"已为您规划导航至前方枫泾服务区，预计{_cn_float(eta_min, 0)}分钟到达。"
        "检测到您偏好咖啡，是否为您提前预订？"
    )
    return dict(
        scores=steps,
        current_score=current,
        trend_per_min=trend_per_min,
        time_to_threshold_min=time_to_threshold,
        scenario_tags=[f"🛣 长途高速", f"⏱ 已行驶 {drive_h}h", "🌤 下午困倦时段"],
        risk_factors=["下午困倦高峰 ×1.3", f"前方 {gap_km*5}km 无服务区", "连续未休息"],
        suggestion=suggestion,
        voice_text=voice_text,
        proactive=dict(
            tool_calls=[
                dict(
                    call=f'navigation_api.plan_route(dest="枫泾服务区", dist_km={gap_km})',
                    result=(
                        f'{{"status": "ready", "route": "G2高速直行·出口32B",'
                        f' "eta_min": {eta_min}}}'
                    ),
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
                dist_km=float(gap_km),
                eta_min=eta_min,
                via=["G2高速直行", "出口32B右转", "服务区匝道入口"],
                layout="highway",
            ),
            voice_yes=(
                f"好的，已为您预订枫泾服务区美式咖啡大杯，到达后直接取餐即可。"
                "导航已启动，请安心驾驶。"
            ),
            voice_no="好的，导航已为您规划好，请根据需要靠站休息。注意保持清醒。",
        ),
    )

# ── Alert rounds ──────────────────────────────────────────────────────────────

def _build_rounds(rng: _random.Random) -> list[dict]:
    # Round 1 — first MILD alert, ignore=0
    p1, y1, s1 = _rand_signals(SeverityTier.MILD, rng)
    sc1         = _compute_score(p1, y1, s1)
    elapsed1    = round(rng.uniform(33, 52), 1)
    minutes1    = rng.randint(165, 205)
    dist1       = round(rng.uniform(9.0, 16.0), 1)
    eta1        = max(5, int(dist1 / 1.6))
    cd1         = BASE_COOLDOWN[SeverityTier.MILD]

    # Round 2 — MILD again, ignore=1, cooldown ×0.8
    p2, y2, s2 = _rand_signals(SeverityTier.MILD, rng)
    # nudge signals slightly higher than round 1 to show persistence
    p2 = max(p2, round(p1 + rng.uniform(0.005, 0.02), 2))
    sc2         = _compute_score(p2, y2, s2)
    cd2_eff     = cd1 * 0.8
    elapsed2    = round(rng.uniform(cd2_eff + 1, cd2_eff + 12), 1)
    minutes2    = minutes1 + rng.randint(8, 15)
    dist2       = round(dist1 - rng.uniform(1.5, 3.0), 1)
    eta2        = max(4, int(dist2 / 1.6))

    # Round 3 — MILD detected but escalated to MODERATE (ignore=2)
    p3, y3, s3 = _rand_signals(SeverityTier.MILD, rng)
    sc3         = _compute_score(p3, y3, s3)
    cd3_eff     = cd1 * 0.6
    elapsed3    = round(rng.uniform(cd3_eff + 1, cd3_eff + 9), 1)
    minutes3    = minutes2 + rng.randint(8, 15)
    coffee_dist = round(rng.uniform(0.8, 1.8), 1)
    coffee_eta  = max(2, int(coffee_dist / 0.45))

    # Round 4 — recovery to NONE, system stays silent
    p4, y4, s4 = _rand_signals(SeverityTier.NONE, rng)
    sc4         = _compute_score(p4, y4, s4)
    minutes4    = minutes3 + rng.randint(5, 12)

    rounds = [
        dict(
            num=1, label="MILD — first alert",
            tier=SeverityTier.MILD,
            score=sc1,
            reasoning=_auto_reasoning(p1, y1, s1),
            ignore=0, elapsed=elapsed1,
            hud_lines=(
                f"建议前往前方服务区休息\n"
                f"沪杭高速服务区  ·  {dist1} 公里"
            ),
            action_label="🖥  Screen alert",
            voice_text=(
                "注意，系统检测到疲劳驾驶。"
                "已为您规划导航：沿高速直行，"
                f"前方三十二号出口右转，约{_cn_float(eta1, 0)}分钟可到达服务区，"
                "建议立即前往稍作休息。"
            ),
            perclos=p1, yawns=y1, steering=s1, minutes=minutes1, road="Highway",
            route=dict(
                road_name="沪杭高速",
                direction="上海方向 →",
                dest_name="沪杭服务区",
                dest_icon="🅿",
                dist_km=dist1,
                eta_min=eta1,
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
            tier=SeverityTier.MILD,
            score=sc2,
            reasoning=_auto_reasoning(p2, y2, s2),
            ignore=1, elapsed=elapsed2,
            hud_lines=(
                f"已连续行驶 {minutes2 // 60} 小时，轻度疲劳\n"
                f"服务区  ·  {dist2} 公里  ·  约 {eta2} 分钟"
            ),
            action_label="🖥  Screen alert",
            voice_text=(
                f"提醒您，已连续驾驶超过{_cn_float(minutes2//60, 0)}小时，检测到轻度疲劳。"
                f"导航已更新，前方服务区距您{_cn_float(dist2)}公里，"
                f"预计{_cn_float(eta2, 0)}分钟到达，请保持注意力，计划靠站休息。"
            ),
            perclos=p2, yawns=y2, steering=s2, minutes=minutes2, road="Highway",
            route=dict(
                road_name="沪杭高速",
                direction="上海方向 →",
                dest_name="沪杭服务区",
                dest_icon="🅿",
                dist_km=dist2,
                eta_min=eta2,
                via=["高速直行", "三十二号出口右转", "服务区匝道入口"],
                layout="highway",
            ),
        ),
        dict(
            num=3, label="MILD detected — escalated to MODERATE",
            tier=SeverityTier.MILD,
            score=sc3,
            reasoning="2 ignores → effective MODERATE.",
            ignore=2, elapsed=elapsed3,
            hud_lines=(
                f"疲劳警报升级  ·  已自动预订咖啡\n"
                f"星巴克来福士广场  ·  {coffee_dist} 公里"
            ),
            action_label="🔊  Voice  ·  ☕ Auto-order",
            voice_text=(
                "警告，疲劳警报升级。"
                f"已为您自动预订附近{_cn_float(coffee_dist)}公里的星巴克美式咖啡，"
                f"导航已启动：前方路口右转，沿滨江大道南行，"
                f"约{_cn_float(coffee_eta, 0)}分钟到达来福士广场，请前往取餐休息。"
            ),
            perclos=p3, yawns=y3, steering=s3, minutes=minutes3, road="Highway",
            route=dict(
                road_name="滨江大道",
                direction="市区方向 ↓",
                dest_name="星巴克（来福士广场）",
                dest_icon="☕",
                dist_km=coffee_dist,
                eta_min=coffee_eta,
                via=["前方路口右转", "滨江大道南行", "来福士广场停车场"],
                layout="city",
            ),
            coffee_order=("星巴克（来福士广场）", "美式（大杯）"),
        ),
        dict(
            num=4, label="Score drops — system stays silent",
            tier=SeverityTier.NONE,
            score=sc4,
            reasoning=_auto_reasoning(p4, y4, s4),
            ignore=0, elapsed=35.0,
            hud_lines="", action_label="", voice_text="",
            perclos=p4, yawns=y4, steering=s4, minutes=minutes4, road="Highway",
            route=None,
        ),
    ]
    return rounds, [
        (sc1, elapsed1, cd1, 1.0),
        (sc2, elapsed2, cd1, 0.8),
        (sc3, elapsed3, cd1, 0.6),
        (sc4, 35.0, 0.0, 0.0),
    ]

# ─────────────────────────────────────────────────────────────────────────────
# Extra proactive scenario panels
# ─────────────────────────────────────────────────────────────────────────────

def _panel_night_watch(
    raw_score: float,
    adj_score: float,
    multiplier: float,
    hour: int,
    minute: int,
    road: str,
    minutes: int,
) -> Panel:
    """Circadian risk amplification: raw score × time multiplier → MILD fires."""
    time_str = f"{hour:02d}:{minute:02d} AM"
    window   = "02:00–05:00 AM  深夜生物钟低谷"

    raw_col = Text()
    raw_col.append(f"  时间      {time_str}\n", style="bold white")
    raw_col.append(f"  危险窗口  {window}\n",   style="yellow")
    raw_col.append(f"\n  {road} · {minutes}min", style="dim")

    amp_col = Text()
    amp_col.append(f"  基础得分  ", style="white")
    amp_col.append_text(_score_bar(raw_score, width=10))
    amp_col.append(f"\n  时间倍率  ×{multiplier}\n", style="bold yellow")
    amp_col.append(f"  调整得分  ", style="white")
    amp_col.append_text(_score_bar(adj_score, width=10))

    dec_col = Text()
    dec_col.append("⚠️   MILD\n", style="bold yellow")
    dec_col.append("调整后触发\n", style="yellow")
    dec_col.append("→ 主动介入", style="bold cyan")

    tbl = Table(box=box.SIMPLE_HEAD, expand=True, show_header=True,
                padding=(0, 2), show_edge=False)
    tbl.add_column("时间上下文", ratio=1)
    tbl.add_column("得分放大",   ratio=1)
    tbl.add_column("判决",       ratio=1)
    tbl.add_row(raw_col, amp_col, dec_col)
    return Panel(
        tbl,
        title="[bold]🌙  Night Watch · 凌晨危险时段[/bold]",
        box=box.DOUBLE_EDGE, border_style="bright_blue", padding=(0, 1),
    )


def _panel_compound_risk(
    fatigue: float,
    w_factor: float,
    fog_factor: float,
    compound: float,
    visibility_m: int,
    rain_level: str,
) -> Panel:
    """Fatigue × weather → compound risk exceeds MILD threshold."""
    eff_tier = SeverityTier.MODERATE if compound >= 0.55 else SeverityTier.MILD

    fat_col = Text()
    fat_col.append("  疲劳得分\n", style="dim")
    fat_col.append_text(_score_bar(fatigue, width=10))
    fat_col.append(f"\n\n  PERCLOS / 打哈欠 / 方向盘", style="dim")

    risk_col = Text()
    risk_col.append(f"  雨天系数  ×{w_factor:.2f}\n",   style="yellow")
    risk_col.append(f"  能见度系数  ×{fog_factor:.2f}\n", style="yellow")
    risk_col.append(f"\n  {rain_level}  ·  能见度 {visibility_m}m\n", style="white")
    risk_col.append("  路面湿滑  ·  制动距离 +40%", style="red")

    cmp_col = Text()
    cmp_col.append(f"  {fatigue:.2f} × {w_factor:.2f} × {fog_factor:.2f}\n",
                   style="dim white")
    cmp_col.append("  = ", style="white")
    cmp_col.append_text(_score_bar(compound, width=10))
    cmp_col.append(f"\n\n  {TIER_ICON[eff_tier]} {eff_tier.name}", style=f"bold {TIER_COLOR[eff_tier]}")
    cmp_col.append("  ↑ 复合升级", style="bold red")

    tbl = Table(box=box.SIMPLE_HEAD, expand=True, show_header=True,
                padding=(0, 2), show_edge=False)
    tbl.add_column("疲劳信号",   ratio=1)
    tbl.add_column("天气风险",   ratio=1)
    tbl.add_column("复合得分",   ratio=1)
    tbl.add_row(fat_col, risk_col, cmp_col)
    return Panel(
        tbl,
        title="[bold]⛈  Compound Risk · 疲劳 × 天气复合风险[/bold]",
        box=box.DOUBLE_EDGE, border_style="yellow", padding=(0, 1),
    )


def _panel_memory_lookup(promise_text: str, elapsed_min: int, promised_min: int) -> Panel:
    """Agent retrieves memory promise and shows elapsed vs promised time."""
    overdue = elapsed_min - promised_min
    lines: list = [Text("")]

    call_ln = Text()
    call_ln.append("  🔧  tool_call  ", style="bold yellow")
    call_ln.append("driver_memory.get_notes()", style="bold white")
    lines.append(call_ln)

    resp_ln = Text()
    resp_ln.append("             → ", style="dim yellow")
    resp_ln.append(f'"{promise_text}"', style="green")
    lines.append(resp_ln)
    lines.append(Text(""))

    time_ln = Text()
    time_ln.append("  🔧  tool_call  ", style="bold yellow")
    time_ln.append("time_tracker.elapsed_since_promise()", style="bold white")
    lines.append(time_ln)

    elapsed_ln = Text()
    elapsed_ln.append("             → ", style="dim yellow")
    elapsed_ln.append(f"elapsed={elapsed_min}min  promised={promised_min}min  ",
                      style="white")
    elapsed_ln.append(f"overdue +{overdue}min", style="bold red")
    lines.append(elapsed_ln)
    lines.append(Text(""))
    return Panel(
        Group(*lines),
        title="[bold]📝  Memory Recall · 承诺追踪[/bold]",
        box=box.ROUNDED, border_style="bright_yellow", padding=(0, 1),
    )


# ── Night Watch runner ────────────────────────────────────────────────────────

async def run_night_watch_round(*, rng: _random.Random, memory: DriverMemory) -> str | None:
    """凌晨危险时段 — score × circadian multiplier fires MILD before raw threshold."""
    raw_score = round(rng.uniform(0.20, 0.27), 2)
    multiplier = 1.6
    adj_score  = round(raw_score * multiplier, 2)
    hour       = rng.choice([2, 3, 4])
    minute     = rng.randint(0, 59)
    minutes    = rng.randint(240, 310)
    road       = "Highway"
    dist_km    = round(rng.uniform(18, 32), 1)
    eta_min    = max(10, int(dist_km / 1.5))

    p, y, s   = _rand_signals(SeverityTier.NONE, rng)
    tier      = SeverityTier.MILD if adj_score >= 0.30 else SeverityTier.NONE
    from_temp = round(rng.uniform(22.0, 25.5), 1)   # current cabin temp
    to_temp   = 20.0

    tool_calls = [
        dict(
            call=f'time_risk_api.get_multiplier(hour={hour})',
            result=f'{{"multiplier": {multiplier}, "reason": "circadian_low", "window": "02:00-05:00"}}',
        ),
        dict(
            call=f'vehicle_api.set_temperature(temp={int(to_temp)}, mode="cool")',
            result=f'{{"status": "ok", "prev_temp": {from_temp}, "new_temp": {int(to_temp)}}}',
        ),
        dict(
            call='vehicle_api.play_music(type="alert_energetic", volume=45)',
            result='{"status": "ok", "track": "Morning Drive Beats Vol.3"}',
        ),
    ]
    question = (
        "已检测到凌晨危险时段 🌙  得分经时间倍率放大已达轻度预警。\n"
        "  Agent 已: 预降车厢温度 20°C + 播放提神音乐 —— 是否确认？"
    )
    voice_text = (
        f"注意，现在是凌晨{_cn_float(hour, 0)}点，生物钟处于最低谷，"
        "疲劳风险经时间倍率放大已触发轻度预警。"
        "系统已为您调低车厢温度并播放提神音乐，建议在前方服务区稍作休整。"
    )

    console.print()
    console.print(Rule(
        "[bold bright_blue]  🌙  Night Watch  ·  凌晨危险时段主动干预  [/bold bright_blue]",
        style="bright_blue", characters="━"))
    console.print()

    audio_task = asyncio.create_task(_fetch_audio(voice_text))

    sections: list = []
    done: set[int] = set()
    with Live(console=console, refresh_per_second=16) as live:
        for i in range(len(PIPELINE_STAGES)):
            sections = [_panel_pipeline(active=i, done=done)]
            live.update(Group(*sections))
            await asyncio.sleep(0.24)
            done.add(i)
        sections = [_panel_pipeline(active=-1, done=done)]
        live.update(Group(*sections))
        await asyncio.sleep(0.15)

        sections.append(_panel_night_watch(raw_score, adj_score, multiplier,
                                           hour, minute, road, minutes))
        live.update(Group(*sections))
        await asyncio.sleep(1.0)

        sections.append(_panel_agent(memory))
        live.update(Group(*sections))
        await asyncio.sleep(0.60)

    audio_path = await audio_task
    await _play_audio(audio_path, fallback_text=voice_text)
    await _animate_tts_waveform("🔊  凌晨危险时段播报")

    console.print()
    console.print(_panel_proactive_actions(tool_calls, question))
    console.print()
    console.print(Align.center(
        Text("  [ Y ]  好，启用提神模式      [ N ]  不用了  ",
             style="bold white on grey23")))
    console.print()

    key = await _wait_keypress()
    _stop_audio()

    if key.lower() == "y":
        conf_text = (
            f"好的，车厢正在从{_cn_float(from_temp)}摄氏度降温至二十摄氏度，提神音乐已播放。"
            "请保持注意力，前方服务区建议稍作休息。"
        )
        # Start TTS fetch in background; run temperature animation immediately
        conf_audio_task = asyncio.create_task(_fetch_audio(conf_text))
        await _animate_temperature_set(from_temp, to_temp)
        # Play confirmation voice after animation completes
        conf_path = await conf_audio_task
        await _play_audio(conf_path, fallback_text=conf_text)
        await _animate_tts_waveform("🔊  提神模式确认播报")
        _stop_audio()
        ok_ln = Text()
        ok_ln.append("  ✅  ", style="bold green")
        ok_ln.append(f"提神模式已启动：车厢 {from_temp}°C → {int(to_temp)}°C · 提神音乐播放中",
                     style="green")
        console.print()
        console.print(Panel(ok_ln, border_style="green", box=box.ROUNDED, padding=(0, 1)))
        response = "ACKNOWLEDGED"
    else:
        response = "IGNORED"

    global _tts_proc
    if _tts_proc is not None:
        try:
            await asyncio.wait_for(_tts_proc.wait(), timeout=12.0)
        except asyncio.TimeoutError:
            pass

    ig_after = 1 if response == "IGNORED" else 0
    console.print()
    console.print(_panel_driver_reaction(response, 0, ig_after, BASE_COOLDOWN[SeverityTier.MILD], None))
    await asyncio.sleep(1.5)
    return response


# ── Promise Keeper runner ─────────────────────────────────────────────────────

async def run_promise_keeper_round(
    *,
    rng: _random.Random,
    memory: DriverMemory,
    elapsed_min: int,
    rest_km: float,
) -> str | None:
    """承诺追踪 — agent retrieves the driver's 30-min promise and follows up."""
    promised_min  = 30
    overdue_min   = elapsed_min - promised_min
    eta_min       = max(3, int(rest_km / 1.4))
    p, y, s       = _rand_signals(SeverityTier.MILD, rng)
    score         = _compute_score(p, y, s)

    promise_note = next(
        (n for n in memory.notes if "承诺" in n or "三十" in n or "30" in n),
        "承诺三十分钟后休息（已超时提醒）",
    )

    tool_calls = [
        dict(
            call="driver_memory.get_notes()",
            result=f'["{promise_note}"]',
        ),
        dict(
            call="time_tracker.elapsed_since_promise()",
            result=(
                f'{{"elapsed_min": {elapsed_min}, "promised_min": {promised_min},'
                f' "overdue_min": {overdue_min}}}'
            ),
        ),
        dict(
            call=f'navigation_api.find_nearest_rest_stop(max_km={rest_km + 2})',
            result=(
                f'{{"name": "沪杭高速服务区", "dist_km": {rest_km},'
                f' "eta_min": {eta_min}, "facilities": ["toilet","coffee","nap_room"]}}'
            ),
        ),
    ]
    question = (
        f"您之前承诺 {promised_min} 分钟后休息，已过 {elapsed_min} 分钟（超时 {overdue_min} min）。\n"
        f"  前方 {rest_km} km 服务区可提供休息 — 是否立即前往？"
    )
    voice_text = (
        f"您之前承诺{_cn_float(promised_min, 0)}分钟后休息，"
        f"现已过{_cn_float(elapsed_min, 0)}分钟，超时{_cn_float(overdue_min, 0)}分钟。"
        f"前方{_cn_float(rest_km)}公里服务区可提供休息，建议立即前往。"
    )

    console.print()
    console.print(Rule(
        "[bold bright_yellow]  📝  Promise Keeper  ·  承诺追踪  [/bold bright_yellow]",
        style="bright_yellow", characters="━"))
    console.print()

    audio_task = asyncio.create_task(_fetch_audio(voice_text))

    sections: list = []
    done: set[int] = set()
    with Live(console=console, refresh_per_second=16) as live:
        for i in range(len(PIPELINE_STAGES)):
            sections = [_panel_pipeline(active=i, done=done)]
            live.update(Group(*sections))
            await asyncio.sleep(0.24)
            done.add(i)
        sections = [_panel_pipeline(active=-1, done=done)]
        live.update(Group(*sections))
        await asyncio.sleep(0.15)

        sections.append(_panel_memory_lookup(promise_note, elapsed_min, promised_min))
        live.update(Group(*sections))
        await asyncio.sleep(1.0)

        sections.append(_panel_problem_id(
            SeverityTier.MILD, score, _auto_reasoning(p, y, s),
            0, 35.0, True, p, y, s, elapsed_min + 170, "Highway",
        ))
        live.update(Group(*sections))
        await asyncio.sleep(0.80)

    audio_path = await audio_task
    await _play_audio(audio_path, fallback_text=voice_text)
    await _animate_tts_waveform("🔊  承诺追踪播报")

    console.print()
    console.print(_panel_proactive_actions(tool_calls, question))

    route = dict(
        road_name="沪杭高速", direction="上海方向 →",
        dest_name="沪杭高速服务区", dest_icon="🅿",
        dist_km=rest_km, eta_min=eta_min,
        via=["高速直行", "服务区出口匝道", "停车场入口"],
        layout="highway",
    )
    await asyncio.sleep(0.60)
    console.print(_panel_route_map(**route))
    console.print()
    console.print(Align.center(
        Text("  [ Y ]  好，马上去休息      [ N ]  再坚持一会  ",
             style="bold white on grey23")))
    console.print()

    key = await _wait_keypress()
    _stop_audio()

    if key.lower() == "y":
        conf_text = "好的，导航已启动，请按提示前往服务区休息，安全抵达后记得休息片刻。"
        path = await _fetch_audio(conf_text)
        await _play_audio(path, fallback_text=conf_text)
        ok_ln = Text()
        ok_ln.append("  ✅  ", style="bold green")
        ok_ln.append("已启动导航前往服务区 · memory 承诺状态已标记为「履行」", style="green")
        console.print()
        console.print(Panel(ok_ln, border_style="green", box=box.ROUNDED, padding=(0, 1)))
        memory.notes.append("承诺已履行：前往服务区休息")
        response = "ACKNOWLEDGED"
    else:
        warn_ln = Text()
        warn_ln.append("  ⚠️  ", style="bold red")
        warn_ln.append(f"已超时 {overdue_min} 分钟 · 下次检测将自动升级为 MODERATE", style="red")
        console.print()
        console.print(Panel(warn_ln, border_style="red", box=box.ROUNDED, padding=(0, 1)))
        response = "IGNORED"

    global _tts_proc
    if _tts_proc is not None:
        try:
            await asyncio.wait_for(_tts_proc.wait(), timeout=12.0)
        except asyncio.TimeoutError:
            pass
    console.print()
    console.print(_panel_driver_reaction(response, 0, 1 if response == "IGNORED" else 0,
                                         BASE_COOLDOWN[SeverityTier.MILD], None))
    await asyncio.sleep(1.5)
    return response


# ── Compound Risk runner ──────────────────────────────────────────────────────

async def run_compound_risk_round(*, rng: _random.Random, memory: DriverMemory) -> str | None:
    """复合风险 — MILD fatigue + rain/fog → compound score escalates to MODERATE."""
    p, y, s       = _rand_signals(SeverityTier.MILD, rng)
    fatigue       = _compute_score(p, y, s)
    rain_choices  = [("小雨", 1.20, "light_rain"), ("中雨", 1.35, "moderate_rain"),
                     ("大雨", 1.55, "heavy_rain")]
    rain_label, w_factor, rain_code = rng.choice(rain_choices)
    visibility    = rng.randint(80, 200)
    fog_factor    = round(1.0 + (200 - visibility) / 200 * 0.4, 2)
    compound      = round(fatigue * w_factor * fog_factor, 2)
    eff_tier      = SeverityTier.MODERATE if compound >= 0.55 else SeverityTier.MILD
    minutes       = rng.randint(180, 230)
    dist_km       = round(rng.uniform(5, 14), 1)
    eta_min       = max(3, int(dist_km / 1.4))
    lat, lon      = round(rng.uniform(30.2, 31.2), 2), round(rng.uniform(120.8, 121.8), 2)

    tool_calls = [
        dict(
            call=f'weather_api.get_conditions(lat={lat}, lon={lon})',
            result=(
                f'{{"weather": "{rain_code}", "visibility_m": {visibility},'
                f' "road_wet": true, "wind_kph": {rng.randint(12, 28)}}}'
            ),
        ),
        dict(
            call=(
                f'risk_model.compound_score(fatigue={fatigue:.2f},'
                f' visibility={visibility}, road_wet=True)'
            ),
            result=(
                f'{{"compound": {compound:.2f}, "w_factor": {w_factor},'
                f' "fog_factor": {fog_factor}, "effective_tier": "{eff_tier.name}"}}'
            ),
        ),
        dict(
            call=f'navigation_api.find_nearest_shelter(max_km={dist_km + 3})',
            result=(
                f'{{"name": "嘉兴南服务区", "dist_km": {dist_km},'
                f' "eta_min": {eta_min}, "has_shelter": true}}'
            ),
        ),
    ]
    question = (
        f"疲劳得分 {fatigue:.2f} 经天气复合放大 → {compound:.2f}，已达 {eff_tier.name}。\n"
        f"  {rain_label} · 能见度 {visibility}m — 建议立即前往有遮蔽的服务区，是否导航？"
    )
    voice_text = (
        f"注意，当前{rain_label}，能见度仅{_cn_float(visibility, 0)}米，"
        "路面湿滑制动距离明显增加。"
        f"系统综合疲劳与天气风险，实际风险等级已提升至{eff_tier.name}。"
        f"前方{_cn_float(dist_km)}公里嘉兴南服务区有遮蔽设施，建议立即前往休整。"
    )

    console.print()
    console.print(Rule(
        "[bold yellow]  ⛈  Compound Risk  ·  疲劳 × 天气复合风险  [/bold yellow]",
        style="yellow", characters="━"))
    console.print()

    audio_task = asyncio.create_task(_fetch_audio(voice_text))
    await _animate_eye_scan(p)

    sections: list = []
    done: set[int] = set()
    with Live(console=console, refresh_per_second=16) as live:
        for i in range(len(PIPELINE_STAGES)):
            sections = [_panel_pipeline(active=i, done=done)]
            live.update(Group(*sections))
            await asyncio.sleep(0.24)
            done.add(i)
        sections = [_panel_pipeline(active=-1, done=done)]
        live.update(Group(*sections))
        await asyncio.sleep(0.15)

        sections.append(_panel_compound_risk(fatigue, w_factor, fog_factor,
                                             compound, visibility, rain_label))
        live.update(Group(*sections))
        await asyncio.sleep(1.0)

        sections.append(_panel_agent(memory))
        live.update(Group(*sections))
        await asyncio.sleep(0.60)

    audio_path = await audio_task
    await _play_audio(audio_path, fallback_text=voice_text)
    await _animate_tts_waveform("🔊  复合风险播报")

    console.print()
    console.print(_panel_proactive_actions(tool_calls, question))

    route = dict(
        road_name="沪杭高速", direction="嘉兴方向 →",
        dest_name="嘉兴南服务区", dest_icon="🏠",
        dist_km=dist_km, eta_min=eta_min,
        via=["高速直行", "嘉兴南出口", "服务区匝道入口"],
        layout="highway",
    )
    await asyncio.sleep(0.60)
    console.print(_panel_car_hud(eff_tier,
        f"复合风险  ·  {rain_label} · 能见度 {visibility}m\n前方 {dist_km}km 服务区 — 建议立即靠站",
        "🔊  Voice  ·  ⛈ 天气风险预警"))
    await asyncio.sleep(0.40)
    console.print(_panel_route_map(**route))
    console.print()
    console.print(Align.center(
        Text("  [ Y ]  好，导航前往      [ N ]  继续行驶  ",
             style="bold white on grey23")))
    console.print()

    key = await _wait_keypress()
    _stop_audio()

    if key.lower() == "y":
        conf_text = "好的，已启动前往嘉兴南服务区的导航，请注意保持安全车距，谨慎驾驶。"
        path = await _fetch_audio(conf_text)
        await _play_audio(path, fallback_text=conf_text)
        ok_ln = Text()
        ok_ln.append("  ✅  ", style="bold green")
        ok_ln.append("导航已启动 · 行驶中请保持安全车距，制动距离 +40%", style="green")
        console.print()
        console.print(Panel(ok_ln, border_style="green", box=box.ROUNDED, padding=(0, 1)))
        response = "ACKNOWLEDGED"
    else:
        warn_ln = Text()
        warn_ln.append("  ⚠️  ", style="bold red")
        warn_ln.append("继续行驶中 · 复合风险保持 MODERATE 级别监控", style="red")
        console.print()
        console.print(Panel(warn_ln, border_style="red", box=box.ROUNDED, padding=(0, 1)))
        response = "IGNORED"

    global _tts_proc
    if _tts_proc is not None:
        try:
            await asyncio.wait_for(_tts_proc.wait(), timeout=12.0)
        except asyncio.TimeoutError:
            pass
    console.print()
    console.print(_panel_driver_reaction(response, 0, 1 if response == "IGNORED" else 0,
                                         BASE_COOLDOWN[eff_tier], None))
    await asyncio.sleep(1.5)
    return response


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Bosch Fatigue Monitor — Proactive Agent Demo")
    ap.add_argument("--night",    action="store_true", help="Add Night Watch scenario (circadian risk)")
    ap.add_argument("--promise",  action="store_true", help="Add Promise Keeper scenario (memory follow-up)")
    ap.add_argument("--compound", action="store_true", help="Add Compound Risk scenario (fatigue × weather)")
    ap.add_argument("--all",      action="store_true", help="Run all extra proactive scenarios")
    args = ap.parse_args()

    do_night    = args.night    or args.all
    do_promise  = args.promise  or args.all
    do_compound = args.compound or args.all

    rng = _random.Random()          # new seed each run → different signals every replay

    console.print()
    extra_tags: list[str] = []
    if do_night:    extra_tags.append("[bold bright_blue]🌙 Night[/bold bright_blue]")
    if do_promise:  extra_tags.append("[bold bright_yellow]📝 Promise[/bold bright_yellow]")
    if do_compound: extra_tags.append("[bold yellow]⛈ Compound[/bold yellow]")
    extras_line = (
        ("  Extra: " + "  ·  ".join(extra_tags))
        if extra_tags else "  [dim]--night / --promise / --compound / --all[/dim]"
    )
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
            Text.from_markup(extras_line, justify="center"),
            Text(""),
        )),
        border_style="blue", box=box.HEAVY,
    ))
    await asyncio.sleep(1.2)

    mem   = _MEM
    pred  = _build_predictive_round(rng)
    rounds, meta = _build_rounds(rng)

    # summary_rows: (label, scenario_tag, resp)
    summary_rows: list[tuple[str, str, str | None]] = []

    # ── Predictive round ──────────────────────────────────────────────────────
    await run_predictive_round(memory=mem, **pred)

    # ── Optional: Night Watch ─────────────────────────────────────────────────
    if do_night:
        resp = await run_night_watch_round(rng=rng, memory=mem)
        summary_rows.append(("Night Watch", "🌙  MILD (×1.6 time)", resp))
        await asyncio.sleep(2.5)

    # ── Standard rounds 1-4, with optional extras injected ───────────────────
    for rd in rounds:
        resp = await run_round(memory=mem, **rd)
        tier = rd["tier"]
        ign  = rd["ignore"]
        if tier == SeverityTier.NONE:
            tag = f"NONE  {rd['score']:.2f}"
        else:
            eff = tier.name if ign < 2 else SeverityTier(int(tier) + 1).name
            tag = (f"{tier.name}→{eff}" if eff != tier.name else tier.name) + f"  {rd['score']:.2f}"
        summary_rows.append((f"Round {rd['num']}", tag, resp))

        if resp == "IGNORED":
            mem.on_alert_ignored()
        elif resp == "ACKNOWLEDGED":
            mem.on_alert_acknowledged()
        await asyncio.sleep(3.0)

        # Promise Keeper fires after Round 1 acknowledgement (driver made a promise)
        if rd["num"] == 1 and do_promise:
            elapsed = rng.randint(31, 46)
            rest_km = round(rng.uniform(4.0, 9.0), 1)
            resp_pk = await run_promise_keeper_round(
                rng=rng, memory=mem, elapsed_min=elapsed, rest_km=rest_km)
            summary_rows.append(("Promise Keeper", f"📝  MILD +{elapsed-30}min overdue", resp_pk))
            await asyncio.sleep(2.5)

        # Compound Risk fires after Round 2
        if rd["num"] == 2 and do_compound:
            resp_cr = await run_compound_risk_round(rng=rng, memory=mem)
            summary_rows.append(("Compound Risk", "⛈  MILD × weather", resp_cr))
            await asyncio.sleep(2.5)

    # ── Summary timeline ──────────────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold]Decision Timeline[/bold]", style="blue", characters="━"))
    console.print()

    tbl = Table(box=box.ROUNDED, expand=True, show_lines=True)
    tbl.add_column("#",       style="bold", width=3,  justify="center")
    tbl.add_column("Scenario",              width=16)
    tbl.add_column("Signal / Tag",          width=22)
    tbl.add_column("Response",              width=16)

    for idx, (scenario, tag, resp) in enumerate(summary_rows, 1):
        if resp == "ACKNOWLEDGED":
            resp_cell = Text("✓  Acknowledged", style="green")
        elif resp == "IGNORED":
            resp_cell = Text("✗  Ignored", style="red")
        else:
            resp_cell = Text("—", style="dim")
        tbl.add_row(str(idx), scenario, tag, resp_cell)

    console.print(tbl)
    console.print()
    console.print(Panel.fit(
        "  [dim]python proactive_demo.py[/dim]              replay  (new random signals)\n"
        "  [dim]python proactive_demo.py --night[/dim]      add Night Watch scenario\n"
        "  [dim]python proactive_demo.py --promise[/dim]    add Promise Keeper scenario\n"
        "  [dim]python proactive_demo.py --compound[/dim]   add Compound Risk scenario\n"
        "  [dim]python proactive_demo.py --all[/dim]        run all extra scenarios\n"
        "  [dim]python demo.py[/dim]                        full 3-driver scenario",
        title="[bold]Run[/bold]", border_style="green",
    ))


if __name__ == "__main__":
    asyncio.run(main())
