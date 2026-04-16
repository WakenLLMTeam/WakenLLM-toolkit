# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Bosch Fatigue Monitor Agent** is a real-time driver fatigue detection system that uses multimodal sensor inputs (text/telemetry, face imagery, audio) to assess driver drowsiness and trigger context-aware interventions. It implements an LLM-as-a-Judge architecture with profile-aware adaptive cooldowns and action escalation.

The system is designed for two driver profiles:
- **Long-Haul**: truck/bus drivers on highways; strict thresholds, early intervention
- **Commuter**: daily city drivers; relaxed thresholds, gentle reminders

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        PIPELINE FLOW                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Sensors → EventBus → StateAggregator → (Judge Queue)            │
│                                           ↓                       │
│                                   ContextEnricher                 │
│                                           ↓                       │
│                              FatigueJudgeAgent (LLM)              │
│                                           ↓                       │
│                                   (Verdict Queue)                 │
│                                           ↓                       │
│                              OrchestratorAgent                    │
│                           (Cooldown + Escalation)                │
│                                           ↓                       │
│                  [Screen] [Voice] [Video] [Push] [Context]       │
│                         Action Agents                            │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Async Pipeline**: All components are async-first using asyncio.Queue for decoupled communication.
2. **Rolling Windows**: Text/image/audio signals are aggregated over a configurable observation window (default 60s) before threshold evaluation.
3. **LLM-as-Judge**: Multimodal reasoning (text prompts + optional face image B64) produces composite fatigue score (0-1) and severity tier (NONE/MILD/MODERATE/SEVERE).
4. **Pyramid Escalation**: SEVERE tier includes all lower-tier actions; ignored alerts trigger adaptive cooldown shortening.
5. **Profile Awareness**: Thresholds and action messages adapt to driver profile (long-haul vs. commuter).
6. **Driver Memory**: Per-driver preferences (coffee lover, prefers naps, etc.) personalize action recommendations.

## Directory Structure

```
bosch_fatigue_monitor/
├── config.py                 # Central config: thresholds, models, cooldowns, intervals
├── main.py                   # Entry point; builds pipeline & sensors; three run modes
├── demo.py                   # Interactive terminal demo (two real scenarios)
├── decision_matrix.py        # Decision heatmap: 20 score → action comparison
├── interactive.py            # Sensor input REPL (manually tune signals, watch decisions)
├── quantify.py              # Score analysis: grids, sweeps, case studies
├── models/
│   ├── fatigue_context.py    # FatigueContext, EnrichedFatigueContext, signal dataclasses
│   ├── judge_verdict.py      # JudgeVerdict, SeverityTier, ModalityScore
│   ├── sensor_event.py       # SensorEvent, SignalType enum
│   ├── action_result.py      # ActionResult (success/error/message)
│   └── driver_memory.py      # DriverMemory (preferences, history); COFFEE_LOVER preset
├── pipeline/
│   ├── event_bus.py          # EventBus (observer pattern for sensor events)
│   ├── state_aggregator.py   # RollingWindow + threshold checking; emits FatigueContext
│   ├── context_enricher.py   # Adds map context (road type, rest/coffee proximity, traffic)
│   ├── driver_profile.py     # classify_profile() & get_thresholds() for profile adaptation
│   └── model_router.py       # LLM endpoint selection (Bosch, OpenAI, or mock)
├── judge/
│   ├── judge_agent.py        # FatigueJudgeAgent: calls LLM; returns JudgeVerdict
│   ├── prompt_template.py    # SYSTEM_PROMPT & build_user_prompt() (vision-aware)
│   └── response_parser.py    # JSON parse verdict from LLM response
├── orchestrator/
│   ├── orchestrator_agent.py # Dispatch logic; calls action agents in tier order
│   └── cooldown_tracker.py   # Per-tier cooldown; tracks ignored counts; escalates effective tier
├── actions/
│   ├── base.py               # ActionAgent ABC
│   ├── screen_display_agent.py       # In-vehicle screen message (profile-aware tone)
│   ├── voice_broadcast_agent.py      # Voice alert (qwen3-tts-flash or mock)
│   ├── video_record_agent.py         # Start/stop video recording
│   ├── phone_push_agent.py           # HTTP POST to phone push endpoint
│   └── context_action_agent.py       # "Smart" suggestion: order coffee, nap, pull over
├── sensors/
│   ├── base.py               # SensorInterface ABC
│   ├── driving_duration_sensor.py    # Tracks cumulative minutes; emits milestone events
│   └── mocks/
│       ├── mock_text_sensor.py       # Simulates telemetry: speed, steering, brake
│       ├── mock_image_sensor.py      # Simulates face PERCLOS + lane deviation
│       └── mock_audio_sensor.py      # Simulates yawns, verbal fatigue, ASR transcript
├── llm/
│   ├── base.py               # LLMClientInterface ABC
│   ├── mock_llm_client.py    # Deterministic response; no API key needed (local dev)
│   ├── openai_client.py      # Bosch gateway (gpt-5, multimodal) — default for demo
│   ├── gpt_client.py         # OpenAI direct (legacy; requires OPENAI_API_KEY)
│   ├── qwen_asr_client.py    # Audio → text (qwen3-asr-flash)
│   └── qwen_tts_client.py    # Text → audio (qwen3-tts-flash)
├── tests/
│   ├── conftest.py           # Pytest fixtures: cfg, alert_fatigue, mock_verdict, etc.
│   ├── test_state_aggregator.py      # RollingWindow, threshold crossing logic
│   ├── test_judge.py         # Response parsing, verdict generation
│   ├── test_cooldown.py      # Cooldown tracker, adaptive escalation
│   ├── test_orchestrator.py  # Action dispatch pyramid, tier ordering
│   ├── test_actions.py       # Profile-aware message generation
│   ├── test_context_enricher.py      # Map context enrichment
│   └── test_rolling_window.py        # RollingWindow edge cases
└── requirements.txt          # openai, python-dotenv
```

## Common Commands

### Run the System

```bash
# Default: MockLLMClient (no API key needed; deterministic output)
cd bosch_fatigue_monitor
python main.py

# Bosch gateway (gpt-5, multimodal) — requires Bosch API key in config.py
python main.py --bosch

# OpenAI direct (gpt-4o or specified model) — requires OPENAI_API_KEY env var
python main.py --real
```

### Run Tests

```bash
# All tests
cd bosch_fatigue_monitor
python -m pytest tests/ -v

# Single test file
python -m pytest tests/test_judge.py -v

# Single test
python -m pytest tests/test_cooldown.py::TestCooldownTracker::test_escalates_after_ignores -v

# With short traceback (useful for quick debugging)
python -m pytest tests/ -v --tb=short
```

### Interactive Demos

```bash
# Terminal demo: two real scenarios (long-haul + commuter) with live rendering
python demo.py

# Single scenario
python demo.py --scenario 1      # long-haul only
python demo.py --scenario 2      # commuter only

# REPL: manually input sensor values, watch agent decisions in real-time
python interactive.py

# Score analysis heatmaps (PERCLOS × yawns, signal sweeps, case studies)
python quantify.py
python quantify.py --grid        # PERCLOS × yawn heatmap only
python quantify.py --sweep       # Single signal scan
python quantify.py --cases       # Real scenario comparison
```

### Decision Matrix

```bash
# Renders a 20-case matrix: different fatigue scores → different agent outputs
# Verifies that score variance actually changes decisions (not just cosmetic)
python decision_matrix.py
```

## Configuration

All tunable parameters are centralized in **`config.py`**:

### Thresholds (per signal)
- **Text**: speed variance, steering magnitude, brake reaction delta
- **Image**: PERCLOS (eye closure ratio; clinical threshold 0.15), lane deviation count
- **Audio**: yawns per minute, verbal fatigue detection flag
- **Composite weights**: text=0.30, image=0.50, audio=0.20
- **Severity tiers**: NONE (<0.30), MILD (<0.55), MODERATE (<0.75), SEVERE (≥0.75)

### Profile Presets
- `LONG_HAUL_THRESHOLDS`: stricter image/audio, earlier composite bar
- `COMMUTER_THRESHOLDS`: looser image (dry eyes), higher composite bar, shorter duration milestones

### Cooldown Policy
- Per-tier minimum seconds between alerts: mild=30s, moderate=60s, severe=120s
- Ignored alert tracking: if driver doesn't improve after Nth alert at tier T, escalate effective tier

### Sensor Intervals (seconds)
- Text sensor: 1.0 (telemetry every 1s)
- Image sensor: 0.5 (face frame every 0.5s)
- Audio sensor: 2.0 (ASR/yawn detection every 2s)
- Driving duration: 60.0 (milestone check every 60s)

### LLM Models
- **Judge**: gpt-5 (Bosch) or gpt-4o (OpenAI) — full reasoning + vision
- **Nano**: gpt-4o-mini — lightweight fallback
- **ASR**: qwen3-asr-flash
- **TTS**: qwen3-tts-flash

## Data Models

### FatigueContext
Raw signal snapshot (no enrichment):
- `text_signals`: speed variance, steering magnitude, brake delta
- `image_signals`: PERCLOS score, lane deviation count, face B64
- `audio_signals`: yawns/min, verbal fatigue bool, transcript snippet
- `window_seconds`: observation window duration
- `driving_duration_min`: cumulative minutes this session

### EnrichedFatigueContext
FatigueContext + context:
- `map`: road type (city/highway), nearest rest/coffee km, traffic density
- `time_risk_multiplier`: 1.0 (normal), 1.3 (post-lunch), 1.6 (deep night 2-5 AM)
- `driver_profile`: LONG_HAUL or COMMUTER
- `driver_memory`: DriverMemory object (preferences, history) or None

### JudgeVerdict
LLM output:
- `composite_score`: 0-1 weighted aggregate
- `severity_tier`: NONE, MILD, MODERATE, or SEVERE
- `modality_scores`: dict of {modality_name → ModalityScore} with rationale & signals_used
- `reasoning`: full LLM explanation
- `context_tags`: e.g., ["highway", "night"]

## Key Integration Points

### StateAggregator
- **Input**: SensorEvents (text, image, audio, duration) from EventBus
- **Logic**: Maintains RollingWindow (deque + timestamp filtering) per signal type; checks thresholds
- **Output**: Emits FatigueContext to judge_queue when any threshold exceeded

### ContextEnricher
- **Input**: FatigueContext
- **Logic**: Adds map context (mocked or real API), applies time-of-day risk multiplier, loads driver profile & memory
- **Output**: EnrichedFatigueContext

### FatigueJudgeAgent
- **Input**: EnrichedFatigueContext
- **Logic**: Builds multimodal prompt (text signals + face B64); calls LLM; parses JSON response
- **Output**: JudgeVerdict

### OrchestratorAgent
- **Input**: (JudgeVerdict, EnrichedFatigueContext) pairs
- **Logic**: 
  - Checks tier cooldown (is_allowed)
  - Updates ignored count (for escalation logic)
  - Dispatches actions in tier order (pyramid: SEVERE includes MILD/MODERATE)
  - Escalates effective tier if ignored_counts[T] > threshold
- **Output**: Calls action agents (screen, voice, video, push, context)

### Action Agents
Each action agent (screen, voice, video, push, context) receives:
- `verdict`: JudgeVerdict (score, tier, reasoning)
- `ctx`: EnrichedFatigueContext (profile, map, memory)

And returns `ActionResult(success, message)`. The **context_action_agent** is special:
- If long-haul + highway + rest nearby → "Pull over at [rest stop name]"
- If commuter + city + coffee nearby + driver likes coffee → Auto-order via coffee API
- If severe + memory.prefers_nap → "Take a 20-min nap"
- If NONE tier → No action

## Testing Strategy

- **test_state_aggregator.py**: RollingWindow correctness, threshold crossing
- **test_judge.py**: Response parsing, multimodal reasoning flow
- **test_cooldown.py**: Cooldown gate logic, adaptive escalation on ignored counts
- **test_orchestrator.py**: Action dispatch pyramid ordering
- **test_actions.py**: Profile-aware message tone (long-haul urgent, commuter gentle)
- **test_context_enricher.py**: Map context injection
- **test_rolling_window.py**: Edge cases (empty, single value, expiration)

All tests use `conftest.py` fixtures for common FatigueContext, verdict, and config mocks.

## Development Notes

### Three LLM Run Modes
1. **MockLLMClient** (default): Returns deterministic verdicts; no API key; fast feedback loop
2. **OpenAI Direct** (--real): Uses OPENAI_API_KEY; calls openai.com directly
3. **Bosch Gateway** (--bosch): Uses Bosch internal LLM gateway (gpt-5); API key in config.py

To switch modes, edit `config.py` or pass CLI flags to `main.py`.

### Multimodal Judge
The judge can reason about:
- **Text signals**: driving telemetry (speed variance, steering, braking)
- **Face image**: PERCLOS score + actual B64 JPEG frame (for vision reasoning)
- **Audio**: yawns, voice quality, transcript snippet

The LLM sees all three modalities at once and produces per-modality scores + composite.

### Profile Adaptation
- **Driver classification**: In `pipeline/driver_profile.py`, classify_profile() reads context tags and driver_memory to assign profile
- **Threshold switching**: Once profile is known, get_thresholds() loads LONG_HAUL or COMMUTER preset
- **Action adaptation**: Each action agent checks driver_profile in EnrichedFatigueContext and adjusts tone/recommendations

### Cooldown Escalation
- If driver ignores MILD alert, next MILD has shorter cooldown
- After N consecutive ignores at MILD, the next MILD alert is treated as MODERATE (escalated)
- Escalation resets when driver acknowledges (score improves) or time passes

## Common Development Patterns

### Adding a New Sensor
1. Create subclass of `SensorInterface` in `sensors/`
2. Implement `stream_to_bus(bus)` to emit `SensorEvent(signal_type, value, timestamp)`
3. Add to sensor list in `main.py`
4. Update `StateAggregator` to handle new signal type if needed

### Adding a New Action Agent
1. Create subclass of `ActionAgent` (base.py)
2. Implement `async def act(verdict, ctx) -> ActionResult`
3. Inject into `OrchestratorAgent.__init__()` in main.py
4. Add dispatch logic to `_dispatch()` or `_pyramid_dispatch()`

### Adding a New LLM Backend
1. Create subclass of `LLMClientInterface` in `llm/`
2. Implement `async def complete(system, user, image_b64) -> str`
3. Add branch in `main._build_llm_client()` for new flag (e.g., --mymodel)
4. Test with mock verdict data first (see `test_judge.py`)

### Tuning Thresholds for a New Profile
1. Create new `ProfileThresholds` in `config.py`
2. Update `driver_profile.classify_profile()` to return new profile name
3. Add new branch in `get_thresholds()` to map profile name → threshold preset
4. Run `python quantify.py --cases` to visualize impact on composite score

## Permissions & Environment

The `.claude/settings.local.json` pre-authorizes common commands:
- `python main.py` (all modes)
- `python -m pytest tests/ -v`
- `python demo.py`, `interactive.py`, `decision_matrix.py`, `quantify.py`
- `pip install`, `pip show`, `pip list`
- WebFetch to boschchina.feishu.cn (internal docs)

If you need to add new commands or tools, update `.claude/settings.local.json` accordingly.
