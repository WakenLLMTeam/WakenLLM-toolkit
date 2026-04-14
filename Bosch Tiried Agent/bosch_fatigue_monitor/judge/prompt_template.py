"""
Single unified prompt template for LLM-as-a-Judge evaluation.
All modality scoring passes through this template — consistency guaranteed.
"""
from models.fatigue_context import EnrichedFatigueContext


SYSTEM_PROMPT = """\
You are a driver safety evaluator for an automotive AI system.
Your task: score the driver's fatigue level based on multimodal sensor data.

Score each modality on a scale of 0.0 (fully alert) to 1.0 (severely fatigued).
Then produce a weighted composite score and select a severity tier.

Severity tiers:
  NONE     = composite < 0.30
  MILD     = 0.30 <= composite < 0.55
  MODERATE = 0.55 <= composite < 0.75
  SEVERE   = composite >= 0.75

Respond ONLY with valid JSON matching this schema:
{
  "text_score":  float,
  "image_score": float,
  "audio_score": float,
  "composite_score": float,
  "severity_tier": "NONE" | "MILD" | "MODERATE" | "SEVERE",
  "text_rationale": string,
  "image_rationale": string,
  "audio_rationale": string,
  "reasoning": string
}
"""


def build_user_prompt(ctx: EnrichedFatigueContext) -> str:
    f = ctx.fatigue
    m = ctx.map
    t = f.text_signals
    i = f.image_signals
    a = f.audio_signals

    return f"""\
=== Observation Window: {f.window_seconds:.0f}s ===

TEXT / TELEMETRY SIGNALS:
  Speed change variance:           {t.speed_change_variance:.4f}  (low = monotonous = fatigued)
  Steering correction magnitude:   {t.steering_correction_magnitude:.2f} deg  (high = sudden correction = fatigued)
  Brake reaction time delta:       {t.brake_reaction_time_delta_ms:+.1f} ms  (positive = slower = fatigued)

IMAGE SIGNALS:
  PERCLOS score:                   {i.perclos_score:.3f}  (clinical threshold: 0.15)
  Lane deviations in window:       {i.lane_deviation_count}
  Face image: {"[ATTACHED]" if i.face_frame_b64 else "[NOT AVAILABLE]"}

AUDIO SIGNALS:
  Yawns per minute:                {a.yawn_count_per_minute:.1f}
  Verbal fatigue expressed:        {"YES" if a.verbal_fatigue_detected else "NO"}
  Transcript snippet:              "{a.transcript_snippet or ""}"

ROAD CONTEXT:
  Road type:                       {m.road_type.value.upper()}
  Nearest rest stop:               {m.rest_spot_name or "unknown"} ({f"{m.nearest_rest_km:.1f} km" if m.nearest_rest_km else "N/A"})
  Nearest coffee:                  {m.coffee_shop_name or "unknown"} ({f"{m.nearest_coffee_km:.1f} km" if m.nearest_coffee_km else "N/A"})

SESSION CONTEXT:
  Continuous driving duration:     {f.driving_duration_min:.0f} min  (2 h / 4 h / 6 h are risk milestones)
  Time-of-day risk multiplier:     {ctx.time_risk_multiplier:.1f}x  (1.6 = deep night 2–5 AM; 1.3 = post-lunch 13–15; 1.0 = normal)
  Driver profile:                  {ctx.driver_profile.value.upper()}

PROFILE GUIDANCE:
  LONG_HAUL  → apply strict thresholds; intervene early; highway accident cost is high;
               duration ≥ 240 min should raise composite even if signals appear mild.
  COMMUTER   → apply lenient thresholds; prefer gentle single reminder; reduce false
               alarms; do NOT escalate solely on PERCLOS without corroborating signals.

NOTE: Always apply the time-of-day risk multiplier to your composite_score before
selecting the severity tier.

Evaluate each modality independently, then produce the composite score and severity tier.
"""
