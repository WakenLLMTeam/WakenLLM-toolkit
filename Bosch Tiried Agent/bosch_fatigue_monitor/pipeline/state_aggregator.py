"""
StateAggregator:
  - Consumes SensorEvents from the EventBus.
  - Maintains one RollingWindow per SignalType.
  - On every event, checks thresholds against config.
  - When any threshold is exceeded, emits a FatigueContext snapshot
    to the judge_queue for evaluation.
"""
import asyncio
import time
from collections import deque
from typing import Deque, Dict, Tuple

from config import AppConfig
from models.sensor_event import SensorEvent, SignalType
from models.fatigue_context import (
    FatigueContext, TextSignals, ImageSignals, AudioSignals, RoadType
)
from pipeline.driver_profile import classify_profile, get_thresholds


class RollingWindow:
    """Time-bounded deque of (timestamp, value) pairs."""

    def __init__(self, window_seconds: float) -> None:
        self._window = window_seconds
        self._data: Deque[Tuple[float, float]] = deque()

    def push(self, value: float, ts: float = 0.0) -> None:
        if ts == 0.0:
            ts = time.time()
        self._data.append((ts, value))
        self._evict(ts)

    def _evict(self, now: float) -> None:
        cutoff = now - self._window
        while self._data and self._data[0][0] < cutoff:
            self._data.popleft()

    def values(self) -> list:
        return [v for _, v in self._data]

    def count(self) -> int:
        return len(self._data)

    def mean(self) -> float:
        vals = self.values()
        return sum(vals) / len(vals) if vals else 0.0

    def variance(self) -> float:
        vals = self.values()
        if len(vals) < 2:
            return 0.0
        m = sum(vals) / len(vals)
        return sum((v - m) ** 2 for v in vals) / len(vals)

    def frequency_per_minute(self) -> float:
        return self.count() / (self._window / 60.0)

    def latest(self) -> float:
        return self._data[-1][1] if self._data else 0.0


class StateAggregator:
    def __init__(self, cfg: AppConfig) -> None:
        self._cfg = cfg
        w = cfg.thresholds.observation_window_seconds
        self._windows: Dict[SignalType, RollingWindow] = {
            st: RollingWindow(w) for st in SignalType
        }
        # Store latest face frame separately (bytes, not numeric)
        self._latest_face_frame_b64: str = ""
        self._latest_transcript: str = ""
        self._verbal_fatigue: bool = False
        self._driving_duration_min: float = 0.0

    def _ingest(self, event: SensorEvent) -> None:
        if event.signal_type == SignalType.LANE_DEVIATION:
            self._windows[SignalType.LANE_DEVIATION].push(1.0, event.timestamp)
        elif event.signal_type == SignalType.YAWN_DETECTED:
            self._windows[SignalType.YAWN_DETECTED].push(1.0, event.timestamp)
        elif event.signal_type == SignalType.DRIVING_DURATION:
            self._driving_duration_min = float(event.value)
        elif event.signal_type == SignalType.VERBAL_FATIGUE:
            self._verbal_fatigue = bool(event.value)
            self._latest_transcript = event.metadata.get("transcript", "")
        elif event.signal_type == SignalType.PERCLOS:
            self._windows[SignalType.PERCLOS].push(float(event.value), event.timestamp)
            if face := event.metadata.get("face_frame_b64"):
                self._latest_face_frame_b64 = face
        else:
            self._windows[event.signal_type].push(float(event.value), event.timestamp)

    def _active_thresholds(self):
        """Return the ThresholdConfig for the currently inferred driver profile."""
        profile = classify_profile(
            driving_duration_min=self._driving_duration_min,
            road_type=RoadType.UNKNOWN,   # StateAggregator has no map access;
                                          # road_type enriched later by ContextEnricher
        )
        return get_thresholds(profile)

    def _threshold_exceeded(self) -> bool:
        t = self._active_thresholds()
        speed_window = self._windows[SignalType.SPEED_CHANGE]
        # Require at least 3 data points before treating low variance as a fatigue signal
        # (empty window variance=0.0 would otherwise always trigger)
        speed_fatigued = (
            speed_window.count() >= 3
            and speed_window.variance() < t.speed_variance_low
        )

        # --- PERCLOS cross-validation (anti-false-positive) ---
        # PERCLOS alone (e.g. dry eyes, bright sun) must NOT trigger an alert.
        # Require at least one corroborating signal before counting it.
        perclos_elevated = self._windows[SignalType.PERCLOS].mean() > t.perclos_threshold
        corroborating_signal = (
            self._windows[SignalType.YAWN_DETECTED].frequency_per_minute() > t.yawn_per_min_high
            or self._verbal_fatigue
            or self._windows[SignalType.STEERING_CORRECTION].latest() > t.steering_magnitude_high
            or self._windows[SignalType.LANE_DEVIATION].frequency_per_minute() > t.lane_deviation_per_min_high
        )
        perclos_confirmed = perclos_elevated and corroborating_signal

        # Driving duration alone triggers a check at the mild milestone
        duration_exceeded = self._driving_duration_min >= t.driving_duration_mild_min

        return (
            speed_fatigued
            or self._windows[SignalType.STEERING_CORRECTION].latest() > t.steering_magnitude_high
            or self._windows[SignalType.BRAKE_REACTION_TIME].latest() > t.brake_reaction_delta_ms_high
            or perclos_confirmed
            or self._windows[SignalType.LANE_DEVIATION].frequency_per_minute() > t.lane_deviation_per_min_high
            or self._windows[SignalType.YAWN_DETECTED].frequency_per_minute() > t.yawn_per_min_high
            or self._verbal_fatigue
            or duration_exceeded
        )

    def _build_context(self) -> FatigueContext:
        w = self._cfg.thresholds.observation_window_seconds
        return FatigueContext(
            text_signals=TextSignals(
                speed_change_variance=self._windows[SignalType.SPEED_CHANGE].variance(),
                steering_correction_magnitude=self._windows[SignalType.STEERING_CORRECTION].latest(),
                brake_reaction_time_delta_ms=self._windows[SignalType.BRAKE_REACTION_TIME].latest(),
            ),
            image_signals=ImageSignals(
                perclos_score=self._windows[SignalType.PERCLOS].mean(),
                lane_deviation_count=self._windows[SignalType.LANE_DEVIATION].count(),
                face_frame_b64=self._latest_face_frame_b64 or None,
            ),
            audio_signals=AudioSignals(
                yawn_count_per_minute=self._windows[SignalType.YAWN_DETECTED].frequency_per_minute(),
                verbal_fatigue_detected=self._verbal_fatigue,
                transcript_snippet=self._latest_transcript or None,
            ),
            window_seconds=w,
            timestamp=time.time(),
            driving_duration_min=self._driving_duration_min,
        )

    async def run(
        self,
        event_queue: asyncio.Queue,
        judge_queue: asyncio.Queue,
    ) -> None:
        while True:
            event: SensorEvent = await event_queue.get()
            self._ingest(event)
            if self._threshold_exceeded():
                ctx = self._build_context()
                await judge_queue.put(ctx)
