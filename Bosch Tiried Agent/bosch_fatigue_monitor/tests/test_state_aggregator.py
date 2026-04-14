"""Tests for StateAggregator threshold detection and context building."""
import asyncio
import time
import pytest
from config import AppConfig
from pipeline.state_aggregator import StateAggregator
from models.sensor_event import SensorEvent, Modality, SignalType


def _push_event(agg: StateAggregator, signal_type: SignalType, value, modality=Modality.TEXT):
    event = SensorEvent(modality=modality, signal_type=signal_type, value=value)
    agg._ingest(event)


class TestStateAggregatorThresholds:
    def test_no_threshold_exceeded_initially(self):
        agg = StateAggregator(AppConfig())
        assert agg._threshold_exceeded() is False

    def test_perclos_above_threshold_triggers(self):
        # PERCLOS requires at least one corroborating signal (anti-false-positive rule)
        agg = StateAggregator(AppConfig())
        _push_event(agg, SignalType.PERCLOS, 0.4, Modality.IMAGE)   # > threshold
        now = time.time()
        for i in range(3):  # 3 yawns/min → corroborating signal
            event = SensorEvent(
                modality=Modality.AUDIO,
                signal_type=SignalType.YAWN_DETECTED,
                value=1.0,
                timestamp=now - i,
            )
            agg._ingest(event)
        assert agg._threshold_exceeded() is True

    def test_perclos_alone_does_not_trigger(self):
        # PERCLOS in isolation (e.g. dry eyes) should NOT fire an alert
        agg = StateAggregator(AppConfig())
        _push_event(agg, SignalType.PERCLOS, 0.4, Modality.IMAGE)
        assert agg._threshold_exceeded() is False

    def test_perclos_below_threshold_does_not_trigger(self):
        agg = StateAggregator(AppConfig())
        _push_event(agg, SignalType.PERCLOS, 0.05, Modality.IMAGE)  # < 0.15
        assert agg._threshold_exceeded() is False

    def test_steering_magnitude_triggers(self):
        agg = StateAggregator(AppConfig())
        _push_event(agg, SignalType.STEERING_CORRECTION, 25.0)      # > 15 deg
        assert agg._threshold_exceeded() is True

    def test_brake_reaction_triggers(self):
        agg = StateAggregator(AppConfig())
        _push_event(agg, SignalType.BRAKE_REACTION_TIME, 300.0)     # > 150ms
        assert agg._threshold_exceeded() is True

    def test_verbal_fatigue_triggers(self):
        agg = StateAggregator(AppConfig())
        event = SensorEvent(
            modality=Modality.AUDIO,
            signal_type=SignalType.VERBAL_FATIGUE,
            value=True,
            metadata={"transcript": "我有点累了"},
        )
        agg._ingest(event)
        assert agg._threshold_exceeded() is True
        assert agg._latest_transcript == "我有点累了"

    def test_yawn_frequency_triggers(self):
        agg = StateAggregator(AppConfig())
        now = time.time()
        # 3 yawns within a 60s window → frequency = 3/min > threshold 2.0
        for i in range(3):
            event = SensorEvent(
                modality=Modality.AUDIO,
                signal_type=SignalType.YAWN_DETECTED,
                value=1.0,
                timestamp=now - i,
            )
            agg._ingest(event)
        assert agg._threshold_exceeded() is True

    def test_lane_deviation_frequency_triggers(self):
        agg = StateAggregator(AppConfig())
        now = time.time()
        # COMMUTER profile threshold is 4/min; push 5 to exceed it
        for i in range(5):
            event = SensorEvent(
                modality=Modality.IMAGE,
                signal_type=SignalType.LANE_DEVIATION,
                value=1.0,
                timestamp=now - i,
            )
            agg._ingest(event)
        assert agg._threshold_exceeded() is True

    def test_build_context_captures_perclos(self):
        agg = StateAggregator(AppConfig())
        _push_event(agg, SignalType.PERCLOS, 0.25, Modality.IMAGE)
        ctx = agg._build_context()
        assert ctx.image_signals.perclos_score == pytest.approx(0.25)

    def test_build_context_captures_face_frame(self):
        agg = StateAggregator(AppConfig())
        event = SensorEvent(
            modality=Modality.IMAGE,
            signal_type=SignalType.PERCLOS,
            value=0.2,
            metadata={"face_frame_b64": "abc123"},
        )
        agg._ingest(event)
        ctx = agg._build_context()
        assert ctx.image_signals.face_frame_b64 == "abc123"

    def test_build_context_audio_fields(self):
        agg = StateAggregator(AppConfig())
        event = SensorEvent(
            modality=Modality.AUDIO,
            signal_type=SignalType.VERBAL_FATIGUE,
            value=True,
            metadata={"transcript": "累了"},
        )
        agg._ingest(event)
        ctx = agg._build_context()
        assert ctx.audio_signals.verbal_fatigue_detected is True
        assert ctx.audio_signals.transcript_snippet == "累了"

    @pytest.mark.asyncio
    async def test_run_emits_to_judge_queue_when_threshold_exceeded(self):
        agg = StateAggregator(AppConfig())
        event_queue: asyncio.Queue = asyncio.Queue()
        judge_queue: asyncio.Queue = asyncio.Queue()

        # Pre-load a threshold-exceeding event (steering — no corroboration needed)
        await event_queue.put(SensorEvent(
            modality=Modality.TEXT,
            signal_type=SignalType.STEERING_CORRECTION,
            value=25.0,   # > 15 deg threshold
        ))

        # Run one iteration of the aggregator (cancel after first event)
        task = asyncio.create_task(agg.run(event_queue, judge_queue))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert not judge_queue.empty()
        ctx = await judge_queue.get()
        assert ctx.text_signals.steering_correction_magnitude == pytest.approx(25.0)
