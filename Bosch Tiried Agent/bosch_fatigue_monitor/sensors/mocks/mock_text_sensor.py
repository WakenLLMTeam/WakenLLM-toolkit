"""
MockTextSensor: simulates dashboard/telemetry signals with a random walk.
After 30 seconds, shifts into 'fatigue mode' to trigger the pipeline.
"""
import asyncio
import random
import time
from sensors.base import SensorInterface
from models.sensor_event import SensorEvent, Modality, SignalType


class MockTextSensor(SensorInterface):
    def __init__(self, interval: float = 1.0) -> None:
        self._interval = interval
        self._running  = False
        self._start_ts = 0.0

    async def start(self) -> None:
        self._running  = True
        self._start_ts = time.time()

    async def stop(self) -> None:
        self._running = False

    async def stream_to_bus(self, bus) -> None:
        await self.start()
        while self._running:
            elapsed = time.time() - self._start_ts
            fatigued = elapsed > 30  # simulate fatigue onset after 30s

            # Speed: low variance when fatigued (monotonous driving)
            speed = random.gauss(60, 0.2 if fatigued else 5.0)
            await bus.publish(SensorEvent(
                modality=Modality.TEXT,
                signal_type=SignalType.SPEED_CHANGE,
                value=speed,
            ))

            # Steering: large sudden corrections when fatigued
            steering = random.gauss(20.0 if fatigued else 3.0, 2.0)
            await bus.publish(SensorEvent(
                modality=Modality.TEXT,
                signal_type=SignalType.STEERING_CORRECTION,
                value=abs(steering),
            ))

            # Brake reaction time: slower when fatigued
            brake_delta = random.gauss(200.0 if fatigued else 30.0, 20.0)
            await bus.publish(SensorEvent(
                modality=Modality.TEXT,
                signal_type=SignalType.BRAKE_REACTION_TIME,
                value=max(0.0, brake_delta),
            ))

            await asyncio.sleep(self._interval)
