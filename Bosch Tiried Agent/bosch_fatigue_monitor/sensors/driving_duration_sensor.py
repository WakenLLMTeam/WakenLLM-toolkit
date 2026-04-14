"""
DrivingDurationSensor: tracks cumulative driving time for the current session.

Emits a DRIVING_DURATION event every `interval` seconds with the total
minutes driven so far.  Downstream consumers use this to:
  - boost composite fatigue scores after 2 h / 4 h / 6 h thresholds
  - inform the Judge and ContextActionAgent of long-haul context
"""
import asyncio
import time

from sensors.base import SensorInterface
from models.sensor_event import SensorEvent, Modality, SignalType


class DrivingDurationSensor(SensorInterface):
    def __init__(self, interval: float = 60.0) -> None:
        self._interval = interval   # how often (seconds) to publish an update
        self._running  = False
        self._start_ts = 0.0

    async def start(self) -> None:
        self._running  = True
        self._start_ts = time.time()

    async def stop(self) -> None:
        self._running = False

    @property
    def elapsed_minutes(self) -> float:
        return (time.time() - self._start_ts) / 60.0

    async def stream_to_bus(self, bus) -> None:
        await self.start()
        while self._running:
            await asyncio.sleep(self._interval)
            await bus.publish(SensorEvent(
                modality=Modality.TEXT,
                signal_type=SignalType.DRIVING_DURATION,
                value=self.elapsed_minutes,
            ))
