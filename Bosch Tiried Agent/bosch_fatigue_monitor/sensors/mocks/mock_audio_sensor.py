"""
MockAudioSensor: simulates yawn events and verbal fatigue detection.
Triggers periodically when in 'fatigue mode' (after 30s).
"""
import asyncio
import random
import time
from sensors.base import SensorInterface
from models.sensor_event import SensorEvent, Modality, SignalType


class MockAudioSensor(SensorInterface):
    def __init__(self, interval: float = 2.0) -> None:
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
            fatigued = elapsed > 30

            if fatigued and random.random() < 0.4:
                await bus.publish(SensorEvent(
                    modality=Modality.AUDIO,
                    signal_type=SignalType.YAWN_DETECTED,
                    value=1.0,
                    metadata={"confidence": round(random.uniform(0.7, 0.99), 2)},
                ))

            if fatigued and random.random() < 0.1:
                await bus.publish(SensorEvent(
                    modality=Modality.AUDIO,
                    signal_type=SignalType.VERBAL_FATIGUE,
                    value=True,
                    metadata={"transcript": "我有点累了"},
                ))

            await asyncio.sleep(self._interval)
