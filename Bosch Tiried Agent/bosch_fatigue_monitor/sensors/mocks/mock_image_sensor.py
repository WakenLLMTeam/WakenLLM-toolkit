"""
MockImageSensor: simulates PERCLOS scores and lane deviation events.
PERCLOS ramps from 0.05 to 0.35 after 30 seconds to simulate fatigue onset.

A minimal valid JPEG is embedded so that vision-capable models receive a real
image frame instead of an empty string — avoids API validation errors.
"""
import asyncio
import random
import time
from sensors.base import SensorInterface
from models.sensor_event import SensorEvent, Modality, SignalType

# Minimal 1×1 white-pixel JPEG (base64) — valid input for any vision model.
# In production, replace with actual camera frame bytes.
_MINIMAL_FACE_JPEG_B64 = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8U"
    "HRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAARC"
    "AABAAEDASIA"
    "AhEBAxEB/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/"
    "xAAUAQEAAAAAAAAAAAAAAAAAAAAA/8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAwDAQACEQ"
    "MRAD8AJQAB/9k="
)


class MockImageSensor(SensorInterface):
    def __init__(self, interval: float = 0.5) -> None:
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
            elapsed  = time.time() - self._start_ts
            fatigued = elapsed > 30

            # PERCLOS: ramp up when fatigued
            base_perclos = 0.30 if fatigued else 0.05
            perclos = min(1.0, random.gauss(base_perclos, 0.03))
            await bus.publish(SensorEvent(
                modality=Modality.IMAGE,
                signal_type=SignalType.PERCLOS,
                value=perclos,
                metadata={"face_frame_b64": _MINIMAL_FACE_JPEG_B64},
            ))

            # Lane deviation: occasional event when fatigued
            if fatigued and random.random() < 0.15:
                await bus.publish(SensorEvent(
                    modality=Modality.IMAGE,
                    signal_type=SignalType.LANE_DEVIATION,
                    value=1.0,
                ))

            await asyncio.sleep(self._interval)
