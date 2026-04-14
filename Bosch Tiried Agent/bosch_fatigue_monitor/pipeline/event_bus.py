"""
Async event bus: sensors publish SensorEvents; aggregator subscribes.
Multiple subscribers can register — each gets its own queue.
"""
import asyncio
from typing import List
from models.sensor_event import SensorEvent


class EventBus:
    def __init__(self) -> None:
        self._subscribers: List[asyncio.Queue] = []

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.append(q)
        return q

    async def publish(self, event: SensorEvent) -> None:
        for q in self._subscribers:
            await q.put(event)
