"""SensorInterface ABC."""
from abc import ABC, abstractmethod
from typing import AsyncIterator
from models.sensor_event import SensorEvent


class SensorInterface(ABC):
    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @abstractmethod
    async def stream_to_bus(self, bus) -> None:
        """Continuously publish SensorEvents to the provided EventBus."""
        ...
