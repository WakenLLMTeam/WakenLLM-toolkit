"""LLM client interface — all backends implement this."""
from abc import ABC, abstractmethod
from typing import Optional


class LLMClientInterface(ABC):
    @abstractmethod
    async def complete(
        self,
        system: str,
        user: str,
        image_b64: Optional[str] = None,
    ) -> str:
        """Returns raw text response from the model."""
        ...
