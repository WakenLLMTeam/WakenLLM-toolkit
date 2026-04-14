"""
GPT-4.5 client for image+text path (judge evaluation).
Requires: OPENAI_API_KEY in environment.
"""
import os
from typing import Optional

from openai import AsyncOpenAI
from llm.base import LLMClientInterface


class GptClient(LLMClientInterface):
    def __init__(self, model: str = "gpt-4.5-preview", max_tokens: int = 512) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    async def complete(
        self,
        system: str,
        user: str,
        image_b64: Optional[str] = None,
    ) -> str:
        content = [{"type": "text", "text": user}]
        if image_b64:
            content.insert(0, {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
            })

        response = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": content},
            ],
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content
