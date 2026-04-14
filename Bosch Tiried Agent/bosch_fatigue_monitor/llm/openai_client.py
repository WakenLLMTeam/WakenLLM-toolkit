"""
Bosch OpenAI-compatible client for the FatigueJudgeAgent.

Uses the standard openai SDK pointed at Bosch's internal LLM gateway.
Supports multimodal input (text + base64 image) for vision-based PERCLOS analysis.

Environment variables (or hardcoded defaults for dev):
  OPENAI_API_KEY   — Bosch gateway key
  OPENAI_BASE_URL  — Bosch gateway base URL
"""
import os
from typing import Optional

from openai import AsyncOpenAI
from llm.base import LLMClientInterface

# Defaults for local dev — override with environment variables in production
_DEFAULT_KEY     = "4a52b2bf90254d29bfb86919924c6d7d"
_DEFAULT_BASE    = "https://aigc.bosch.com.cn/llmservice/api/v1"


class OpenAIClient(LLMClientInterface):
    """
    Drop-in replacement for GptClient that routes through the Bosch gateway.
    Accepts the same (system, user, image_b64) interface as every other backend.
    """

    def __init__(
        self,
        model:      str = "gpt-5",
        max_tokens: int = 512,
        api_key:    Optional[str] = None,
        base_url:   Optional[str] = None,
    ) -> None:
        self._model      = model
        self._max_tokens = max_tokens
        self._client     = AsyncOpenAI(
            api_key  = api_key  or os.environ.get("OPENAI_API_KEY",  _DEFAULT_KEY),
            base_url = base_url or os.environ.get("OPENAI_BASE_URL", _DEFAULT_BASE),
        )

    async def complete(
        self,
        system:    str,
        user:      str,
        image_b64: Optional[str] = None,
    ) -> str:
        content: list = []

        # Attach face image first if available (vision models process it alongside text)
        if image_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
            })

        content.append({"type": "text", "text": user})

        response = await self._client.chat.completions.create(
            model       = self._model,
            max_tokens  = self._max_tokens,
            response_format = {"type": "json_object"},
            messages    = [
                {"role": "system", "content": system},
                {"role": "user",   "content": content},
            ],
        )
        return response.choices[0].message.content
