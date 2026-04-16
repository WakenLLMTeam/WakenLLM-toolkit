"""
Qwen3-TTS-Flash client: text-to-speech via Bosch AI gateway.

Uses the OpenAI-compatible /audio/speech endpoint proxied through
https://aigc.bosch.com.cn/llmservice/api/v1.

Returns WAV (PCM 16-bit 24kHz) bytes that can be written to a .wav file and played with afplay.
"""
import os
from typing import Optional
from llm.base import LLMClientInterface

_BOSCH_API_KEY  = "4a52b2bf90254d29bfb86919924c6d7d"
_BOSCH_BASE_URL = "https://aigc.bosch.com.cn/llmservice/api/v1"


class QwenTtsClient(LLMClientInterface):
    def __init__(
        self,
        model: str = "qwen3-tts-flash",
        voice: str = "longxiaochun",   # Mandarin female voice
        api_key: str = _BOSCH_API_KEY,
        base_url: str = _BOSCH_BASE_URL,
    ) -> None:
        self._model   = model
        self._voice   = voice
        self._api_key = api_key
        self._base_url = base_url

    async def complete(
        self,
        system: str,
        user: str,
        image_b64: Optional[str] = None,
    ) -> str:
        raise NotImplementedError(
            "QwenTtsClient is for TTS only. Use synthesize(text) instead."
        )

    async def synthesize(self, text: str) -> bytes:
        """Call Qwen3-TTS-Flash → return MP3 audio bytes."""
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)
        response = await client.audio.speech.create(
            model=self._model,
            voice=self._voice,
            input=text,
        )
        return response.content
