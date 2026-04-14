"""
Qwen3-tts-flash client: text-to-speech for voice broadcast alerts.
Requires: DASHSCOPE_API_KEY in environment.

Docs: https://help.aliyun.com/zh/model-studio/developer-reference/qwen-tts
"""
import os
from typing import Optional
from llm.base import LLMClientInterface


class QwenTtsClient(LLMClientInterface):
    def __init__(
        self,
        model: str = "qwen3-tts-flash",
        voice: str = "default",
    ) -> None:
        self._model = model
        self._voice = voice
        self._api_key = os.environ.get("DASHSCOPE_API_KEY", "")

    async def complete(
        self,
        system: str,
        user: str,
        image_b64: Optional[str] = None,
    ) -> str:
        raise NotImplementedError(
            "QwenTtsClient.complete() is for text tasks. "
            "Use synthesize(text) for TTS."
        )

    async def synthesize(self, text: str) -> bytes:
        """
        Submit text → return audio bytes (PCM/WAV).
        Replace this stub with the actual DashScope SDK call.
        """
        # TODO: replace with real DashScope call
        # import dashscope
        # response = dashscope.audio.tts.SpeechSynthesizer.call(
        #     model=self._model,
        #     text=text,
        #     voice=self._voice,
        #     api_key=self._api_key,
        # )
        # return response.get_audio_data()
        raise NotImplementedError("Connect DashScope API key to activate.")
