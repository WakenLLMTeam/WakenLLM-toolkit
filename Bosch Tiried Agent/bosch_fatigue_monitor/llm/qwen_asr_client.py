"""
Qwen3-asr-flash client: speech-to-text for yawn/verbal fatigue detection.
Requires: DASHSCOPE_API_KEY in environment.

Docs: https://help.aliyun.com/zh/model-studio/developer-reference/qwen-audio
"""
import os
from typing import Optional
from llm.base import LLMClientInterface


class QwenAsrClient(LLMClientInterface):
    """
    Thin wrapper around DashScope ASR API.
    In production: submit audio bytes, get transcript back.
    """

    def __init__(self, model: str = "qwen3-asr-flash") -> None:
        self._model = model
        self._api_key = os.environ.get("DASHSCOPE_API_KEY", "")

    async def complete(
        self,
        system: str,
        user: str,
        image_b64: Optional[str] = None,
    ) -> str:
        raise NotImplementedError(
            "QwenAsrClient.complete() is for text tasks. "
            "Use transcribe(audio_bytes) for ASR."
        )

    async def transcribe(self, audio_bytes: bytes) -> str:
        """
        Submit raw audio bytes → return transcript string.
        Replace this stub with the actual DashScope SDK call.
        """
        # TODO: replace with real DashScope call
        # import dashscope
        # response = dashscope.audio.asr.Recognition.call(
        #     model=self._model,
        #     file=audio_bytes,
        #     format="wav",
        #     api_key=self._api_key,
        # )
        # return response.output.sentence[0]["text"]
        raise NotImplementedError("Connect DashScope API key to activate.")
