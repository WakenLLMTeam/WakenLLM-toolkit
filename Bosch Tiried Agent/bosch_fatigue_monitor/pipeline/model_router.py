"""
ModelRouter: single routing table mapping modality path → LLM client.
Swapping a model means editing one file — nothing else changes.

  image + text  →  GptClient       (GPT-4.5, judge evaluation)
  audio ASR     →  QwenAsrClient   (qwen3-asr-flash)
  audio TTS     →  QwenTtsClient   (qwen3-tts-flash)
"""
from config import AppConfig
from llm.base import LLMClientInterface
from llm.gpt_client import GptClient
from llm.qwen_asr_client import QwenAsrClient
from llm.qwen_tts_client import QwenTtsClient


class ModelRouter:
    def __init__(self, cfg: AppConfig) -> None:
        self._judge_client = GptClient(
            model=cfg.models.judge_model,
            max_tokens=cfg.models.judge_max_tokens,
        )
        self._asr_client = QwenAsrClient(model=cfg.models.asr_model)
        self._tts_client = QwenTtsClient(
            model=cfg.models.tts_model,
            voice=cfg.models.tts_voice,
        )

    @property
    def judge(self) -> LLMClientInterface:
        """Image + text path: GPT-4.5"""
        return self._judge_client

    @property
    def asr(self) -> QwenAsrClient:
        """Audio ASR: qwen3-asr-flash"""
        return self._asr_client

    @property
    def tts(self) -> QwenTtsClient:
        """Audio TTS: qwen3-tts-flash"""
        return self._tts_client
