"""
Entry point: assembles and runs the proactive fatigue monitoring pipeline.

Pipeline flow:
  Sensors → EventBus → StateAggregator → ContextEnricher
  → FatigueJudgeAgent → OrchestratorAgent → ActionAgents

Run modes:
  python main.py              # MockLLMClient — no API key needed, for local dev
  python main.py --bosch      # Bosch gateway (gpt-5, multimodal) — default for demo
  python main.py --real       # OpenAI direct (legacy, requires OPENAI_API_KEY)
"""
import asyncio
import logging
import sys

from config import config
from pipeline.event_bus import EventBus
from pipeline.state_aggregator import StateAggregator
from pipeline.context_enricher import ContextEnricher, MockMapClient
from judge.judge_agent import FatigueJudgeAgent
from orchestrator.orchestrator_agent import OrchestratorAgent
from actions.screen_display_agent import ScreenDisplayAgent
from actions.voice_broadcast_agent import VoiceBroadcastAgent
from actions.video_record_agent import VideoRecordAgent
from actions.phone_push_agent import PhonePushAgent
from actions.context_action_agent import ContextActionAgent
from sensors.mocks.mock_text_sensor import MockTextSensor
from sensors.mocks.mock_image_sensor import MockImageSensor
from sensors.mocks.mock_audio_sensor import MockAudioSensor
from sensors.driving_duration_sensor import DrivingDurationSensor
from llm.mock_llm_client import MockLLMClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _build_llm_client():
    args = sys.argv[1:]

    if "--bosch" in args:
        from llm.openai_client import OpenAIClient
        client = OpenAIClient(
            model      = config.models.judge_model,   # gpt-5
            max_tokens = config.models.judge_max_tokens,
            api_key    = config.bosch_api.api_key,
            base_url   = config.bosch_api.base_url,
        )
        logger.info(
            "Using Bosch gateway: %s @ %s",
            config.models.judge_model,
            config.bosch_api.base_url,
        )
        return client

    if "--real" in args:
        from llm.gpt_client import GptClient
        client = GptClient(
            model      = config.models.judge_model,
            max_tokens = config.models.judge_max_tokens,
        )
        logger.info("Using OpenAI direct: %s", config.models.judge_model)
        return client

    logger.info("Using MockLLMClient (no API key required)")
    return MockLLMClient()


async def enrich_and_judge_loop(
    enricher:      ContextEnricher,
    judge:         FatigueJudgeAgent,
    judge_queue:   asyncio.Queue,
    verdict_queue: asyncio.Queue,
) -> None:
    while True:
        fatigue_ctx = await judge_queue.get()
        enriched    = await enricher.enrich(fatigue_ctx)
        verdict     = await judge.evaluate(enriched)
        await verdict_queue.put((verdict, enriched))


async def main() -> None:
    llm_client = _build_llm_client()

    # --- Build pipeline ---
    bus      = EventBus()
    agg      = StateAggregator(config)
    enricher = ContextEnricher(MockMapClient())
    judge    = FatigueJudgeAgent(llm_client)
    orch     = OrchestratorAgent(
        screen_agent  = ScreenDisplayAgent(),
        voice_agent   = VoiceBroadcastAgent(),
        video_agent   = VideoRecordAgent(),
        push_agent    = PhonePushAgent(config.phone_push_url),
        context_agent = ContextActionAgent(),
        cfg           = config,
    )

    event_queue:   asyncio.Queue = bus.subscribe()
    judge_queue:   asyncio.Queue = asyncio.Queue()
    verdict_queue: asyncio.Queue = asyncio.Queue()

    sensors = [
        MockTextSensor(config.text_sensor_interval),
        MockImageSensor(config.image_sensor_interval),
        MockAudioSensor(config.audio_sensor_interval),
        DrivingDurationSensor(config.driving_duration_sensor_interval),
    ]

    logger.info("Starting Bosch Fatigue Monitor Agent...")
    logger.info("Fatigue simulation triggers after ~30 s. Press Ctrl-C to stop.")

    await asyncio.gather(
        *[s.stream_to_bus(bus) for s in sensors],
        agg.run(event_queue, judge_queue),
        enrich_and_judge_loop(enricher, judge, judge_queue, verdict_queue),
        orch.run(verdict_queue),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down.")
