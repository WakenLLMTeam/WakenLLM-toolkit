import asyncio
import argparse
from src.config_loader import load_config
from src.data_handler import DataHandler
from src.llm_handler import LLMHandler
from src.evaluator import Evaluator
# --- Fix: Ensure class name is consistent with the definition in pipeline.py ---
from src.pipeline import WakenllmPipeline

async def main_async():
    """The true entry point for all async logic."""
    parser = argparse.ArgumentParser(description="WAKENLLM Toolkit")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/experiment.yaml',
        help='Path to the experiment configuration file.'
    )
    args = parser.parse_args()

    # 1. Load configuration
    config = load_config(args.config)

    # 2. Initialize all service modules
    data_handler = DataHandler(config)
    llm_handler = LLMHandler(config)
    evaluator = Evaluator()

    # 3. Assemble and start the Pipeline!
    pipeline = WakenllmPipeline(config, data_handler, llm_handler, evaluator)
    await pipeline.run()

    print("\n--- Experiment workflow completed ---")

def main():
    """Synchronous launcher responsible for creating and running the event loop."""
    try:
        # This is the only place in the program where asyncio.run() is used
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    except Exception as e:
        print(f"\nUncaught exception in program: {e}")

if __name__ == "__main__":
    main()