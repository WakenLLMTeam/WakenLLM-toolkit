import asyncio
import argparse
from src.config_loader import load_config
from src.data_handler import DataHandler
from src.llm_handler import LLMHandler
from src.evaluator import Evaluator
# --- 修正：确保类名与 pipeline.py 文件中的定义完全一致 ---
from src.pipeline import WakenllmPipeline

async def main_async():
    """所有异步逻辑的真正入口点。"""
    parser = argparse.ArgumentParser(description="WAKENLLM Toolkit")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/experiment.yaml',
        help='Path to the experiment configuration file.'
    )
    args = parser.parse_args()

    # 1. 加载配置
    config = load_config(args.config)

    # 2. 初始化所有服务模块
    data_handler = DataHandler(config)
    llm_handler = LLMHandler(config)
    evaluator = Evaluator()

    # 3. 组装并启动Pipeline！
    pipeline = WakenllmPipeline(config, data_handler, llm_handler, evaluator)
    await pipeline.run()

    print("\n--- 实验流程执行完毕 ---")

def main():
    """同步的启动器，负责创建和运行事件循环。"""
    try:
        # 这是程序中唯一一处 asyncio.run()
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n程序被用户中断。")
    except Exception as e:
        print(f"\n程序发生未捕获的异常: {e}")

if __name__ == "__main__":
    main()