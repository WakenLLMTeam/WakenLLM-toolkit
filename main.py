import argparse
from src.config_loader import load_config
from src.data_handler import DataHandler
from src.llm_handler import LLMHandler
from src.evaluator import Evaluator
from src.pipeline import WakenLLMPipeline # 导入我们的主角

def main():
    parser = argparse.ArgumentParser(description="WAKENLLM Toolkit")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/experiment.yaml',
        help='Path to the experiment configuration file.'
    )
    args = parser.parse_args()

    # --- 1. 加载配置 ---
    config = load_config(args.config)

    # --- 2. 初始化所有服务模块 ---
    data_handler = DataHandler(config)
    llm_handler = LLMHandler(config)
    evaluator = Evaluator()

    # --- 3. 组装并启动Pipeline！ ---
    pipeline = WakenLLMPipeline(config, data_handler, llm_handler, evaluator)
    pipeline.run()

    print("\n--- 实验流程执行完毕 ---")

if __name__ == "__main__":
    main()