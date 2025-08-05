import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. 日志设置 (建议 #3) ---
# 与主项目风格保持一致，提供清晰的日志输出
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- 2. 默认路径定义 ---
# 将路径定义放在脚本开头，便于管理
DEFAULT_RESULTS_DIR = Path(__file__).parent.parent / "results"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / "visual_summary"


def aggregate_results(results_dir: Path) -> Optional[pd.DataFrame]:
    """
    扫描指定目录下的所有 'summary_*.json' 文件，并将它们聚合成一个Pandas DataFrame。

    Args:
        results_dir (Path): 包含实验结果摘要文件的目录。

    Returns:
        Optional[pd.DataFrame]: 如果找到摘要文件，则返回聚合后的DataFrame；否则返回None。
    """
    logger.info(f"正在扫描目录 '{results_dir}' 以查找结果摘要...")

    summaries: List[Dict[str, Any]] = []
    for f in results_dir.glob("summary_*.json"):
        with open(f, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
                # 将嵌套的'metrics'字典展平，便于DataFrame处理
                metrics = data.pop('metrics', {})
                data.update(metrics)
                summaries.append(data)
            except json.JSONDecodeError:
                logger.warning(f"无法解析文件 '{f}'，已跳过。")

    if not summaries:
        logger.error(f"在 '{results_dir}' 中未找到任何有效的 'summary_*.json' 文件。")
        logger.error("请先运行至少一个实验，并确保它能正确生成摘要文件。")
        return None

    logger.info(f"成功聚合了 {len(summaries)} 个实验的结果。")
    return pd.DataFrame(summaries)


def generate_visuals(df: pd.DataFrame, metric: str, output_dir: Path) -> None:
    """
    根据聚合的数据，生成一个CSV报告和一张对比图表。

    Args:
        df (pd.DataFrame): 包含所有实验结果的DataFrame。
        metric (str): 需要可视化的指标名称 (例如, 'OCR', 'TCR1')。
        output_dir (Path): 保存生成的报告和图表的目录。
    """
    # --- 3. 确保输出目录存在 (建议 #5) ---
    output_dir.mkdir(exist_ok=True)

    # 3.1 保存完整的CSV聚合报告
    csv_path = output_dir / "all_experiments_summary.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"✔️  完整的聚合报告已保存至: {csv_path}")

    # --- 4. 容错：检查指标是否存在 (建议 #4) ---
    if metric not in df.columns:
        logger.error(f"错误：指定的指标 '{metric}' 不存在于结果数据中。")
        logger.warning(f"可用的指标包括: {', '.join(df.columns)}")
        return

    # 3.2 生成可视化图表
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    # 筛选出包含有效指标值的数据
    plot_df = df.dropna(subset=[metric]).copy()
    if plot_df.empty:
        logger.warning(f"没有找到指标 '{metric}' 的有效数据，跳过绘图。")
        return

    plot_df[metric] = plot_df[metric] * 100  # 转换为百分比

    sns.barplot(data=plot_df, x='dataset', y=metric, hue='model', ax=ax, palette='viridis')

    ax.set_title(f'Comparison of {metric} by Model and Dataset', fontsize=18, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=14)
    ax.set_ylabel(f'{metric} (%)', fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(0, 100)
    ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')

    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', fontsize=10, padding=3)

    plt.tight_layout()

    # --- 5. 保存为PNG和矢量图 (建议 #6) ---
    chart_path_png = output_dir / f"{metric.lower()}_comparison.png"
    chart_path_pdf = output_dir / f"{metric.lower()}_comparison.pdf"

    plt.savefig(chart_path_png, dpi=300)
    plt.savefig(chart_path_pdf, bbox_inches='tight')

    logger.info(f"✔️  对比图已保存为PNG格式: {chart_path_png}")
    logger.info(f"✔️  对比图已保存为PDF矢量格式: {chart_path_pdf}")
    plt.show()


def main() -> None:
    """主函数：解析参数并驱动整个流程。"""
    # --- 6. 使用 argparse 解析命令行参数 (建议 #1) ---
    parser = argparse.ArgumentParser(
        description="Aggregate and visualize results for the WAKENLLM Toolkit.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="OCR",
        help="The primary metric to visualize in the bar chart."
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="The directory where experiment summary files are stored."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="The directory where visualizations and reports will be saved."
    )
    args = parser.parse_args()

    logger.info("--- WAKENLLM 结果聚合与可视化脚本启动 ---")
    results_df = aggregate_results(args.results_dir)
    if results_df is not None:
        generate_visuals(results_df, args.metric, args.output_dir)
    logger.info("--- 脚本执行完毕 ---")


if __name__ == "__main__":
    main()