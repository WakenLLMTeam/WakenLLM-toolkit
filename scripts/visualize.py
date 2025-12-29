import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Logging Configuration (Suggestion #3) ---
# Keep consistent with main project style, provide clear log output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- 2. Default Path Definitions ---
# Place path definitions at the beginning of the script for easy management
DEFAULT_RESULTS_DIR = Path(__file__).parent.parent / "results"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / "visual_summary"


def aggregate_results(results_dir: Path) -> Optional[pd.DataFrame]:
    """
    Scan all 'summary_*.json' files in the specified directory and aggregate them into a Pandas DataFrame.

    Args:
        results_dir (Path): Directory containing experiment result summary files.

    Returns:
        Optional[pd.DataFrame]: Returns aggregated DataFrame if summary files are found; otherwise returns None.
    """
    logger.info(f"Scanning directory '{results_dir}' for result summaries...")

    summaries: List[Dict[str, Any]] = []
    for f in results_dir.glob("summary_*.json"):
        with open(f, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
                # Flatten nested 'metrics' dictionary for easier DataFrame processing
                metrics = data.pop('metrics', {})
                data.update(metrics)
                summaries.append(data)
            except json.JSONDecodeError:
                logger.warning(f"Unable to parse file '{f}', skipping.")

    if not summaries:
        logger.error(f"No valid 'summary_*.json' files found in '{results_dir}'.")
        logger.error("Please run at least one experiment and ensure it correctly generates summary files.")
        return None

    logger.info(f"Successfully aggregated results from {len(summaries)} experiments.")
    return pd.DataFrame(summaries)


def generate_visuals(df: pd.DataFrame, metric: str, output_dir: Path) -> None:
    """
    Generate a CSV report and a comparison chart based on aggregated data.

    Args:
        df (pd.DataFrame): DataFrame containing all experiment results.
        metric (str): Name of the metric to visualize (e.g., 'OCR', 'TCR1').
        output_dir (Path): Directory to save generated reports and charts.
    """
    # --- 3. Ensure output directory exists (Suggestion #5) ---
    output_dir.mkdir(exist_ok=True)

    # 3.1 Save complete CSV aggregation report
    csv_path = output_dir / "all_experiments_summary.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"✔️  Complete aggregation report saved to: {csv_path}")

    # --- 4. Error handling: Check if metric exists (Suggestion #4) ---
    if metric not in df.columns:
        logger.error(f"Error: Specified metric '{metric}' does not exist in result data.")
        logger.warning(f"Available metrics include: {', '.join(df.columns)}")
        return

    # 3.2 Generate visualization chart
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    # Filter data with valid metric values
    plot_df = df.dropna(subset=[metric]).copy()
    if plot_df.empty:
        logger.warning(f"No valid data found for metric '{metric}', skipping plot.")
        return

    plot_df[metric] = plot_df[metric] * 100  # Convert to percentage

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

    # --- 5. Save as PNG and vector format (Suggestion #6) ---
    chart_path_png = output_dir / f"{metric.lower()}_comparison.png"
    chart_path_pdf = output_dir / f"{metric.lower()}_comparison.pdf"

    plt.savefig(chart_path_png, dpi=300)
    plt.savefig(chart_path_pdf, bbox_inches='tight')

    logger.info(f"✔️  Comparison chart saved as PNG format: {chart_path_png}")
    logger.info(f"✔️  Comparison chart saved as PDF vector format: {chart_path_pdf}")
    plt.show()


def main() -> None:
    """Main function: Parse arguments and drive the entire workflow."""
    # --- 6. Use argparse to parse command line arguments (Suggestion #1) ---
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

    logger.info("--- WAKENLLM Result Aggregation and Visualization Script Started ---")
    results_df = aggregate_results(args.results_dir)
    if results_df is not None:
        generate_visuals(results_df, args.metric, args.output_dir)
    logger.info("--- Script Execution Completed ---")


if __name__ == "__main__":
    main()