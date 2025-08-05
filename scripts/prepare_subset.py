import json
import argparse
from pathlib import Path


def create_subset(original_dataset_path: Path, subset_path: Path, num_samples: int):
    """从原始数据集中提取指定数量的样本并保存为新的子集文件。"""
    if not original_dataset_path.exists():
        print(f"错误: 原始数据集文件 '{original_dataset_path}' 不存在。")
        exit(1)

    print(f"正在从 '{original_dataset_path}' 读取数据...")
    with open(original_dataset_path, 'r', encoding='utf-8') as f:
        full_data = json.load(f)

    if not isinstance(full_data, list):
        print("错误: 数据集格式不正确，期望是一个JSON数组。")
        exit(1)

    if num_samples > len(full_data):
        print(f"警告: 请求的样本数量 ({num_samples}) 大于总样本数 ({len(full_data)})。将使用所有样本。")
        num_samples = len(full_data)

    subset_data = full_data[:num_samples]

    # 确保输出目录存在
    subset_path.parent.mkdir(parents=True, exist_ok=True)

    with open(subset_path, 'w', encoding='utf-8') as f:
        json.dump(subset_data, f, indent=2)

    print(f"✔️  成功创建子集，包含 {len(subset_data)} 条样本，已保存至: '{subset_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为WAKENLLM工具包创建数据集子集。")
    parser.add_argument("--dataset", required=True, help="原始数据集的名称 (例如, FLD, FOLIO)")
    parser.add_argument("--samples", required=True, type=int, help="要提取的样本数量")

    args = parser.parse_args()

    data_dir = Path(__file__).parent.parent / "data"
    original_path = data_dir / f"{args.dataset}.json"

    # 定义子集文件的命名规则
    subset_filename = f"{args.dataset}_subset_{args.samples}.json"
    subset_path = data_dir / subset_filename

    create_subset(original_path, subset_path, args.samples)