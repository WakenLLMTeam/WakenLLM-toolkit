import json
from pathlib import Path
from typing import List, Dict, Any


class DataHandler:
    """
    负责项目中所有文件输入/输出（I/O）的专职管家。
    它解析配置中的路径模板，并提供统一的读写方法。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化时接收合并后的配置字典。
        """
        self.config = config
        self.results_dir = Path("results")
        self.data_dir = Path("data")

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, template_key: str, **kwargs) -> Path:
        """
        一个内部辅助方法，根据配置动态生成完整的文件路径。
        """
        template = self.config['paths'][template_key]

        # 使用传入的参数和配置来填充模板
        format_args = {
            "dataset_name": self.config['dataset_name'],
            "model_name": self.config['model_name'],
            **kwargs  # 允许传入额外的格式化参数，如 situation
        }
        return Path(template.format(**format_args))

    # --- 关键修正：添加这个被遗漏的辅助方法 ---
    def save_json(self, data: Any, file_path: Path):
        """通用的JSON保存方法，供其他保存函数调用。"""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"数据已成功保存到: {file_path}")

    # -----------------------------------------

    def load_and_filter_dataset(self) -> List[Dict[str, Any]]:
        """加载并筛选原始数据集中的可验证样本。(使用严格的 '[]' 访问器)"""
        raw_dataset_path = self._get_path('raw_dataset_template')
        print(f"正在从 {raw_dataset_path} 加载数据...")

        try:
            with open(raw_dataset_path, "r", encoding='utf-8') as file:
                full_dataset = json.load(file)
        except FileNotFoundError:
            print(f"错误：找不到数据集文件 {raw_dataset_path}。请检查您的data目录和配置文件。")
            return []

        # --- 关键修正：将 .get("proof_label") 修改为 ["proof_label"] ---
        filtered_dataset = [
            element for element in full_dataset
            if element["proof_label"] in ["__PROVED__", "__DISPROVED__"]
        ]
        print(f"数据加载完成，筛选出 {len(filtered_dataset)} 条可验证样本。")
        return filtered_dataset

    def load_unverifiable_dataset(self) -> List[Dict[str, Any]]:
        """加载并筛选出原始数据集中的不可验证样本。(使用严格的 '[]' 访问器)"""
        raw_dataset_path = self._get_path('raw_dataset_template')
        print(f"正在从 {raw_dataset_path} 加载数据以寻找不可验证样本...")

        try:
            with open(raw_dataset_path, "r", encoding='utf-8') as file:
                full_dataset = json.load(file)
        except FileNotFoundError:
            print(f"错误：找不到数据集文件 {raw_dataset_path}。")
            return []

        # --- 关键修正：将 .get("proof_label") 修改为 ["proof_label"] ---
        unverifiable_dataset = [
            element for element in full_dataset
            if element["proof_label"] == "__UNKNOWN__"
        ]
        print(f"数据加载完成，筛选出 {len(unverifiable_dataset)} 条不可验证样本。")
        return unverifiable_dataset

    def save_stage1_stimulation_output(self, data: List[Dict[str, Any]]):
        """保存Stage 1 Stimulation处理后的数据集。"""
        output_path = self._get_path('step4_postprocess_template')
        self.save_json(data, output_path)

    def save_stage2_reflection_output(self, data: List[Dict[str, Any]]):
        """保存Stage 2 Reflection处理后的最终数据集。"""
        output_path = self._get_path('step5_final_output_template')
        self.save_json(data, output_path)

    def save_rtg_label_situation_results(self, data: Dict[str, Any], situation: str):
        """按情况保存RtG Label测试的评估结果。"""
        output_path = self._get_path('rtg_label_eval_template', situation=situation)
        self.save_json(data, output_path)

    # ... (为未来的setting3预留的保存方法) ...