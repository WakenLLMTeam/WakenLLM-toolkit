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

        Args:
            config: 包含了路径、模型名、数据集名等所有信息的配置字典。
        """
        self.config = config
        self.results_dir = Path("results")
        self.data_dir = Path("data")

        # 确保输出目录存在
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, template_key: str) -> Path:
        """
        一个内部辅助方法，根据配置动态生成完整的文件路径。

        Args:
            template_key: 配置中`paths`字典下的路径模板键名。

        Returns:
            一个完整的、可用的Path对象。
        """
        template = self.config['paths'][template_key]

        # 使用配置中的信息填充路径模板中的占位符
        return Path(template.format(
            dataset_name=self.config['dataset_name'],
            model_name=self.config['model_name']
        ))

    def load_and_filter_dataset(self) -> List[Dict[str, Any]]:
        """
        加载并筛选原始数据集。
        这部分逻辑来自于 step1.py 的数据加载部分。
        """
        raw_dataset_path = self._get_path('raw_dataset_template')
        print(f"正在从 {raw_dataset_path} 加载数据...")

        try:
            with open(raw_dataset_path, "r", encoding='utf-8') as file:
                full_dataset = json.load(file)
        except FileNotFoundError:
            print(f"错误：找不到数据集文件 {raw_dataset_path}。请检查您的data目录和配置文件。")
            return []

        # 仅筛选出需要评估的可验证样本
        filtered_dataset = [
            element for element in full_dataset
            if element.get("proof_label") in ["__PROVED__", "__DISPROVED__"]
        ]
        print(f"数据加载完成，筛选出 {len(filtered_dataset)} 条可验证样本。")
        return filtered_dataset

    def save_step1_output(self, data: List[Dict[str, Any]]):
        """
        保存Step1处理后的，标注了"False UNKNOWN"的样本集。
        """
        output_path = self._get_path('step1_output_template')
        print(f"正在将Step 1的输出保存到 {output_path}...")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print("保存成功。")

    def save_stage1_stimulation_output(self, data: List[Dict[str, Any]]):
        """保存Stage 1 Stimulation处理后的数据集。"""
        output_path = self._get_path('step4_postprocess_template')
        print(f"正在将Stage 1 Stimulation的输出保存到 {output_path}...")
        self.save_json(data, output_path)  # 复用已有的save_json方法
        print("保存成功。")

    def save_evaluation_results(self, data: Dict[str, Any], step_name: str):
        """
        保存评估结果，如准确率、预测详情等。
        """
        # 我们可以让路径模板更通用
        eval_path_template = self.config['paths'].get(
            f'evaluation_{step_name}_template',
            f"results/evaluation_{step_name}_{self.config['dataset_name']}_{self.config['model_name']}.json"
        )
        eval_path = Path(eval_path_template)

        print(f"正在将 {step_name} 的评估结果保存到 {eval_path}...")
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print("保存成功。")

    def save_stage2_reflection_output(self, data: List[Dict[str, Any]]):
        """保存Stage 2 Reflection处理后的最终数据集。"""
        output_path = self._get_path('step5_final_output_template')
        print(f"正在将Stage 2 Reflection的最终输出保存到 {output_path}...")
        self.save_json(data, output_path)
        print("保存成功。")

    def load_unverifiable_dataset(self) -> List[Dict[str, Any]]:
        """
        加载并筛选出原始数据集中的不可验证（__UNKNOWN__）样本。
        这部分逻辑来自于 step2.py。
        """
        raw_dataset_path = self._get_path('raw_dataset_template')
        print(f"正在从 {raw_dataset_path} 加载数据以寻找不可验证样本...")

        try:
            with open(raw_dataset_path, "r", encoding='utf-8') as file:
                full_dataset = json.load(file)
        except FileNotFoundError:
            print(f"错误：找不到数据集文件 {raw_dataset_path}。")
            return []

        # 仅筛选出标签为 __UNKNOWN__ 的样本
        unverifiable_dataset = [
            element for element in full_dataset
            if element.get("proof_label") == "__UNKNOWN__"
        ]
        print(f"数据加载完成，筛选出 {len(unverifiable_dataset)} 条不可验证样本。")
        return unverifiable_dataset

    def save_rtg_label_situation_results(self, data: Dict[str, Any], situation: str):
        """按情况保存RtG Label测试的评估结果。"""
        # 我们需要一个更灵活的路径生成方法
        template = self.config['paths']['rtg_label_eval_template']
        output_path = Path(template.format(
            dataset_name=self.config['dataset_name'],
            model_name=self.config['model_name'],
            situation=situation
        ))
        print(f"正在将 Situation '{situation}' 的评估结果保存到 {output_path}...")
        self.save_json(data, output_path)
        print("保存成功。")

    def save_rtg_process_step4_output(self, data: List[Dict[str, Any]]):
        """保存RtG Process测试第一步（step4）的输出。"""
        output_path = self._get_path('rtg_process_step4_output_template')
        print(f"正在将 RtG Process-Step4 的输出保存到 {output_path}...")
        self.save_json(data, output_path)
        print("保存成功。")

    def save_rtg_process_step5_output(self, data: List[Dict[str, Any]]):
        """保存RtG Process测试第二步（step5）的输出。"""
        output_path = self._get_path('rtg_process_step5_output_template')
        print(f"正在将 RtG Process-Step5 的输出保存到 {output_path}...")
        self.save_json(data, output_path)
        print("保存成功。")

    def save_rtg_process_final_evaluation(self, data: Dict[str, Any]):
        """保存RtG Process测试的最终评估指标。"""
        output_path = self._get_path('rtg_process_final_eval_template')
        print(f"正在将 RtG Process 的最终评估结果保存到 {output_path}...")
        self.save_json(data, output_path)
        print("保存成功。")