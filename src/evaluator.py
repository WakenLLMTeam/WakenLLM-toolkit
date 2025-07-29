import re
from typing import List, Dict, Any

class Evaluator:
    """
    负责所有评估指标的计算。
    """
    def parse_llm_output(self, text: str) -> str:
        """
        使用正则表达式从模型的文本输出中提取标准标签。
        """
        if not isinstance(text, str):
            return "__UNKNOWN__"
        # 增加了对“__PROVED__”等格式的直接匹配，以防模型直接输出正确格式
        m = re.search(r"__(DISPROVED|PROVED|UNKNOWN)__", text.upper())
        if m:
            return m.group(0)

        m = re.search(r"(DISPROVED|PROVED|UNKNOWN)", text.upper())
        return f"__{m.group(1)}__" if m else "__UNKNOWN__"

    def calculate_accuracy(self, predictions: List[str], ground_truths: List[str]) -> Dict[str, Any]:
        """
        计算准确率等基本指标。
        """
        correct_count = sum(p == gt for p, gt in zip(predictions, ground_truths))
        total_count = len(predictions)
        accuracy = correct_count / total_count if total_count > 0 else 0

        print(f"\n准确数量: {correct_count} / {total_count}")
        print(f"准确率: {accuracy:.2%}")

        return {
            "correct_count": correct_count,
            "total_count": total_count,
            "accuracy": accuracy
        }

    def parse_binary_answer(self, text: str) -> str:
        """从模型的输出中解析出'True'或'False'。"""
        if not isinstance(text, str):
            return "False"  # 发生错误时默认为推理能力问题

        # 将模型输出转为大写并去除首尾空格
        processed_text = text.strip().upper()

        if "TRUE" in processed_text:
            return "True"
        elif "FALSE" in processed_text:
            return "False"
        else:
            return "False"  # 如果没有明确的True/False，也认为是推理能力问题