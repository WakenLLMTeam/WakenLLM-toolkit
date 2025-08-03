import re
from typing import List, Dict, Any

class Evaluator:
    """
    负责所有评估指标的计算。
    """

    def parse_llm_output(self, text: str) -> str:
        """
        从模型的文本输出中提取标准标签。
        此最终版本与旧版 step1.py 的 extract_label 函数逻辑完全一致。
        """
        if not isinstance(text, str):
            return "__UNKNOWN__"

        # 使用与 step1.py 完全相同的正则表达式逻辑，查找第一个出现的关键词
        match = re.search(r"(DISPROVED|PROVED|UNKNOWN)", text.upper())

        if match:
            # 返回找到的第一个匹配项，并格式化
            return f"__{match.group(1)}__"
        else:
            # 如果找不到任何关键词，则返回UNKNOWN
            return "__UNKNOWN__"

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
        """从模型的输出中解析出'True'或'False'，与旧版step2.py逻辑保持严格一致。"""
        if not isinstance(text, str):
            return "False"

        processed_text = text.strip().upper()

        # 使用严格的 '==' 来匹配旧版逻辑，而不是宽容的 'in'
        if processed_text == "TRUE":
            return "True"
        else:
            return "False"