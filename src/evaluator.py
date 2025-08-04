import re
from typing import List, Dict, Any

class Evaluator:
    """
    Responsible for calculating all evaluation metrics.
    """

    def parse_llm_output(self, text: str) -> str:
        """
        Extract standard labels from model text output.
        This final version is completely consistent with the extract_label function logic from the old step1.py.
        """
        if not isinstance(text, str):
            return "__UNKNOWN__"

        # Use the exact same regex logic as step1.py to find the first occurring keyword
        match = re.search(r"(DISPROVED|PROVED|UNKNOWN)", text.upper())

        if match:
            # Return the first found match and format it
            return f"__{match.group(1)}__"
        else:
            # If no keywords are found, return UNKNOWN
            return "__UNKNOWN__"

    def calculate_accuracy(self, predictions: List[str], ground_truths: List[str]) -> Dict[str, Any]:
        """
        Calculate basic metrics such as accuracy.
        """
        correct_count = sum(p == gt for p, gt in zip(predictions, ground_truths))
        total_count = len(predictions)
        accuracy = correct_count / total_count if total_count > 0 else 0

        print(f"\nCorrect count: {correct_count} / {total_count}")
        print(f"Accuracy: {accuracy:.2%}")

        return {
            "correct_count": correct_count,
            "total_count": total_count,
            "accuracy": accuracy
        }

    def parse_binary_answer(self, text: str) -> str:
        """Parse 'True' or 'False' from model output, maintaining strict consistency with old step2.py logic."""
        if not isinstance(text, str):
            return "False"

        processed_text = text.strip().upper()

        # Use strict '==' to match old logic, not lenient 'in'
        if processed_text == "TRUE":
            return "True"
        else:
            return "False"