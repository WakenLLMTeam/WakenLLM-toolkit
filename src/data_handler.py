import json
from pathlib import Path
from typing import List, Dict, Any


class DataHandler:
    """
    Dedicated manager responsible for all file input/output (I/O) operations in the project.
    It parses path templates from configuration and provides unified read/write methods.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with the merged configuration dictionary.
        """
        self.config = config
        self.results_dir = Path("results")
        self.data_dir = Path("data")

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, template_key: str, **kwargs) -> Path:
        """
        An internal helper method that dynamically generates complete file paths based on configuration.
        """
        template = self.config['paths'][template_key]

        # Use passed parameters and configuration to fill the template
        format_args = {
            "dataset_name": self.config['dataset_name'],
            "model_name": self.config['model_name'],
            **kwargs  # Allow additional formatting parameters, such as situation
        }
        return Path(template.format(**format_args))

    # --- Critical fix: Add this missing helper method ---
    def save_json(self, data: Any, file_path: Path):
        """Generic JSON save method for other save functions to call."""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Data successfully saved to: {file_path}")

    # -----------------------------------------

    def load_and_filter_dataset(self) -> List[Dict[str, Any]]:
        """Load and filter verifiable samples from the original dataset. (Using strict '[]' accessor)"""
        raw_dataset_path = self._get_path('raw_dataset_template')
        print(f"Loading data from {raw_dataset_path}...")

        try:
            with open(raw_dataset_path, "r", encoding='utf-8') as file:
                full_dataset = json.load(file)
        except FileNotFoundError:
            print(f"Error: Dataset file {raw_dataset_path} not found. Please check your data directory and configuration file.")
            return []

        # --- Critical fix: Change .get("proof_label") to ["proof_label"] ---
        filtered_dataset = [
            element for element in full_dataset
            if element["proof_label"] in ["__PROVED__", "__DISPROVED__"]
        ]
        print(f"Data loading completed, filtered {len(filtered_dataset)} verifiable samples.")
        return filtered_dataset

    def load_unverifiable_dataset(self) -> List[Dict[str, Any]]:
        """Load and filter unverifiable samples from the original dataset. (Using strict '[]' accessor)"""
        raw_dataset_path = self._get_path('raw_dataset_template')
        print(f"Loading data from {raw_dataset_path} to find unverifiable samples...")

        try:
            with open(raw_dataset_path, "r", encoding='utf-8') as file:
                full_dataset = json.load(file)
        except FileNotFoundError:
            print(f"Error: Dataset file {raw_dataset_path} not found.")
            return []

        # --- Critical fix: Change .get("proof_label") to ["proof_label"] ---
        unverifiable_dataset = [
            element for element in full_dataset
            if element["proof_label"] == "__UNKNOWN__"
        ]
        print(f"Data loading completed, filtered {len(unverifiable_dataset)} unverifiable samples.")
        return unverifiable_dataset

    def save_stage1_stimulation_output(self, data: List[Dict[str, Any]]):
        """Save the dataset after Stage 1 Stimulation processing."""
        output_path = self._get_path('step4_postprocess_template')
        self.save_json(data, output_path)

    def save_stage2_reflection_output(self, data: List[Dict[str, Any]]):
        """Save the final dataset after Stage 2 Reflection processing."""
        output_path = self._get_path('step5_final_output_template')
        self.save_json(data, output_path)

    def save_rtg_label_situation_results(self, data: Dict[str, Any], situation: str):
        """Save RtG Label test evaluation results by situation."""
        output_path = self._get_path('rtg_label_eval_template', situation=situation)
        self.save_json(data, output_path)

    # ... (Reserved save method for future setting3) ...