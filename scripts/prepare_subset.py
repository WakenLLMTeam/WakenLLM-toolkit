import json
import argparse
from pathlib import Path


def create_subset(original_dataset_path: Path, subset_path: Path, num_samples: int):
    if not original_dataset_path.exists():
        print(f"Error: Original dataset file '{original_dataset_path}' does not exist.")
        exit(1)

    print(f"Reading data from '{original_dataset_path}'...")
    with open(original_dataset_path, 'r', encoding='utf-8') as f:
        full_data = json.load(f)

    if not isinstance(full_data, list):
        print("Error: Dataset format is incorrect. Expected a JSON array.")
        exit(1)

    if num_samples > len(full_data):
        print(f"Warning: Requested number of samples ({num_samples}) exceeds total samples ({len(full_data)}). Will use all samples.")
        num_samples = len(full_data)

    subset_data = full_data[:num_samples]

    # Ensure output directory exists
    subset_path.parent.mkdir(parents=True, exist_ok=True)

    with open(subset_path, 'w', encoding='utf-8') as f:
        json.dump(subset_data, f, indent=2)

    print(f"✔️  Successfully created subset with {len(subset_data)} samples, saved to: '{subset_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a dataset subset for the WAKENLLM toolkit.")
    parser.add_argument("--dataset", required=True, help="Name of the original dataset (e.g., FLD, FOLIO)")
    parser.add_argument("--samples", required=True, type=int, help="Number of samples to extract")

    args = parser.parse_args()

    data_dir = Path(__file__).parent.parent / "data"
    original_path = data_dir / f"{args.dataset}.json"

    # Define naming convention for subset files
    subset_filename = f"{args.dataset}_subset_{args.samples}.json"
    subset_path = data_dir / subset_filename

    create_subset(original_path, subset_path, args.samples)