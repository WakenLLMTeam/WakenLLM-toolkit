import yaml
from pathlib import Path

def load_config(config_path: str) -> dict:
    """
    Load the main configuration file and optional secrets file, then merge them.

    Args:
        config_path: Path to the main configuration file (e.g., 'configs/experiment.yaml').

    Returns:
        A dictionary containing all configuration information.
    """
    # Load main configuration file
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Try to load the secrets file from the project root directory
    secrets_path = Path(__file__).parent.parent / "secrets.yaml"
    if secrets_path.exists():
        with open(secrets_path, 'r', encoding='utf-8') as f:
            secrets = yaml.safe_load(f)
        # Merge secrets information into the main configuration
        if secrets:
            config.update(secrets)
        print("Successfully loaded secrets file (secrets.yaml).")
    else:
        print(f"Warning: Secrets file {secrets_path} not found. Please ensure it has been created from the template.")

    return config