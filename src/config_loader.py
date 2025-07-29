import yaml
from pathlib import Path

def load_config(config_path: str) -> dict:
    """
    加载主配置文件和可选的密钥文件，并合并它们。

    Args:
        config_path: 主配置文件的路径 (e.g., 'configs/experiment.yaml').

    Returns:
        一个包含所有配置信息的字典。
    """
    # 加载主配置文件
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 尝试加载项目根目录的密钥文件
    secrets_path = Path(__file__).parent.parent / "secrets.yaml"
    if secrets_path.exists():
        with open(secrets_path, 'r', encoding='utf-8') as f:
            secrets = yaml.safe_load(f)
        # 将密钥信息合并到主配置中
        if secrets:
            config.update(secrets)
        print("成功加载密钥文件 (secrets.yaml)。")
    else:
        print(f"警告: 密钥文件 {secrets_path} 未找到。请确保已根据模板创建。")

    return config