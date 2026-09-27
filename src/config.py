import yaml
from pathlib import Path

def load_config(config_path: str) -> dict:
    """Read Conf file and return it as a dictionary"""

    path=Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(path,"r") as f:
        config=yaml.safe_load(f)

    return config