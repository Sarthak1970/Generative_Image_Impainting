import os
import yaml
from pathlib import Path
from typing import Optional, Union

DEFAULT_CONFIG_RELATIVE_PATH = Path("configs") / "config.yaml"
PROJECT_ROOT_ENV_VAR = "REPAINT_PROJECT_ROOT"


def load_config(config_path: str) -> dict:
    """Read Conf file and return it as a dictionary"""

    path=Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(path,"r") as f:
        config=yaml.safe_load(f)

    return config


def _looks_like_project_root(path: Path) -> bool:
    return (path / "src" / "repaint.py").is_file() and (path / DEFAULT_CONFIG_RELATIVE_PATH).is_file()


def find_project_root(start: Optional[Union[str, Path]] = None) -> Path:
    """Locate the repository root (the folder holding ``src/`` and ``configs/``).

    Order: the ``REPAINT_PROJECT_ROOT`` environment variable, then ``start``
    (default: the current working directory) and each of its parents, then
    the parent of this file's ``src/`` folder. Raises a clear error if none
    contains both ``src/repaint.py`` and ``configs/config.yaml``.
    """
    candidates = []
    env_root = os.environ.get(PROJECT_ROOT_ENV_VAR)
    if env_root:
        candidates.append(Path(env_root).expanduser())
    start_path = Path(start).expanduser() if start is not None else Path.cwd()
    start_path = start_path.resolve()
    candidates.extend([start_path, *start_path.parents])
    candidates.append(Path(__file__).resolve().parent.parent)
    for candidate in candidates:
        if _looks_like_project_root(candidate):
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find the project root (a folder containing src/repaint.py and "
        f"{DEFAULT_CONFIG_RELATIVE_PATH.as_posix()}). Searched: "
        + ", ".join(str(c) for c in candidates)
        + f". Set the {PROJECT_ROOT_ENV_VAR} environment variable to the repository folder."
    )


def resolve_project_path(relative: Union[str, Path], project_root: Optional[Path] = None,
                         must_exist: bool = False) -> Path:
    """Resolve ``relative`` against the project root (absolute paths pass through)."""
    path = Path(relative).expanduser()
    if not path.is_absolute():
        path = (project_root or find_project_root()) / path
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Required project path does not exist: {path}")
    return path


def load_project_config(config_path: Optional[Union[str, Path]] = None,
                        project_root: Optional[Path] = None) -> dict:
    """Load ``configs/config.yaml`` (or ``config_path``) relative to the project root."""
    root = project_root or find_project_root()
    path = resolve_project_path(config_path or DEFAULT_CONFIG_RELATIVE_PATH, root)
    if not path.is_file():
        raise FileNotFoundError(
            f"Config file not found at {path}. Expected {DEFAULT_CONFIG_RELATIVE_PATH.as_posix()} "
            f"under the project root {root}."
        )
    return load_config(str(path))
