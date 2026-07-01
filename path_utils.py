from pathlib import Path
from typing import Union

PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_path(path: Union[str, Path]) -> Path:
    """Resolve a path relative to the project root when needed."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path
