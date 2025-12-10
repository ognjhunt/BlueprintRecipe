"""Path utilities."""

import os
from pathlib import Path
from typing import Optional


def resolve_path(
    path: str,
    base_path: Optional[str] = None,
    must_exist: bool = False
) -> Path:
    """
    Resolve a path, handling relative paths and environment variables.

    Args:
        path: Path to resolve
        base_path: Base path for relative paths
        must_exist: Raise error if path doesn't exist

    Returns:
        Resolved Path object
    """
    # Expand environment variables
    path = os.path.expandvars(path)

    # Expand user home
    path = os.path.expanduser(path)

    # Create Path object
    resolved = Path(path)

    # Handle relative paths
    if not resolved.is_absolute() and base_path:
        resolved = Path(base_path) / resolved

    # Make absolute
    resolved = resolved.resolve()

    # Check existence if required
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved}")

    return resolved


def ensure_dir(path: str) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object for the directory
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_relative_path(path: str, base: str) -> str:
    """
    Get relative path from base.

    Args:
        path: Target path
        base: Base path

    Returns:
        Relative path string
    """
    try:
        return str(Path(path).relative_to(Path(base)))
    except ValueError:
        return path
