"""Configuration utilities."""

import json
import os
from pathlib import Path
from typing import Any, Optional

import yaml


def load_config(path: str) -> dict[str, Any]:
    """
    Load configuration from JSON or YAML file.

    Args:
        path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        if path.suffix in [".yaml", ".yml"]:
            return yaml.safe_load(f)
        elif path.suffix == ".json":
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")


def save_config(config: dict[str, Any], path: str) -> None:
    """
    Save configuration to JSON or YAML file.

    Args:
        config: Configuration dictionary
        path: Path to save file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        if path.suffix in [".yaml", ".yml"]:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        elif path.suffix == ".json":
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")


def get_env_config() -> dict[str, Any]:
    """Get configuration from environment variables."""
    return {
        "google_api_key": os.environ.get("GOOGLE_API_KEY"),
        "google_cloud_project": os.environ.get("GOOGLE_CLOUD_PROJECT"),
        "asset_root": os.environ.get("ASSET_ROOT", "/mnt/assets"),
        "storage_bucket": os.environ.get("STORAGE_BUCKET", "blueprint-8c1ca.appspot.com"),
    }
