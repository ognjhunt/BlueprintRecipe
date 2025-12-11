"""Lightweight helpers for validating JSON payloads against schemas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator


def load_schema(schema_path: str | Path) -> dict[str, Any]:
    path = Path(schema_path)
    return json.loads(path.read_text())


def validate_payload(payload: dict[str, Any], schema_path: str | Path) -> dict[str, Any]:
    """Validate a payload against the provided JSON schema."""

    schema = load_schema(schema_path)
    validator = Draft7Validator(schema)
    errors = sorted(validator.iter_errors(payload), key=lambda e: e.path)

    return {
        "valid": len(errors) == 0,
        "errors": [f"{'/'.join([str(p) for p in error.path])}: {error.message}" for error in errors],
    }


def validate_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a scene manifest using the canonical schema."""

    repo_root = Path(__file__).resolve().parents[2]
    schema_path = repo_root / "schemas" / "scene_manifest_schema.json"
    return validate_payload(payload, schema_path)

