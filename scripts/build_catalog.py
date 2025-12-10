"""CLI wrapper for building an asset catalog from a pack directory.

This script resolves relative paths against the repository root and writes the
catalog JSON to the requested output location.

Example:
    python scripts/build_catalog.py \
        --pack-path /path/to/ResidentialAssetsPack \
        --pack-name ResidentialAssetsPack \
        --output asset_index.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.asset_catalog import AssetCatalogBuilder


def find_repo_root(start: Path) -> Path:
    """Locate the repository root by searching for a .git directory."""
    for candidate in [start] + list(start.parents):
        if (candidate / ".git").exists():
            return candidate
    # Fallback to the directory containing the script
    return start


def resolve_path(path_str: str, repo_root: Path) -> Path:
    """Resolve a possibly relative path against the repository root."""
    path = Path(path_str)
    return path if path.is_absolute() else repo_root / path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build an asset catalog JSON file.")
    parser.add_argument(
        "--pack-path",
        required=True,
        help="Path to the asset pack directory to index."
    )
    parser.add_argument(
        "--pack-name",
        required=True,
        help="Name to record for the asset pack."
    )
    parser.add_argument(
        "--output",
        default="asset_index.json",
        help="Output path for the catalog JSON (relative paths use repo root)."
    )

    args = parser.parse_args(argv)

    repo_root = find_repo_root(Path(__file__).resolve())
    pack_path = resolve_path(args.pack_path, repo_root)
    output_path = resolve_path(args.output, repo_root)

    if not pack_path.exists():
        print(f"Pack path does not exist: {pack_path}", file=sys.stderr)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        builder = AssetCatalogBuilder(pack_path=str(pack_path), pack_name=args.pack_name)
        catalog = builder.build()
        builder.save(catalog, str(output_path))
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to build catalog: {exc}", file=sys.stderr)
        return 1

    print(f"Catalog saved to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
