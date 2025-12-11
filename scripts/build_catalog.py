"""CLI wrapper for building an asset catalog from a pack directory.

This script resolves relative paths against the repository root and writes the
catalog JSON to the requested output location. It can also upsert the assets
into Firestore using schemas/firestore_asset_schema.json, storing pack metadata
and sim-ready flags as metadata-only references for NVIDIA packs.

Example:
    python scripts/build_catalog.py \
        --pack-path /path/to/ResidentialAssetsPack \
        --pack-name ResidentialAssetsPack \
        --output asset_index.json \
        --push-to-firestore
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from src.asset_catalog import AssetCatalogBuilder
from src.asset_catalog.ingestion import AssetIngestionService, StorageURIs

try:
    from google.cloud import firestore
except Exception:  # pragma: no cover - dependency guard
    firestore = None


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
        help="Path to the asset pack directory to index.",
    )
    parser.add_argument(
        "--pack-name",
        required=True,
        help="Name to record for the asset pack.",
    )
    parser.add_argument(
        "--output",
        default="asset_index.json",
        help="Output path for the catalog JSON (relative paths use repo root).",
    )
    parser.add_argument(
        "--push-to-firestore",
        action="store_true",
        help="When set, upsert catalog entries into Firestore using the configured collection.",
    )
    parser.add_argument(
        "--firestore-project",
        default=os.getenv("FIRESTORE_PROJECT"),
        help="Optional Firestore project ID (defaults to FIRESTORE_PROJECT env variable).",
    )
    parser.add_argument(
        "--firestore-collection",
        default=os.getenv("FIRESTORE_COLLECTION", "assets"),
        help="Firestore collection to upsert into (defaults to FIRESTORE_COLLECTION env variable or 'assets').",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
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

        if args.push_to_firestore:
            if not firestore:
                logging.warning("google-cloud-firestore is not installed; skipping Firestore sync")
            else:
                client = firestore.Client(project=args.firestore_project) if args.firestore_project else firestore.Client()
                ingestion = AssetIngestionService(
                    firestore_client=client,
                    collection_path=args.firestore_collection,
                )
                for asset in catalog.assets:
                    storage = StorageURIs(
                        usd_uri=f"pack://{catalog.pack_name}/{asset.relative_path}",
                        thumbnail_uri=asset.thumbnail_path,
                        gcs_bucket=None,
                    )
                    try:
                        ingestion.ingest_pack_asset(asset, catalog, storage=storage)
                    except Exception as exc:  # pragma: no cover - best effort ingest
                        logging.error("Failed to upsert %s: %s", asset.asset_id, exc)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to build catalog: {exc}", file=sys.stderr)
        return 1

    print(f"Catalog saved to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
