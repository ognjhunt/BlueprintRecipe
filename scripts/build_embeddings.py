#!/usr/bin/env python3
"""
Utility script to build a text-embedding database for the asset catalog.

The script loads a catalog JSON file, generates embeddings over the
asset names, descriptions, and tags using the configured model, and
writes the resulting database to disk.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.asset_catalog.catalog_builder import AssetCatalogBuilder
from src.asset_catalog.embeddings import AssetEmbeddings, EmbeddingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an embeddings DB for an asset catalog")
    parser.add_argument(
        "--catalog",
        required=True,
        type=Path,
        help="Path to the catalog JSON file (e.g., asset_index.json)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to write the embeddings DB (e.g., data/asset_embeddings.json)",
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformers model name to load",
    )
    parser.add_argument(
        "--backend",
        default="sentence-transformers",
        choices=["sentence-transformers", "vertex-ai", "openai"],
        help="Embedding backend to use",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    catalog = AssetCatalogBuilder.load(str(args.catalog))

    config = EmbeddingConfig(
        model_name=args.model,
        backend=args.backend,
    )
    embeddings = AssetEmbeddings(config)

    embeddings.build_index(catalog, batch_size=args.batch_size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    embeddings.save(str(args.output))

    print(
        f"Saved embeddings for {len(catalog.assets)} assets to {args.output} "
        f"using model '{args.model}'"
    )


if __name__ == "__main__":
    main()
