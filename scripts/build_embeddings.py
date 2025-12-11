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
        "--api-key",
        help="API key for remote embedding providers (OpenAI, Vertex/Gemini)",
    )
    parser.add_argument(
        "--project-id",
        help="Project identifier for cloud backends (if required)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation",
    )
    parser.add_argument(
        "--include-thumbnails",
        action="store_true",
        help="Also generate embeddings for asset thumbnails (when available)",
    )
    parser.add_argument(
        "--image-model",
        default=None,
        help="Image embedding model name (e.g., clip-ViT-B-32 or SigLIP)",
    )
    parser.add_argument(
        "--image-backend",
        default=None,
        choices=["sentence-transformers", "vertex-ai", "openai"],
        help="Backend to use for image embeddings (defaults to text backend)",
    )
    parser.add_argument(
        "--thumbnail-root",
        default=None,
        help="Optional base directory for resolving thumbnail paths",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    catalog = AssetCatalogBuilder.load(str(args.catalog))

    config = EmbeddingConfig(
        model_name=args.model,
        backend=args.backend,
        api_key=args.api_key,
        project_id=args.project_id,
        image_model_name=args.image_model,
        image_backend=args.image_backend,
    )
    embeddings = AssetEmbeddings(config)
    try:
        embeddings.build_index(
            catalog,
            batch_size=args.batch_size,
            thumbnail_root=args.thumbnail_root,
        )
        if not args.include_thumbnails:
            embeddings.thumbnail_embeddings = {}
            embeddings.thumbnail_index_matrix = None
            embeddings.thumbnail_asset_ids = []
    except Exception as exc:
        raise SystemExit(f"Failed to build embeddings: {exc}") from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    embeddings.save(str(args.output))

    print(
        f"Saved embeddings for {len(catalog.assets)} assets to {args.output} "
        f"using model '{args.model}'"
    )


if __name__ == "__main__":
    main()
