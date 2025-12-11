#!/usr/bin/env python3
"""
Utility script to build a text-embedding database for the asset catalog.

The script loads a catalog JSON file, generates embeddings over the
asset names, descriptions, and tags using the configured model, and
writes the resulting database to disk.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

from src.asset_catalog.catalog_builder import AssetCatalogBuilder
from src.asset_catalog.embeddings import AssetEmbeddings, EmbeddingConfig
from src.asset_catalog.vector_store import VectorStoreClient, VectorStoreConfig

try:
    from google.cloud import firestore
except Exception:  # pragma: no cover - dependency guard
    firestore = None


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
        type=Path,
        help="Optional path to write the embeddings DB (e.g., data/asset_embeddings.json)",
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
    parser.add_argument(
        "--vector-store-provider",
        choices=["in-memory", "vertex-ai", "pgvector"],
        default=os.getenv("VECTOR_STORE_PROVIDER"),
        help="If provided, also push embeddings to a vector DB instead of only writing JSON",
    )
    parser.add_argument(
        "--vector-store-uri",
        default=os.getenv("VECTOR_STORE_URI"),
        help="Connection string for the vector DB (for pgvector)",
    )
    parser.add_argument(
        "--vector-store-collection",
        default=os.getenv("VECTOR_STORE_COLLECTION", "asset-embeddings"),
        help="Collection or namespace to upsert embeddings into",
    )
    parser.add_argument(
        "--firestore-project",
        default=os.getenv("FIRESTORE_PROJECT"),
        help="Optional Firestore project for updating embedding references",
    )
    parser.add_argument(
        "--firestore-collection",
        default=os.getenv("FIRESTORE_COLLECTION", "assets"),
        help="Firestore collection where asset documents live",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.INFO)
    catalog = AssetCatalogBuilder.load(str(args.catalog))

    vector_store = None
    if args.vector_store_provider:
        vs_config = VectorStoreConfig(
            provider=args.vector_store_provider,
            collection=args.vector_store_collection,
            connection_uri=args.vector_store_uri,
            project_id=args.project_id,
        )
        vector_store = VectorStoreClient(vs_config)

    config = EmbeddingConfig(
        model_name=args.model,
        backend=args.backend,
        api_key=args.api_key,
        project_id=args.project_id,
        image_model_name=args.image_model,
        image_backend=args.image_backend,
    )
    embeddings = AssetEmbeddings(config, vector_store=vector_store)
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

    firestore_client = None
    if firestore:
        try:
            firestore_client = firestore.Client(project=args.firestore_project) if args.firestore_project else firestore.Client()
        except Exception as exc:  # pragma: no cover - dependency guard
            logging.error("Failed to initialize Firestore client: %s", exc)

    if firestore_client and vector_store:
        _update_firestore_embeddings(
            firestore_client=firestore_client,
            collection=args.firestore_collection,
            embeddings=embeddings,
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        embeddings.save(str(args.output))
        print(
            f"Saved embeddings for {len(catalog.assets)} assets to {args.output} "
            f"using model '{args.model}'"
        )
    elif vector_store:
        print(
            f"Upserted embeddings for {len(catalog.assets)} assets to vector store "
            f"{vector_store.config.collection} using model '{args.model}'"
        )
    else:
        raise SystemExit("No output path or vector store provided; nothing to persist")


def _update_firestore_embeddings(firestore_client: Any, collection: str, embeddings: AssetEmbeddings) -> None:
    """Persist embedding vector references back into Firestore documents."""

    if not embeddings.vector_store:
        logging.info("No vector store configured; skipping Firestore embedding updates")
        return

    provider = embeddings.vector_store.config.provider
    text_dim = embeddings.config.dimension
    thumb_dim = embeddings.config.image_dimension

    for asset_id in embeddings.asset_ids:
        payload: dict[str, dict[str, int | str]] = {"embeddings": {}}
        if asset_id in embeddings.embeddings:
            payload["embeddings"]["text"] = {
                "vector_id": f"{asset_id}:text",
                "provider": provider,
                "dimension": text_dim,
            }
        if asset_id in embeddings.thumbnail_embeddings:
            payload["embeddings"]["thumbnail"] = {
                "vector_id": f"{asset_id}:thumbnail",
                "provider": provider,
                "dimension": thumb_dim,
            }

        if not payload["embeddings"]:
            continue

        try:
            firestore_client.collection(collection).document(asset_id).set(payload, merge=True)
        except Exception as exc:  # pragma: no cover - best effort update
            logging.error("Failed to update embeddings for %s: %s", asset_id, exc)


if __name__ == "__main__":
    main()
