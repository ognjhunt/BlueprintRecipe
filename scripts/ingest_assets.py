#!/usr/bin/env python3
"""Ingest asset metadata and embeddings into Firestore and a vector DB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.asset_catalog import (
    AssetCatalogBuilder,
    AssetEmbeddings,
    AssetIngestionService,
    StorageURIs,
    VectorStoreClient,
    VectorStoreConfig,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest assets into Firestore and a vector DB")
    parser.add_argument("--catalog", type=Path, help="Optional asset catalog JSON to ingest")
    parser.add_argument(
        "--zeroscene", type=Path, help="Optional ZeroScene asset payload (JSON dict or list)"
    )
    parser.add_argument("--gcs-bucket", required=True, help="Destination GCS bucket for USDs and thumbnails")
    parser.add_argument(
        "--usd-prefix", default="usd", help="Prefix under the bucket where USD files are stored"
    )
    parser.add_argument(
        "--thumbnail-prefix",
        default="thumbnails",
        help="Prefix under the bucket where thumbnails are stored",
    )
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Embedding model for ZeroScene assets")
    parser.add_argument(
        "--vector-store-provider",
        choices=["in-memory", "vertex-ai", "pgvector"],
        default="in-memory",
        help="Vector store provider for embeddings",
    )
    parser.add_argument("--vector-store-collection", default="asset-embeddings", help="Vector store collection")
    parser.add_argument("--vector-store-uri", help="Connection URI for pgvector")
    parser.add_argument("--project-id", help="Cloud project for managed providers")
    parser.add_argument("--api-key", help="API key for cloud embedding backends if required")
    parser.add_argument(
        "--thumbnail-root",
        default=None,
        help="Local directory to resolve thumbnail paths when ingesting catalog assets",
    )
    return parser.parse_args()


def _build_storage_uris(bucket: str, usd_prefix: str, thumb_prefix: str, relative_path: str, thumbnail_path: str | None) -> StorageURIs:
    usd_uri = f"gs://{bucket}/{usd_prefix.rstrip('/')}/{relative_path}"
    thumb_uri = None
    if thumbnail_path:
        thumb_uri = f"gs://{bucket}/{thumb_prefix.rstrip('/')}/{Path(thumbnail_path).name}"
    return StorageURIs(usd_uri=usd_uri, thumbnail_uri=thumb_uri, gcs_bucket=bucket)


def _load_zeroscene_payload(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError("ZeroScene payload must be a dict or list")


def _ingest_catalog_assets(args: argparse.Namespace, ingestion: AssetIngestionService, embeddings: AssetEmbeddings) -> None:
    if not args.catalog:
        return

    catalog = AssetCatalogBuilder.load(str(args.catalog))
    embeddings.build_index(catalog, thumbnail_root=args.thumbnail_root)

    for asset in catalog.assets:
        storage = _build_storage_uris(
            args.gcs_bucket,
            args.usd_prefix,
            args.thumbnail_prefix,
            asset.relative_path,
            asset.thumbnail_path,
        )
        text_emb = embeddings.embeddings.get(asset.asset_id)
        thumb_emb = embeddings.thumbnail_embeddings.get(asset.asset_id)
        ingestion.ingest_pack_asset(
            asset,
            catalog,
            storage=storage,
            text_embedding=text_emb,
            thumbnail_embedding=thumb_emb,
        )


def _ingest_zeroscene_assets(args: argparse.Namespace, ingestion: AssetIngestionService, embeddings: AssetEmbeddings) -> None:
    if not args.zeroscene:
        return

    payloads = _load_zeroscene_payload(args.zeroscene)

    for payload in payloads:
        description = payload.get("description") or payload.get("display_name") or payload.get("title")
        text_emb = embeddings.embed_text(description) if description else None

        thumbnail_embedding = None
        raw_thumb = payload.get("thumbnail_embedding")
        if raw_thumb is not None:
            thumbnail_embedding = np.array(raw_thumb, dtype=np.float32)

        storage = StorageURIs(
            usd_uri=payload.get("usd_uri")
            or f"gs://{args.gcs_bucket}/{args.usd_prefix.rstrip('/')}/{payload.get('asset_id', payload.get('id'))}.usd",
            thumbnail_uri=payload.get("thumbnail_uri"),
            gcs_bucket=args.gcs_bucket,
        )

        ingestion.ingest_zeroscene_asset(
            payload,
            storage=storage,
            text_embedding=text_emb,
            thumbnail_embedding=thumbnail_embedding,
        )


def main() -> None:
    args = _parse_args()

    vector_config = VectorStoreConfig(
        provider=args.vector_store_provider,
        collection=args.vector_store_collection,
        connection_uri=args.vector_store_uri,
        project_id=args.project_id,
    )
    vector_store = VectorStoreClient(vector_config)

    ingestion = AssetIngestionService(vector_store=vector_store)
    embedding_helper = AssetEmbeddings(vector_store=vector_store)
    embedding_helper.config.model_name = args.model
    if args.api_key:
        embedding_helper.config.api_key = args.api_key

    _ingest_catalog_assets(args, ingestion, embedding_helper)
    _ingest_zeroscene_assets(args, ingestion, embedding_helper)

    print("Ingestion complete")


if __name__ == "__main__":
    main()
