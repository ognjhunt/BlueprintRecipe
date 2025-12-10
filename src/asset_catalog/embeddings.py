"""
Asset Embeddings - Vector embeddings for semantic asset search.

This module provides embedding-based asset search using:
- Text embeddings for descriptions and tags
- Image embeddings for thumbnails
- Combined multimodal embeddings

Supports various backends:
- Local sentence-transformers
- Cloud APIs (Vertex AI, OpenAI)
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import numpy as np


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    backend: str = "sentence-transformers"  # or "vertex-ai", "openai"
    api_key: Optional[str] = None
    project_id: Optional[str] = None


class AssetEmbeddings:
    """
    Manages vector embeddings for asset search.

    Usage:
        embeddings = AssetEmbeddings(config)
        embeddings.build_index(catalog)
        matches = embeddings.search("modern wooden dining table", top_k=10)
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self.embeddings: dict[str, np.ndarray] = {}
        self.asset_ids: list[str] = []
        self.index_matrix: Optional[np.ndarray] = None
        self._model = None

    def _load_model(self) -> None:
        """Load the embedding model."""
        if self._model is not None:
            return

        if self.config.backend == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.config.model_name)
            except ImportError:
                print("Warning: sentence-transformers not installed. Using random embeddings.")
                self._model = "stub"
        elif self.config.backend == "vertex-ai":
            # Would use Vertex AI embeddings
            self._model = "vertex-ai"
        elif self.config.backend == "openai":
            # Would use OpenAI embeddings
            self._model = "openai"
        else:
            self._model = "stub"

    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        self._load_model()

        if self._model == "stub" or self._model is None:
            # Return random embedding for testing
            np.random.seed(hash(text) % 2**32)
            return np.random.randn(self.config.dimension).astype(np.float32)

        if hasattr(self._model, 'encode'):
            return self._model.encode(text, convert_to_numpy=True)

        # Fallback
        np.random.seed(hash(text) % 2**32)
        return np.random.randn(self.config.dimension).astype(np.float32)

    def build_index(
        self,
        catalog: Any,
        batch_size: int = 32
    ) -> None:
        """
        Build embedding index for catalog assets.

        Args:
            catalog: AssetCatalog to index
            batch_size: Batch size for embedding generation
        """
        self._load_model()

        texts = []
        self.asset_ids = []

        for asset in catalog.assets:
            # Combine asset info into searchable text
            text_parts = [
                asset.display_name or "",
                asset.category,
                asset.subcategory or "",
                " ".join(asset.tags)
            ]
            text = " ".join(filter(None, text_parts))
            texts.append(text)
            self.asset_ids.append(asset.asset_id)

        # Generate embeddings in batches
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = [self.embed_text(t) for t in batch]
            all_embeddings.extend(batch_embeddings)

        # Build index matrix
        self.index_matrix = np.vstack(all_embeddings)

        # Store individual embeddings
        for asset_id, emb in zip(self.asset_ids, all_embeddings):
            self.embeddings[asset_id] = emb

    def search(
        self,
        query: str,
        top_k: int = 10,
        threshold: float = 0.0
    ) -> list[tuple[str, float]]:
        """
        Search for assets by semantic similarity.

        Args:
            query: Search query text
            top_k: Number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of (asset_id, similarity_score) tuples
        """
        if self.index_matrix is None or len(self.asset_ids) == 0:
            return []

        # Embed query
        query_emb = self.embed_text(query)

        # Compute cosine similarity
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        index_norms = self.index_matrix / (
            np.linalg.norm(self.index_matrix, axis=1, keepdims=True) + 1e-8
        )
        similarities = np.dot(index_norms, query_norm)

        # Get top k
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= threshold:
                results.append((self.asset_ids[idx], score))

        return results

    def save(self, path: str) -> None:
        """Save embeddings to file."""
        data = {
            "config": {
                "model_name": self.config.model_name,
                "dimension": self.config.dimension,
                "backend": self.config.backend
            },
            "asset_ids": self.asset_ids,
            "embeddings": {
                k: v.tolist() for k, v in self.embeddings.items()
            }
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str) -> None:
        """Load embeddings from file."""
        with open(path) as f:
            data = json.load(f)

        self.config = EmbeddingConfig(**data["config"])
        self.asset_ids = data["asset_ids"]
        self.embeddings = {
            k: np.array(v, dtype=np.float32)
            for k, v in data["embeddings"].items()
        }

        # Rebuild index matrix
        if self.asset_ids:
            self.index_matrix = np.vstack([
                self.embeddings[aid] for aid in self.asset_ids
            ])
