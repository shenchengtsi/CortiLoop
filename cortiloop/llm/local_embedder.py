"""
Local embedding + reranking using sentence-transformers.

Auto-downloads models from HuggingFace on first use. No API key needed.
Supports 100+ languages with bge-m3 default.

Environment variables:
    CORTILOOP_EMBEDDING_MODEL   — override embedding model (default: BAAI/bge-m3)
    CORTILOOP_RERANKER_MODEL    — override reranker model (default: BAAI/bge-reranker-v2-m3)
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("cortiloop.embedder")

# Defaults — multilingual, good for both Chinese and English
_DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
_DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Lighter alternatives for resource-constrained environments
_LIGHT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
_LIGHT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class LocalEmbedder:
    """
    Local embedding using sentence-transformers.
    Model is loaded lazily on first call and cached.
    """

    def __init__(self, model_name: str | None = None):
        self._model_name = (
            model_name
            or os.environ.get("CORTILOOP_EMBEDDING_MODEL")
            or _DEFAULT_EMBEDDING_MODEL
        )
        self._model: Any = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
            return self._model
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for local embedding. "
                "Install with: pip install sentence-transformers"
            )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._load_model()
        embeddings = model.encode(texts, normalize_embeddings=True)
        return [e.tolist() for e in embeddings]

    async def embed_one(self, text: str) -> list[float]:
        results = await self.embed([text])
        return results[0]


class LocalReranker:
    """
    Local cross-encoder reranking using sentence-transformers.
    Model is loaded lazily on first call and cached.
    """

    def __init__(self, model_name: str | None = None):
        self._model_name = (
            model_name
            or os.environ.get("CORTILOOP_RERANKER_MODEL")
            or _DEFAULT_RERANKER_MODEL
        )
        self._model: Any = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
            logger.info("Loading reranker model: %s", self._model_name)
            self._model = CrossEncoder(self._model_name)
            return self._model
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for local reranking. "
                "Install with: pip install sentence-transformers"
            )

    async def rerank(
        self, query: str, documents: list[str], top_k: int = 10
    ) -> list[tuple[int, float]]:
        model = self._load_model()
        pairs = [[query, doc] for doc in documents]
        scores = model.predict(pairs)
        indexed = [(i, float(s)) for i, s in enumerate(scores)]
        indexed.sort(key=lambda x: x[1], reverse=True)
        return indexed[:top_k]
