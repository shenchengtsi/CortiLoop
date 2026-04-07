"""
Vector index abstraction — pluggable ANN backends.

Supports:
- numpy brute-force (default, zero extra deps, good for <100K vectors)
- usearch HNSW (optional, pip install usearch, good for millions)

Brain analogy: upgrading from sequential hippocampal scan to
parallel cortical column activation for pattern matching.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

logger = logging.getLogger("cortiloop.vector_index")


class VectorIndex(ABC):
    """Abstract vector index — search by cosine similarity."""

    @abstractmethod
    def add(self, key: str, vector: list[float]) -> None:
        """Add or update a vector."""

    @abstractmethod
    def remove(self, key: str) -> None:
        """Remove a vector by key."""

    @abstractmethod
    def search(self, query: list[float], top_k: int = 20) -> list[tuple[str, float]]:
        """Return (key, similarity) pairs sorted by descending similarity."""

    @abstractmethod
    def count(self) -> int:
        """Number of vectors in the index."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all vectors."""

    def bulk_add(self, items: list[tuple[str, list[float]]]) -> None:
        """Add multiple vectors. Override for batch optimization."""
        for key, vec in items:
            self.add(key, vec)


class NumpyIndex(VectorIndex):
    """
    Brute-force cosine similarity using numpy.
    Simple, zero dependencies beyond numpy. Good for <100K vectors.
    """

    def __init__(self):
        self._keys: list[str] = []
        self._matrix: np.ndarray | None = None  # (N, dim) float32
        self._key_to_idx: dict[str, int] = {}
        self._dirty = False

    def add(self, key: str, vector: list[float]) -> None:
        vec = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm  # pre-normalize for fast cosine

        if key in self._key_to_idx:
            idx = self._key_to_idx[key]
            self._matrix[idx] = vec
        else:
            self._key_to_idx[key] = len(self._keys)
            self._keys.append(key)
            if self._matrix is None:
                self._matrix = vec.reshape(1, -1)
            else:
                self._matrix = np.vstack([self._matrix, vec.reshape(1, -1)])

    def remove(self, key: str) -> None:
        if key not in self._key_to_idx:
            return
        idx = self._key_to_idx.pop(key)
        self._keys.pop(idx)
        if self._matrix is not None and len(self._keys) > 0:
            self._matrix = np.delete(self._matrix, idx, axis=0)
            # Rebuild index mapping
            self._key_to_idx = {k: i for i, k in enumerate(self._keys)}
        else:
            self._matrix = None
            self._key_to_idx = {}

    def search(self, query: list[float], top_k: int = 20) -> list[tuple[str, float]]:
        if self._matrix is None or len(self._keys) == 0:
            return []
        q = np.array(query, dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm == 0:
            return []
        q = q / norm

        # Batch cosine similarity (vectors are pre-normalized)
        sims = self._matrix @ q  # (N,)
        k = min(top_k, len(self._keys))
        top_indices = np.argpartition(sims, -k)[-k:]
        top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]

        return [(self._keys[i], float(sims[i])) for i in top_indices]

    def count(self) -> int:
        return len(self._keys)

    def clear(self) -> None:
        self._keys.clear()
        self._matrix = None
        self._key_to_idx.clear()

    def bulk_add(self, items: list[tuple[str, list[float]]]) -> None:
        """Optimized batch add — builds matrix in one shot."""
        if not items:
            return
        for key, vec in items:
            self.add(key, vec)


class UsearchIndex(VectorIndex):
    """
    HNSW approximate nearest neighbor using usearch.
    Scales to millions of vectors with sub-millisecond search.

    Requires: pip install usearch
    """

    def __init__(self, dim: int, metric: str = "cos"):
        try:
            from usearch.index import Index
        except ImportError:
            raise ImportError(
                "usearch is required for UsearchIndex. "
                "Install with: pip install usearch"
            )

        self._dim = dim
        self._index = Index(ndim=dim, metric=metric, dtype="f32")
        self._key_to_label: dict[str, int] = {}
        self._label_to_key: dict[int, str] = {}
        self._next_label = 0

    def _assign_label(self, key: str) -> int:
        if key in self._key_to_label:
            return self._key_to_label[key]
        label = self._next_label
        self._next_label += 1
        self._key_to_label[key] = label
        self._label_to_key[label] = key
        return label

    def add(self, key: str, vector: list[float]) -> None:
        label = self._assign_label(key)
        vec = np.array(vector, dtype=np.float32)
        self._index.add(label, vec)

    def remove(self, key: str) -> None:
        if key not in self._key_to_label:
            return
        label = self._key_to_label.pop(key)
        self._label_to_key.pop(label, None)
        try:
            self._index.remove(label)
        except Exception:
            pass  # usearch may not support remove in all versions

    def search(self, query: list[float], top_k: int = 20) -> list[tuple[str, float]]:
        if len(self._key_to_label) == 0:
            return []
        q = np.array(query, dtype=np.float32)
        k = min(top_k, len(self._key_to_label))
        results = self._index.search(q, k)

        pairs = []
        for label, distance in zip(results.keys, results.distances):
            label = int(label)
            key = self._label_to_key.get(label)
            if key is None:
                continue
            # usearch cosine distance = 1 - similarity
            similarity = 1.0 - float(distance)
            pairs.append((key, similarity))
        return pairs

    def count(self) -> int:
        return len(self._key_to_label)

    def clear(self) -> None:
        from usearch.index import Index
        self._index = Index(ndim=self._dim, metric="cos", dtype="f32")
        self._key_to_label.clear()
        self._label_to_key.clear()
        self._next_label = 0

    def bulk_add(self, items: list[tuple[str, list[float]]]) -> None:
        """Batch insert — usearch handles this efficiently."""
        if not items:
            return
        labels = []
        vectors = []
        for key, vec in items:
            label = self._assign_label(key)
            labels.append(label)
            vectors.append(vec)
        mat = np.array(vectors, dtype=np.float32)
        self._index.add(np.array(labels, dtype=np.int64), mat)


def create_vector_index(dim: int, backend: str = "auto") -> VectorIndex:
    """
    Factory: create best available vector index.

    backend:
      "auto"   — usearch if available, else numpy
      "numpy"  — force numpy brute-force
      "usearch" — force usearch HNSW (raises if not installed)
    """
    if backend == "numpy":
        logger.info("Using NumpyIndex (brute-force)")
        return NumpyIndex()

    if backend == "usearch":
        logger.info("Using UsearchIndex (HNSW)")
        return UsearchIndex(dim=dim)

    # auto: try usearch first
    try:
        from usearch.index import Index
        logger.info("Using UsearchIndex (HNSW) — usearch detected")
        return UsearchIndex(dim=dim)
    except ImportError:
        logger.info("Using NumpyIndex (brute-force) — install usearch for HNSW")
        return NumpyIndex()
