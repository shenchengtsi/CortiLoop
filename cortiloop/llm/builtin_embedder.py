"""
Built-in hash-based embedder — zero dependencies, works offline.

Used automatically when the Agent's LLM doesn't provide embedding.
Produces deterministic, locality-sensitive vectors from character n-grams.
"""

from __future__ import annotations

import hashlib
import math
import re


class BuiltinEmbedder:
    """Hash-based embedder. Similar texts produce similar vectors."""

    def __init__(self, dim: int = 256):
        self._dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_embed(t) for t in texts]

    async def embed_one(self, text: str) -> list[float]:
        return self._hash_embed(text)

    def _hash_embed(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        text_lower = text.lower().strip()

        # Character n-grams (2,3,4) for language-agnostic similarity
        ngrams = []
        for n in (2, 3, 4):
            for i in range(len(text_lower) - n + 1):
                ngrams.append(text_lower[i:i + n])

        # Whole words
        words = re.findall(r'[a-zA-Z]+', text_lower)
        cjk = re.findall(r'[\u4e00-\u9fff]', text)
        ngrams.extend(words)
        ngrams.extend(cjk)

        for gram in ngrams:
            h = int(hashlib.md5(gram.encode()).hexdigest(), 16)
            idx = h % self._dim
            sign = 1.0 if (h // self._dim) % 2 == 0 else -1.0
            vec[idx] += sign

        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


class BuiltinReranker:
    """Word-overlap reranker. No external dependencies."""

    async def rerank(
        self, query: str, documents: list[str], top_k: int = 10
    ) -> list[tuple[int, float]]:
        q_words = set(re.findall(r'[a-zA-Z]+', query.lower()))
        q_words.update(re.findall(r'[\u4e00-\u9fff]', query))
        scored = []
        for i, doc in enumerate(documents):
            d_words = set(re.findall(r'[a-zA-Z]+', doc.lower()))
            d_words.update(re.findall(r'[\u4e00-\u9fff]', doc))
            overlap = len(q_words & d_words)
            score = overlap / max(len(q_words | d_words), 1)
            scored.append((i, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
