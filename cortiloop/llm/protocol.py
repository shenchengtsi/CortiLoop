"""
MemoryLLM Protocol — the interface CortiLoop needs from any LLM.

Designed for zero-friction integration with Agent frameworks:
- Agent only needs to provide chat completion (complete/complete_json)
- Embedding and reranking are optional — CortiLoop has built-in fallbacks

Usage:
    loop = CortiLoop(llm=agent.llm)   # agent.llm just needs complete()
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class MemoryLLM(Protocol):
    """
    Minimal LLM interface — only chat completion is required.

    CortiLoop uses built-in hash-based embedding when embed() is not provided.
    Agent frameworks typically only need to implement complete/complete_json.
    """

    async def complete(self, system: str, user: str, response_format: str = "json") -> str:
        """Chat completion. Returns raw text."""
        ...

    async def complete_json(self, system: str, user: str) -> dict:
        """Chat completion, parsed as JSON dict."""
        ...


@runtime_checkable
class Embedder(Protocol):
    """Optional embedding interface. If not provided, CortiLoop uses built-in hash embeddings."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        ...

    async def embed_one(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        ...


@runtime_checkable
class Reranker(Protocol):
    """Optional reranking interface. If not provided, CortiLoop uses word-overlap scoring."""

    async def rerank(
        self, query: str, documents: list[str], top_k: int = 10
    ) -> list[tuple[int, float]]:
        """Rerank documents by relevance. Returns (index, score) pairs."""
        ...
