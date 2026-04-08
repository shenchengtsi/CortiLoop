"""
MemoryLLM Protocol — the interface CortiLoop needs from any LLM.

Any object that implements these 5 methods can power CortiLoop's memory.
Agent frameworks just pass in their existing LLM client — no extra config needed.

Usage:
    loop = CortiLoop(llm=my_agent.llm)   # any object with these methods
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class MemoryLLM(Protocol):
    """
    Minimal LLM interface for CortiLoop.

    Any LLM client that implements these methods can be used.
    Agent frameworks can wrap their existing clients with a thin adapter.
    """

    async def complete(self, system: str, user: str, response_format: str = "json") -> str:
        """Chat completion. Returns raw text."""
        ...

    async def complete_json(self, system: str, user: str) -> dict:
        """Chat completion, parsed as JSON dict."""
        ...

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        ...

    async def embed_one(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        ...

    async def rerank(
        self, query: str, documents: list[str], top_k: int = 10
    ) -> list[tuple[int, float]]:
        """Rerank documents by relevance. Returns (index, score) pairs."""
        ...
