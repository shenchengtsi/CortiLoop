"""
LLM abstraction layer for CortiLoop.

Supports:
- openai   — OpenAI API (default)
- anthropic — Anthropic Claude API
- ollama   — Local models via Ollama (OpenAI-compatible endpoint)
- litellm  — Universal adapter (100+ models, optional dependency)
"""

from __future__ import annotations

import json
import os
from typing import Any

from cortiloop.config import LLMConfig


class LLMClient:
    """Unified LLM client for fact extraction, consolidation, and scoring."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._chat_client: Any = None
        self._embed_client: Any = None

    def _get_api_key(self) -> str:
        if self.config.api_key:
            return self.config.api_key
        if self.config.provider == "ollama":
            return "ollama"  # Ollama doesn't need a real key
        if self.config.provider == "litellm":
            # litellm reads keys from env vars automatically
            return "litellm"
        env_key = "ANTHROPIC_API_KEY" if self.config.provider == "anthropic" else "OPENAI_API_KEY"
        key = os.environ.get(env_key, "")
        if not key:
            raise ValueError(f"Set {env_key} or provide api_key in config")
        return key

    def _get_chat_client(self):
        if self._chat_client:
            return self._chat_client

        if self.config.provider == "anthropic":
            import anthropic
            self._chat_client = anthropic.Anthropic(api_key=self._get_api_key())
        elif self.config.provider == "ollama":
            import openai
            base_url = self.config.base_url or "http://localhost:11434/v1"
            self._chat_client = openai.OpenAI(
                api_key="ollama",
                base_url=base_url,
            )
        elif self.config.provider == "litellm":
            # litellm is used differently — no persistent client needed
            self._chat_client = "litellm"
        else:
            import openai
            kwargs: dict[str, Any] = {"api_key": self._get_api_key(), "timeout": 120.0}
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            self._chat_client = openai.OpenAI(**kwargs)
        return self._chat_client

    def _get_embed_client(self):
        if self._embed_client:
            return self._embed_client

        if self.config.provider == "ollama":
            import openai
            base_url = self.config.base_url or "http://localhost:11434/v1"
            self._embed_client = openai.OpenAI(
                api_key="ollama",
                base_url=base_url,
            )
        elif self.config.provider == "litellm":
            self._embed_client = "litellm"
        else:
            import openai
            kwargs: dict[str, Any] = {"api_key": self._get_api_key()}
            if self.config.base_url and self.config.provider == "openai":
                kwargs["base_url"] = self.config.base_url
            self._embed_client = openai.OpenAI(**kwargs)
        return self._embed_client

    async def complete(self, system: str, user: str, response_format: str = "json") -> str:
        """Send a chat completion request. Returns raw text."""
        client = self._get_chat_client()

        if self.config.provider == "anthropic":
            msg = client.messages.create(
                model=self.config.model,
                max_tokens=2048,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return msg.content[0].text

        elif self.config.provider == "litellm":
            import litellm
            kwargs: dict[str, Any] = {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.1,
            }
            if response_format == "json":
                kwargs["response_format"] = {"type": "json_object"}
            resp = litellm.completion(**kwargs)
            return resp.choices[0].message.content or ""

        else:
            # openai or ollama (both use OpenAI-compatible API)
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            # For JSON mode: append instruction to system prompt instead of
            # using response_format, which some providers don't support
            if response_format == "json":
                messages[0]["content"] += "\n\nYou MUST respond with valid JSON only. No markdown, no explanation, just JSON."

            kwargs: dict[str, Any] = {
                "model": self.config.model,
                "messages": messages,
                "temperature": 0.1,
            }
            # Only use response_format for actual OpenAI API (not compatible endpoints)
            is_real_openai = (
                self.config.provider == "openai"
                and (not self.config.base_url or "api.openai.com" in self.config.base_url)
            )
            if response_format == "json" and is_real_openai:
                kwargs["response_format"] = {"type": "json_object"}

            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""

    async def complete_json(self, system: str, user: str) -> dict:
        """Complete and parse as JSON."""
        text = await self.complete(system, user, response_format="json")
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(text)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        if not texts:
            return []

        if self.config.provider == "litellm":
            import litellm
            resp = litellm.embedding(
                model=self.config.embedding_model,
                input=texts,
            )
            return [item["embedding"] for item in resp.data]

        client = self._get_embed_client()
        resp = client.embeddings.create(
            model=self.config.embedding_model,
            input=texts,
        )
        return [item.embedding for item in resp.data]

    async def embed_one(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        results = await self.embed([text])
        return results[0] if results else []

    async def rerank(self, query: str, documents: list[str], top_k: int = 10) -> list[tuple[int, float]]:
        """
        Rerank documents by relevance to query.
        Returns list of (original_index, relevance_score) sorted by descending score.

        Uses LLM-based scoring when no dedicated reranker is available.
        """
        if not documents:
            return []

        # Try dedicated reranker APIs first
        if self.config.provider == "litellm":
            try:
                import litellm
                resp = litellm.rerank(
                    model=self.config.rerank_model or "rerank-english-v3.0",
                    query=query,
                    documents=documents,
                    top_n=top_k,
                )
                return [(r.index, r.relevance_score) for r in resp.results]
            except Exception:
                pass  # fall through to LLM-based reranking

        # LLM-based reranking: score each document
        return await self._llm_rerank(query, documents, top_k)

    async def _llm_rerank(
        self, query: str, documents: list[str], top_k: int
    ) -> list[tuple[int, float]]:
        """Rerank using the chat model to score relevance 0-10."""
        # Batch documents into a single LLM call for efficiency
        doc_list = "\n".join(
            f"[{i}] {doc[:200]}" for i, doc in enumerate(documents)
        )
        system = (
            "You are a relevance scorer. Given a query and numbered documents, "
            "rate each document's relevance to the query from 0.0 to 1.0. "
            "Return JSON: {\"scores\": [{\"index\": 0, \"score\": 0.8}, ...]}"
        )
        user_msg = f"Query: {query}\n\nDocuments:\n{doc_list}"

        try:
            result = await self.complete_json(system, user_msg)
            scores = result.get("scores", [])
            pairs = [(s["index"], float(s["score"])) for s in scores]
            pairs.sort(key=lambda x: x[1], reverse=True)
            return pairs[:top_k]
        except Exception:
            # Fallback: return original order
            return [(i, 1.0 - i * 0.01) for i in range(min(top_k, len(documents)))]
