"""CortiLoop engine client — direct Python import for Hermes plugin.

Uses nest_asyncio to bridge sync Hermes hooks with async CortiLoop engine.

Required env vars:
    CORTILOOP_DB_PATH           SQLite database path (e.g. ~/.cortiloop/cortiloop.db)
    CORTILOOP_NAMESPACE         Namespace for memory isolation (default: default)

Optional env vars:
    CORTILOOP_LLM_PROVIDER      LLM provider: local, openai, anthropic (default: local)
    CORTILOOP_LLM_MODEL         Model name for attention gate scoring
    CORTILOOP_API_KEY            API key for the LLM provider
    CORTILOOP_BASE_URL           Custom base URL for the LLM provider
    CORTILOOP_LLM_HEADERS        JSON string of extra HTTP headers
    CORTILOOP_EMBEDDING_MODEL   Path or name of embedding model
    CORTILOOP_EMBEDDING_DIM     Embedding dimension (default: auto-detected)
    CORTILOOP_ATTENTION_THRESHOLD  Attention gate threshold (default: 0.3)
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from threading import Lock
from typing import Any

logger = logging.getLogger("hermes.plugin.cortiloop")

_engine = None
_lock = Lock()

# ── Config from env vars (no hardcoded defaults for secrets/paths) ──

SHARED_DB_PATH = os.environ.get("CORTILOOP_DB_PATH", "cortiloop.db")
SHARED_NAMESPACE = os.environ.get("CORTILOOP_NAMESPACE", "default")


def _run_async(coro) -> Any:
    """Run an async coroutine from sync context.

    Uses nest_asyncio to allow calling from within an already-running
    event loop (hermes hooks run inside an async context).
    """
    import nest_asyncio

    nest_asyncio.apply()
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def _get_engine():
    """Lazy-init shared CortiLoop engine singleton."""
    global _engine
    if _engine is not None:
        return _engine

    with _lock:
        if _engine is not None:
            return _engine

        from cortiloop.config import CortiLoopConfig
        from cortiloop.engine import CortiLoop

        config = CortiLoopConfig()
        config.db_path = os.path.expanduser(SHARED_DB_PATH)
        config.namespace = SHARED_NAMESPACE

        # Attention gate threshold
        if os.environ.get("CORTILOOP_ATTENTION_THRESHOLD"):
            config.attention_gate.threshold = float(
                os.environ["CORTILOOP_ATTENTION_THRESHOLD"]
            )

        # Embedding config
        embedding_model = os.environ.get("CORTILOOP_EMBEDDING_MODEL")
        if os.environ.get("CORTILOOP_EMBEDDING_DIM"):
            config.llm.embedding_dim = int(os.environ["CORTILOOP_EMBEDDING_DIM"])

        # Build embedder/reranker
        embedder = None
        reranker = None
        if embedding_model:
            try:
                from cortiloop.llm.local_embedder import LocalEmbedder, LocalReranker

                embedder = LocalEmbedder(model_name=embedding_model)
                reranker = LocalReranker()
                config.llm.embedding_dim = embedder.dim
            except ImportError:
                pass

        # LLM provider
        provider = os.environ.get("CORTILOOP_LLM_PROVIDER", "local")

        if provider == "local":
            from cortiloop.llm.local_client import LocalLLMClient

            llm = LocalLLMClient(embedding_dim=config.llm.embedding_dim)
            if embedder is None:
                try:
                    from cortiloop.llm.local_embedder import LocalEmbedder, LocalReranker

                    embedder = LocalEmbedder()
                    reranker = LocalReranker()
                except ImportError:
                    pass
            _engine = CortiLoop(config, llm=llm, embedder=embedder, reranker=reranker)
        else:
            config.llm.provider = provider
            if os.environ.get("CORTILOOP_LLM_MODEL"):
                config.llm.model = os.environ["CORTILOOP_LLM_MODEL"]
            if os.environ.get("CORTILOOP_API_KEY"):
                config.llm.api_key = os.environ["CORTILOOP_API_KEY"]
            if os.environ.get("CORTILOOP_BASE_URL"):
                config.llm.base_url = os.environ["CORTILOOP_BASE_URL"]
            if os.environ.get("CORTILOOP_LLM_HEADERS"):
                import json as _json

                config.llm.headers = _json.loads(os.environ["CORTILOOP_LLM_HEADERS"])
            _engine = CortiLoop(config, embedder=embedder, reranker=reranker)

        logger.info(
            "CortiLoop engine initialized (db=%s, ns=%s, provider=%s)",
            config.db_path,
            config.namespace,
            provider,
        )
        return _engine


def retain(text: str, session_id: str = "", task_context: str = "") -> dict[str, Any]:
    """Store a memory."""
    try:
        engine = _get_engine()
        return _run_async(
            engine.retain(text=text, session_id=session_id, task_context=task_context)
        )
    except Exception as e:
        logger.warning("cortiloop retain failed: %s", e)
        return {"error": str(e)}


def recall(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Retrieve relevant memories."""
    try:
        engine = _get_engine()
        return _run_async(engine.recall(query=query, top_k=top_k))
    except Exception as e:
        logger.warning("cortiloop recall failed: %s", e)
        return []


def reflect() -> dict[str, Any]:
    """Trigger deep consolidation cycle."""
    try:
        engine = _get_engine()
        return _run_async(engine.reflect())
    except Exception as e:
        logger.warning("cortiloop reflect failed: %s", e)
        return {"error": str(e)}


def stats() -> dict[str, Any]:
    """Get memory system statistics."""
    try:
        engine = _get_engine()
        return _run_async(engine.stats())
    except Exception as e:
        logger.warning("cortiloop stats failed: %s", e)
        return {"error": str(e)}


def health() -> bool:
    """Check if CortiLoop engine is initialized."""
    return _engine is not None
