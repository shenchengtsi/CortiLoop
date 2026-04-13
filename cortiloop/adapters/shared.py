"""Shared engine singleton for adapters (MCP, HTTP, nanobot hook)."""

from __future__ import annotations

import os

from cortiloop.config import CortiLoopConfig
from cortiloop.engine import CortiLoop

_engine: CortiLoop | None = None


def get_engine() -> CortiLoop:
    global _engine
    if _engine is not None:
        return _engine

    config = CortiLoopConfig()

    db_path = os.environ.get("CORTILOOP_DB_PATH", "~/.nanobot/cortiloop.db")
    config.db_path = os.path.expanduser(db_path)

    if os.environ.get("CORTILOOP_NAMESPACE"):
        config.namespace = os.environ["CORTILOOP_NAMESPACE"]

    # Attention gate threshold — lower = more permissive
    if os.environ.get("CORTILOOP_ATTENTION_THRESHOLD"):
        config.attention_gate.threshold = float(os.environ["CORTILOOP_ATTENTION_THRESHOLD"])

    # Embedding config — must be set before engine init (affects vector index dim)
    if os.environ.get("CORTILOOP_EMBEDDING_MODEL"):
        config.llm.embedding_model = os.environ["CORTILOOP_EMBEDDING_MODEL"]
    if os.environ.get("CORTILOOP_EMBEDDING_DIM"):
        config.llm.embedding_dim = int(os.environ["CORTILOOP_EMBEDDING_DIM"])

    provider = os.environ.get("CORTILOOP_LLM_PROVIDER", "local")

    # Build embedder/reranker from env (shared across all providers)
    embedder = None
    reranker = None
    embedding_model = os.environ.get("CORTILOOP_EMBEDDING_MODEL")
    if embedding_model:
        try:
            from cortiloop.llm.local_embedder import LocalEmbedder, LocalReranker
            embedder = LocalEmbedder(model_name=embedding_model)
            reranker = LocalReranker()
        except ImportError:
            pass

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
        _engine = CortiLoop(config, embedder=embedder, reranker=reranker)

    return _engine
