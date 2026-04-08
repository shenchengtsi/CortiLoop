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

    provider = os.environ.get("CORTILOOP_LLM_PROVIDER", "local")

    if provider == "local":
        from cortiloop.llm.local_client import LocalLLMClient
        llm = LocalLLMClient(embedding_dim=config.llm.embedding_dim)
        _engine = CortiLoop(config, llm=llm)
    else:
        config.llm.provider = provider
        if os.environ.get("CORTILOOP_LLM_MODEL"):
            config.llm.model = os.environ["CORTILOOP_LLM_MODEL"]
        if os.environ.get("CORTILOOP_API_KEY"):
            config.llm.api_key = os.environ["CORTILOOP_API_KEY"]
        if os.environ.get("CORTILOOP_BASE_URL"):
            config.llm.base_url = os.environ["CORTILOOP_BASE_URL"]
        _engine = CortiLoop(config)

    return _engine
