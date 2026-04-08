"""
CortiLoop nanobot hook — automatic retain/recall without LLM decision.

Monkey-patches nanobot's AgentLoop._process_message to:
1. Before LLM loop: recall relevant memories → inject into system context
2. After LLM loop: retain user message + response asynchronously

Usage (in sitecustomize.py or startup script):
    from cortiloop.adapters.nanobot_hook import install
    install()
"""

from __future__ import annotations

import asyncio
import logging
import os
from functools import wraps
from typing import Any

logger = logging.getLogger("cortiloop.nanobot_hook")

# ── Engine singleton ──

_engine = None
_engine_lock = asyncio.Lock()


async def _get_engine():
    global _engine
    if _engine is not None:
        return _engine

    async with _engine_lock:
        if _engine is not None:
            return _engine

        from cortiloop.config import CortiLoopConfig
        from cortiloop.engine import CortiLoop

        config = CortiLoopConfig()
        db_path = os.environ.get("CORTILOOP_DB_PATH", "~/.nanobot/cortiloop.db")
        config.db_path = os.path.expanduser(db_path)

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

        logger.info("CortiLoop engine initialized (db=%s, provider=%s)", config.db_path, provider)
        return _engine


# ── Recall: inject memories into context ──

async def _recall_for_context(user_message: str) -> str | None:
    """Recall relevant memories and format as context block."""
    try:
        engine = await _get_engine()
        memories = await engine.recall(query=user_message, top_k=5)
        if not memories:
            return None

        lines = []
        for m in memories:
            content = m.get("content", "")
            score = m.get("score", 0)
            if content:
                lines.append(f"- {content} (relevance: {score:.2f})")

        if not lines:
            return None

        return (
            "\n\n[Long-term Memory Context]\n"
            "The following are relevant memories from past conversations:\n"
            + "\n".join(lines)
            + "\n[End Memory Context]"
        )
    except Exception as e:
        logger.warning("CortiLoop recall failed: %s", e)
        return None


# ── Retain: store conversation asynchronously ──

async def _retain_conversation(user_message: str, assistant_response: str, session_key: str):
    """Retain the conversation turn in background."""
    try:
        engine = await _get_engine()

        # Retain user message
        if user_message and len(user_message.strip()) > 5:
            await engine.retain(
                text=user_message,
                session_id=session_key,
                source_type="user_said",
            )

        # Retain assistant response (as LLM-inferred facts)
        if assistant_response and len(assistant_response.strip()) > 20:
            # Only retain substantive responses, skip short acknowledgments
            await engine.retain(
                text=f"[Assistant response to '{user_message[:50]}...']: {assistant_response[:500]}",
                session_id=session_key,
                source_type="llm_inferred",
            )
    except Exception as e:
        logger.warning("CortiLoop retain failed: %s", e)


# ── Monkey-patch installer ──

_installed = False


def install():
    """Monkey-patch nanobot's AgentLoop._process_message."""
    global _installed
    if _installed:
        return
    _installed = True

    # Set env vars from nanobot config if available
    try:
        import json
        config_path = os.path.expanduser("~/.nanobot/config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            mcp_env = cfg.get("tools", {}).get("mcpServers", {}).get("cortiloop", {}).get("env", {})
            for k, v in mcp_env.items():
                os.environ.setdefault(k, v)
    except Exception:
        pass

    try:
        from nanobot.agent.loop import AgentLoop
    except ImportError:
        logger.warning("nanobot not found, skipping CortiLoop hook installation")
        return

    original_process = AgentLoop._process_message

    @wraps(original_process)
    async def patched_process_message(self, msg, **kwargs):
        # Skip system messages and slash commands
        if msg.channel == "system" or msg.content.strip().startswith("/"):
            return await original_process(self, msg, **kwargs)

        user_text = msg.content.strip()
        session_key = kwargs.get("session_key") or msg.session_key

        # Pre-LLM: recall memories and inject into context
        memory_context = await _recall_for_context(user_text)
        if memory_context:
            # Append memory context to user message
            original_content = msg.content
            msg.content = msg.content + memory_context
            logger.info("CortiLoop: injected %d chars of memory context", len(memory_context))

        # Run original processing
        result = await original_process(self, msg, **kwargs)

        # Restore original message content
        if memory_context:
            msg.content = original_content

        # Post-LLM: retain conversation in background
        response_text = result.content if result else ""
        if user_text:
            asyncio.create_task(
                _retain_conversation(user_text, response_text, session_key)
            )

        return result

    AgentLoop._process_message = patched_process_message
    logger.info("CortiLoop: nanobot hook installed (auto recall + retain)")
