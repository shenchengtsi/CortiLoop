"""
Nanobot Adapter — Direct Python integration for nanobot.

nanobot is Python-based and supports:
1. MCP tools (use mcp_server.py for that)
2. Direct Python import with lifecycle hooks

This adapter provides nanobot-native integration:
- Hook into agent lifecycle (on_message, on_tool_call)
- Auto-retain user messages
- Auto-recall before agent response
- Periodic reflect via cron

Usage in nanobot config.json:
{
  "agents": {
    "defaults": {
      "memory": {
        "provider": "cortiloop",
        "config": {
          "db_path": "~/.nanobot/cortiloop.db",
          "namespace": "default"
        }
      }
    }
  }
}

Or use as MCP tool in nanobot's MCP config:
{
  "mcp": {
    "servers": {
      "cortiloop": {
        "command": "python",
        "args": ["-m", "cortiloop.adapters.mcp_server"]
      }
    }
  }
}
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from cortiloop.config import CortiLoopConfig
from cortiloop.engine import CortiLoop

logger = logging.getLogger("cortiloop.nanobot")


class NanobotMemoryPlugin:
    """
    Nanobot-native memory plugin.

    Integrates CortiLoop directly into nanobot's agent lifecycle.
    Can be used as a drop-in replacement for nanobot's built-in memory.
    """

    def __init__(self, config: dict | CortiLoopConfig | None = None):
        if isinstance(config, dict):
            self.config = CortiLoopConfig.from_dict(config)
        else:
            self.config = config or CortiLoopConfig()
        self.engine = CortiLoop(self.config)
        self._reflect_task: asyncio.Task | None = None

    # ── Lifecycle hooks ──

    async def on_user_message(
        self,
        message: str,
        session_id: str = "",
        task_context: str = "",
    ) -> dict[str, Any]:
        """
        Call this when a user message arrives.
        Automatically retains memorable content.
        """
        result = await self.engine.retain(
            text=message,
            session_id=session_id,
            task_context=task_context,
            source_type="user_said",
        )
        logger.debug("Retained %d facts from user message", result.get("stored", 0))
        return result

    async def on_before_response(
        self,
        user_message: str,
        top_k: int = 10,
    ) -> str:
        """
        Call before generating agent response.
        Returns relevant memory context to inject into the prompt.
        """
        memories = await self.engine.recall(user_message, top_k)
        if not memories:
            return ""

        lines = ["[Relevant memories from CortiLoop:]"]
        for m in memories:
            prefix = "💡" if m["type"] == "observation" else "📝" if m["type"] == "unit" else "⚙️"
            lines.append(f"  {prefix} {m['content']}")
        return "\n".join(lines)

    async def on_agent_response(
        self,
        response: str,
        session_id: str = "",
    ) -> dict[str, Any]:
        """
        Call after agent generates a response.
        Optionally retain LLM-generated insights.
        """
        return await self.engine.retain(
            text=response,
            session_id=session_id,
            source_type="llm_inferred",
        )

    # ── Periodic tasks ──

    def start_reflect_scheduler(self, interval_seconds: int = 3600):
        """Start periodic deep consolidation in the background."""
        async def _loop():
            while True:
                await asyncio.sleep(interval_seconds)
                try:
                    result = await self.engine.reflect()
                    logger.info("Periodic reflect completed: %s", result)
                except Exception as e:
                    logger.warning("Periodic reflect failed: %s", e)

        self._reflect_task = asyncio.create_task(_loop())

    def stop(self):
        """Stop the plugin and clean up."""
        if self._reflect_task:
            self._reflect_task.cancel()
        self.engine.close()

    # ── Direct API access ──

    async def retain(self, text: str, **kwargs) -> dict:
        return await self.engine.retain(text, **kwargs)

    async def recall(self, query: str, **kwargs) -> list[dict]:
        return await self.engine.recall(query, **kwargs)

    async def reflect(self) -> dict:
        return await self.engine.reflect()

    async def stats(self) -> dict:
        return await self.engine.stats()
