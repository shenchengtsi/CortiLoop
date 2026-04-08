"""
MCP Server — Standard MCP SDK integration for nanobot / openclaw / any MCP host.

Exposes CortiLoop as 4 MCP tools:
  - cortiloop_retain: Store a memory
  - cortiloop_recall: Retrieve memories
  - cortiloop_reflect: Trigger deep consolidation
  - cortiloop_stats: Get system stats

Uses LocalLLMClient by default (offline, no API key needed).
Embedding handled by sentence-transformers or hash fallback.

Run standalone:
  python -m cortiloop.adapters.mcp_server

Configure in nanobot:
  "mcpServers": {
    "cortiloop": {
      "command": "python",
      "args": ["-m", "cortiloop.adapters.mcp_server"],
      "env": { "CORTILOOP_DB_PATH": "~/.nanobot/cortiloop.db" }
    }
  }
"""

from __future__ import annotations

import asyncio
import json
import logging
import os

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from cortiloop.config import CortiLoopConfig
from cortiloop.engine import CortiLoop
from cortiloop.llm.local_client import LocalLLMClient

logger = logging.getLogger("cortiloop.mcp")

# Global engine instance (created on first use)
_engine: CortiLoop | None = None


def _get_engine() -> CortiLoop:
    global _engine
    if _engine is not None:
        return _engine

    config = CortiLoopConfig()

    # Config from environment variables
    db_path = os.environ.get("CORTILOOP_DB_PATH", "~/.nanobot/cortiloop.db")
    config.db_path = os.path.expanduser(db_path)

    if os.environ.get("CORTILOOP_NAMESPACE"):
        config.namespace = os.environ["CORTILOOP_NAMESPACE"]

    # LLM: use external provider if configured, otherwise offline local client
    provider = os.environ.get("CORTILOOP_LLM_PROVIDER", "local")

    if provider == "local":
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


# Create MCP server
app = Server("cortiloop")


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="cortiloop_retain",
            description=(
                "Store information into long-term memory. Uses bioinspired attention gating "
                "to filter noise, extracts structured facts, builds association graph, "
                "and triggers consolidation. Returns number of facts stored."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text content to remember",
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Optional session/conversation ID",
                        "default": "",
                    },
                    "task_context": {
                        "type": "string",
                        "description": "Optional current task description for relevance scoring",
                        "default": "",
                    },
                },
                "required": ["text"],
            },
        ),
        types.Tool(
            name="cortiloop_recall",
            description=(
                "Retrieve relevant memories using multi-probe search: semantic similarity, "
                "keyword matching, graph traversal, and temporal filtering. "
                "Results are fused via Reciprocal Rank Fusion."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to search memories for",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="cortiloop_reflect",
            description=(
                "Trigger deep consolidation cycle. Detects procedural patterns, "
                "generates mental models, runs decay sweep, and prunes duplicates."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        types.Tool(
            name="cortiloop_stats",
            description="Get memory system statistics (unit count, observation count, etc.)",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    engine = _get_engine()

    if name == "cortiloop_retain":
        result = await engine.retain(
            text=arguments["text"],
            session_id=arguments.get("session_id", ""),
            task_context=arguments.get("task_context", ""),
        )
    elif name == "cortiloop_recall":
        result = await engine.recall(
            query=arguments["query"],
            top_k=arguments.get("top_k", 10),
        )
    elif name == "cortiloop_reflect":
        result = await engine.reflect()
    elif name == "cortiloop_stats":
        result = await engine.stats()
    else:
        result = {"error": f"Unknown tool: {name}"}

    return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False, default=str))]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
