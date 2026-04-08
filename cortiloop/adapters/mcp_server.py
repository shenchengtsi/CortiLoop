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

from cortiloop.adapters.shared import get_engine as _get_engine

logger = logging.getLogger("cortiloop.mcp")


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
