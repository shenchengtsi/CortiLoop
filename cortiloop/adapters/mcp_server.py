"""
MCP Server — Universal adapter for any MCP-compatible agent framework.

Both nanobot and openclaw support MCP, so this is the primary integration point.
Exposes CortiLoop as an MCP tool server with 4 tools:
  - cortiloop_retain: Store a memory
  - cortiloop_recall: Retrieve memories
  - cortiloop_reflect: Trigger deep consolidation
  - cortiloop_stats: Get system stats

Run standalone:
  python -m cortiloop.adapters.mcp_server

Or configure in nanobot/openclaw MCP settings.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any

from cortiloop.config import CortiLoopConfig
from cortiloop.engine import CortiLoop

logger = logging.getLogger("cortiloop.mcp")

# MCP protocol constants
JSONRPC_VERSION = "2.0"
MCP_PROTOCOL_VERSION = "2024-11-05"

# Tool definitions
TOOLS = [
    {
        "name": "cortiloop_retain",
        "description": (
            "Store information into long-term memory. Uses bioinspired attention gating "
            "to filter noise, extracts structured facts, builds association graph, "
            "and triggers consolidation. Returns number of facts stored."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text content to remember",
                },
                "session_id": {
                    "type": "string",
                    "description": "Optional session/conversation ID for context tracking",
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
    },
    {
        "name": "cortiloop_recall",
        "description": (
            "Retrieve relevant memories using multi-probe search: semantic similarity, "
            "keyword matching, graph traversal (spreading activation), and temporal filtering. "
            "Results are fused via Reciprocal Rank Fusion."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to search memories for",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "cortiloop_reflect",
        "description": (
            "Trigger deep consolidation cycle (the 'sleep mode'). "
            "Detects procedural patterns, generates mental models, "
            "runs decay sweep, and prunes duplicates. "
            "Call periodically or during low-activity periods."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "cortiloop_stats",
        "description": "Get memory system statistics (unit count, observation count, etc.)",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]


class CortiLoopMCPServer:
    """MCP server that wraps the CortiLoop engine."""

    def __init__(self, config: CortiLoopConfig | None = None):
        self.engine = CortiLoop(config)

    async def handle_request(self, request: dict) -> dict:
        """Handle a single JSON-RPC request."""
        method = request.get("method", "")
        req_id = request.get("id")
        params = request.get("params", {})

        try:
            if method == "initialize":
                return self._success(req_id, {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": "cortiloop",
                        "version": "0.1.0",
                    },
                })

            elif method == "notifications/initialized":
                return {}  # no response needed for notifications

            elif method == "tools/list":
                return self._success(req_id, {"tools": TOOLS})

            elif method == "tools/call":
                tool_name = params.get("name", "")
                args = params.get("arguments", {})
                result = await self._call_tool(tool_name, args)
                return self._success(req_id, {
                    "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}],
                })

            else:
                return self._error(req_id, -32601, f"Method not found: {method}")

        except Exception as e:
            logger.exception("Error handling request")
            return self._error(req_id, -32603, str(e))

    async def _call_tool(self, name: str, args: dict) -> Any:
        if name == "cortiloop_retain":
            return await self.engine.retain(
                text=args["text"],
                session_id=args.get("session_id", ""),
                task_context=args.get("task_context", ""),
            )
        elif name == "cortiloop_recall":
            return await self.engine.recall(
                query=args["query"],
                top_k=args.get("top_k", 10),
            )
        elif name == "cortiloop_reflect":
            return await self.engine.reflect()
        elif name == "cortiloop_stats":
            return await self.engine.stats()
        else:
            raise ValueError(f"Unknown tool: {name}")

    @staticmethod
    def _success(req_id: Any, result: Any) -> dict:
        return {"jsonrpc": JSONRPC_VERSION, "id": req_id, "result": result}

    @staticmethod
    def _error(req_id: Any, code: int, message: str) -> dict:
        return {"jsonrpc": JSONRPC_VERSION, "id": req_id, "error": {"code": code, "message": message}}

    async def run_stdio(self):
        """Run as stdio MCP server (standard MCP transport)."""
        logger.info("CortiLoop MCP server starting on stdio")

        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)

        writer_transport, writer_protocol = await asyncio.get_event_loop().connect_write_pipe(
            asyncio.streams.FlowControlMixin, sys.stdout
        )
        writer = asyncio.StreamWriter(writer_transport, writer_protocol, reader, asyncio.get_event_loop())

        while True:
            try:
                line = await reader.readline()
                if not line:
                    break
                request = json.loads(line.decode().strip())
                response = await self.handle_request(request)
                if response:
                    out = json.dumps(response, ensure_ascii=False) + "\n"
                    writer.write(out.encode())
                    await writer.drain()
            except json.JSONDecodeError:
                continue
            except Exception as e:
                logger.exception("Stdio loop error")
                break

        self.engine.close()


def main():
    """Entry point for running the MCP server."""
    import os

    config = CortiLoopConfig()

    # Allow config via env vars
    if os.environ.get("CORTILOOP_DB_PATH"):
        config.db_path = os.environ["CORTILOOP_DB_PATH"]
    if os.environ.get("CORTILOOP_NAMESPACE"):
        config.namespace = os.environ["CORTILOOP_NAMESPACE"]
    if os.environ.get("CORTILOOP_LLM_PROVIDER"):
        config.llm.provider = os.environ["CORTILOOP_LLM_PROVIDER"]
    if os.environ.get("CORTILOOP_LLM_MODEL"):
        config.llm.model = os.environ["CORTILOOP_LLM_MODEL"]

    server = CortiLoopMCPServer(config)
    asyncio.run(server.run_stdio())


if __name__ == "__main__":
    main()
