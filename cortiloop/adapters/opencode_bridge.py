"""
OpenCode Bridge — one-shot CLI for CortiLoop operations.

Called by OpenCode custom tools and plugins via subprocess:
  python -m cortiloop.adapters.opencode_bridge retain '{"text":"hello"}'
  python -m cortiloop.adapters.opencode_bridge recall '{"query":"hello"}'
  python -m cortiloop.adapters.opencode_bridge reflect
  python -m cortiloop.adapters.opencode_bridge stats

Returns JSON to stdout. Errors go to stderr.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys

from cortiloop.adapters.shared import get_engine

logger = logging.getLogger("cortiloop.opencode_bridge")

# Suppress noisy logs to stderr so stdout stays clean JSON
logging.basicConfig(level=logging.WARNING, stream=sys.stderr)


async def _run(command: str, args: dict) -> dict:
    engine = get_engine()

    if command == "retain":
        text = args.get("text", "")
        if not text:
            return {"error": "text is required"}
        return await engine.retain(
            text=text,
            session_id=args.get("session_id", ""),
            task_context=args.get("task_context", ""),
        )

    elif command == "recall":
        query = args.get("query", "")
        if not query:
            return {"error": "query is required"}
        results = await engine.recall(
            query=query,
            top_k=args.get("top_k", 10),
        )
        return {"results": results, "count": len(results)}

    elif command == "reflect":
        return await engine.reflect()

    elif command == "stats":
        return await engine.stats()

    else:
        return {"error": f"Unknown command: {command}"}


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: opencode_bridge <retain|recall|reflect|stats> [args_json]"}))
        sys.exit(1)

    command = sys.argv[1]

    args: dict = {}
    if len(sys.argv) > 2:
        try:
            args = json.loads(sys.argv[2])
        except json.JSONDecodeError as e:
            print(json.dumps({"error": f"Invalid JSON: {e}"}))
            sys.exit(1)

    try:
        result = asyncio.run(_run(command, args))
    except Exception as e:
        logger.error("Bridge error: %s", e, exc_info=True)
        result = {"error": str(e)}

    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
