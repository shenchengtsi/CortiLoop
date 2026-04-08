"""
CortiLoop HTTP API — lightweight REST server for plugin integrations.

Exposes retain/recall/reflect/stats as JSON endpoints.
Designed for OpenClaw plugins, browser extensions, or any HTTP client.

Usage:
    python -m cortiloop.adapters.http_server
    # or with env vars:
    CORTILOOP_HTTP_PORT=8766 CORTILOOP_DB_PATH=~/.openclaw/cortiloop.db python -m cortiloop.adapters.http_server

Endpoints:
    POST /retain   {"text": "...", "session_id": "", "task_context": ""}
    POST /recall   {"query": "...", "top_k": 5}
    POST /reflect  {}
    GET  /stats
    GET  /health
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any

logger = logging.getLogger("cortiloop.http")

# Async event loop for running coroutines from sync handler
_loop: asyncio.AbstractEventLoop | None = None


def _run_async(coro) -> Any:
    """Run an async coroutine from the sync HTTP handler."""
    global _loop
    if _loop is None:
        _loop = asyncio.new_event_loop()
    return _loop.run_until_complete(coro)


class CortiLoopHTTPHandler(BaseHTTPRequestHandler):
    """HTTP request handler for CortiLoop API."""

    def do_GET(self):
        if self.path == "/health":
            self._json_response(200, {"status": "ok"})
        elif self.path == "/stats":
            from cortiloop.adapters.shared import get_engine
            engine = get_engine()
            result = _run_async(engine.stats())
            self._json_response(200, result)
        else:
            self._json_response(404, {"error": "not found"})

    def do_POST(self):
        body = self._read_body()
        if body is None:
            return

        from cortiloop.adapters.shared import get_engine
        engine = get_engine()

        if self.path == "/retain":
            text = body.get("text", "")
            if not text:
                self._json_response(400, {"error": "text is required"})
                return
            result = _run_async(engine.retain(
                text=text,
                session_id=body.get("session_id", ""),
                task_context=body.get("task_context", ""),
            ))
            self._json_response(200, result)

        elif self.path == "/recall":
            query = body.get("query", "")
            if not query:
                self._json_response(400, {"error": "query is required"})
                return
            result = _run_async(engine.recall(
                query=query,
                top_k=body.get("top_k", 5),
            ))
            self._json_response(200, result)

        elif self.path == "/reflect":
            result = _run_async(engine.reflect())
            self._json_response(200, result)

        else:
            self._json_response(404, {"error": "not found"})

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def _read_body(self) -> dict | None:
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError) as e:
            self._json_response(400, {"error": f"invalid JSON: {e}"})
            return None

    def _json_response(self, status: int, data: Any):
        body = json.dumps(data, ensure_ascii=False, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, format, *args):
        logger.debug(format, *args)


def run_http_server(host: str = "127.0.0.1", port: int = 8766):
    """Start the HTTP API server."""
    server = HTTPServer((host, port), CortiLoopHTTPHandler)
    print(f"CortiLoop HTTP API — http://{host}:{port}")
    print(f"  POST /retain  POST /recall  POST /reflect  GET /stats  GET /health")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="CortiLoop HTTP API Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("CORTILOOP_HTTP_PORT", "8766")))
    args = parser.parse_args()
    run_http_server(args.host, args.port)


if __name__ == "__main__":
    main()
