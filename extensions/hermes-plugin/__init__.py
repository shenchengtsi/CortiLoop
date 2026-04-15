"""CortiLoop memory plugin for Hermes Agent.

Auto-recall (pre_llm_call) and auto-retain (post_llm_call) via plugin hooks,
plus manual tools for the LLM to call on demand.

Install:
    cp -r extensions/hermes-plugin ~/.hermes/plugins/cortiloop

Configure env vars (in ~/.hermes/.env or shell):
    CORTILOOP_DB_PATH=~/.cortiloop/cortiloop.db
    CORTILOOP_NAMESPACE=default
    CORTILOOP_LLM_PROVIDER=openai
    CORTILOOP_LLM_MODEL=your-model
    CORTILOOP_API_KEY=your-key
    CORTILOOP_BASE_URL=https://api.example.com/v1
    CORTILOOP_EMBEDDING_MODEL=path-or-name
    CORTILOOP_EMBEDDING_DIM=1024
"""

from __future__ import annotations

import json
import logging
import threading

from . import client, schemas

logger = logging.getLogger("hermes.plugin.cortiloop")

# ── Auto hooks (code-level, mandatory) ──────────────────────────────


def _auto_recall(session_id: str, user_message: str, **kwargs):
    """pre_llm_call: auto-recall relevant memories and inject into context.

    Fires every turn, before the LLM sees the message.
    Returns {"context": ...} to inject recalled memories into user message.
    """
    if not user_message:
        return None

    memories = client.recall(query=user_message, top_k=5)
    if not memories:
        return None

    # Format recalled memories for injection
    lines = ["[CortiLoop — recalled from long-term memory:]"]
    for m in memories:
        if isinstance(m, dict):
            content = m.get("content", m.get("text", ""))
            mem_type = m.get("type", "")
            score = m.get("score", "")
            prefix = {"observation": "~", "unit": "-", "procedural": ">"}.get(mem_type, "-")
            line = f"  {prefix} {content}"
            if score and isinstance(score, float):
                line += f"  (relevance: {score:.2f})"
            lines.append(line)
        elif isinstance(m, str):
            lines.append(f"  - {m}")

    context = "\n".join(lines)
    logger.info("Auto-recall injected %d memories for session %s", len(memories), session_id)
    return {"context": context}


def _auto_retain(
    session_id: str,
    user_message: str,
    assistant_response: str,
    **kwargs,
):
    """post_llm_call: auto-retain the completed exchange into CortiLoop.

    Fires every turn, after the LLM produces a final response.
    Runs in a background thread to avoid blocking (retain calls LLM for
    attention gate scoring which can take 10-30s).
    """
    if not user_message or not assistant_response:
        return

    combined = f"User: {user_message}\n\nAssistant: {assistant_response}"
    # Truncate to avoid overwhelming the embedder
    if len(combined) > 4000:
        combined = combined[:4000] + "\n...[truncated]"

    def _bg():
        try:
            result = client.retain(
                text=combined,
                session_id=session_id,
                task_context="hermes auto-retain",
            )
            stored = result.get("stored", 0) if isinstance(result, dict) else 0
            logger.info("Auto-retain stored %d facts for session %s", stored, session_id)
        except Exception as e:
            logger.warning("Auto-retain background failed: %s", e)

    t = threading.Thread(target=_bg, daemon=True, name="cortiloop-retain")
    t.start()
    logger.info("Auto-retain dispatched to background thread for session %s", session_id)


# ── Manual tool handlers (LLM-initiated) ───────────────────────────


def _handle_recall(args: dict, **kwargs) -> str:
    query = args.get("query", "")
    if not query:
        return json.dumps({"error": "query is required"})
    top_k = args.get("top_k", 5)
    memories = client.recall(query=query, top_k=top_k)
    return json.dumps(memories, ensure_ascii=False, default=str)


def _handle_retain(args: dict, **kwargs) -> str:
    text = args.get("text", "")
    if not text:
        return json.dumps({"error": "text is required"})
    result = client.retain(
        text=text,
        task_context=args.get("task_context", ""),
    )
    return json.dumps(result, ensure_ascii=False, default=str)


def _handle_reflect(args: dict, **kwargs) -> str:
    result = client.reflect()
    return json.dumps(result, ensure_ascii=False, default=str)


def _handle_stats(args: dict, **kwargs) -> str:
    result = client.stats()
    return json.dumps(result, ensure_ascii=False, default=str)


# ── CLI command ─────────────────────────────────────────────────────


def _cli_handler(args):
    sub = getattr(args, "cortiloop_cmd", None)
    if sub == "status":
        ok = client.health()
        if ok:
            st = client.stats()
            print("CortiLoop: online")
            print(f"  db: {client.SHARED_DB_PATH}")
            print(f"  namespace: {client.SHARED_NAMESPACE}")
            for k, v in st.items():
                print(f"  {k}: {v}")
        else:
            print("CortiLoop: not initialized")
            print("  Check: pip install cortiloop, env vars")
    else:
        print("Usage: hermes cortiloop <status>")


def _setup_argparse(subparser):
    subs = subparser.add_subparsers(dest="cortiloop_cmd")
    subs.add_parser("status", help="Check CortiLoop server status and stats")
    subparser.set_defaults(func=_cli_handler)


# ── Registration ────────────────────────────────────────────────────


def register(ctx):
    """Wire hooks and tools into Hermes."""

    # Mandatory hooks — code-level, always fire
    ctx.register_hook("pre_llm_call", _auto_recall)
    ctx.register_hook("post_llm_call", _auto_retain)

    # Optional tools — LLM can also manually recall/retain/reflect/stats
    ctx.register_tool(
        name="cortiloop_recall",
        toolset="cortiloop",
        schema=schemas.CORTILOOP_RECALL,
        handler=_handle_recall,
    )
    ctx.register_tool(
        name="cortiloop_retain",
        toolset="cortiloop",
        schema=schemas.CORTILOOP_RETAIN,
        handler=_handle_retain,
    )
    ctx.register_tool(
        name="cortiloop_reflect",
        toolset="cortiloop",
        schema=schemas.CORTILOOP_REFLECT,
        handler=_handle_reflect,
    )
    ctx.register_tool(
        name="cortiloop_stats",
        toolset="cortiloop",
        schema=schemas.CORTILOOP_STATS,
        handler=_handle_stats,
    )

    # CLI: hermes cortiloop status
    ctx.register_cli_command(
        name="cortiloop",
        help="CortiLoop memory system",
        setup_fn=_setup_argparse,
        handler_fn=_cli_handler,
    )

    # Don't initialize engine here — lazy-init on first hook call
    logger.info("CortiLoop plugin registered (engine will init on first use)")
