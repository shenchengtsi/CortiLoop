# CortiLoop Plugin for Hermes Agent

Integrates CortiLoop bioinspired memory into [Hermes Agent](https://github.com/nous-research/hermes-agent) via the plugin system.

## What it does

- **Auto-recall** (`pre_llm_call` hook) — every turn, before the LLM sees the user message, relevant memories are recalled and injected into the context. Code-level, mandatory — the LLM cannot skip it.
- **Auto-retain** (`post_llm_call` hook) — every turn, after the LLM responds, the exchange is retained into CortiLoop in a background thread. Code-level, mandatory.
- **Manual tools** — the LLM can also call `cortiloop_recall`, `cortiloop_retain`, `cortiloop_reflect`, and `cortiloop_stats` on demand.
- **CLI** — `hermes cortiloop status` to check the memory system.

## Installation

```bash
cp -r extensions/hermes-plugin ~/.hermes/plugins/cortiloop
pip install cortiloop nest_asyncio sentence-transformers
```

## Configuration

Set environment variables in `~/.hermes/.env` or your shell:

```bash
# Required
CORTILOOP_DB_PATH=~/.cortiloop/cortiloop.db

# Optional — defaults shown
CORTILOOP_NAMESPACE=default
CORTILOOP_LLM_PROVIDER=local          # local, openai, anthropic
CORTILOOP_LLM_MODEL=                  # model name for attention gate scoring
CORTILOOP_API_KEY=                    # API key for non-local providers
CORTILOOP_BASE_URL=                   # custom endpoint URL
CORTILOOP_EMBEDDING_MODEL=            # path or HuggingFace model name
CORTILOOP_EMBEDDING_DIM=              # auto-detected if using local embedder
CORTILOOP_ATTENTION_THRESHOLD=0.3     # lower = more permissive retention
```

### Sharing memory with nanobot / opencode

To share the same memory database across all three agents, point them all to the same `CORTILOOP_DB_PATH` and `CORTILOOP_NAMESPACE`. Use the same `CORTILOOP_EMBEDDING_MODEL` and `CORTILOOP_EMBEDDING_DIM` for consistent vector search results.

## Architecture

```
User message arrives
    │
    ▼
┌─────────────────────────────────┐
│ pre_llm_call → _auto_recall()  │  Code-level, mandatory
│ Recall top-5 memories           │  Injected into user message
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ LLM processing + tool loop     │  Can also manually call
│                                 │  cortiloop_recall/retain/
│                                 │  reflect/stats tools
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ post_llm_call → _auto_retain() │  Code-level, mandatory
│ Retain exchange (background)    │  Fire-and-forget thread
└─────────────────────────────────┘
```

## Dependencies

- `cortiloop` — the memory engine
- `nest_asyncio` — bridges sync Hermes hooks with async CortiLoop engine
- `sentence-transformers` — local embedding (required if using `CORTILOOP_EMBEDDING_MODEL`)
