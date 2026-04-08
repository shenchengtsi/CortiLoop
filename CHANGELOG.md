# Changelog

## v0.4.1 (2026-04-08)

### Bug Fixes

- **response_format compatibility**: Non-OpenAI providers (doubao, volcengine, etc.) no longer fail with `json_object is not supported`. JSON output is now enforced via system prompt instruction instead of `response_format` parameter. Only actual OpenAI API (`api.openai.com`) uses the native `response_format`.
- **Embedding fallback chain**: Added `_FallbackEmbedder` wrapper that automatically switches from LLM-based embedding to local/builtin fallback on first failure. Previously, if the LLM provider didn't support embedding (e.g. doubao coding endpoint), the entire retain pipeline would crash.
- **`_create_default_embedder` import check**: Now verifies `sentence-transformers` availability at creation time (via `import` check) rather than at runtime, ensuring the fallback chain correctly reaches `BuiltinEmbedder` when the package is not installed.
- **Local timestamps**: All `datetime.utcnow()` replaced with `datetime.now()` across models, storage, and all 7 bioinspired layers. Memory timestamps now reflect local system time.

### Improvements

- **Retain error handling**: Steps 1-3 (attention gate, encoding, storage) now wrapped in try/except. Attention gate failure falls back to default importance (0.5); encoding/storage failure returns graceful error instead of crashing.
- **Recall error handling**: Recall failure returns empty list `[]` instead of propagating exception, preventing agent flow disruption.
- **Viz drilldown**: Statistics panel cards are now clickable. Clicking a stat card opens a detail overlay showing the corresponding memory list with content, strength, access count, entities, and timestamps. New API endpoint: `/api/drilldown/{category}`.

### New

- `examples/nanobot_hook.py` — Reference implementation for automatic retain/recall via monkey-patching nanobot's `_process_message`. Not included in the package; intended as integration example.

## v0.4.0 (2026-04-08)

### Breaking Changes

- **Agent-First architecture**: `CortiLoop(config, llm=, embedder=, reranker=)` — constructor now accepts external LLM, embedder, and reranker instances.
- **Protocol-driven**: New `MemoryLLM`, `Embedder`, `Reranker` protocols replace hard dependency on `LLMClient`.
- All sub-modules now receive separated dependencies (encoder gets embedder, retriever gets embedder+reranker, etc.).

### New

- `cortiloop/llm/protocol.py` — Three runtime-checkable protocols
- `cortiloop/llm/local_embedder.py` — Local sentence-transformers (BAAI/bge-m3 + bge-reranker-v2-m3)
- `cortiloop/llm/builtin_embedder.py` — Zero-dependency hash n-gram embedding + word-overlap reranking
- 4-level auto-detection chain: user-provided > LLM built-in > sentence-transformers > hash fallback
- MCP server rewritten with standard `mcp.server.Server` SDK (replaces hand-rolled JSON-RPC)

## v0.3.0

- PostgreSQL + pgvector support
- Multi-tenant authentication
- LongMemEval 92% benchmark
- Web visualization panel

## v0.2.0

- usearch HNSW index
- Ollama support
- litellm adapter
- Background consolidation worker

## v0.1.0

- Initial release: 7-layer bioinspired architecture
- SQLite storage, MCP server, bilingual attention gate
