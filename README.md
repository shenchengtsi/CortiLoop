# CortiLoop

**Bioinspired Agent Memory Engine** — modeled after the full lifecycle of human brain memory.

[Chinese Version / 中文文档](README_zh.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-48%20passed-brightgreen.svg)]()
[![Benchmark](https://img.shields.io/badge/LongMemEval-92%25-blue.svg)]()

> A memory plugin for AI agents. Works with [nanobot](https://github.com/HKUDS/nanobot), [openclaw](https://github.com/openclaw/openclaw), and any MCP-compatible agent framework.

---

## Why CortiLoop?

Most agent memory systems are flat key-value stores or simple RAG. Real brains manage memories through **encoding, consolidation, retrieval, association, forgetting, and reconsolidation** — a full lifecycle that keeps knowledge accurate, relevant, and manageable over time.

CortiLoop implements this full lifecycle:

| Problem | How the brain solves it | How CortiLoop implements it |
|---------|------------------------|---------------------------|
| Noise flooding | Prefrontal attention gate | 5-dimension importance scoring — corrections & explicit marks get highest weight |
| Stale knowledge | Reconsolidation window | Conflict detection with supersede / merge / coexist resolution |
| Retrieval degradation | Forgetting curve | Ebbinghaus decay with differential rates per memory tier |
| Fragmented recall | Pattern completion (CA3) | 4-route multi-probe search + Reciprocal Rank Fusion |
| No association | Hebbian learning | Dynamic knowledge graph with spreading activation |
| Information overload | Sleep consolidation | Background worker for periodic deep consolidation + pruning |

## Architecture

```
Agent Input → [Attention Gate] → [Encoder] → [Hippocampal Store]
                                                    │
                                    ┌───────────────┤
                                    ↓               ↓
                            [Synaptic Consol.]  [Association Graph]
                            (units→observations) (Hebbian edges)
                                    │
                                    ↓ (periodic)
                            [Systems Consol.]
                            (mental models, procedural detection)
                                    │
            [Multi-Probe Recall] ←──┘
            (semantic+keyword+graph+temporal → RRF fusion)
                                    │
                            [Reconsolidation]     [Forgetting]
                            (conflict detection)  (decay+prune)
```

### 7 Bioinspired Layers

| Layer | Brain Analogy | What It Does |
|-------|--------------|--------------|
| **Attention Gate** | Prefrontal cortex + dopamine novelty signal | Scores importance; filters noise before encoding |
| **Encoder** | Hippocampal encoding + entity binding | Extracts structured facts, entities, embeddings |
| **Consolidation** | Sleep-driven hippocampus→neocortex transfer | Synaptic (immediate) + Systems (deep/periodic) |
| **Association** | Hebbian learning + spreading activation | Knowledge graph with co-occurrence/temporal/causal edges |
| **Retrieval** | CA3 pattern completion + multi-modal fusion | 4-route search + RRF + cross-encoder reranking |
| **Forgetting** | Ebbinghaus curve + microglia pruning | Strength decay, deduplication, capacity management |
| **Reconsolidation** | Memory destabilization + restabilization | Conflict detection, safe update, history preservation |

## Features

### Core (v0.1)
- 7-layer bioinspired memory lifecycle
- MCP server + nanobot plugin + openclaw skill
- SQLite zero-dependency storage
- Bilingual attention gate (English + Chinese)

### Scale (v0.2)
- Pluggable vector index (usearch HNSW / numpy fallback)
- Ollama local LLM support (fully offline)
- litellm universal adapter (100+ LLM providers)
- Cross-encoder reranking
- Background consolidation worker

### Production (v0.3)
- PostgreSQL + pgvector storage backend
- Multi-tenant authentication (API key → namespace isolation)
- LongMemEval benchmark harness (5 dimensions, 13 test cases)
- Web visualization panel (D3.js knowledge graph + dashboard)
- `BaseStore` abstraction for custom storage backends

### Agent-First (v0.4)
- **`MemoryLLM` Protocol** — Agent only provides `complete()` + `complete_json()`, nothing else
- **Separated `Embedder` / `Reranker` Protocols** — chat, embedding, reranking are independent concerns
- **Local sentence-transformers** — `BAAI/bge-m3` embedding + `BAAI/bge-reranker-v2-m3` cross-encoder, auto-downloads from HuggingFace, no API key needed
- **4-level auto-detection** — user-provided → LLM built-in → sentence-transformers → hash fallback
- **Environment variable config** — `CORTILOOP_EMBEDDING_MODEL` / `CORTILOOP_RERANKER_MODEL`
- **48 tests** passing, **92% LongMemEval** benchmark score

## Quick Start

```bash
pip install cortiloop

# Optional:
pip install cortiloop[local]       # sentence-transformers (recommended for quality)
pip install cortiloop[usearch]     # HNSW vector index
pip install cortiloop[postgres]    # PostgreSQL + pgvector
pip install cortiloop[all]         # Everything
```

### Use Your Agent's LLM (Recommended)

```python
from cortiloop import CortiLoop

# Your agent already has an LLM — just pass it in
loop = CortiLoop(llm=agent.llm)

await loop.retain("Alice is the PM of ProjectX, using React + TypeScript")
await loop.retain("ok")  # filtered out by attention gate

results = await loop.recall("What's Alice's project?")
for r in results:
    print(f"[{r['type']}] {r['content']} (score: {r['score']:.3f})")
```

Your LLM only needs **chat completion** — just 2 methods:

```python
from cortiloop import MemoryLLM

class MyAgentLLM:  # implements MemoryLLM protocol
    async def complete(self, system: str, user: str, response_format: str = "json") -> str: ...
    async def complete_json(self, system: str, user: str) -> dict: ...
```

**Embedding and reranking are handled automatically.** CortiLoop selects the best available backend:

| Priority | Embedding | Reranking | When |
|----------|-----------|-----------|------|
| 1 | User-provided `embedder=` | User-provided `reranker=` | Explicit override |
| 2 | LLM's built-in `embed()` | LLM's built-in `rerank()` | LLM supports it (e.g. LLMClient) |
| 3 | `BAAI/bge-m3` (local) | `BAAI/bge-reranker-v2-m3` (local) | `sentence-transformers` installed |
| 4 | Hash-based n-gram | Word-overlap scoring | Zero dependencies (fallback) |

Override with environment variables:

```bash
CORTILOOP_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5      # lighter English-only model
CORTILOOP_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2  # faster reranker
```

Or pass explicitly:

```python
loop = CortiLoop(llm=agent.llm, embedder=my_embedder, reranker=my_reranker)
```

### Standalone (with built-in LLM config)

```python
from cortiloop import CortiLoop, CortiLoopConfig

# No existing LLM? CortiLoop can create one from config
config = CortiLoopConfig(db_path="memory.db")
config.llm.provider = "openai"  # or "ollama", "anthropic", "litellm"
loop = CortiLoop(config=config)
```

### With Ollama (fully local, no API key)

```python
config = CortiLoopConfig(db_path="memory.db")
config.llm.provider = "ollama"
config.llm.model = "llama3.1"
loop = CortiLoop(config=config)
# Embedding handled by sentence-transformers or hash fallback — no config needed
```

### With PostgreSQL (production scale)

```bash
pip install cortiloop[postgres]
```

```python
config = CortiLoopConfig(
    db_path="postgresql://user:pass@localhost:5432/cortiloop",
    storage_backend="postgres",  # uses pgvector HNSW natively
)
loop = CortiLoop(config=config)
```

### MCP Server

```bash
export OPENAI_API_KEY=sk-...
cortiloop-mcp
```

### Visualization Panel

```bash
cortiloop-viz --db cortiloop.db --port 8765
# Open http://localhost:8765
```

Features: force-directed knowledge graph, statistics dashboard, memory timeline, decay curve charts.

### Benchmark

```bash
# Quick smoke test (13 hand-crafted cases, offline)
python -m benchmarks.longmemeval --provider local

# Official LongMemEval (500 questions from ICLR 2025 paper)
python -m benchmarks.download_longmemeval --variant s   # download dataset first
python -m benchmarks.longmemeval_official --variant s --provider openai

# Run specific question types
python -m benchmarks.longmemeval_official --variant s --types knowledge-update temporal-reasoning

# Run a subset for quick iteration
python -m benchmarks.longmemeval_official --variant s --max-items 20

# Save results as JSON
python -m benchmarks.longmemeval_official --variant s --output results.json
```

**Quick benchmark** (13 cases): Information Extraction, Temporal Reasoning, Knowledge Update, Associative Retrieval, Multi-Session Reasoning.

**Official LongMemEval** (500 questions, 3 variants):
| Variant | Sessions/Question | Tokens | Use Case |
|---------|-------------------|--------|----------|
| oracle | Answer-relevant only | ~small | Debugging |
| s | ~40 | ~115K | Recommended |
| m | ~500 | ~1.5M | Stress test |

6 question types: single-session-user, single-session-assistant, single-session-preference, temporal-reasoning, knowledge-update, multi-session.

## Integration

### nanobot

```json
{
  "mcp": {
    "servers": {
      "cortiloop": {
        "command": "python",
        "args": ["-m", "cortiloop.adapters.mcp_server"],
        "env": { "CORTILOOP_DB_PATH": "~/.nanobot/cortiloop.db" }
      }
    }
  }
}
```

### openclaw

```json
{
  "cortiloop": {
    "command": "python",
    "args": ["-m", "cortiloop.adapters.mcp_server"],
    "env": { "CORTILOOP_DB_PATH": "~/.openclaw/cortiloop.db" }
  }
}
```

### nanobot Direct Plugin (Python)

```python
from cortiloop.adapters.nanobot_plugin import NanobotMemoryPlugin

memory = NanobotMemoryPlugin({"db_path": "memory.db"})
await memory.on_user_message("I prefer TypeScript strict mode")
context = await memory.on_before_response("Write a React component")
# context contains relevant memories to inject into prompt
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `cortiloop_retain` | Store text into long-term memory with attention gating |
| `cortiloop_recall` | Multi-probe retrieval with RRF fusion |
| `cortiloop_reflect` | Deep consolidation cycle (procedural detection + decay + pruning) |
| `cortiloop_stats` | Memory system statistics |

## Configuration

See [config.example.yaml](config.example.yaml) for all options.

```yaml
storage_backend: "sqlite"       # "sqlite" | "postgres"
vector_backend: "auto"          # "auto" | "numpy" | "usearch"

attention_gate:
  threshold: 0.2
  weights:
    correction: 0.30            # strongest signal
    novelty: 0.25
    explicit_mark: 0.20

retrieval:
  rerank_enabled: false         # cross-encoder reranking
  rerank_top_k: 50

decay:
  episodic_rate: 0.1            # fast: conversation details
  semantic_rate: 0.03           # moderate: extracted knowledge
  procedural_rate: 0.005        # slow: learned habits

auth:
  enabled: false
  api_keys: {}                  # key → namespace mapping
```

Environment variables for embedding/reranking model selection:

```bash
CORTILOOP_EMBEDDING_MODEL=BAAI/bge-m3                    # default, multilingual
CORTILOOP_RERANKER_MODEL=BAAI/bge-reranker-v2-m3         # default, multilingual
```

## Design Principles

1. **Not everything is worth remembering** — attention gate filters noise
2. **Write fast, refine slow** — immediate encoding + async consolidation
3. **Accumulate, don't overwrite** — raw facts are immutable; observations evolve
4. **Use it or lose it** — retrieval strengthens; disuse decays
5. **Forgetting is a feature** — active pruning prevents retrieval degradation
6. **Partial cue, full recall** — multi-probe search maximizes recall
7. **Neurons that fire together wire together** — Hebbian graph strengthening
8. **Safe updates, never delete originals** — reconsolidation with full history
9. **Agent-first** — zero config when used as a plugin; Agent's LLM is the only requirement

## Project Structure

```
cortiloop/
├── encoding/          # Attention gate + encoder
├── consolidation/     # Synaptic (immediate) + Systems (deep)
├── retrieval/         # Multi-probe + RRF + reranking
├── association/       # Hebbian knowledge graph
├── forgetting/        # Ebbinghaus decay + pruner
├── reconsolidation/   # Conflict detection + safe update
├── storage/           # BaseStore ABC + SQLite + PostgreSQL
├── llm/
│   ├── protocol.py        # MemoryLLM / Embedder / Reranker protocols
│   ├── client.py          # Built-in LLM client (OpenAI/Anthropic/Ollama/litellm)
│   ├── local_client.py    # Offline rule-based client (for testing/benchmark)
│   ├── local_embedder.py  # sentence-transformers embedding + cross-encoder
│   └── builtin_embedder.py # Hash-based embedding fallback (zero deps)
├── workers/           # Background consolidation worker
├── adapters/          # MCP server + nanobot plugin + openclaw skill
├── viz/               # Web visualization panel
└── auth.py            # Multi-tenant authentication
benchmarks/
├── longmemeval.py             # Quick benchmark (5 dimensions, 13 cases)
├── longmemeval_official.py    # Official LongMemEval (500 questions, ICLR 2025)
├── download_longmemeval.py    # Dataset downloader (HuggingFace)
└── data/                      # Downloaded datasets (gitignored)
```

## Development

```bash
git clone https://github.com/shenchengtsi/CortiLoop.git
cd CortiLoop
pip install -e ".[dev]"
pytest  # 48 tests
python -m benchmarks.longmemeval --provider local  # 92% benchmark
```

## License

MIT
