# CortiLoop OpenClaw Plugin

Bioinspired long-term memory for OpenClaw, powered by the CortiLoop engine.

## Architecture

```
OpenClaw (TypeScript)  ──HTTP──>  CortiLoop HTTP API (Python)
  before_agent_start   ──────>    POST /recall
  agent_end            ──────>    POST /retain
```

All heavy lifting (encoding, embedding, consolidation, graph traversal, decay) runs in the Python backend.
The plugin is a thin HTTP client that hooks into OpenClaw's lifecycle.

## Setup

### 1. Start the CortiLoop HTTP server

```bash
# Default: http://127.0.0.1:8766
python -m cortiloop.adapters.http_server

# Or with custom config
CORTILOOP_HTTP_PORT=8766 \
CORTILOOP_LLM_PROVIDER=openai \
CORTILOOP_LLM_MODEL=doubao-seed-2.0-pro \
CORTILOOP_API_KEY=your-key \
CORTILOOP_BASE_URL=https://ark.cn-beijing.volces.com/api/coding/v3 \
python -m cortiloop.adapters.http_server
```

### 2. Install the plugin in OpenClaw

Add to your OpenClaw config:

```json
{
  "plugins": {
    "slots": {
      "memory": "memory-cortiloop"
    },
    "entries": {
      "memory-cortiloop": {
        "enabled": true,
        "config": {
          "cortiloopUrl": "http://127.0.0.1:8766",
          "autoRecall": true,
          "autoCapture": true,
          "recallTopK": 5,
          "captureMaxChars": 2000
        }
      }
    }
  }
}
```

**Important**: Custom config must be nested under `"config"`, and `"slots.memory"` must point to `"memory-cortiloop"` to override the default memory plugin.

## Features

### Auto-Recall (before_agent_start)
Before each agent turn, the plugin calls `POST /recall` with the user's prompt and injects relevant memories as `<relevant-memories>` context.

### Auto-Capture (agent_end)
After each successful agent turn, user messages are sent to `POST /retain` for encoding and storage. CortiLoop handles dedup, importance scoring, and fact extraction.

### Manual Tools
- `cortiloop_recall` — Search memories with multi-probe retrieval
- `cortiloop_retain` — Store information with bioinspired attention gating
- `cortiloop_reflect` — Trigger deep consolidation cycle
- `cortiloop_stats` — Get memory system statistics

### CLI Commands
```bash
openclaw cortiloop search "query"
openclaw cortiloop stats
openclaw cortiloop reflect
openclaw cortiloop health
```

## Config Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `cortiloopUrl` | string | `http://127.0.0.1:8766` | CortiLoop HTTP API URL |
| `autoRecall` | boolean | `true` | Auto-inject memories before agent turns |
| `autoCapture` | boolean | `true` | Auto-store user messages after agent turns |
| `recallTopK` | number | `5` | Max memories to inject per recall |
| `captureMaxChars` | number | `2000` | Max message length for auto-capture |
