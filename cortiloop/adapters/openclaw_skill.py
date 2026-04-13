"""
OpenClaw Adapter — Skill + MCP integration for openclaw.

openclaw is TypeScript-based and supports:
1. MCP Registry for external tools (primary integration)
2. Skills platform (bundled/managed/workspace skills)
3. Extensions (channel-level integrations)

Since openclaw is TypeScript and CortiLoop is Python, the primary integration
is via the MCP server. This module provides:
1. Skill manifest generation for openclaw's skill registry
2. A helper to set up the MCP connection config

Usage — add to openclaw MCP config (in settings or .env):
  MCP_CORTILOOP_COMMAND=python -m cortiloop.adapters.mcp_server

Or use the skill manifest for openclaw's skill system.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# OpenClaw skill manifest
SKILL_MANIFEST = {
    "name": "cortiloop",
    "displayName": "CortiLoop Memory",
    "description": (
        "Bioinspired agent memory engine modeled after human brain mechanisms. "
        "Provides long-term memory with attention gating, multi-layer consolidation, "
        "Hebbian association graphs, Ebbinghaus decay, and safe reconsolidation."
    ),
    "version": "0.1.0",
    "author": "CortiLoop",
    "license": "MIT",
    "type": "mcp",
    "mcp": {
        "command": "python",
        "args": ["-m", "cortiloop.adapters.mcp_server"],
        "env": {
            "CORTILOOP_DB_PATH": "~/.openclaw/cortiloop.db",
            "CORTILOOP_NAMESPACE": "openclaw",
        },
    },
    "tools": [
        {
            "name": "cortiloop_retain",
            "description": "Store information into bioinspired long-term memory",
            "when": "After each user message or when important information is shared",
        },
        {
            "name": "cortiloop_recall",
            "description": "Retrieve relevant memories using multi-probe search",
            "when": "Before generating a response, to enrich context with past knowledge",
        },
        {
            "name": "cortiloop_reflect",
            "description": "Trigger deep consolidation (procedural detection, mental models, cleanup)",
            "when": "Periodically or during idle time",
        },
        {
            "name": "cortiloop_stats",
            "description": "Get memory system statistics",
            "when": "When user asks about memory status",
        },
    ],
    "systemPromptAddition": (
        "You have access to CortiLoop, a bioinspired long-term memory system. "
        "Use cortiloop_retain to store important user information after each message. "
        "Use cortiloop_recall before responding to check for relevant past context. "
        "Use cortiloop_reflect periodically to consolidate knowledge."
    ),
}


def generate_skill_manifest(output_dir: str | Path = ".") -> Path:
    """Generate the openclaw skill manifest file."""
    output_path = Path(output_dir) / "cortiloop-skill.json"
    with open(output_path, "w") as f:
        json.dump(SKILL_MANIFEST, f, indent=2, ensure_ascii=False)
    return output_path


def generate_nanobot_mcp_config() -> dict[str, Any]:
    """Generate the MCP server config snippet for nanobot's config.json."""
    return {
        "cortiloop": {
            "command": "python",
            "args": ["-m", "cortiloop.adapters.mcp_server"],
            "env": {
                "CORTILOOP_DB_PATH": "~/.nanobot/cortiloop.db",
                "CORTILOOP_NAMESPACE": "nanobot",
            },
        }
    }


def print_setup_guide():
    """Print setup instructions for both platforms."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                 CortiLoop Setup Guide                       ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  ── nanobot integration ──                                   ║
║                                                              ║
║  Option A: MCP (recommended)                                 ║
║  Add to ~/.nanobot/config.json → "mcp.servers":              ║
║  {                                                           ║
║    "cortiloop": {                                            ║
║      "command": "python",                                    ║
║      "args": ["-m", "cortiloop.adapters.mcp_server"]         ║
║    }                                                         ║
║  }                                                           ║
║                                                              ║
║  Option B: Direct Python import                              ║
║  from cortiloop.adapters.nanobot_plugin import               ║
║      NanobotMemoryPlugin                                     ║
║  memory = NanobotMemoryPlugin()                              ║
║                                                              ║
║  ── opencode integration ──                                  ║
║                                                              ║
║  cortiloop-opencode --mode full                              ║
║  (or: cortiloop-opencode --mode mcp)                         ║
║                                                              ║
║  ── openclaw integration ──                                  ║
║                                                              ║
║  Option A: MCP Registry                                      ║
║  Add to openclaw MCP config:                                 ║
║  {                                                           ║
║    "cortiloop": {                                            ║
║      "command": "python",                                    ║
║      "args": ["-m", "cortiloop.adapters.mcp_server"]         ║
║    }                                                         ║
║  }                                                           ║
║                                                              ║
║  Option B: Skill manifest                                    ║
║  python -c "from cortiloop.adapters.openclaw_skill import    ║
║      generate_skill_manifest; generate_skill_manifest()"     ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    print_setup_guide()
