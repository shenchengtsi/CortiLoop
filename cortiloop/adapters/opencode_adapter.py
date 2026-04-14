"""
OpenCode Adapter — Setup utility for CortiLoop + OpenCode integration.

Generates custom tools, auto-memory plugin, and config for any OpenCode project.

Three integration modes:
  1. Custom Tools only  — LLM explicitly calls retain/recall/reflect/status
  2. Plugin only        — Auto-retain/recall/reflect via event hooks
  3. Full (recommended) — Both tools + plugin for manual + automatic memory

Usage:
  cortiloop-opencode                     # Interactive setup in current dir
  cortiloop-opencode --target ~/myproject # Setup in specific project
  cortiloop-opencode --mode full         # Non-interactive, full mode

Can also use MCP mode (no custom tools needed):
  cortiloop-opencode --mode mcp          # Just add MCP config to opencode.json
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

# ── File content templates ──

PACKAGE_JSON = {
    "name": "cortiloop-opencode-integration",
    "private": True,
    "dependencies": {
        "@opencode-ai/plugin": "latest",
    },
}


def _opencode_mcp_config(
    db_path: str = "~/.config/opencode/cortiloop.db",
    namespace: str = "opencode",
    provider: str = "local",
) -> dict[str, Any]:
    """Generate MCP server config for opencode.json."""
    return {
        "cortiloop": {
            "type": "local",
            "command": ["python", "-m", "cortiloop.adapters.mcp_server"],
            "environment": {
                "CORTILOOP_DB_PATH": db_path,
                "CORTILOOP_NAMESPACE": namespace,
                "CORTILOOP_LLM_PROVIDER": provider,
            },
            "enabled": True,
        }
    }


def _copy_example_file(src_name: str, dst_path: Path):
    """Copy an example file from the examples/opencode directory."""
    examples_dir = Path(__file__).parent.parent.parent / "examples" / "opencode"
    src = examples_dir / src_name
    if src.exists():
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_path)
        return True
    return False


def setup_tools(target: Path):
    """Install custom tools into target project."""
    tools_dir = target / ".opencode" / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)

    if _copy_example_file("tools/memory.ts", tools_dir / "memory.ts"):
        print(f"  [ok] {tools_dir / 'memory.ts'}")
    else:
        print("  [warn] Could not find example tools/memory.ts")


def setup_plugin(target: Path):
    """Install auto-memory plugin into target project."""
    plugins_dir = target / ".opencode" / "plugins"
    plugins_dir.mkdir(parents=True, exist_ok=True)

    if _copy_example_file("plugins/cortiloop-memory.ts", plugins_dir / "cortiloop-memory.ts"):
        print(f"  [ok] {plugins_dir / 'cortiloop-memory.ts'}")
    else:
        print("  [warn] Could not find example plugins/cortiloop-memory.ts")


def setup_package_json(target: Path):
    """Ensure .opencode/package.json has the plugin dependency."""
    pkg_path = target / ".opencode" / "package.json"
    pkg_path.parent.mkdir(parents=True, exist_ok=True)

    if pkg_path.exists():
        with open(pkg_path) as f:
            existing = json.load(f)
        deps = existing.setdefault("dependencies", {})
        if "@opencode-ai/plugin" not in deps:
            deps["@opencode-ai/plugin"] = "latest"
            with open(pkg_path, "w") as f:
                json.dump(existing, f, indent=2)
            print(f"  [ok] Updated {pkg_path} (added @opencode-ai/plugin)")
        else:
            print(f"  [ok] {pkg_path} (already has @opencode-ai/plugin)")
    else:
        with open(pkg_path, "w") as f:
            json.dump(PACKAGE_JSON, f, indent=2)
        print(f"  [ok] Created {pkg_path}")


def setup_opencode_json(target: Path, mode: str, provider: str = "local"):
    """Merge CortiLoop config into opencode.json."""
    config_path = target / "opencode.json"
    config: dict[str, Any] = {}

    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    # Add MCP config
    if mode == "mcp":
        mcp = config.setdefault("mcp", {})
        mcp.update(_opencode_mcp_config(provider=provider))
        print(f"  [ok] Added MCP server config to {config_path}")

    # Add permission for memory tools
    if mode in ("tools", "full"):
        perms = config.setdefault("permission", {})
        perms["memory_*"] = "allow"
        print(f"  [ok] Added memory_* permission to {config_path}")

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"  [ok] Wrote {config_path}")


def setup(target: Path, mode: str = "full", provider: str = "local"):
    """Run full setup for the given mode."""
    print(f"\n  CortiLoop + OpenCode Setup")
    print(f"  Target: {target}")
    print(f"  Mode:   {mode}")
    print(f"  LLM:    {provider}")
    print()

    if mode in ("tools", "full"):
        print("Installing custom tools...")
        setup_tools(target)
        setup_package_json(target)
        print()

    if mode in ("plugin", "full"):
        print("Installing auto-memory plugin...")
        setup_plugin(target)
        setup_package_json(target)
        print()

    if mode == "mcp":
        print("Configuring MCP server...")
        setup_opencode_json(target, mode, provider)
        print()
    elif mode in ("tools", "full"):
        setup_opencode_json(target, mode, provider)
        print()

    print("  Done! Start OpenCode in your project to use CortiLoop memory.")
    print()

    if mode in ("tools", "full"):
        print("  Available tools (LLM can call these):")
        print("    memory_retain  — store information")
        print("    memory_recall  — retrieve memories")
        print("    memory_reflect — consolidate knowledge")
        print("    memory_status  — check memory stats")
        print()

    if mode in ("plugin", "full"):
        print("  Auto-memory (background):")
        print("    - Auto-retain user + assistant messages")
        print("    - Auto-recall during context compaction")
        print("    - Auto-reflect when session goes idle")
        print()

    if mode == "mcp":
        print("  MCP tools (LLM can call these):")
        print("    cortiloop_retain  — store information")
        print("    cortiloop_recall  — retrieve memories")
        print("    cortiloop_reflect — consolidate knowledge")
        print("    cortiloop_stats   — check memory stats")
        print()


def print_setup_guide():
    """Print setup instructions for all modes."""
    print("""
+==============================================================+
|             CortiLoop + OpenCode Integration                  |
+==============================================================+

  Prerequisites:
    pip install cortiloop

  -- Mode 1: Custom Tools (recommended for explicit control) --

    cortiloop-opencode --mode tools

    Installs .opencode/tools/memory.ts with 4 tools:
      memory_retain, memory_recall, memory_reflect, memory_status

  -- Mode 2: Auto-Memory Plugin (recommended for hands-free) --

    cortiloop-opencode --mode plugin

    Installs .opencode/plugins/cortiloop-memory.ts that:
      - Auto-retains user + assistant messages
      - Auto-recalls during context compaction
      - Auto-reflects when session is idle

  -- Mode 3: Full = Tools + Plugin (recommended) --

    cortiloop-opencode --mode full

    Both explicit tools AND automatic background memory.

  -- Mode 4: MCP Only (minimal, no custom code) --

    cortiloop-opencode --mode mcp

    Adds CortiLoop as MCP server in opencode.json.
    Uses standard MCP protocol (same as nanobot/openclaw).

  -- Env vars --

    CORTILOOP_DB_PATH       SQLite path (default: ~/.config/opencode/cortiloop.db)
    CORTILOOP_LLM_PROVIDER  openai | anthropic | ollama | local
    CORTILOOP_LLM_MODEL     Model name
    CORTILOOP_API_KEY        API key (for openai/anthropic)
    CORTILOOP_NAMESPACE     Tenant namespace (default: opencode)

+==============================================================+
""")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Set up CortiLoop memory for an OpenCode project",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=".",
        help="Target project directory (default: current dir)",
    )
    parser.add_argument(
        "--mode",
        choices=["tools", "plugin", "full", "mcp", "guide"],
        default="full",
        help="Integration mode (default: full)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=os.environ.get("CORTILOOP_LLM_PROVIDER", "local"),
        help="LLM provider (default: local)",
    )
    args = parser.parse_args()

    if args.mode == "guide":
        print_setup_guide()
        return

    target = Path(args.target).resolve()
    if not target.is_dir():
        print(f"Error: {target} is not a directory", file=sys.stderr)
        sys.exit(1)

    setup(target, args.mode, args.provider)


if __name__ == "__main__":
    main()
