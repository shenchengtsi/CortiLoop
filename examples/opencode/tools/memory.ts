/**
 * CortiLoop Memory — OpenCode Custom Tools
 *
 * Provides 4 tools for bioinspired long-term memory management:
 *   memory_retain  — Store information with attention gating
 *   memory_recall  — Multi-probe retrieval (semantic + keyword + graph + temporal)
 *   memory_reflect — Trigger deep consolidation cycle
 *   memory_status  — Get memory system statistics
 *
 * Install:
 *   1. pip install cortiloop
 *   2. Copy this file to .opencode/tools/memory.ts in your project
 *   3. Copy ../package.json to .opencode/package.json
 *
 * Env vars (optional):
 *   CORTILOOP_DB_PATH     — SQLite path (default: ~/.config/opencode/cortiloop.db)
 *   CORTILOOP_LLM_PROVIDER — openai | anthropic | ollama | local (default: local)
 *   CORTILOOP_LLM_MODEL   — Model name (default: provider default)
 *   CORTILOOP_NAMESPACE   — Tenant namespace (default: opencode)
 */

import { tool } from "@opencode-ai/plugin"

// ── Helper: call CortiLoop Python bridge ──

async function bridge(
  command: string,
  args: Record<string, unknown> = {},
): Promise<string> {
  const payload = JSON.stringify(args)
  const env = {
    ...process.env,
    CORTILOOP_DB_PATH:
      process.env.CORTILOOP_DB_PATH || "~/.config/opencode/cortiloop.db",
    CORTILOOP_NAMESPACE: process.env.CORTILOOP_NAMESPACE || "opencode",
  }

  const proc = Bun.spawn(
    [
      "python",
      "-m",
      "cortiloop.adapters.opencode_bridge",
      command,
      payload,
    ],
    { stdout: "pipe", stderr: "pipe", env },
  )

  const [stdout, stderr] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
  ])
  await proc.exited

  if (proc.exitCode !== 0) {
    const errMsg = stderr.trim() || stdout.trim() || "Unknown error"
    throw new Error(`cortiloop bridge failed (exit ${proc.exitCode}): ${errMsg}`)
  }

  return stdout.trim()
}

// ── Tool: memory_retain ──

export const retain = tool({
  description:
    "Store information into CortiLoop bioinspired long-term memory. " +
    "Uses attention gating (novelty, correction, emotion, task relevance) to filter noise, " +
    "extracts structured facts, builds a Hebbian association graph, " +
    "and triggers synaptic consolidation. " +
    "Use this when the user shares important facts, preferences, decisions, or corrections.",
  args: {
    text: tool.schema.string().describe("The text content to remember"),
    session_id: tool.schema
      .string()
      .optional()
      .describe("Session/conversation identifier"),
    task_context: tool.schema
      .string()
      .optional()
      .describe("Current task description for relevance scoring"),
  },
  async execute(args, ctx) {
    const result = await bridge("retain", {
      text: args.text,
      session_id: args.session_id || ctx.sessionID || "",
      task_context: args.task_context || "",
    })

    const parsed = JSON.parse(result)
    if (parsed.skipped) {
      return `Memory filtered by attention gate (importance: ${parsed.importance?.toFixed(2)}). Content was not novel or relevant enough to store.`
    }
    return `Stored ${parsed.stored} fact(s) (importance: ${parsed.importance?.toFixed(2)}). Entities: ${(parsed.entities || []).join(", ") || "none"}`
  },
})

// ── Tool: memory_recall ──

export const recall = tool({
  description:
    "Retrieve relevant memories from CortiLoop using multi-probe search: " +
    "semantic similarity, keyword matching, knowledge graph traversal, " +
    "and temporal filtering, fused via Reciprocal Rank Fusion. " +
    "Use this to recall past conversations, user preferences, project context, " +
    "or any previously stored information.",
  args: {
    query: tool.schema.string().describe("The query to search memories for"),
    top_k: tool.schema
      .number()
      .optional()
      .describe("Maximum number of results (default: 5)"),
  },
  async execute(args, ctx) {
    const result = await bridge("recall", {
      query: args.query,
      top_k: args.top_k || 5,
    })

    const parsed = JSON.parse(result)
    if (!parsed.results || parsed.results.length === 0) {
      return "No relevant memories found."
    }

    const lines = parsed.results.map(
      (m: { type: string; content: string; score: number; entities?: string[] }, i: number) => {
        const icon =
          m.type === "observation" ? "[insight]" :
          m.type === "procedural" ? "[skill]" :
          "[fact]"
        const entities = m.entities?.length ? ` (${m.entities.join(", ")})` : ""
        return `${i + 1}. ${icon} ${m.content}${entities} — relevance: ${m.score.toFixed(2)}`
      },
    )

    return `Found ${parsed.count} memories:\n${lines.join("\n")}`
  },
})

// ── Tool: memory_reflect ──

export const reflect = tool({
  description:
    "Trigger CortiLoop deep consolidation cycle (like brain sleep). " +
    "Runs systems consolidation (detects procedural patterns, generates mental models), " +
    "Ebbinghaus decay sweep (weakens unused memories), " +
    "and pruning (deduplication + capacity management). " +
    "Call this periodically or during idle time.",
  args: {},
  async execute(_args, _ctx) {
    const result = await bridge("reflect")
    const parsed = JSON.parse(result)

    const parts: string[] = ["Deep consolidation completed:"]
    for (const [key, value] of Object.entries(parsed)) {
      parts.push(`  ${key}: ${value}`)
    }
    return parts.join("\n")
  },
})

// ── Tool: memory_status ──

export const status = tool({
  description: "Get CortiLoop memory system statistics: unit count, observation count, namespace, worker status.",
  args: {},
  async execute(_args, _ctx) {
    const result = await bridge("stats")
    const parsed = JSON.parse(result)

    return [
      "CortiLoop Memory Status:",
      `  Memory units: ${parsed.memory_units}`,
      `  Observations: ${parsed.observations}`,
      `  Namespace: ${parsed.namespace}`,
      `  Worker: ${parsed.worker_running ? "running" : "stopped"}`,
    ].join("\n")
  },
})
