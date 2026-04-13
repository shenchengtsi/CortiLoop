/**
 * CortiLoop Auto-Memory Plugin for OpenCode
 *
 * Automatic memory management via event hooks:
 *   - Auto-retain: saves user messages and assistant responses after each turn
 *   - Auto-recall: injects relevant memories into context before compaction
 *   - Auto-reflect: triggers deep consolidation when session goes idle
 *
 * Install:
 *   1. pip install cortiloop
 *   2. Copy to .opencode/plugins/cortiloop-memory.ts
 *   3. Copy ../package.json to .opencode/package.json (if not already there)
 *
 * This plugin works independently of the custom tools in tools/memory.ts.
 * You can use both together: the plugin handles automatic background memory,
 * while the tools let the LLM explicitly retain/recall when needed.
 */

import type { Plugin } from "@opencode-ai/plugin"

// ── Config ──

const CORTILOOP_DB =
  process.env.CORTILOOP_DB_PATH || "~/.opencode/cortiloop.db"
const CORTILOOP_NAMESPACE = process.env.CORTILOOP_NAMESPACE || "opencode"
const AUTO_RETAIN_MIN_LENGTH = 10 // skip trivial messages
const AUTO_RECALL_TOP_K = 5
const REFLECT_DEBOUNCE_MS = 5 * 60 * 1000 // 5 minutes after idle

// ── Bridge helper ──

async function bridge(
  command: string,
  args: Record<string, unknown> = {},
): Promise<Record<string, unknown>> {
  const payload = JSON.stringify(args)
  const env = {
    ...process.env,
    CORTILOOP_DB_PATH: CORTILOOP_DB,
    CORTILOOP_NAMESPACE: CORTILOOP_NAMESPACE,
  }

  const proc = Bun.spawn(
    ["python", "-m", "cortiloop.adapters.opencode_bridge", command, payload],
    { stdout: "pipe", stderr: "pipe", env },
  )

  const stdout = await new Response(proc.stdout).text()
  await proc.exited

  if (proc.exitCode !== 0) return { error: "bridge failed" }

  try {
    return JSON.parse(stdout.trim())
  } catch {
    return { error: "invalid JSON from bridge" }
  }
}

// ── Track state ──

let reflectTimer: ReturnType<typeof setTimeout> | null = null
const retainedMessages = new Set<string>() // deduplicate by messageID

function scheduleReflect() {
  if (reflectTimer) clearTimeout(reflectTimer)
  reflectTimer = setTimeout(async () => {
    try {
      await bridge("reflect")
      console.log("[cortiloop] auto-reflect completed")
    } catch (e) {
      console.error("[cortiloop] auto-reflect failed:", e)
    }
  }, REFLECT_DEBOUNCE_MS)
}

// ── Plugin export ──

export const CortiLoopAutoMemory: Plugin = async (ctx) => {
  console.log(
    `[cortiloop] auto-memory plugin loaded (db: ${CORTILOOP_DB}, ns: ${CORTILOOP_NAMESPACE})`,
  )

  return {
    // ── Auto-retain: capture conversation after message updates ──
    "message.part.updated": async (input, _output) => {
      try {
        const part = (input as Record<string, unknown>).part as
          | Record<string, unknown>
          | undefined
        const messageId = (input as Record<string, unknown>).messageID as
          | string
          | undefined
        const sessionId = (input as Record<string, unknown>).sessionID as
          | string
          | undefined

        if (!part || !messageId) return
        // Deduplicate — message.part.updated fires multiple times per part
        const dedupeKey = `${messageId}:${(part as Record<string, unknown>).index || 0}`
        if (retainedMessages.has(dedupeKey)) return

        // Extract text content
        const type = part.type as string | undefined
        if (type !== "text") return

        const text = (part.content as string) || ""
        if (text.length < AUTO_RETAIN_MIN_LENGTH) return

        const role = (input as Record<string, unknown>).role as
          | string
          | undefined
        const sourceType =
          role === "user" ? "user_said" : "llm_inferred"

        retainedMessages.add(dedupeKey)
        // Retain in background — don't block the message flow
        bridge("retain", {
          text:
            sourceType === "user_said"
              ? text
              : `[Assistant]: ${text.slice(0, 500)}`,
          session_id: sessionId || "",
          task_context: "",
        }).catch((e) => console.error("[cortiloop] auto-retain failed:", e))

        // Reset reflect timer on activity
        scheduleReflect()
      } catch (e) {
        // Never crash the host — swallow all errors
        console.error("[cortiloop] message hook error:", e)
      }
    },

    // ── Auto-recall: inject memories during context compaction ──
    "experimental.session.compacting": async (_input, output) => {
      try {
        // Find the most recent user message to use as recall query
        const context = (output as Record<string, unknown[]>).context
        if (!context || !Array.isArray(context)) return

        // Use the last context entry as a proxy for what the user is working on
        const lastEntry = context[context.length - 1] as string | undefined
        const query =
          lastEntry?.slice(0, 200) || "recent conversation context"

        const result = await bridge("recall", {
          query,
          top_k: AUTO_RECALL_TOP_K,
        })

        const memories = (result.results as Array<Record<string, unknown>>) || []
        if (memories.length === 0) return

        const lines = memories.map((m) => {
          const icon =
            m.type === "observation"
              ? "[insight]"
              : m.type === "procedural"
                ? "[skill]"
                : "[fact]"
          return `  ${icon} ${m.content}`
        })

        context.push(
          [
            "## Long-term Memory (CortiLoop)",
            "The following are relevant memories from past sessions:",
            ...lines,
          ].join("\n"),
        )

        console.log(
          `[cortiloop] injected ${memories.length} memories into compacted context`,
        )
      } catch (e) {
        console.error("[cortiloop] compaction hook error:", e)
      }
    },

    // ── Auto-reflect on session idle ──
    "session.idle": async (_input, _output) => {
      scheduleReflect()
    },

    // ── Log session creation ──
    "session.created": async (input, _output) => {
      const sessionId = (input as Record<string, unknown>).sessionID as
        | string
        | undefined
      console.log(`[cortiloop] new session: ${sessionId || "unknown"}`)
      retainedMessages.clear()
    },
  }
}
