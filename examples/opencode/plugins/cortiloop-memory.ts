/**
 * CortiLoop Auto-Memory Plugin for OpenCode (V1 format)
 *
 * Automatic memory management via plugin hooks:
 *   - Auto-retain (chat.message): saves user message text on each chat turn
 *   - Auto-retain (event): saves assistant text parts via bus events
 *   - Auto-recall (experimental.session.compacting): injects memories during compaction
 */

import { appendFileSync } from "fs"

// ── Logging ──

const LOG_FILE = "/tmp/cortiloop-plugin.log"
function log(msg: string) {
  const ts = new Date().toISOString()
  const line = `${ts} ${msg}\n`
  try { appendFileSync(LOG_FILE, line) } catch {}
}

// ── Config ──

const CORTILOOP_DB =
  process.env.CORTILOOP_DB_PATH || "~/.config/opencode/cortiloop.db"
const CORTILOOP_NAMESPACE = process.env.CORTILOOP_NAMESPACE || "opencode"
const AUTO_RETAIN_MIN_LENGTH = 20
const AUTO_RECALL_TOP_K = 5
const REFLECT_DEBOUNCE_MS = 5 * 60 * 1000
const PYTHON_PATH = process.env.CORTILOOP_PYTHON || "python"

// ── Bridge queue (serialize to avoid concurrent model loading, non-blocking) ──

let bridgeQueue: Promise<any> = Promise.resolve()

function enqueueBridge(
  command: string,
  args: Record<string, unknown> = {},
): void {
  bridgeQueue = bridgeQueue
    .then(() => bridge(command, args))
    .catch((e) => log(`queued bridge error: ${e}`))
}

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
    CORTILOOP_LLM_PROVIDER: process.env.CORTILOOP_LLM_PROVIDER || "openai",
    CORTILOOP_LLM_MODEL: process.env.CORTILOOP_LLM_MODEL || "",
    CORTILOOP_API_KEY: process.env.CORTILOOP_API_KEY || "",
    CORTILOOP_BASE_URL: process.env.CORTILOOP_BASE_URL || "",
    CORTILOOP_EMBEDDING_MODEL: process.env.CORTILOOP_EMBEDDING_MODEL || "",
    CORTILOOP_EMBEDDING_DIM: process.env.CORTILOOP_EMBEDDING_DIM || "1024",
    CORTILOOP_ATTENTION_THRESHOLD: process.env.CORTILOOP_ATTENTION_THRESHOLD || "0.15",
    CORTILOOP_LLM_HEADERS: process.env.CORTILOOP_LLM_HEADERS || "",
  }

  log(`bridge: ${command} (${payload.slice(0, 100)})`)

  let proc
  try {
    proc = Bun.spawn(
      [PYTHON_PATH, "-m", "cortiloop.adapters.opencode_bridge", command, payload],
      { stdout: "pipe", stderr: "pipe", env },
    )
  } catch (e: any) {
    log(`bridge spawn failed: ${e?.message || e}`)
    return { error: `spawn failed: ${e?.message}` }
  }

  const stdout = await new Response(proc.stdout).text()
  const stderr = await new Response(proc.stderr).text()
  await proc.exited

  if (proc.exitCode !== 0) {
    log(`bridge exit=${proc.exitCode} stderr=${stderr.slice(0, 300)}`)
    return { error: "bridge failed", exitCode: proc.exitCode, stderr: stderr.slice(0, 300) }
  }

  try {
    const result = JSON.parse(stdout.trim())
    log(`bridge ${command} OK: ${JSON.stringify(result).slice(0, 200)}`)
    return result
  } catch {
    log(`bridge invalid JSON: stdout=${stdout.slice(0, 200)}`)
    return { error: "invalid JSON from bridge" }
  }
}

// ── Track state ──

let reflectTimer: ReturnType<typeof setTimeout> | null = null
const retainedParts = new Set<string>()

function scheduleReflect() {
  if (reflectTimer) clearTimeout(reflectTimer)
  reflectTimer = setTimeout(() => {
    enqueueBridge("reflect")
  }, REFLECT_DEBOUNCE_MS)
}

// ── V1 Plugin export ──

export default {
  id: "cortiloop-memory",
  server: async (_ctx: any) => {
    log(`plugin init (db: ${CORTILOOP_DB}, ns: ${CORTILOOP_NAMESPACE})`)

    return {
      // ── Auto-retain user messages via chat.message hook ──
      "chat.message": async (input: any, output: any) => {
        log(`>>> chat.message hook fired! sessionID=${input?.sessionID}`)
        try {
          const sessionId: string = input?.sessionID || ""
          const parts: any[] = output?.parts || []

          for (const part of parts) {
            if (part.type !== "text") continue

            const partId: string = part.id || ""
            if (partId && retainedParts.has(partId)) continue

            const text: string = part.text || ""
            if (text.length < AUTO_RETAIN_MIN_LENGTH) continue
            if (part.synthetic || part.ignored) continue

            if (partId) retainedParts.add(partId)

            log(`auto-retain user message (${text.length} chars, session: ${sessionId})`)

            // Fire-and-forget: enqueue retain, don't block message flow
            enqueueBridge("retain", {
              text: text.slice(0, 1000),
              session_id: sessionId,
              task_context: "user message",
            })
          }

          scheduleReflect()
        } catch (e) {
          log(`chat.message hook error: ${e}`)
        }
      },

      // ── Auto-retain assistant responses via generic event hook ──
      "event": async (input: any) => {
        try {
          const event = input?.event
          if (!event || event.type !== "message.part.updated") return

          const props = event.properties
          if (!props) return

          const part = props.part
          if (!part || part.type !== "text") return

          const partId: string = part.id || ""
          if (!partId || retainedParts.has(partId)) return

          const text: string = part.text || ""
          if (text.length < AUTO_RETAIN_MIN_LENGTH) return
          if (part.synthetic || part.ignored) return

          retainedParts.add(partId)

          const sessionId: string = props.sessionID || part.sessionID || ""

          log(`auto-retain assistant part (${text.length} chars, session: ${sessionId})`)

          // Fire-and-forget: enqueue retain, don't block event processing
          enqueueBridge("retain", {
            text: text.slice(0, 1000),
            session_id: sessionId,
            task_context: "assistant response",
          })

          scheduleReflect()
        } catch (e) {
          log(`event hook error: ${e}`)
        }
      },

      // ── Auto-recall during context compaction (must await — result needed) ──
      "experimental.session.compacting": async (input: any, output: any) => {
        log(`>>> compacting hook fired! sessionID=${input?.sessionID}`)
        try {
          const context: string[] = output?.context
          if (!context || !Array.isArray(context)) return

          const sessionId: string = input?.sessionID || ""
          const query =
            context.length > 0
              ? context[context.length - 1].slice(0, 200)
              : "recent conversation context"

          log(`auto-recall for compaction (session: ${sessionId})`)

          const result = await bridge("recall", {
            query,
            top_k: AUTO_RECALL_TOP_K,
          })

          const memories =
            (result.results as Array<Record<string, unknown>>) || []
          if (memories.length === 0) return

          const lines = memories.map((m) => `  - ${m.content}`)

          context.push(
            [
              "## Long-term Memory (CortiLoop)",
              "Relevant memories from past sessions:",
              ...lines,
            ].join("\n"),
          )

          log(`injected ${memories.length} memories into compacted context`)
        } catch (e) {
          log(`compaction hook error: ${e}`)
        }
      },
    }
  },
}
