import type { Hooks, PluginInput } from "@opencode-ai/plugin"
import type { PromptFileAttachment, SessionMessageAssistant, V2Event } from "@opencode/client/promise"
import type { Plugin } from "@opencode/plugin"
import { loadConfig, parseModel, type FallbackModelObject } from "./config.ts"
import { createLogger } from "./log.ts"

// ── Types ────────────────────────────────────────────────────────────────────

interface SessionState {
  // Rate limit fallback state
  fallbackActive: boolean
  cooldownEndTime: number

  // Disconnect resume state
  resumeAttempts: number
  resumeActive: boolean
  originalModel: FallbackModelObject | null
  // Timestamp (Date.now()) of the last plugin-initiated abort.
  // Retry events whose `next` timestamp predates this are stale and ignored.
  pluginAbortedAt: number

  // Ralph loop state
  loopActive: boolean

  // V2 only: Date.now() of the last idle handling pass for this session, used
  // to collapse the transient session.status(idle) and durable session.idle
  // events that describe the same logical idle.
  lastIdleHandledAt: number

  // V2 only: the model reference the plugin switched the session to when the
  // rate-limit fallback fired, or null when no fallback switch is applied.
  appliedFallbackModel: FallbackModelObject | null

  // V2 only: the session's model as last observed, updated by our own
  // switchModel calls and by session.model.selected events; used to detect a
  // manual user switch between the fallback and the cooldown expiry.
  lastKnownModel: FallbackModelObject | null
}

interface MessageInfo {
  id: string
  role: "user" | "assistant"
  sessionID: string
  model?: {
    providerID: string
    modelID: string
  }
  agent?: string
}

interface MessagePart {
  id: string
  type: string
  text?: string
  mime?: string
  filename?: string
  url?: string
  name?: string
}

interface MessageWithParts {
  info: MessageInfo
  parts: MessagePart[]
}

// ── Helpers ──────────────────────────────────────────────────────────────────

const sessionStates = new Map<string, SessionState>()

function createDefaultSessionState(): SessionState {
  return {
    fallbackActive: false,
    cooldownEndTime: 0,
    resumeAttempts: 0,
    resumeActive: false,
    originalModel: null,
    pluginAbortedAt: 0,
    loopActive: false,
    lastIdleHandledAt: 0,
    appliedFallbackModel: null,
    lastKnownModel: null,
  }
}

function getSessionState(sessionID: string): SessionState {
  let state = sessionStates.get(sessionID)
  if (!state) {
    state = createDefaultSessionState()
    sessionStates.set(sessionID, state)
  }
  return state
}

/**
 * Creates a case-insensitive substring matcher (suitable for multi-word phrases
 * like "rate limit" that don't need word-boundary protection).
 */
export function createSubstringMatcher(patterns: string[]) {
  return (message: string): boolean => {
    const lower = message.toLowerCase()
    return patterns.some(pattern => lower.includes(pattern.toLowerCase()))
  }
}

/**
 * Creates a case-insensitive word-boundary matcher using regex \b.
 * This prevents false positives from patterns like "eof" matching "thereof"
 * or "502" matching inside longer numbers. Patterns that look like HTTP
 * status codes (pure digits) are matched with \b on both sides.
 */
export function createWordBoundaryMatcher(patterns: string[]) {
  const regexes = patterns.map(pattern => {
    // Escape regex special characters in the pattern
    const escaped = pattern.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
    return new RegExp(`\\b${escaped}\\b`, "i")
  })
  return (message: string): boolean => {
    return regexes.some(regex => regex.test(message))
  }
}

/**
 * Extract all text content from an assistant message's parts.
 * Returns the concatenated text, or null if no text was found.
 */
function extractPartialContent(message: MessageWithParts): string | null {
  const textParts = message.parts
    .filter(p => p.type === "text" && p.text)
    .map(p => p.text!)

  if (textParts.length === 0) return null

  const content = textParts.join("")
  // Only consider it partial content if there's something meaningful
  return content.trim().length > 0 ? content : null
}

/**
 * Build the continuation prompt by injecting partial content into the template.
 * If the partial content exceeds maxLength, only the tail is kept (preceded by
 * a "[...truncated]" marker) so the model can find its place without the
 * conversation history ballooning.
 */
export function buildContinuationPrompt(template: string, partialContent: string, maxLength: number): string {
  let content = partialContent
  if (content.length > maxLength) {
    content = "[...truncated]\n" + content.slice(-maxLength)
  }
  return template.replace("{{partial_content}}", content)
}

// ── Plugin ───────────────────────────────────────────────────────────────────

export async function createPlugin(context: PluginInput): Promise<Hooks> {
  const { config, createdDefaultConfig } = loadConfig()
  const logger = createLogger(config.logging)

  if (createdDefaultConfig) {
    await logger.info("Created default config file on first run", {
      path: createdDefaultConfig,
    })
  }

  const isRateLimitMessage = createSubstringMatcher(config.rateLimitPatterns)
  const isDisconnectMessage = createWordBoundaryMatcher(config.disconnectPatterns)
  const fallbackModel = parseModel(config.fallbackModel)

  await logger.info("Plugin initialized", {
    enabled: config.enabled,
    fallbackModel: config.fallbackModel,
    rateLimitPatterns: config.rateLimitPatterns,
    disconnectPatterns: config.disconnectPatterns,
    resumeWithSameModel: config.resumeWithSameModel,
    maxResumeAttempts: config.maxResumeAttempts,
    cooldownMs: config.cooldownMs,
    loopEnabled: config.loopEnabled,
    loopStartPhrase: config.loopStartPhrase,
    loopStopPhrase: config.loopStopPhrase,
  })

  if (!config.enabled) {
    await logger.info("Plugin disabled via config")
    return {}
  }

  // ── Rate Limit Recovery ──────────────────────────────────────────────────
  // Original behavior: abort, revert to last user message, re-send with fallback model.

  async function handleRateLimitRecovery(sessionID: string, retryMessage: string) {
    const state = getSessionState(sessionID)

    if (state.fallbackActive && Date.now() < state.cooldownEndTime) {
      await logger.info("Skipping rate-limit fallback, cooldown active", {
        sessionID,
        cooldownRemaining: state.cooldownEndTime - Date.now(),
      })
      return
    }

    await logger.info("Rate limit detected, switching to fallback model", {
      sessionID,
      message: retryMessage,
      fallbackModel: config.fallbackModel,
    })

    state.fallbackActive = true
    state.cooldownEndTime = Date.now() + config.cooldownMs
    state.pluginAbortedAt = Date.now()

    try {
      await logger.info("Aborting session", { sessionID })
      await context.client.session.abort({ path: { id: sessionID } })
      await new Promise(resolve => setTimeout(resolve, 200))

      await logger.info("Fetching messages", { sessionID })
      const messagesResponse = await context.client.session.messages({ path: { id: sessionID } })
      const messages = messagesResponse.data as MessageWithParts[] | undefined

      if (!messages || messages.length === 0) {
        await logger.error("No messages found in session", { sessionID })
        return
      }

      const lastUserMessage = [...messages].reverse().find(m => m.info.role === "user")
      if (!lastUserMessage) {
        await logger.error("No user message found in session", { sessionID })
        return
      }

      await logger.info("Reverting to last user message", {
        sessionID,
        messageId: lastUserMessage.info.id,
        totalMessages: messages.length,
      })
      await context.client.session.revert({
        path: { id: sessionID },
        body: { messageID: lastUserMessage.info.id },
      })
      await new Promise(resolve => setTimeout(resolve, 500))

      const originalParts = lastUserMessage.parts
        .filter(p => !isSyntheticPart(p))
        .map(p => convertToPromptPart(p))
        .filter((p): p is NonNullable<typeof p> => p !== null)

      if (originalParts.length === 0) {
        await logger.error("No valid parts found in user message", { sessionID })
        return
      }

      await logger.info("Sending prompt with fallback model", {
        sessionID,
        model: fallbackModel,
        partsCount: originalParts.length,
      })
      await context.client.session.prompt({
        path: { id: sessionID },
        body: {
          model: fallbackModel,
          agent: lastUserMessage.info.agent,
          parts: originalParts,
        },
      })

      await logger.info("Rate-limit fallback prompt sent successfully", { sessionID })
    } catch (err) {
      await logger.error("Failed to send rate-limit fallback prompt", {
        sessionID,
        error: err instanceof Error ? err.message : String(err),
      })
    }
  }

  // ── Disconnect Resume Recovery ───────────────────────────────────────────
  // New behavior: capture partial assistant response, then send a continuation
  // prompt that includes the partial content so the model picks up where it
  // left off.

  async function handleDisconnectRecovery(sessionID: string, retryMessage: string) {
    const state = getSessionState(sessionID)

    // Guard: max resume attempts
    if (state.resumeAttempts >= config.maxResumeAttempts) {
      await logger.warn("Max resume attempts reached, giving up", {
        sessionID,
        attempts: state.resumeAttempts,
        maxAttempts: config.maxResumeAttempts,
      })
      return
    }

    // Guard: if we're already actively resuming, don't stack
    if (state.resumeActive) {
      await logger.info("Resume already in progress, skipping", { sessionID })
      return
    }

    await logger.info("Disconnect detected, attempting to resume", {
      sessionID,
      message: retryMessage,
      attempt: state.resumeAttempts + 1,
      maxAttempts: config.maxResumeAttempts,
    })

    state.resumeActive = true
    state.resumeAttempts += 1
    state.pluginAbortedAt = Date.now()

    try {
      // Exponential backoff: 2s, 4s, 8s for attempts 1, 2, 3
      const backoffMs = Math.min(2000 * Math.pow(2, state.resumeAttempts - 1), 30000)
      if (state.resumeAttempts > 1) {
        await logger.info("Applying exponential backoff before retry", {
          sessionID,
          attempt: state.resumeAttempts,
          backoffMs,
        })
        await new Promise(resolve => setTimeout(resolve, backoffMs))
      }

      // 1. Abort the current retry loop
      await logger.info("Aborting session for disconnect recovery", { sessionID })
      await context.client.session.abort({ path: { id: sessionID } })
      await new Promise(resolve => setTimeout(resolve, 200))

      // 2. Fetch messages to find partial assistant response
      await logger.info("Fetching messages for partial content capture", { sessionID })
      const messagesResponse = await context.client.session.messages({ path: { id: sessionID } })
      const messages = messagesResponse.data as MessageWithParts[] | undefined

      if (!messages || messages.length === 0) {
        await logger.error("No messages found in session", { sessionID })
        return
      }

      // 3. Find the last assistant message (the interrupted one) and last user message
      const reversedMessages = [...messages].reverse()
      const lastAssistantMessage = reversedMessages.find(m => m.info.role === "assistant")
      const lastUserMessage = reversedMessages.find(m => m.info.role === "user")

      if (!lastUserMessage) {
        await logger.error("No user message found in session", { sessionID })
        return
      }

      // 4. Extract partial content from the interrupted assistant response
      const partialContent = lastAssistantMessage ? extractPartialContent(lastAssistantMessage) : null

      // 5. Capture the original model that was responding
      if (lastAssistantMessage?.info.model && !state.originalModel) {
        state.originalModel = {
          providerID: lastAssistantMessage.info.model.providerID,
          modelID: lastAssistantMessage.info.model.modelID,
        }
      }

      await logger.info("Partial content captured", {
        sessionID,
        hasPartialContent: !!partialContent,
        partialContentLength: partialContent?.length ?? 0,
        originalModel: state.originalModel,
      })

      // 6. Revert to the last user message (removes the incomplete assistant response)
      await logger.info("Reverting session for disconnect recovery", {
        sessionID,
        messageId: lastUserMessage.info.id,
      })
      await context.client.session.revert({
        path: { id: sessionID },
        body: { messageID: lastUserMessage.info.id },
      })
      await new Promise(resolve => setTimeout(resolve, 500))

      // 7. Determine which model to use
      // First attempt: try the same model (it was a disconnect, not a model issue)
      // Subsequent attempts: escalate to fallback model
      let modelToUse: FallbackModelObject
      if (config.resumeWithSameModel && state.originalModel && state.resumeAttempts <= 1) {
        modelToUse = state.originalModel
        await logger.info("Retrying with same model", { sessionID, model: modelToUse })
      } else {
        modelToUse = fallbackModel
        await logger.info("Escalating to fallback model", { sessionID, model: modelToUse })
      }

      // 8. Build the prompt parts
      let promptParts: Array<{ type: "text"; text: string } | { type: "file"; mime: string; filename?: string; url: string } | { type: "agent"; name: string }>

      if (partialContent) {
        // We have partial content: send a continuation prompt
        const continuationText = buildContinuationPrompt(config.continuationPromptTemplate, partialContent, config.maxPartialContentLength)
        promptParts = [{ type: "text", text: continuationText }]

        await logger.info("Sending continuation prompt with partial content", {
          sessionID,
          model: modelToUse,
          partialContentLength: partialContent.length,
        })
      } else {
        // No partial content captured: fall back to re-sending the original user message
        promptParts = lastUserMessage.parts
          .filter(p => !isSyntheticPart(p))
          .map(p => convertToPromptPart(p))
          .filter((p): p is NonNullable<typeof p> => p !== null)

        if (promptParts.length === 0) {
          await logger.error("No valid parts to re-send", { sessionID })
          return
        }

        await logger.info("No partial content, re-sending original user message", {
          sessionID,
          model: modelToUse,
          partsCount: promptParts.length,
        })
      }

      // 9. Send the prompt
      await context.client.session.prompt({
        path: { id: sessionID },
        body: {
          model: modelToUse,
          agent: lastUserMessage.info.agent,
          parts: promptParts,
        },
      })

      await logger.info("Disconnect recovery prompt sent successfully", {
        sessionID,
        attempt: state.resumeAttempts,
        usedContinuation: !!partialContent,
        model: modelToUse,
      })
    } catch (err) {
      await logger.error("Failed to send disconnect recovery prompt", {
        sessionID,
        error: err instanceof Error ? err.message : String(err),
        attempt: state.resumeAttempts,
      })
    } finally {
      state.resumeActive = false
    }
  }

  // ── Event Handler ────────────────────────────────────────────────────────

  // Loop phrase matchers (case-insensitive)
  const loopStartLower = config.loopStartPhrase.toLowerCase()
  const loopStopLower = config.loopStopPhrase.toLowerCase()

  /**
   * Check the last user message in a session for loop control phrases.
   * Returns "start" if the start phrase is found, "stop" if the stop phrase is
   * found, or null if neither is found. Stop takes priority over start.
   */
  async function checkLoopPhrase(sessionID: string): Promise<"start" | "stop" | null> {
    if (!config.loopEnabled) return null

    try {
      const messagesResponse = await context.client.session.messages({ path: { id: sessionID } })
      const messages = messagesResponse.data as MessageWithParts[] | undefined
      if (!messages || messages.length === 0) return null

      const lastUserMessage = [...messages].reverse().find(m => m.info.role === "user")
      if (!lastUserMessage) return null

      const text = lastUserMessage.parts
        .filter(p => p.type === "text" && p.text)
        .map(p => p.text!)
        .join("")
        .toLowerCase()

      // Stop takes priority: if user says both (unlikely), stop wins
      if (text.includes(loopStopLower)) return "stop"
      if (text.includes(loopStartLower)) return "start"
      return null
    } catch {
      return null
    }
  }

  return {
    event: async ({ event }) => {
      if (event.type === "session.status") {
        const props = event.properties as {
          sessionID: string
          status: {
            type: "idle" | "retry" | "busy"
            attempt?: number
            message?: string
            next?: number
          }
        }

        if (props.status.type === "retry" && props.status.message) {
          const sessionID = props.sessionID
          const state = getSessionState(sessionID)

          // If the plugin recently initiated an abort, ignore retry events
          // that were likely queued before our abort took effect.
          // We use a 5-second grace window: any retry event arriving within
          // 5 seconds of our abort is assumed to be stale.
          const ABORT_GRACE_MS = 5000
          if (state.pluginAbortedAt > 0 && Date.now() - state.pluginAbortedAt < ABORT_GRACE_MS) {
            await logger.info("Ignoring retry event within abort grace window", {
              sessionID,
              abortedAt: state.pluginAbortedAt,
              elapsed: Date.now() - state.pluginAbortedAt,
            })
            return
          }

          const message = props.status.message

          // Check disconnect patterns first (more specific recovery)
          if (isDisconnectMessage(message)) {
            await handleDisconnectRecovery(sessionID, message)
            return
          }

          // Then check rate limit patterns
          if (isRateLimitMessage(message)) {
            await handleRateLimitRecovery(sessionID, message)
            return
          }

          // Unknown retry reason -- log but don't act
          await logger.info("Retry detected but no matching pattern", {
            sessionID,
            message,
          })
        }

        if (props.status.type === "idle") {
          const sessionID = props.sessionID
          const state = sessionStates.get(sessionID) ?? getSessionState(sessionID)

          // Reset rate-limit cooldown if expired
          if (state.fallbackActive && Date.now() >= state.cooldownEndTime) {
            state.fallbackActive = false
            await logger.info("Rate-limit cooldown expired, fallback reset", { sessionID })
          }

          // Reset disconnect resume state on successful completion
          if (state.resumeAttempts > 0 && !state.resumeActive) {
            await logger.info("Session idle after disconnect recovery, resetting resume state", {
              sessionID,
              totalAttempts: state.resumeAttempts,
            })
            state.resumeAttempts = 0
            state.originalModel = null
            state.pluginAbortedAt = 0
          }

          // Ralph loop: check for control phrases and auto-continue
          if (config.loopEnabled) {
            const loopAction = await checkLoopPhrase(sessionID)

            if (loopAction === "start") {
              state.loopActive = true
              await logger.info("Ralph loop activated for session", { sessionID })
              // Don't loop on the activation message itself -- wait for next idle
              return
            }

            if (loopAction === "stop") {
              state.loopActive = false
              await logger.info("Ralph loop deactivated for session", { sessionID })
              return
            }

            // If loop is active and no control phrase was found, send the loop prompt
            if (state.loopActive) {
              await logger.info("Ralph loop firing, sending continuation prompt", { sessionID })
              try {
                await context.client.session.prompt({
                  path: { id: sessionID },
                  body: {
                    parts: [{ type: "text", text: config.loopPrompt }],
                  },
                })
                await logger.info("Ralph loop prompt sent successfully", { sessionID })
              } catch (err) {
                await logger.error("Failed to send Ralph loop prompt", {
                  sessionID,
                  error: err instanceof Error ? err.message : String(err),
                })
              }
            }
          }
        }
      }

      if (event.type === "session.deleted") {
        const props = event.properties as { info?: { id?: string } }
        if (props.info?.id) {
          sessionStates.delete(props.info.id)
          await logger.info("Session cleaned up", { sessionID: props.info.id })
        }
      }
    },
  }
}

// ── Utility Functions ────────────────────────────────────────────────────────

function isSyntheticPart(part: MessagePart): boolean {
  return (part as any).synthetic === true
}

function convertToPromptPart(part: MessagePart): { type: "text"; text: string } | { type: "file"; mime: string; filename?: string; url: string } | { type: "agent"; name: string } | null {
  switch (part.type) {
    case "text":
      if (part.text) {
        return { type: "text", text: part.text }
      }
      return null
    case "file":
      if (part.url && part.mime) {
        return { type: "file", mime: part.mime, filename: part.filename, url: part.url }
      }
      return null
    case "agent":
      if (part.name) {
        return { type: "agent", name: part.name }
      }
      return null
    default:
      return null
  }
}

// ── V2 (OpenCode 2.x) Setup ─────────────────────────────────────────────────
// The V2 runtime hands plugins a client context instead of the V1 hook bus.
// Recovery is performed through ctx.session primitives: interrupt, context,
// revert.stage/commit, switchModel and prompt. Everything here is best-effort:
// an error in one recovery attempt is logged and swallowed, never thrown into
// the event loop.

/**
 * Find the last message of a given type in a V2 context() listing, scanning
 * from the end. Typed so passing ctx.session.context() output narrows the
 * returned message to the matching member of the message union.
 */
function findLastMessageByType<T extends { type: string }, K extends T["type"]>(
  messages: readonly T[],
  kind: K,
): Extract<T, { type: K }> | undefined {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index]
    if (message && message.type === kind) {
      return message as Extract<T, { type: K }>
    }
  }
  return undefined
}

/**
 * Re-express a stored user-message file attachment as a prompt FileAttachment.
 * URI-sourced files keep their source URI; inline files are re-embedded as a
 * data URL built from the stored base64 payload.
 */
function convertV2FilesToAttachments(files: readonly PromptFileAttachment[] | undefined): Array<{ uri: string; name?: string }> {
  if (!files || files.length === 0) return []
  const attachments: Array<{ uri: string; name?: string }> = []
  for (const file of files) {
    let uri: string
    if (file.source.type === "uri") {
      uri = file.source.uri
    } else {
      uri = `data:${file.mime};base64,${file.data}`
    }
    if (file.name) {
      attachments.push({ uri, name: file.name })
    } else {
      attachments.push({ uri })
    }
  }
  return attachments
}

/** Extract the concatenated text of a V2 assistant message's text content blocks. */
function extractV2PartialContent(message: SessionMessageAssistant): string | null {
  const textParts: string[] = []
  for (const block of message.content) {
    if (block.type === "text" && block.text) {
      textParts.push(block.text)
    }
  }
  if (textParts.length === 0) return null
  const content = textParts.join("")
  return content.trim().length > 0 ? content : null
}

/**
 * The revert sub-API of the full session client. V2's plugin context types and
 * runtime both expose only a Pick of the session API that omits revert (the
 * promise adapter builds ctx.session from fifteen methods plus hook), so a
 * staged-then-committed rewind is feature-detected and skipped when absent.
 */
interface V2RevertApi {
  stage(input: { sessionID: string; messageID: string }): Promise<unknown>
  commit(input: { sessionID: string }): Promise<void>
  clear(input: { sessionID: string }): Promise<void>
}

interface V2SessionWithRevert {
  revert?: V2RevertApi
}

/**
 * Rewind the session to just before the given user message when the runtime
 * exposes revert, best-effort: stage then commit, clearing a half-staged revert
 * if committing fails. Returns whether the rewind actually happened; callers
 * proceed without it (the divergent path) rather than abandoning recovery.
 */
async function tryRevertToMessage(
  session: V2SessionWithRevert,
  sessionID: string,
  messageID: string,
  logger: ReturnType<typeof createLogger>,
): Promise<boolean> {
  const revert = session.revert
  if (!revert || typeof revert.stage !== "function" || typeof revert.commit !== "function") {
    return false
  }
  try {
    await revert.stage({ sessionID, messageID })
    await revert.commit({ sessionID })
    return true
  } catch (err) {
    await logger.warn("Revert failed, clearing staged state", {
      sessionID,
      error: err instanceof Error ? err.message : String(err),
    })
    try {
      await revert.clear({ sessionID })
    } catch {
      // Nothing left to do; the next stage overwrites or the server recovers.
    }
    return false
  }
}

function sameModelRef(a: FallbackModelObject, b: FallbackModelObject): boolean {
  return a.providerID === b.providerID && a.modelID === b.modelID
}

function toSwitchModelRef(model: FallbackModelObject): { providerID: string; id: string } {
  return { providerID: model.providerID, id: model.modelID }
}

/**
 * Decide what to do with an active rate-limit fallback when a session goes
 * idle after (or before) the cooldown window.
 *
 * Manual-switch heuristic: the plugin tracked the model it applied
 * (appliedFallbackModel) and the session's model as last observed
 * (lastKnownModel, fed by our own switchModel calls and by
 * session.model.selected events). If the last observed model differs from the
 * one we applied, somebody switched in between and we leave it alone. Limits:
 * events missed while the subscription was down are invisible (optimistic
 * restore), variant-only switches compare equal because the config format has
 * no variant slot, and with no remembered original model there is nothing to
 * restore.
 */
export function decideFallbackRestore(input: {
  fallbackActive: boolean
  nowMs: number
  cooldownEndTime: number
  originalModel: FallbackModelObject | null
  appliedFallbackModel: FallbackModelObject | null
  lastKnownModel: FallbackModelObject | null
}): "cooldown-active" | "restore" | "skip-manual-switch" | "skip-no-original-model" {
  if (!input.fallbackActive || input.nowMs < input.cooldownEndTime) {
    return "cooldown-active"
  }
  if (!input.originalModel) {
    return "skip-no-original-model"
  }
  if (input.appliedFallbackModel && input.lastKnownModel && !sameModelRef(input.lastKnownModel, input.appliedFallbackModel)) {
    return "skip-manual-switch"
  }
  return "restore"
}

export function createV2Setup(ctx: Plugin.Context): Plugin.Cleanup {
  const abortController = new AbortController()
  const { config, createdDefaultConfig } = loadConfig()
  const logger = createLogger(config.logging)

  if (createdDefaultConfig) {
    void logger.info("Created default config file on first run", { path: createdDefaultConfig })
  }
  void logger.info("V2 plugin initialized", {
    enabled: config.enabled,
    fallbackModel: config.fallbackModel,
    rateLimitPatterns: config.rateLimitPatterns,
    disconnectPatterns: config.disconnectPatterns,
    resumeWithSameModel: config.resumeWithSameModel,
    maxResumeAttempts: config.maxResumeAttempts,
    cooldownMs: config.cooldownMs,
    loopEnabled: config.loopEnabled,
  })

  if (!config.enabled) {
    void logger.info("Plugin disabled via config")
    return () => {}
  }

  const isRateLimitMessage = createSubstringMatcher(config.rateLimitPatterns)
  const isDisconnectMessage = createWordBoundaryMatcher(config.disconnectPatterns)
  const configuredFallbackModel = parseModel(config.fallbackModel)
  const loopStartLower = config.loopStartPhrase.toLowerCase()
  const loopStopLower = config.loopStopPhrase.toLowerCase()

  // V2 publishes both the transient session.status (idle variant) and the
  // durable session.idle for one logical idle. Two consecutive idles for the
  // same session closer than this window are treated as one event, so the
  // ralph loop prompt is never fired twice per turn.
  const IDLE_DEDUPE_MS = 2000

  // ── Rate Limit Recovery (V2) ─────────────────────────────────────────────

  async function handleRateLimitRecovery(sessionID: string, retryMessage: string): Promise<void> {
    const state = getSessionState(sessionID)

    if (state.fallbackActive && Date.now() < state.cooldownEndTime) {
      await logger.info("Skipping rate-limit fallback, cooldown active", {
        sessionID,
        cooldownRemaining: state.cooldownEndTime - Date.now(),
      })
      return
    }

    await logger.info("Rate limit detected, switching to fallback model", {
      sessionID,
      message: retryMessage,
      fallbackModel: config.fallbackModel,
    })

    state.fallbackActive = true
    state.cooldownEndTime = Date.now() + config.cooldownMs
    state.pluginAbortedAt = Date.now()

    try {
      await logger.info("Interrupting session", { sessionID })
      await ctx.session.interrupt({ sessionID })
      await new Promise(resolve => setTimeout(resolve, 200))

      await logger.info("Fetching context", { sessionID })
      const messages = await ctx.session.context({ sessionID })

      if (messages.length === 0) {
        await logger.error("No messages found in session", { sessionID })
        return
      }

      const lastUserMessage = findLastMessageByType(messages, "user")
      if (!lastUserMessage) {
        await logger.error("No user message found in session", { sessionID })
        return
      }

      await logger.info("Reverting to last user message", {
        sessionID,
        messageId: lastUserMessage.id,
        totalMessages: messages.length,
      })
      // Divergence from V1's single destructive revert call: V2 splits revert
      // into stage + commit, and the plugin context does not expose revert at
      // all (its session API is a Pick that omits it), so the rewind is
      // best-effort. When revert is unavailable the aborted assistant turn
      // stays in history and the re-prompt below duplicates the user turn.
      const reverted = await tryRevertToMessage(
        ctx.session as unknown as V2SessionWithRevert,
        sessionID,
        lastUserMessage.id,
        logger,
      )
      if (reverted) {
        await new Promise(resolve => setTimeout(resolve, 500))
      }

      // Remember which model was answering before we overwrite it, so the
      // idle-after-cooldown pass can restore it. The last assistant message's
      // model is the best synchronous source available here.
      const lastAssistantMessage = findLastMessageByType(messages, "assistant")
      if (lastAssistantMessage) {
        state.originalModel = {
          providerID: lastAssistantMessage.model.providerID,
          modelID: lastAssistantMessage.model.id,
        }
      }

      const fallbackModel = configuredFallbackModel

      await logger.info("Switching session model", { sessionID, model: fallbackModel })
      // Divergence from V1: the model is carried by per-prompt request config in
      // V1; in V2 it becomes the session's model for every later turn. The
      // idle-after-cooldown pass restores the remembered original model, so the
      // blast radius is the cooldown window (and only while the user has not
      // switched models themselves). The Model.Ref variant has no slot in the
      // two-segment fallbackModel config format, so it is not sent.
      await ctx.session.switchModel({
        sessionID,
        model: toSwitchModelRef(fallbackModel),
      })
      state.appliedFallbackModel = fallbackModel
      state.lastKnownModel = fallbackModel

      const files = convertV2FilesToAttachments(lastUserMessage.files)
      const agents = (lastUserMessage.agents ?? []).map(agent => ({ name: agent.name }))

      await logger.info("Sending prompt with fallback model", {
        sessionID,
        model: fallbackModel,
        textLength: lastUserMessage.text.length,
        filesCount: files.length,
      })
      // Divergence from V1: prompt() enqueues the message for queued delivery
      // instead of running it inline, and returns once accepted.
      await ctx.session.prompt({
        sessionID,
        text: lastUserMessage.text,
        ...(files.length > 0 ? { files } : {}),
        ...(agents.length > 0 ? { agents } : {}),
      })

      await logger.info("Rate-limit fallback prompt sent successfully", { sessionID })
    } catch (err) {
      await logger.error("Failed to send rate-limit fallback prompt", {
        sessionID,
        error: err instanceof Error ? err.message : String(err),
      })
    }
  }

  // ── Disconnect Resume Recovery (V2) ──────────────────────────────────────

  async function handleDisconnectRecovery(sessionID: string, retryMessage: string): Promise<void> {
    const state = getSessionState(sessionID)

    if (state.resumeAttempts >= config.maxResumeAttempts) {
      await logger.warn("Max resume attempts reached, giving up", {
        sessionID,
        attempts: state.resumeAttempts,
        maxAttempts: config.maxResumeAttempts,
      })
      return
    }

    if (state.resumeActive) {
      await logger.info("Resume already in progress, skipping", { sessionID })
      return
    }

    await logger.info("Disconnect detected, attempting to resume", {
      sessionID,
      message: retryMessage,
      attempt: state.resumeAttempts + 1,
      maxAttempts: config.maxResumeAttempts,
    })

    state.resumeActive = true
    state.resumeAttempts += 1
    state.pluginAbortedAt = Date.now()

    try {
      const backoffMs = Math.min(2000 * Math.pow(2, state.resumeAttempts - 1), 30000)
      if (state.resumeAttempts > 1) {
        await logger.info("Applying exponential backoff before retry", {
          sessionID,
          attempt: state.resumeAttempts,
          backoffMs,
        })
        await new Promise(resolve => setTimeout(resolve, backoffMs))
      }

      await logger.info("Interrupting session for disconnect recovery", { sessionID })
      await ctx.session.interrupt({ sessionID })
      await new Promise(resolve => setTimeout(resolve, 200))

      await logger.info("Fetching context for partial content capture", { sessionID })
      const messages = await ctx.session.context({ sessionID })

      if (messages.length === 0) {
        await logger.error("No messages found in session", { sessionID })
        return
      }

      const lastAssistantMessage = findLastMessageByType(messages, "assistant")
      const lastUserMessage = findLastMessageByType(messages, "user")

      if (!lastUserMessage) {
        await logger.error("No user message found in session", { sessionID })
        return
      }

      const partialContent = lastAssistantMessage ? extractV2PartialContent(lastAssistantMessage) : null

      if (lastAssistantMessage && !state.originalModel) {
        state.originalModel = {
          providerID: lastAssistantMessage.model.providerID,
          modelID: lastAssistantMessage.model.id,
        }
      }

      await logger.info("Partial content captured", {
        sessionID,
        hasPartialContent: !!partialContent,
        partialContentLength: partialContent?.length ?? 0,
        originalModel: state.originalModel,
      })

      await logger.info("Reverting session for disconnect recovery", {
        sessionID,
        messageId: lastUserMessage.id,
      })
      // Same best-effort stage + commit shape (and same divergence caveat) as
      // the rate-limit path: V2 splits V1's destructive revert into an
      // explicit two-step, and the plugin context may not expose revert at all.
      const reverted = await tryRevertToMessage(
        ctx.session as unknown as V2SessionWithRevert,
        sessionID,
        lastUserMessage.id,
        logger,
      )
      if (reverted) {
        await new Promise(resolve => setTimeout(resolve, 500))
      }

      let modelToUse: FallbackModelObject
      if (config.resumeWithSameModel && state.originalModel && state.resumeAttempts <= 1) {
        modelToUse = state.originalModel
        await logger.info("Retrying with same model", { sessionID, model: modelToUse })
      } else {
        modelToUse = configuredFallbackModel
        await logger.info("Escalating to fallback model", { sessionID, model: modelToUse })
      }

      let text: string
      let files: Array<{ uri: string; name?: string }> = []
      const agents = (lastUserMessage.agents ?? []).map(agent => ({ name: agent.name }))

      if (partialContent) {
        // Divergence from V1: when the revert ran, the continuation prompt runs
        // on top of a context that still holds the user message (V1 reverted it
        // away); when revert was unavailable, the history additionally still
        // holds the partial assistant turn itself, so the pasted partial is
        // redundant with what the model already sees. Harmless either way.
        text = buildContinuationPrompt(config.continuationPromptTemplate, partialContent, config.maxPartialContentLength)
        await logger.info("Sending continuation prompt with partial content", {
          sessionID,
          model: modelToUse,
          partialContentLength: partialContent.length,
        })
      } else {
        text = lastUserMessage.text
        files = convertV2FilesToAttachments(lastUserMessage.files)
        if (text.length === 0 && files.length === 0) {
          await logger.error("No text or files to re-send", { sessionID })
          return
        }
        await logger.info("No partial content, re-sending original user message", {
          sessionID,
          model: modelToUse,
          textLength: text.length,
          filesCount: files.length,
        })
      }

      await ctx.session.switchModel({
        sessionID,
        model: toSwitchModelRef(modelToUse),
      })
      state.lastKnownModel = modelToUse

      await ctx.session.prompt({
        sessionID,
        text,
        ...(files.length > 0 ? { files } : {}),
        ...(agents.length > 0 ? { agents } : {}),
      })

      await logger.info("Disconnect recovery prompt sent successfully", {
        sessionID,
        attempt: state.resumeAttempts,
        usedContinuation: !!partialContent,
        model: modelToUse,
      })
    } catch (err) {
      await logger.error("Failed to send disconnect recovery prompt", {
        sessionID,
        error: err instanceof Error ? err.message : String(err),
        attempt: state.resumeAttempts,
      })
    } finally {
      state.resumeActive = false
    }
  }

  // ── Ralph Loop (V2) ──────────────────────────────────────────────────────

  async function checkLoopPhraseV2(sessionID: string): Promise<"start" | "stop" | null> {
    if (!config.loopEnabled) return null

    try {
      const messages = await ctx.session.context({ sessionID })
      const lastUserMessage = findLastMessageByType(messages, "user")
      if (!lastUserMessage) return null

      const text = lastUserMessage.text.toLowerCase()

      if (text.includes(loopStopLower)) return "stop"
      if (text.includes(loopStartLower)) return "start"
      return null
    } catch {
      return null
    }
  }

  async function handleIdle(sessionID: string): Promise<void> {
    const state = getSessionState(sessionID)

    if (state.fallbackActive && Date.now() >= state.cooldownEndTime) {
      const decision = decideFallbackRestore({
        fallbackActive: true,
        nowMs: Date.now(),
        cooldownEndTime: state.cooldownEndTime,
        originalModel: state.originalModel,
        appliedFallbackModel: state.appliedFallbackModel,
        lastKnownModel: state.lastKnownModel,
      })

      // Clearing fallbackActive in every branch makes the restore fire at most
      // once per fallback cycle; the idle dedupe window already collapses the
      // double idle event pair, and this only ever runs from an idle pass, so
      // the switch back can never land mid-run.
      state.fallbackActive = false
      state.appliedFallbackModel = null

      if (decision === "restore" && state.originalModel) {
        try {
          await ctx.session.switchModel({ sessionID, model: toSwitchModelRef(state.originalModel) })
          state.lastKnownModel = state.originalModel
          await logger.info("Rate-limit cooldown expired, restored original model", {
            sessionID,
            model: state.originalModel,
          })
        } catch (err) {
          await logger.error("Failed to restore original model after cooldown", {
            sessionID,
            error: err instanceof Error ? err.message : String(err),
          })
        }
      } else if (decision === "skip-manual-switch") {
        await logger.info("Rate-limit cooldown expired, user switched models in between, leaving session model alone", {
          sessionID,
          lastKnownModel: state.lastKnownModel,
        })
      } else {
        await logger.info("Rate-limit cooldown expired, fallback reset", { sessionID })
      }
    }

    if (state.resumeAttempts > 0 && !state.resumeActive) {
      await logger.info("Session idle after disconnect recovery, resetting resume state", {
        sessionID,
        totalAttempts: state.resumeAttempts,
      })
      state.resumeAttempts = 0
      state.originalModel = null
      state.pluginAbortedAt = 0
    }

    if (!config.loopEnabled) return

    const loopAction = await checkLoopPhraseV2(sessionID)

    if (loopAction === "start") {
      state.loopActive = true
      await logger.info("Ralph loop activated for session", { sessionID })
      return
    }

    if (loopAction === "stop") {
      state.loopActive = false
      await logger.info("Ralph loop deactivated for session", { sessionID })
      return
    }

    if (state.loopActive) {
      await logger.info("Ralph loop firing, sending continuation prompt", { sessionID })
      try {
        // Divergence from V1: the loop prompt is queued delivery rather than an
        // inline prompt. The idle-after-that-prompt event starts a fresh idle
        // handling pass well outside the dedupe window, so no suppression flag
        // is needed here.
        await ctx.session.prompt({ sessionID, text: config.loopPrompt })
        await logger.info("Ralph loop prompt sent successfully", { sessionID })
      } catch (err) {
        await logger.error("Failed to send Ralph loop prompt", {
          sessionID,
          error: err instanceof Error ? err.message : String(err),
        })
      }
    }
  }

  // ── Event Loop ───────────────────────────────────────────────────────────

  function getIdleSessionID(event: V2Event): string | null {
    if (event.type === "session.idle") {
      return event.data.sessionID
    }
    if (event.type === "session.status" && event.data.status.type === "idle") {
      return event.data.sessionID
    }
    return null
  }

  const subscriptionLoop = (async (): Promise<void> => {
    try {
      for await (const event of ctx.event.subscribe({ signal: abortController.signal })) {
        try {
          if (event.type === "session.deleted") {
            sessionStates.delete(event.data.sessionID)
            await logger.info("Session cleaned up", { sessionID: event.data.sessionID })
            continue
          }

          if (event.type === "session.model.selected") {
            const state = getSessionState(event.data.sessionID)
            // Durable event: fires for user switches and echoes our own
            // switchModel calls alike. Feeds the manual-switch heuristic in
            // decideFallbackRestore; the echo of our own switch is harmless
            // because it re-observes exactly the model we just applied.
            state.lastKnownModel = {
              providerID: event.data.model.providerID,
              modelID: event.data.model.id,
            }
            continue
          }

          const idleSessionID = getIdleSessionID(event)
          if (idleSessionID !== null) {
            const state = getSessionState(idleSessionID)
            if (Date.now() - state.lastIdleHandledAt < IDLE_DEDUPE_MS) {
              continue
            }
            state.lastIdleHandledAt = Date.now()
            await handleIdle(idleSessionID)
            continue
          }

          if (event.type === "session.status" && event.data.status.type === "retry") {
            const sessionID = event.data.sessionID
            const status = event.data.status
            const state = getSessionState(sessionID)

            // Same stale-retry discipline as V1: retry events that were already
            // in flight when the plugin initiated its own interrupt are ignored
            // for the grace window, so plugin-initiated recovery never loops.
            const ABORT_GRACE_MS = 5000
            if (state.pluginAbortedAt > 0 && Date.now() - state.pluginAbortedAt < ABORT_GRACE_MS) {
              await logger.info("Ignoring retry event within abort grace window", {
                sessionID,
                abortedAt: state.pluginAbortedAt,
                elapsed: Date.now() - state.pluginAbortedAt,
              })
              continue
            }

            const message = status.message

            if (isDisconnectMessage(message)) {
              await handleDisconnectRecovery(sessionID, message)
              continue
            }

            if (isRateLimitMessage(message)) {
              await handleRateLimitRecovery(sessionID, message)
              continue
            }

            await logger.info("Retry detected but no matching pattern", { sessionID, message })
          }
        } catch (err) {
          await logger.error("Failed to handle event", {
            eventType: event.type,
            error: err instanceof Error ? err.message : String(err),
          })
        }
      }
    } catch (err) {
      if (!abortController.signal.aborted) {
        await logger.error("Event subscription stopped", {
          error: err instanceof Error ? err.message : String(err),
        })
      }
    }
  })()

  void subscriptionLoop

  return () => {
    abortController.abort()
  }
}
