import type { Hooks, PluginInput } from "@opencode-ai/plugin"
import { loadConfig, parseModel, type FallbackModelObject } from "./config"
import { createLogger } from "./log"

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
function createSubstringMatcher(patterns: string[]) {
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
function createWordBoundaryMatcher(patterns: string[]) {
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
function buildContinuationPrompt(template: string, partialContent: string, maxLength: number): string {
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
