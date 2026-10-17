import { expect, test } from "bun:test"
import { buildContinuationPrompt, createSubstringMatcher, createWordBoundaryMatcher, decideFallbackRestore } from "./plugin.ts"

const ORIGINAL = { providerID: "frame", modelID: "local-agentic" }
const FALLBACK = { providerID: "anthropic", modelID: "claude-opus-4-5" }
const USER_CHOICE = { providerID: "openai", modelID: "gpt-5" }

test("rate-limit matcher: case-insensitive substring patterns match retry messages", () => {
  const isRateLimit = createSubstringMatcher(["rate limit", "too many requests"])

  expect(isRateLimit("Error: Rate Limit exceeded")).toBe(true)
  expect(isRateLimit("Too Many Requests")).toBe(true)
  expect(isRateLimit("503 Service Unavailable")).toBe(false)
})

test("disconnect matcher: word-boundary patterns match disconnects without false positives", () => {
  const isDisconnect = createWordBoundaryMatcher(["eof", "502", "connection reset"])

  expect(isDisconnect("stream disconnected: connection reset by peer")).toBe(true)
  expect(isDisconnect("upstream returned 502 bad gateway")).toBe(true)
  expect(isDisconnect("thereof we spoke, at 150200 units")).toBe(false)
})

test("disconnect recovery: continuation prompt embeds partial content and truncates its tail", () => {
  const template = "Continue here:\n{{partial_content}}\nEnd."

  const embedded = buildContinuationPrompt(template, "Hello wor", 100)
  expect(embedded).toBe("Continue here:\nHello wor\nEnd.")

  const long = buildContinuationPrompt(template, "x".repeat(50), 20)
  expect(long).toBe(`Continue here:\n[...truncated]\n${"x".repeat(20)}\nEnd.`)
})

test("fallback restore: cooldown active, no fallback running", () => {
  expect(decideFallbackRestore({
    fallbackActive: false, nowMs: 10_000, cooldownEndTime: 5_000,
    originalModel: ORIGINAL, appliedFallbackModel: FALLBACK, lastKnownModel: FALLBACK,
  })).toBe("cooldown-active")

  expect(decideFallbackRestore({
    fallbackActive: true, nowMs: 4_000, cooldownEndTime: 5_000,
    originalModel: ORIGINAL, appliedFallbackModel: FALLBACK, lastKnownModel: FALLBACK,
  })).toBe("cooldown-active")
})

test("fallback restore: expired cooldown with untouched fallback restores the original model", () => {
  expect(decideFallbackRestore({
    fallbackActive: true, nowMs: 6_000, cooldownEndTime: 5_000,
    originalModel: ORIGINAL, appliedFallbackModel: FALLBACK, lastKnownModel: FALLBACK,
  })).toBe("restore")

  // A missed model event (lastKnownModel never observed) restores optimistically.
  expect(decideFallbackRestore({
    fallbackActive: true, nowMs: 6_000, cooldownEndTime: 5_000,
    originalModel: ORIGINAL, appliedFallbackModel: FALLBACK, lastKnownModel: null,
  })).toBe("restore")
})

test("fallback restore: a user model switch in between skips the restore", () => {
  expect(decideFallbackRestore({
    fallbackActive: true, nowMs: 6_000, cooldownEndTime: 5_000,
    originalModel: ORIGINAL, appliedFallbackModel: FALLBACK, lastKnownModel: USER_CHOICE,
  })).toBe("skip-manual-switch")
})

test("fallback restore: no remembered original model skips the restore", () => {
  expect(decideFallbackRestore({
    fallbackActive: true, nowMs: 6_000, cooldownEndTime: 5_000,
    originalModel: null, appliedFallbackModel: FALLBACK, lastKnownModel: FALLBACK,
  })).toBe("skip-no-original-model")
})
