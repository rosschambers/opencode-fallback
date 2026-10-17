// Dual OpenCode V1 + V2 plugin entrypoint. The default export is the ONLY
// export of this module: V1 (1.18.29+) calls every export of a plugin file as a
// plugin, so a named function export here would double-register it.
//
// - V2 (2.x) reads `id` and calls `setup`, listening on the event bus through
//   ctx.event.subscribe and recovering through the session client primitives.
// - V1 calls the `server` entry with its PluginInput and wires the returned
//   hook object (unchanged V1 behaviour).
//
// The default export is a plain object with exactly the shape that
// Plugin.define from @opencode/plugin produces (Plugin.define is an identity
// function). It is written as a literal on purpose: a value import of
// @opencode/plugin would make the file unresolvable for any loader without that
// package installed, while type-only imports are erased at load time and cost
// nothing in either runtime.
import type { PluginInput, Hooks } from "@opencode-ai/plugin"
import type { Plugin } from "@opencode/plugin"
import { createPlugin, createV2Setup } from "./src/plugin.ts"

export default {
  id: "opencode-fallback",
  setup(ctx: Plugin.Context): Plugin.Cleanup {
    return createV2Setup(ctx)
  },
  async server(context: PluginInput): Promise<Hooks> {
    return createPlugin(context)
  },
}
