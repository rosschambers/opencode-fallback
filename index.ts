// Only export the plugin function - OpenCode calls ALL exports as plugins
// WARNING: Only ONE named function export is allowed. OpenCode calls every export as a plugin.
export { createPlugin as FallbackPlugin } from "./src/plugin"

// Type exports are fine (stripped at runtime)
export type {
  FallbackConfig,
  RateLimitFallbackConfig,
  FallbackModel,
  FallbackModelObject,
} from "./src/config"

import { createPlugin } from "./src/plugin"
export default createPlugin
