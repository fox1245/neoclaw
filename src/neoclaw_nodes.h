// neoclaw/src/neoclaw_nodes.h — register custom NeoGraph node types,
// conditions, and reducers needed by neoclaw's bundled topologies.
//
// These extend the built-in NeoGraph factory (`llm_call`,
// `tool_dispatch`, `intent_classifier`, `subgraph`) with the pieces
// required to mimic ouroboros-style multi-stage reviewers and
// specification-first workflows entirely from JSON:
//
//   - `llm_with_prompt` — `llm_call` variant whose system prompt is read
//     from the JSON node config (`config.prompt`), letting one topology
//     run several LLM stages with different reviewer angles. Optional
//     `config.model` pins a model per-stage (the foundation for an
//     ouroboros-style 1×/10×/30× tiered routing once neoclaw grows
//     multi-Provider support).
//
//   - `score_extract` — parses `{"score": float}` (with prose
//     tolerance) from the last assistant message and writes the float
//     to the `score` channel. Used as the bridge between an LLM-produced
//     ambiguity estimate and a downstream threshold condition.
//
//   - `score_below_0_2` (condition) — reads the `score` channel,
//     returns `"ready"` if ≤ 0.2, else `"more"`. Mirrors ouroboros's
//     canonical Ambiguity gate.
//
// Registration is process-wide (NeoGraph's three registries are
// singletons — see include/neograph/graph/loader.h) and idempotent.
#pragma once

namespace neoclaw {

/// Register all neoclaw-specific node types and conditions with
/// NeoGraph's process-wide registries. Idempotent. Call once at
/// startup, before any GraphEngine::compile.
void register_nodes();

} // namespace neoclaw
