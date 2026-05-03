// neoclaw/src/topology.h — JSON-defined topology loader.
//
// neoclaw v0.5+ can boot its agent loop from a JSON file instead of the
// hardcoded ReAct path. The same binary, the same model, and the same
// tool set become a different *agent persona* per file:
//
//   topologies/pair.json         — ReAct pair-programmer (the default)
//   topologies/code-review.json  — single-LLM diff reviewer, no write tools
//   topologies/planner-executor.json
//   topologies/debate.json
//   topologies/deep-research.json
//
// This is harness-engineering territory — the actual product surface
// lives here, not in the inference runtime. NeoGraph's GraphEngine does
// the real work; this module is the neoclaw-specific glue (path
// resolution, NodeContext build, REPL-friendly streaming bridge).
#pragma once

#include <neograph/graph/engine.h>
#include <neograph/provider.h>
#include <neograph/tool.h>

#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace neoclaw {

/// Resolve a `topology:` config value to an absolute path.
///
/// Absolute paths are returned unchanged. Relative paths are searched
/// in this order (first hit wins):
///   1. CWD / <path>
///   2. <exe-dir> / <path>
///   3. <exe-dir>/topologies/ <basename(path)>     (so a bundle that ships
///                                                  `topologies/pair.json`
///                                                  resolves `pair.json`
///                                                  invocations)
///   4. <exe-dir>/../share/neoclaw/topologies/<basename(path)>
///                                                 (FHS install layout)
///
/// Throws std::runtime_error if no candidate exists.
std::filesystem::path resolve_topology_path(const std::string& spec);

/// Compile a JSON topology file into a runnable GraphEngine. The
/// `tools` argument is used for both NodeContext.tools (raw pointers
/// the engine will dispatch through) and engine ownership transfer
/// via own_tools(...) — neoclaw hands the tool ownership over so the
/// engine controls lifetime cleanly.
///
/// `system_prompt` becomes NodeContext.instructions; LLMCallNode
/// auto-prepends it as a system message every turn (mirroring
/// neograph::llm::Agent's behaviour).
std::unique_ptr<neograph::graph::GraphEngine> compile_topology(
    const std::filesystem::path&                 json_path,
    std::shared_ptr<neograph::Provider>          provider,
    std::vector<std::unique_ptr<neograph::Tool>> tools,
    const std::string&                            system_prompt);

/// Per-turn driver. Treats `conversation` as the canonical message
/// history (JSON array of {role, content, tool_calls?, tool_call_id?}
/// objects), runs the engine once, prints LLM tokens through
/// `on_token` as they arrive, and returns the updated conversation
/// (post-run engine state for the "messages" channel).
///
/// Caller owns the conversation across turns: pass it in, get the
/// next iteration's conversation back. The engine itself is stateless
/// per call (no checkpoint thread is held).
neograph::json run_topology_turn(
    neograph::graph::GraphEngine&                engine,
    neograph::json                                conversation,
    const std::function<void(const std::string&)>& on_token);

} // namespace neoclaw
