#include "topology.h"

#include <neograph/graph/loader.h>  // ReducerRegistry / NodeFactory (process-wide singletons)

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

#if defined(__linux__)
#  include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace neoclaw {

namespace {

// Read the directory the running executable lives in. Used to resolve
// `topology: pair.json` against `<exe-dir>/topologies/pair.json` so a
// shipped bundle works without any prefix gymnastics from the user.
fs::path exe_dir() {
#if defined(__linux__)
    char buf[4096];
    const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0) {
        buf[n] = '\0';
        return fs::path(buf).parent_path();
    }
#endif
    return fs::current_path();
}

// Load a JSON document from disk. Errors include the file path so a
// typo in the YAML's `topology:` field is immediately obvious.
neograph::json read_json_file(const fs::path& p) {
    std::ifstream in(p);
    if (!in) {
        throw std::runtime_error(
            "neoclaw: cannot open topology file " + p.string());
    }
    std::ostringstream os;
    os << in.rdbuf();
    try {
        return neograph::json::parse(os.str());
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "neoclaw: JSON parse error in " + p.string() + ": " + e.what());
    }
}

} // namespace

// =======================================================================
// Path resolution
// =======================================================================
fs::path resolve_topology_path(const std::string& spec) {
    if (spec.empty())
        throw std::runtime_error("neoclaw: empty topology spec");

    fs::path requested(spec);
    if (requested.is_absolute()) {
        if (fs::exists(requested)) return requested;
        throw std::runtime_error(
            "neoclaw: topology not found at absolute path " + spec);
    }

    const fs::path here = fs::current_path();
    const fs::path bin  = exe_dir();
    const fs::path base = requested.filename();

    const std::vector<fs::path> candidates = {
        here / requested,
        bin  / requested,
        bin  / "topologies" / base,
        bin  / ".." / "share" / "neoclaw" / "topologies" / base,
    };

    for (const auto& c : candidates) {
        std::error_code ec;
        if (fs::exists(c, ec) && !ec) {
            return fs::weakly_canonical(c);
        }
    }

    std::ostringstream tried;
    tried << "neoclaw: topology '" << spec << "' not found. Tried:\n";
    for (const auto& c : candidates) tried << "  " << c.string() << "\n";
    throw std::runtime_error(tried.str());
}

// =======================================================================
// compile
// =======================================================================
std::unique_ptr<neograph::graph::GraphEngine> compile_topology(
    const fs::path&                              json_path,
    std::shared_ptr<neograph::Provider>          provider,
    std::vector<std::unique_ptr<neograph::Tool>> tools,
    const std::string&                            system_prompt) {

    auto definition = read_json_file(json_path);

    // NodeContext wants raw Tool* pointers (lifetime contract: pointees
    // must outlive the engine). We transfer ownership to the engine
    // immediately after compile() so the unique_ptr vector going out of
    // scope here doesn't dangle.
    std::vector<neograph::Tool*> tool_ptrs;
    tool_ptrs.reserve(tools.size());
    for (auto& t : tools) tool_ptrs.push_back(t.get());

    neograph::graph::NodeContext ctx;
    ctx.provider     = std::move(provider);
    ctx.tools        = std::move(tool_ptrs);
    ctx.instructions = system_prompt;
    // ctx.model left empty — the LocalProvider/GemmaProvider both ignore
    // it (they were configured at construction time).

    auto engine = neograph::graph::GraphEngine::compile(definition, ctx);
    engine->own_tools(std::move(tools));
    return engine;
}

// =======================================================================
// run_topology_turn
// =======================================================================
neograph::json run_topology_turn(
    neograph::graph::GraphEngine&                engine,
    neograph::json                                conversation,
    const std::function<void(const std::string&)>& on_token) {

    if (!conversation.is_array()) {
        throw std::runtime_error(
            "neoclaw: conversation must be a JSON array of message objects");
    }

    const size_t n_before = conversation.size();

    neograph::graph::RunConfig cfg;
    // Stateless thread per turn — checkpointing isn't wired here yet.
    // When we add `/save` / `/resume` REPL commands, switch to a
    // persistent thread_id + a SqliteCheckpointStore on the engine.
    cfg.thread_id = "neoclaw-repl";
    cfg.input     = neograph::json{{"messages", conversation}};

    // We use the non-streaming `run()` here (not run_stream). NeoGraph's
    // LLMCallNode overrides `execute_stream` (sync) but NOT
    // `execute_stream_async`, so under the engine's async coroutine path
    // (the default) the GraphStreamCallback is silently dropped — no
    // LLM_TOKEN events ever fire. Rather than depend on that being
    // fixed upstream, we run to completion and print any messages added
    // during the turn after the fact. Trade-off: no token-by-token
    // streaming on the topology path. The Agent path (cfg.topology
    // empty) still streams normally. v0.5+ TODO is to either subclass
    // LLMCallNode in neoclaw with the missing override, or push the fix
    // into NeoGraph.
    auto result = engine.run(cfg);

    // Set NEOCLAW_TRACE_GRAPH=1 to see the per-turn execution trace —
    // useful when authoring a new topology JSON to verify the nodes
    // fire in the order you expect.
    if (const char* dbg = std::getenv("NEOCLAW_TRACE_GRAPH");
        dbg && *dbg && *dbg != '0') {
        std::fprintf(stderr, "[topology] trace:");
        for (const auto& n : result.execution_trace)
            std::fprintf(stderr, " %s", n.c_str());
        std::fprintf(stderr, "\n");
    }

    // GraphState serializes as
    //   { "channels": { "<name>": { "value": ..., "version": N } },
    //     "global_version": N, "final_response": "..." }
    // so the messages array lives at result.output.channels.messages.value.
    auto msgs = conversation;
    if (result.output.contains("channels")
        && result.output["channels"].contains("messages")
        && result.output["channels"]["messages"].contains("value")
        && result.output["channels"]["messages"]["value"].is_array()) {
        msgs = result.output["channels"]["messages"]["value"];
    }

    if (on_token) {
        for (size_t i = n_before; i < msgs.size(); ++i) {
            const auto& m   = msgs[i];
            const auto role = m.value("role", std::string{});
            if (role == "assistant") {
                const auto content = m.value("content", std::string{});
                if (!content.empty()) on_token(content);
                // Tool-call announcements come from the local_provider's
                // ui::print_tool_start at JSON-parse time, so we don't
                // re-render them here.
            }
            // role=="tool" results are surfaced indirectly via the next
            // assistant turn's content; skipping the verbose dump here.
        }
    }

    return msgs;
}

} // namespace neoclaw
