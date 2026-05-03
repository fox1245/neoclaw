#include "topology.h"

#include <neograph/graph/loader.h>  // ReducerRegistry / NodeFactory (process-wide singletons)

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

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

// Sliding-window stream-output suppressor.
//
// Hides occurrences of `<open>...<close>` spans from the user-visible
// token stream while still letting the *full* span land in the engine's
// message channel (where score_extract / downstream nodes can parse
// it). Used to keep "control" payloads emitted by the LLM out of the
// REPL — e.g. spec-first.json's interviewer wraps its ambiguity score
// in `<score>0.6</score>`, plan-then-act.json wraps its plan in
// `<plan>…</plan>`, and we don't want either bleeding into the prose
// the user reads.
//
// Implementation: buffer the streamed text. Emit prefixes that are
// definitely outside any span. When an `<open>` lands, swallow until
// the matching `<close>`. Trailing bytes that *could* be a partial
// open-tag prefix are held back until enough bytes arrive to decide.
class StreamSuppressor {
public:
    using Sink = std::function<void(const std::string&)>;

    explicit StreamSuppressor(std::vector<std::pair<std::string, std::string>> spans)
        : spans_(std::move(spans)) {}

    void feed(const std::string& chunk, const Sink& sink) {
        buf_ += chunk;
        while (true) {
            if (inside_idx_ >= 0) {
                const auto& close = spans_[static_cast<size_t>(inside_idx_)].second;
                const auto pos = buf_.find(close);
                if (pos == std::string::npos) {
                    // Still inside — buffer everything, emit nothing.
                    return;
                }
                buf_.erase(0, pos + close.size());
                inside_idx_ = -1;
            }
            // Outside. Find earliest open tag, OR earliest "could be a
            // partial open" suffix at end of buf_.
            size_t earliest = std::string::npos;
            int    which    = -1;
            for (size_t i = 0; i < spans_.size(); ++i) {
                auto p = buf_.find(spans_[i].first);
                if (p != std::string::npos && (earliest == std::string::npos
                                                || p < earliest)) {
                    earliest = p;
                    which    = static_cast<int>(i);
                }
            }
            if (earliest != std::string::npos) {
                if (earliest > 0 && sink) sink(buf_.substr(0, earliest));
                buf_.erase(0, earliest + spans_[static_cast<size_t>(which)].first.size());
                inside_idx_ = which;
                continue;  // loop to look for close
            }
            // No full open tag. Compute the safe-to-emit length:
            // anything except a trailing prefix that could grow into
            // an open tag on the next chunk.
            size_t safe_end = buf_.size();
            for (const auto& [open, _close] : spans_) {
                for (size_t k = 1; k < open.size() && k <= buf_.size(); ++k) {
                    if (buf_.compare(buf_.size() - k, k, open, 0, k) == 0) {
                        if (buf_.size() - k < safe_end) safe_end = buf_.size() - k;
                        break;
                    }
                }
            }
            if (safe_end > 0 && sink) sink(buf_.substr(0, safe_end));
            buf_.erase(0, safe_end);
            return;
        }
    }

    void flush(const Sink& sink) {
        // End-of-stream: emit only content that's *outside* any span.
        // Trailing unterminated-span content is dropped — the model
        // started a control region and ran out of turn before closing
        // it; safer to hide than to leak the half-baked metadata.
        if (inside_idx_ < 0 && !buf_.empty() && sink) sink(buf_);
        buf_.clear();
        inside_idx_ = -1;
    }

    // Per-node reset. Called at NODE_END so an unterminated span in
    // node A doesn't continue to swallow node B's streamed tokens.
    // Any inside-span buffer is dropped silently (it was control
    // payload anyway); outside-span trailing safe content gets
    // emitted before the reset.
    void reset_at_node_boundary(const Sink& sink) {
        if (inside_idx_ < 0 && !buf_.empty() && sink) sink(buf_);
        buf_.clear();
        inside_idx_ = -1;
    }

private:
    std::vector<std::pair<std::string, std::string>> spans_;
    std::string buf_;
    int         inside_idx_ = -1;
};

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
fs::path default_topologies_dir() {
    const fs::path here = fs::current_path();
    const fs::path bin  = exe_dir();
    const std::vector<fs::path> candidates = {
        here / "topologies",
        bin  / "topologies",
        bin  / ".." / "share" / "neoclaw" / "topologies",
    };
    for (const auto& c : candidates) {
        std::error_code ec;
        if (fs::exists(c, ec) && fs::is_directory(c, ec) && !ec)
            return fs::weakly_canonical(c);
    }
    return {};
}

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

    // Stream tokens through to the user via run_stream. Requires NeoGraph
    // ≥ the commit that adds `LLMCallNode::execute_stream_async` (without
    // it the default async path silently drops the GraphStreamCallback
    // and no LLM_TOKEN events fire). neoclaw's CMake pins a NeoGraph tag
    // that includes that override.
    //
    // The StreamSuppressor wrap hides span markers used by spec-first /
    // plan-then-act / 3-pass-review topologies (`<score>…</score>`,
    // `<plan>…</plan>`) from the user's terminal while keeping the full
    // text in the message channel for downstream nodes to parse.
    StreamSuppressor suppressor({
        {"<score>", "</score>"},
        {"<plan>",  "</plan>"},
    });
    auto result = engine.run_stream(cfg, [&](const auto& evt) {
        using T = neograph::graph::GraphEvent::Type;
        if (evt.type == T::LLM_TOKEN && on_token && evt.data.is_string()) {
            suppressor.feed(evt.data.template get<std::string>(), on_token);
        } else if (evt.type == T::NODE_END) {
            // Reset suppressor at every node boundary so an unterminated
            // span in one node (e.g. planner's `<plan>` minus `</plan>`)
            // doesn't continue to swallow the next node's tokens.
            if (on_token) suppressor.reset_at_node_boundary(on_token);
        }
    });
    if (on_token) suppressor.flush(on_token);

    // Set NEOCLAW_TRACE_GRAPH=1 to see the per-turn execution trace —
    // useful when authoring a new topology JSON to verify the nodes
    // fire in the order you expect.
    if (const char* dbg = std::getenv("NEOCLAW_TRACE_GRAPH");
        dbg && *dbg && *dbg != '0') {
        std::fprintf(stderr, "[topology] trace:");
        for (const auto& n : result.execution_trace)
            std::fprintf(stderr, " %s", n.c_str());
        std::fprintf(stderr, "\n");
        if (const char* dump = std::getenv("NEOCLAW_DUMP_STATE");
            dump && *dump && *dump != '0') {
            std::fprintf(stderr, "[topology] state: %s\n",
                          result.output.dump().c_str());
        }
    }

    // GraphState serializes as
    //   { "channels": { "<name>": { "value": ..., "version": N } },
    //     "global_version": N, "final_response": "..." }
    // so the messages array lives at result.output.channels.messages.value.
    // NB: NOT `result.output["messages"]` — that always misses on the
    // engine's wrapped state shape. (Cost: 5 minutes the first time round.)
    if (result.output.contains("channels")
        && result.output["channels"].contains("messages")
        && result.output["channels"]["messages"].contains("value")
        && result.output["channels"]["messages"]["value"].is_array()) {
        return result.output["channels"]["messages"]["value"];
    }
    // Fallback: pathological topology that doesn't carry a `messages`
    // channel — hand back what we sent so the REPL doesn't lose state.
    (void)n_before;
    return conversation;
}

} // namespace neoclaw
