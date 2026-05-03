#include "neoclaw_nodes.h"

#include <neograph/graph/loader.h>
#include <neograph/graph/node.h>
#include <neograph/graph/state.h>
#include <neograph/graph/types.h>
#include <neograph/json.h>

#include <atomic>
#include <memory>
#include <string>
#include <vector>

namespace neoclaw {

namespace {

using neograph::ChatMessage;
using neograph::json;
using neograph::graph::ChannelWrite;
using neograph::graph::ConditionRegistry;
using neograph::graph::GraphNode;
using neograph::graph::GraphState;
using neograph::graph::LLMCallNode;
using neograph::graph::NodeContext;
using neograph::graph::NodeFactory;

// =======================================================================
// `llm_with_prompt` — LLMCallNode subclass that reads a per-node prompt
// (and optional model override) from JSON config.
//
// Construction trick: NodeContext is a value struct (provider shared_ptr,
// vector<Tool*>, strings, json), so we just clone-and-overwrite before
// handing it to the LLMCallNode constructor. No private-field surgery.
// =======================================================================
class LLMWithPrompt : public LLMCallNode {
public:
    LLMWithPrompt(const std::string& name,
                  const json& config,
                  const NodeContext& ctx)
        : LLMCallNode(name, override_ctx(config, ctx)) {}

private:
    static NodeContext override_ctx(const json& config,
                                    const NodeContext& ctx) {
        NodeContext out = ctx;
        if (config.is_object()) {
            if (config.contains("prompt") && config["prompt"].is_string()) {
                // Replace the inherited cfg.agent.system_prompt for this
                // node only. Other nodes in the same graph keep their
                // own prompt (inherited or per-node-overridden).
                out.instructions = config["prompt"].template get<std::string>();
            }
            if (config.contains("model") && config["model"].is_string()) {
                out.model = config["model"].template get<std::string>();
            }
            // `tools_off: true` strips the tool list for this node only.
            // The local_provider's system-prompt rewriter prepends a
            // tool-protocol prompt whenever NodeContext.tools is non-
            // empty — for planner / reviewer / interviewer / seed-writer
            // nodes that should produce prose only, this pollutes the
            // model's context with "to call a tool, emit JSON…" rules
            // and small models (Gemma 4B) get visibly confused. Setting
            // tools_off=true here keeps tools available to *other*
            // nodes (e.g. an executor subgraph) while sparing the
            // single-shot prose stages.
            if (config.contains("tools_off") && config["tools_off"].is_boolean()
                && config["tools_off"].template get<bool>()) {
                out.tools.clear();
            }
        }
        return out;
    }
};

// =======================================================================
// `score_extract` — parse `{"score": <float>}` from the last assistant
// message and write the float to the `score` channel.
//
// Tolerates models that:
//   (a) emit pure JSON,
//   (b) wrap JSON in markdown ```json fences,
//   (c) bury the JSON in surrounding prose (we scan for the first
//       `"score"` key occurrence).
//
// Default = 1.0 (max ambiguity / fail-safe loop) when nothing parseable
// is found — the threshold condition will keep us in interview mode
// rather than prematurely committing.
// =======================================================================
class ScoreExtract : public GraphNode {
public:
    explicit ScoreExtract(std::string name) : name_(std::move(name)) {}

    std::vector<ChannelWrite> execute(const GraphState& state) override {
        auto msgs = state.get_messages();
        double score = 1.0;
        if (!msgs.empty()) {
            const auto& last = msgs.back();
            if (last.role == "assistant" && !last.content.empty()) {
                score = parse_score_lenient(last.content);
            }
        }
        return {ChannelWrite{"score", json(score)}};
    }

    std::string get_name() const override { return name_; }

private:
    static double parse_score_lenient(const std::string& text) {
        // Preferred form: `<score>0.6</score>` — easy to stream-strip
        // from user-visible output, easy to parse here. Also handles
        // the open-only case `<score>0.6` where a small model
        // truncated before emitting `</score>`: we parse as far as
        // the next non-numeric character.
        if (const auto a = text.find("<score>");
            a != std::string::npos) {
            const size_t p   = a + 7;  // past `<score>`
            const auto   b   = text.find("</score>", p);
            const size_t end = (b != std::string::npos) ? b : text.size();
            size_t q = p;
            while (q < end) {
                char c = text[q];
                if ((c >= '0' && c <= '9') || c == '.' || c == '-'
                     || c == '+' || c == 'e' || c == 'E') ++q;
                else break;
            }
            if (q > p) {
                try {
                    return std::stod(text.substr(p, q - p));
                } catch (...) { /* fall through to JSON form */ }
            }
        }

        // Backward-compat: `{"score": 0.6}`. Try strict full-parse
        // first, then a tolerant substring scan for prose-wrapped JSON.
        try {
            const auto j = json::parse(text);
            if (j.is_object() && j.contains("score")
                && j["score"].is_number()) {
                return j["score"].template get<double>();
            }
        } catch (...) { /* fall through */ }

        const auto k = text.find("\"score\"");
        if (k == std::string::npos) return 1.0;
        size_t p = k + 7;  // past `"score"`
        // Skip whitespace + colon.
        while (p < text.size() && (text[p] == ' ' || text[p] == '\t'
                                    || text[p] == ':' || text[p] == '\n')) ++p;
        if (p >= text.size()) return 1.0;
        size_t q = p;
        while (q < text.size()) {
            char c = text[q];
            if ((c >= '0' && c <= '9') || c == '.' || c == '-'
                 || c == '+' || c == 'e' || c == 'E') ++q;
            else break;
        }
        if (q == p) return 1.0;
        try {
            return std::stod(text.substr(p, q - p));
        } catch (...) {
            return 1.0;
        }
    }

    std::string name_;
};

} // namespace

void register_nodes() {
    static std::atomic<bool> registered{false};
    bool expected = false;
    if (!registered.compare_exchange_strong(expected, true)) return;

    // ---- node types ----
    NodeFactory::instance().register_type("llm_with_prompt",
        [](const std::string& name, const json& config,
           const NodeContext& ctx) -> std::unique_ptr<GraphNode> {
            return std::make_unique<LLMWithPrompt>(name, config, ctx);
        });

    NodeFactory::instance().register_type("score_extract",
        [](const std::string& name, const json& /*config*/,
           const NodeContext& /*ctx*/) -> std::unique_ptr<GraphNode> {
            return std::make_unique<ScoreExtract>(name);
        });

    // ---- conditions ----
    //
    // Ambiguity gate at ouroboros's canonical 0.2 threshold. Reads the
    // `score` channel; returns "ready" when score ≤ 0.2 (proceed to
    // seed/execute), "more" otherwise (stay in interview). Default is
    // "more" when the channel is absent / non-numeric — fail-safe
    // toward more questioning rather than premature commitment.
    ConditionRegistry::instance().register_condition("score_below_0_2",
        [](const GraphState& state) -> std::string {
            try {
                const auto v = state.get("score");
                if (v.is_number()) {
                    return v.template get<double>() <= 0.2 ? "ready" : "more";
                }
            } catch (...) { /* channel missing — fall through */ }
            return "more";
        });

    // Looser variant for less rigorous spec phases (vibe-mode-with-spec
    // etc.). Same shape, threshold 0.5.
    ConditionRegistry::instance().register_condition("score_below_0_5",
        [](const GraphState& state) -> std::string {
            try {
                const auto v = state.get("score");
                if (v.is_number()) {
                    return v.template get<double>() <= 0.5 ? "ready" : "more";
                }
            } catch (...) { /* channel missing — fall through */ }
            return "more";
        });
}

} // namespace neoclaw
