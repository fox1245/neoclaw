#include "gemma_provider.h"

#include <neograph/json.h>

#include <algorithm>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace neoclaw {

namespace {

using neograph::ChatCompletion;
using neograph::ChatMessage;
using neograph::ChatTool;
using neograph::CompletionParams;
using neograph::StreamCallback;
using neograph::ToolCall;
using neograph::json;

// ----------------------------------------------------------------------
// Tool-protocol system prompt.
//
// The model sees one system message listing every tool, and a strict
// contract for emitting a single-turn tool call. On the wire, tool
// results come back as role="user" messages prefixed with "TOOL_RESULT:"
// because most public GGUF chat templates (Gemma, Llama, Qwen) don't
// distinguish a `role:"tool"` slot.
// ----------------------------------------------------------------------
std::string build_tool_protocol(const std::vector<ChatTool>& tools,
                                const std::string& user_system) {
    std::ostringstream os;
    if (!user_system.empty()) {
        os << user_system << "\n\n";
    }
    os << "## Tool protocol\n\n";
    if (tools.empty()) {
        os << "You have no tools available.";
        return os.str();
    }

    os << "You have access to the following tools:\n\n";
    for (const auto& t : tools) {
        os << "### " << t.name << "\n";
        os << t.description << "\n";
        os << "Parameters (JSON Schema): " << t.parameters.dump() << "\n\n";
    }

    os << "When you decide to call a tool, respond with ONLY a single JSON "
          "object in this exact shape, with no prose, no markdown fences, "
          "no explanation before or after:\n\n"
       << R"({"tool_call": {"name": "<tool_name>", "arguments": {<args>}}})"
       << "\n\n"
          "A tool result will come back as a user message prefixed with "
          "`TOOL_RESULT:`. Read it, then either call another tool or "
          "respond to the original user question in plain prose. Never "
          "emit a tool call for conversational or code-generation requests "
          "that don't actually need a tool.";
    return os.str();
}

// ----------------------------------------------------------------------
// Message rewriting. Gemma / Llama / Qwen chat templates don't have a
// role:"tool" slot — fold those into user messages with a "TOOL_RESULT:"
// prefix. Assistant messages that carried tool_calls get their call
// re-serialised into content so the next turn has context.
// ----------------------------------------------------------------------
std::vector<ChatMessage> rewrite_for_gemma(
    const std::vector<ChatMessage>& in,
    const std::vector<ChatTool>&    tools,
    const std::string&              user_system) {

    std::vector<ChatMessage> out;
    out.reserve(in.size() + 1);

    // Extract any caller-supplied system message and fold it into the
    // tool protocol — we always want exactly one system message at the
    // top, carrying the tool contract.
    std::string caller_system = user_system;
    bool inserted_system = false;

    for (const auto& m : in) {
        if (m.role == "system") {
            // Merge: user_system (from CompletionParams/Agent) + the caller's.
            caller_system = caller_system.empty()
                ? m.content
                : caller_system + "\n\n" + m.content;
            continue;
        }

        if (!inserted_system) {
            ChatMessage sys;
            sys.role = "system";
            sys.content = build_tool_protocol(tools, caller_system);
            out.push_back(std::move(sys));
            inserted_system = true;
        }

        if (m.role == "tool") {
            ChatMessage u;
            u.role = "user";
            u.content = "TOOL_RESULT: " + m.content;
            out.push_back(std::move(u));
        } else if (m.role == "assistant" && !m.tool_calls.empty()) {
            // Re-serialise the tool call into content so the model can
            // see its own prior action in the transcript.
            ChatMessage a;
            a.role = "assistant";
            const auto& tc = m.tool_calls.front();
            json body;
            body["tool_call"]["name"]      = tc.name;
            body["tool_call"]["arguments"] = json::parse(
                tc.arguments.empty() ? std::string("{}") : tc.arguments);
            a.content = body.dump();
            out.push_back(std::move(a));
        } else {
            out.push_back(m);
        }
    }

    if (!inserted_system) {
        ChatMessage sys;
        sys.role = "system";
        sys.content = build_tool_protocol(tools, caller_system);
        out.insert(out.begin(), std::move(sys));
    }

    return out;
}

// ----------------------------------------------------------------------
// Find the first balanced JSON object in `text` whose body contains the
// substring `"tool_call"`. Returns (start, end) offsets with end
// exclusive; `std::nullopt` if no match.
// ----------------------------------------------------------------------
std::optional<std::pair<size_t, size_t>> find_tool_call_block(
    const std::string& text) {

    for (size_t i = 0; i < text.size(); ++i) {
        if (text[i] != '{') continue;
        // Scan the balanced object starting at i.
        int depth = 0;
        bool in_string = false;
        bool escape    = false;
        for (size_t j = i; j < text.size(); ++j) {
            char c = text[j];
            if (in_string) {
                if (escape) escape = false;
                else if (c == '\\') escape = true;
                else if (c == '"') in_string = false;
                continue;
            }
            if (c == '"') in_string = true;
            else if (c == '{') ++depth;
            else if (c == '}') {
                --depth;
                if (depth == 0) {
                    const auto slice = text.substr(i, j - i + 1);
                    if (slice.find("\"tool_call\"") != std::string::npos) {
                        return std::make_pair(i, j + 1);
                    }
                    break; // non-match — try the next top-level `{`
                }
            }
        }
    }
    return std::nullopt;
}

// ----------------------------------------------------------------------
// Parse `{"tool_call": {"name": "...", "arguments": {...}}}` into a
// NeoGraph ToolCall. Returns std::nullopt on any error (malformed JSON,
// missing fields). The parser is intentionally strict — we'd rather
// treat a broken tool call as plain prose than misdispatch.
// ----------------------------------------------------------------------
std::optional<ToolCall> parse_tool_call(const std::string& body) {
    try {
        auto j = json::parse(body);
        if (!j.contains("tool_call")) return std::nullopt;
        auto tc = j["tool_call"];
        if (!tc.contains("name")) return std::nullopt;

        ToolCall out;
        out.id   = "call_gemma_0";
        out.name = tc["name"].template get<std::string>();
        if (tc.contains("arguments")) {
            auto args = tc["arguments"];
            out.arguments = args.is_string()
                ? args.template get<std::string>()
                : args.dump();
        } else {
            out.arguments = "{}";
        }
        return out;
    } catch (...) {
        return std::nullopt;
    }
}

} // namespace

// -----------------------------------------------------------------------
// GemmaProvider implementation.
// -----------------------------------------------------------------------
GemmaProvider::GemmaProvider(Config cfg) : cfg_(std::move(cfg)) {
    neograph::llm::OpenAIProvider::Config oc;
    oc.api_key       = "neoclaw-local";
    oc.base_url      = cfg_.endpoint;
    oc.default_model = cfg_.model_name;
    delegate_ = neograph::llm::OpenAIProvider::create(oc);
}

ChatCompletion GemmaProvider::complete(const CompletionParams& params) {
    return complete_stream(params, nullptr);
}

ChatCompletion GemmaProvider::complete_stream(
    const CompletionParams&       params_in,
    const StreamCallback&         on_chunk) {

    // 1. Rewrite messages + inject tool-protocol system prompt.
    CompletionParams p;
    p.model        = cfg_.model_name;
    p.temperature  = (params_in.temperature > 0) ? params_in.temperature
                                                  : cfg_.temperature;
    p.max_tokens   = (params_in.max_tokens > 0) ? params_in.max_tokens
                                                 : cfg_.max_tokens;
    // We deliberately drop params_in.tools from the wire — the server
    // ignores it and the model sees tools only via our injected prompt.
    p.messages = rewrite_for_gemma(
        params_in.messages, params_in.tools,
        /*user_system=*/""  // Agent already folded its system prompt
                             // into params_in.messages[0] by the time we
                             // get here; rewrite_for_gemma merges it.
    );

    // 2. Stream through the delegate, accumulating tokens so we can
    //    inspect the full response for a tool-call JSON block.
    std::string buffered;
    auto capture = [&](const std::string& tok) {
        buffered += tok;
        if (on_chunk) on_chunk(tok);
    };
    auto resp = delegate_->complete_stream(p, capture);

    // Delegate builds resp.message.content from the chunks too, but its
    // parser may have already stripped control tokens. Prefer our own
    // buffered text — it's what the model actually said.
    const std::string& raw = buffered.empty() ? resp.message.content : buffered;

    // 3. Look for a tool-call JSON block. Accept it only when it's the
    //    dominant shape of the response (first substantive JSON object).
    if (auto span = find_tool_call_block(raw)) {
        auto body = raw.substr(span->first, span->second - span->first);
        if (auto tc = parse_tool_call(body)) {
            resp.message.tool_calls.clear();
            resp.message.tool_calls.push_back(*tc);
            // Keep any prose that preceded the JSON — useful when the
            // model narrated before the call. Strip the JSON itself.
            std::string pre = raw.substr(0, span->first);
            resp.message.content = pre;
            return resp;
        }
    }

    resp.message.content = raw;
    return resp;
}

} // namespace neoclaw
