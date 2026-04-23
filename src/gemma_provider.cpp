#include "gemma_provider.h"
#include "ui.h"

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
// See local_provider.cpp for the rationale — this is the HTTP-path twin.
// Kept duplicated so the two backends can drift independently (e.g. when
// a future model ships a native tool-call token, only one side moves).
std::string compact_signature(const json& params) {
    if (!params.is_object() || !params.contains("properties"))
        return std::string();
    std::ostringstream os;
    bool first = true;
    for (auto [name, spec] : params["properties"].items()) {
        if (!first) os << ", ";
        first = false;
        os << name;
        if (spec.is_object() && spec.contains("type"))
            os << ": " << spec["type"].template get<std::string>();
    }
    return os.str();
}

std::string build_tool_protocol(const std::vector<ChatTool>& tools,
                                const std::string& user_system) {
    std::ostringstream os;
    if (!user_system.empty()) os << user_system << "\n\n";
    if (tools.empty()) {
        os << "You have no tools available.";
        return os.str();
    }
    os << "Tools:\n";
    for (const auto& t : tools) {
        os << "- " << t.name << "(" << compact_signature(t.parameters) << ") — "
           << t.description << "\n";
    }
    os << "\nWhen to call a tool:\n"
          "- write_file: ONLY when the user gave an explicit file path to save to.\n"
          "- read_file / grep / glob: ONLY when the user references existing files\n"
          "  or asks about project contents.\n"
          "- bash: ONLY when the user explicitly asks to run a command.\n"
          "\nWhen NOT to call a tool — answer in prose, with code inside markdown\n"
          "```fences``` (never via write_file):\n"
          "- \"write/show/draft/explain me <code>\" without a file path.\n"
          "- Questions, designs, reasoning, math.\n"
          "- Greetings and short follow-ups.\n"
          "\nTo call a tool, reply with ONLY this JSON (no prose, no fences):\n"
       << R"({"tool_call":{"name":"<tool>","arguments":{...}}})" << "\n\n"
          "Tool results arrive as `TOOL_RESULT: ...` user messages. Then call\n"
          "another tool or answer in prose.";
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
// Tolerates up to three missing trailing `}` — see local_provider.cpp
// for the Gemma-4 truncation rationale.
std::optional<std::pair<size_t, size_t>> find_tool_call_block(
    const std::string& text) {
    for (size_t i = 0; i < text.size(); ++i) {
        if (text[i] != '{') continue;
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
                    break;
                }
            }
        }
    }

    const size_t anchor = text.find("{\"tool_call\"");
    if (anchor == std::string::npos) return std::nullopt;
    int depth = 0;
    bool in_str = false, esc = false;
    for (size_t j = anchor; j < text.size(); ++j) {
        char c = text[j];
        if (in_str) {
            if (esc) esc = false;
            else if (c == '\\') esc = true;
            else if (c == '"') in_str = false;
            continue;
        }
        if (c == '"')      in_str = true;
        else if (c == '{') ++depth;
        else if (c == '}') --depth;
    }
    if (depth > 0 && depth <= 3) {
        return std::make_pair(anchor, text.size());
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
        int depth = 0;
        bool in_str = false, esc = false;
        for (char c : body) {
            if (in_str) {
                if (esc) esc = false;
                else if (c == '\\') esc = true;
                else if (c == '"') in_str = false;
                continue;
            }
            if (c == '"')      in_str = true;
            else if (c == '{') ++depth;
            else if (c == '}') --depth;
        }
        std::string fixed = body;
        while (depth-- > 0) fixed += '}';
        auto j = json::parse(fixed);
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

    // 2. Stream through the delegate. Two things happen per chunk:
    //    a) feed into the raw buffer so we can parse tool_call later,
    //    b) strip chat-template artifacts (e.g. `<|im_end|>`, `<end_of_turn>`)
    //       before handing bytes to the user callback, so the user never
    //       sees the model's internal control tokens mid-stream.
    std::string raw_buffered;       ///< Unfiltered — what the model really said.
    std::string filter_carry;       ///< Cross-chunk boundary holdback for strip.
    std::string pending;            ///< Trailing holdback — see below.
    bool        saw_tool_prefix = false;

    // Trailing holdback: always keep the last HOLDBACK bytes in `pending`
    // so a mid-stream `{"tool_call"` opener is fully disambiguated
    // before any byte leaks to stdout. The earlier leading-edge-only
    // version leaked mid-stream tool calls that came after a prose
    // preamble (e.g. "I will write the file. {"tool_call": ...}").
    constexpr size_t HOLDBACK = std::char_traits<char>::length(
        "{\"tool_call\"");  // 13

    auto capture = [&](const std::string& tok) {
        raw_buffered += tok;

        std::string out_tok = tok;
        ui::strip_chat_artifacts(out_tok, filter_carry);

        if (saw_tool_prefix) return;

        if (raw_buffered.find("{\"tool_call\"") != std::string::npos) {
            saw_tool_prefix = true;
            pending.clear();
            return;
        }

        pending += out_tok;
        if (pending.size() > HOLDBACK) {
            const size_t safe = pending.size() - HOLDBACK;
            if (on_chunk && safe > 0) on_chunk(pending.substr(0, safe));
            pending.erase(0, safe);
        }
    };

    auto resp = delegate_->complete_stream(p, capture);
    // Flush the holdback tail — if the stream ended without a match it's
    // all normal prose.
    if (!saw_tool_prefix && on_chunk && !pending.empty()) {
        on_chunk(pending);
        pending.clear();
    }
    if (!saw_tool_prefix && on_chunk && !filter_carry.empty()) {
        std::string tail = filter_carry;
        filter_carry.clear();
        ui::strip_chat_artifacts(tail, filter_carry);
        if (!tail.empty()) on_chunk(tail);
    }

    // Delegate builds resp.message.content from chunks too, but its
    // parser may have already stripped things. Prefer our raw buffer.
    const std::string& raw = raw_buffered.empty() ? resp.message.content
                                                  : raw_buffered;

    // 3. Look for a tool-call JSON block.
    if (auto span = find_tool_call_block(raw)) {
        auto body = raw.substr(span->first, span->second - span->first);
        if (auto tc = parse_tool_call(body)) {
            resp.message.tool_calls.clear();
            resp.message.tool_calls.push_back(*tc);

            std::string pre = raw.substr(0, span->first);
            std::string carry;
            ui::strip_chat_artifacts(pre, carry);
            resp.message.content = pre;

            std::string args_preview = tc->arguments;
            if (args_preview.size() > 80) args_preview.resize(80);
            ui::print_tool_start(tc->name, args_preview);
            return resp;
        }
    }

    // Same handling as local_provider: if a tool_call opener was seen
    // but the JSON didn't parse, return an empty turn so the garbled
    // bytes don't contaminate the conversation history.
    if (saw_tool_prefix) {
        std::fprintf(stderr,
            "[neoclaw] warning: model started a tool_call but emitted "
            "unparseable JSON (%zu bytes); turn dropped.\n",
            raw.size());
        resp.message.content = "";
        return resp;
    }

    std::string clean = raw;
    std::string carry;
    ui::strip_chat_artifacts(clean, carry);
    resp.message.content = clean;
    return resp;
}

} // namespace neoclaw
