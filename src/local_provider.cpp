#include "local_provider.h"
#include "ui.h"

#include <transformercpp/generation.h>
#include <transformercpp/hub.h>
#include <transformercpp/model.h>
#include <transformercpp/tokenizer.h>

// llama_log_set lives here. We include the C header directly (not
// through a TransformerCPP wrapper) so we can silence llama.cpp's
// chatty per-load diagnostics.
#include <llama.h>

#include <neograph/json.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace neoclaw {

// =======================================================================
// Tool-protocol system prompt + message rewriting.
//
// Identical contract to gemma_provider.cpp: the model is told about
// each tool and the strict JSON shape it must emit. Role="tool" turns
// get folded into "TOOL_RESULT: ..." user messages because the GGUF
// chat templates (Gemma, Llama, Qwen) don't carry a dedicated tool
// slot. Duplicated rather than shared with gemma_provider.cpp because
// that one lives under the http path; keeping them independent keeps
// the two providers free to diverge as the prompt format evolves
// (e.g. native tool-call tokens in a future Gemma).
// =======================================================================
namespace {

using neograph::ChatCompletion;
using neograph::ChatMessage;
using neograph::ChatTool;
using neograph::CompletionParams;
using neograph::StreamCallback;
using neograph::ToolCall;
using neograph::json;

// Render a JSON Schema parameter block as a compact signature string.
// e.g. `path: str, content: str` — no schema verbosity, just names+types.
// This shaves the tool-injection prompt from ~520 tokens to ~120 tokens
// for the default 4-tool set, which is a 4x prefill reduction. Most
// instruction-tuned models (Gemma, Qwen, Llama) infer arg types from
// the signature just fine.
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

// Gemma / Llama / Qwen chat template fed through an in-process model
// doesn't have a `role:"tool"` slot. Fold tool results into user turns
// with a "TOOL_RESULT:" prefix; keep assistant tool-call turns as
// content so the transcript stays self-describing on resume.
std::vector<ChatMessage> rewrite_for_local(
    const std::vector<ChatMessage>& in,
    const std::vector<ChatTool>&    tools,
    const std::string&              user_system) {

    std::vector<ChatMessage> out;
    out.reserve(in.size() + 1);

    std::string caller_system = user_system;
    bool inserted = false;

    for (const auto& m : in) {
        if (m.role == "system") {
            caller_system = caller_system.empty()
                ? m.content
                : caller_system + "\n\n" + m.content;
            continue;
        }
        if (!inserted) {
            ChatMessage sys;
            sys.role = "system";
            sys.content = build_tool_protocol(tools, caller_system);
            out.push_back(std::move(sys));
            inserted = true;
        }
        if (m.role == "tool") {
            ChatMessage u;
            u.role = "user";
            u.content = "TOOL_RESULT: " + m.content;
            out.push_back(std::move(u));
        } else if (m.role == "assistant" && !m.tool_calls.empty()) {
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
    if (!inserted) {
        ChatMessage sys;
        sys.role = "system";
        sys.content = build_tool_protocol(tools, caller_system);
        out.insert(out.begin(), std::move(sys));
    }
    return out;
}

// Render a message list into a Gemma-style chat prompt. Real
// production would pull the template out of the GGUF metadata
// (`tokenizer.chat_template`). For v0.2 the hard-coded Gemma shape
// is close enough for every instruction-tuned model we've tested.
std::string render_chat(const std::vector<ChatMessage>& msgs) {
    std::ostringstream os;
    os << "<bos>";
    for (const auto& m : msgs) {
        const std::string role =
            (m.role == "system")    ? "system" :
            (m.role == "assistant") ? "model"  : "user";
        os << "<start_of_turn>" << role << "\n"
           << m.content
           << "<end_of_turn>\n";
    }
    os << "<start_of_turn>model\n";
    return os.str();
}

// Balanced-brace scan for {"tool_call": ...}. Tolerates up to three
// missing trailing `}` because Gemma-4 E4B (and similar smaller models)
// routinely emit their turn-stop sequence (`<end_of_turn>`) one brace
// early — the JSON opens `{{{ … }}` (three opens, two closes) and the
// stop sequence fires before the model can type the outer `}`. When we
// detect that shape we treat the tail as implicitly closed; json::parse
// below operates on a synthesised balanced slice.
std::optional<std::pair<size_t, size_t>> find_tool_call_block(
    const std::string& text) {
    // Standard strict scan first — covers happy-path models (GPT-4,
    // Claude, Qwen-Coder) that ship well-formed JSON.
    for (size_t i = 0; i < text.size(); ++i) {
        if (text[i] != '{') continue;
        int depth = 0;
        bool in_str = false, esc = false;
        for (size_t j = i; j < text.size(); ++j) {
            char c = text[j];
            if (in_str) {
                if (esc) esc = false;
                else if (c == '\\') esc = true;
                else if (c == '"') in_str = false;
                continue;
            }
            if (c == '"')      in_str = true;
            else if (c == '{') ++depth;
            else if (c == '}') {
                --depth;
                if (depth == 0) {
                    auto slice = text.substr(i, j - i + 1);
                    if (slice.find("\"tool_call\"") != std::string::npos)
                        return std::make_pair(i, j + 1);
                    break;
                }
            }
        }
    }

    // Lenient pass: find the first `{"tool_call"` and measure brace
    // imbalance through end of text. If we're short by 1-3 `}`, return
    // a span that covers the text plus a note that the caller will
    // append the missing braces before JSON-parsing.
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
        // Return the full remaining tail; caller will close braces.
        return std::make_pair(anchor, text.size());
    }
    return std::nullopt;
}

std::optional<ToolCall> parse_tool_call(const std::string& body) {
    try {
        // Count brace imbalance and pad with trailing `}` if needed.
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
        out.id   = "call_local_0";
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

// Pretty-print progress during HubClient::download. We reserve one
// line on stderr and repaint with a \r so the user sees "downloading
// X% of Y MB" without spamming the scrollback.
struct DownloadProgress {
    std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
    void operator()(size_t got, size_t total) {
        const auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(
                now - last).count() < 200 && got != total) return;
        last = now;
        const double mb   = got   / (1024.0 * 1024.0);
        const double tmb  = total / (1024.0 * 1024.0);
        if (total > 0) {
            const double pct = 100.0 * got / total;
            std::fprintf(stderr,
                "\r[neoclaw] downloading model: %6.2f%% (%7.1f / %7.1f MB)  ",
                pct, mb, tmb);
        } else {
            std::fprintf(stderr,
                "\r[neoclaw] downloading model: %7.1f MB  ", mb);
        }
        std::fflush(stderr);
        if (got == total) std::fputc('\n', stderr);
    }
};

} // namespace

// =======================================================================
// LocalProvider
// =======================================================================
LocalProvider::LocalProvider(std::shared_ptr<transformercpp::Model> model,
                             Config cfg)
    : model_(std::move(model)), cfg_(std::move(cfg)) {}

ChatCompletion LocalProvider::complete(const CompletionParams& params) {
    return complete_stream(params, nullptr);
}

ChatCompletion LocalProvider::complete_stream(
    const CompletionParams&       params_in,
    const StreamCallback&         on_chunk) {

    auto msgs = rewrite_for_local(
        params_in.messages, params_in.tools, /*user_system=*/"");
    std::string prompt = render_chat(msgs);
    auto input = model_->tokenizer().encode(prompt);

    transformercpp::GenerationConfig gc;
    gc.max_new_tokens = (params_in.max_tokens > 0) ? params_in.max_tokens
                                                    : cfg_.max_tokens;
    gc.temperature    = (params_in.temperature > 0) ? params_in.temperature
                                                     : cfg_.temperature;
    gc.stop_sequences = {"<end_of_turn>", "<eos>"};

    std::string raw_buffered, filter_carry, pending;
    bool saw_tool_prefix = false;

    // Trailing holdback: always keep the last HOLDBACK bytes in `pending`
    // and never flush them to stdout yet — they might grow into a
    // `{"tool_call"` match on the next chunk. When the accumulated buffer
    // completes the match, `pending` is dropped entirely (the partial
    // JSON that leaked the earlier 12 chars through a naïve approach is
    // safely held here). If generation ends without a match, the tail
    // flushes cleanly. Covers both leading-edge and mid-stream tool
    // calls with identical code.
    constexpr size_t HOLDBACK = std::char_traits<char>::length(
        "{\"tool_call\"");  // 13

    gc.streamer = [&](const std::string& tok) {
        raw_buffered += tok;

        std::string out_tok = tok;
        ui::strip_chat_artifacts(out_tok, filter_carry);

        if (saw_tool_prefix) return;

        if (raw_buffered.find("{\"tool_call\"") != std::string::npos) {
            saw_tool_prefix = true;
            pending.clear();   // whatever was in flight is the JSON opener
            return;
        }

        pending += out_tok;
        if (pending.size() > HOLDBACK) {
            const size_t safe = pending.size() - HOLDBACK;
            if (on_chunk && safe > 0) on_chunk(pending.substr(0, safe));
            pending.erase(0, safe);
        }
    };

    try {
        (void)model_->generate(input, gc);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("local inference failed: ") + e.what());
    }

    // Flush the trailing holdback buffer. If the stream ended without
    // a tool-call match, everything in `pending` is normal prose.
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

    ChatCompletion resp;
    resp.message.role = "assistant";

    if (auto span = find_tool_call_block(raw_buffered)) {
        auto body = raw_buffered.substr(span->first, span->second - span->first);
        if (auto tc = parse_tool_call(body)) {
            resp.message.tool_calls.push_back(*tc);
            std::string pre = raw_buffered.substr(0, span->first);
            std::string carry;
            ui::strip_chat_artifacts(pre, carry);
            resp.message.content = pre;

            std::string args_preview = tc->arguments;
            if (args_preview.size() > 80) args_preview.resize(80);
            ui::print_tool_start(tc->name, args_preview);
            return resp;
        }
    }

    // If we saw a `{"tool_call"` opener but couldn't find a balanced /
    // parseable JSON block, the model started emitting a tool call and
    // then botched it (truncated mid-JSON, spurious stop-sequence, etc.).
    // Return an empty turn rather than bleeding the garbled raw bytes
    // into the conversation history — otherwise the next turn shows the
    // model hallucinating a "yes I wrote the file" narrative based on
    // its own malformed prior output. Also log to stderr so the user
    // knows their request actually did something, it just crashed on
    // the way out.
    if (saw_tool_prefix) {
        std::fprintf(stderr,
            "[neoclaw] warning: model started a tool_call but emitted "
            "unparseable JSON (%zu bytes); turn dropped.\n",
            raw_buffered.size());
        if (const char* dump_env = std::getenv("NEOCLAW_DUMP_BAD_JSON");
            dump_env && *dump_env && *dump_env != '0') {
            std::fprintf(stderr, "----- raw model output -----\n%s\n----- end -----\n",
                         raw_buffered.c_str());
        }
        resp.message.content = "";
        return resp;
    }

    std::string clean = raw_buffered;
    std::string carry;
    ui::strip_chat_artifacts(clean, carry);
    resp.message.content = clean;
    return resp;
}

// =======================================================================
// resolve_model + load_model
// =======================================================================
std::string resolve_model(const std::string& model_id,
                          const std::string& filename) {
    // If the caller passed a path (absolute or obvious file) and it
    // exists, treat it as local.
    fs::path as_path(model_id);
    if (fs::exists(as_path) && fs::is_regular_file(as_path)) {
        return fs::canonical(as_path).string();
    }

    // HubClient download — checks its own cache first.
    DownloadProgress progress;
    try {
        if (filename.empty()) {
            return transformercpp::HubClient::download_best_gguf(
                model_id, std::ref(progress));
        }
        return transformercpp::HubClient::download(
            model_id, filename, std::ref(progress));
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string(
            "neoclaw: failed to fetch '") + model_id
            + (filename.empty() ? "" : ("/" + filename))
            + "': " + e.what());
    }
}

// Silence llama.cpp's per-load spam. We keep GGML_LOG_LEVEL_ERROR
// passthrough so real failures still surface; everything below error
// gets dropped. Set NEOCLAW_LLAMA_VERBOSE=1 to restore the firehose.
static void llama_log_silent(ggml_log_level level, const char* text, void*) {
    static const bool verbose = [] {
        if (const char* v = std::getenv("NEOCLAW_LLAMA_VERBOSE"))
            return v[0] != '\0' && v[0] != '0';
        return false;
    }();
    if (verbose || level >= GGML_LOG_LEVEL_ERROR) {
        std::fputs(text, stderr);
    }
}

std::shared_ptr<transformercpp::Model> load_model(const std::string& path) {
    // Register the quiet log callback BEFORE AutoModel::from_pretrained
    // so the first-load tensor manifest dump gets muted.
    llama_log_set(llama_log_silent, nullptr);

    std::cerr << "[neoclaw] loading model...";
    std::cerr.flush();
    const auto t0 = std::chrono::steady_clock::now();
    auto uptr = transformercpp::AutoModel::from_pretrained(path);
    const long ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();
    std::cerr << " done (" << ms << " ms)\n";
    return std::shared_ptr<transformercpp::Model>(uptr.release());
}

} // namespace neoclaw
