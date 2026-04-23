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

std::string build_tool_protocol(const std::vector<ChatTool>& tools,
                                const std::string& user_system) {
    std::ostringstream os;
    if (!user_system.empty()) os << user_system << "\n\n";
    os << "## Tool protocol\n\n";
    if (tools.empty()) {
        os << "You have no tools available.";
        return os.str();
    }
    os << "You have access to the following tools:\n\n";
    for (const auto& t : tools) {
        os << "### " << t.name << "\n" << t.description << "\n"
           << "Parameters (JSON Schema): " << t.parameters.dump() << "\n\n";
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

// Balanced-brace scan for {"tool_call": ...}, identical semantics to
// the http-path parser.
std::optional<std::pair<size_t, size_t>> find_tool_call_block(
    const std::string& text) {
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
    return std::nullopt;
}

std::optional<ToolCall> parse_tool_call(const std::string& body) {
    try {
        auto j = json::parse(body);
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

    std::string raw_buffered, filter_carry;
    bool saw_tool_prefix = false;

    gc.streamer = [&](const std::string& tok) {
        raw_buffered += tok;

        std::string out_tok = tok;
        ui::strip_chat_artifacts(out_tok, filter_carry);

        if (!saw_tool_prefix) {
            if (raw_buffered.find("{\"tool_call\"") != std::string::npos)
                saw_tool_prefix = true;
            if (on_chunk && !out_tok.empty() && !saw_tool_prefix)
                on_chunk(out_tok);
        }
    };

    try {
        (void)model_->generate(input, gc);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("local inference failed: ") + e.what());
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
