// neoclaw/src/main.cpp — REPL loop.
//
// Reads one user turn at a time from stdin, drives the NeoGraph Agent,
// streams tokens back, shows tool invocations inline. Ctrl-D (EOF) or
// `/quit` exits. `/help` prints commands, `/run <cmd>` short-circuits
// the LLM and runs a shell command under the sandbox directly.

#include "config.h"
#include "gemma_provider.h"
#include "local_provider.h"
#include "neoclaw_nodes.h"
#include "sandbox.h"
#include "tools.h"
#include "topology.h"
#include "ui.h"

#if NEOCLAW_HAVE_TUI
#  include "mode_picker.h"
#endif

#include <neograph/graph/engine.h>
#include <neograph/llm/agent.h>
#include <neograph/tool.h>

#include <algorithm>
#include <cctype>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct CliArgs {
    fs::path    config_path;        ///< empty = discovery
    fs::path    project_root;       ///< empty = config value
    bool        no_sandbox = false; ///< override tools.bash.sandbox to "none"
};

CliArgs parse_cli(int argc, char** argv) {
    CliArgs a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if ((s == "--config" || s == "-c") && i + 1 < argc) {
            a.config_path = argv[++i];
        } else if ((s == "--project-root" || s == "-p") && i + 1 < argc) {
            a.project_root = argv[++i];
        } else if (s == "--no-sandbox") {
            a.no_sandbox = true;
        } else if (s == "--help" || s == "-h") {
            std::cout
              << "neoclaw — local C++ coding agent\n\n"
              << "Usage: neoclaw [options]\n"
              << "  -c, --config <file>       Path to YAML config (default: discover)\n"
              << "  -p, --project-root <dir>  Override session.project_root\n"
              << "      --no-sandbox          Disable bwrap for bash tool (DANGEROUS)\n"
              << "  -h, --help                Show this help and exit\n\n"
              << "Config discovery order:\n"
              << "  $NEOCLAW_CONFIG, ./neoclaw.yaml,\n"
              << "  $XDG_CONFIG_HOME/neoclaw/config.yaml, ~/.config/neoclaw/config.yaml\n";
            std::exit(0);
        } else {
            std::cerr << "neoclaw: unknown argument '" << s
                      << "', use --help\n";
            std::exit(2);
        }
    }
    return a;
}

// Simple y/N prompt. Defaults to N on EOF or empty input.
bool confirm_yn(const std::string& prompt) {
    std::cout << prompt << " [y/N] " << std::flush;
    std::string line;
    if (!std::getline(std::cin, line)) return false;
    if (line.empty()) return false;
    return line[0] == 'y' || line[0] == 'Y';
}

void print_banner(const neoclaw::Config& cfg) {
    neoclaw::ui::BannerLines b;
    b.title    = "neoclaw v0.5.0 - local C++ coding agent";
    b.model    = cfg.model.id;
    b.endpoint = (cfg.backend == neoclaw::BackendKind::Local)
                     ? std::string("in-process (llama.cpp)")
                     : cfg.server.endpoint;
    b.project  = cfg.session.project_root.string();
    b.bash_line = std::string(cfg.tools.bash.enabled ? "enabled" : "disabled")
                + " (sandbox=" + cfg.tools.bash.sandbox + ")";
    b.hint     = "type /help for commands, Ctrl-D or /quit to exit";
    neoclaw::ui::print_banner(b);
}

} // namespace

int main(int argc, char** argv) {
    // Enable the user's locale so wcwidth() actually knows how wide a
    // Korean / em-dash / emoji codepoint is when painting the banner.
    // Fall back to C.UTF-8 when the environment doesn't set LANG — a
    // plain "C" locale rejects multi-byte UTF-8 in mbrtowc() and every
    // em-dash in the banner prints as "???" + a broken width.
    const char* picked = std::setlocale(LC_ALL, "");
    auto is_utf8 = [](const char* l) {
        if (!l) return false;
        std::string s = l;
        for (auto& c : s) c = (char)std::tolower((unsigned char)c);
        return s.find("utf-8") != std::string::npos
            || s.find("utf8")  != std::string::npos;
    };
    if (!is_utf8(picked)) {
        if (!std::setlocale(LC_ALL, "C.UTF-8")) {
            std::setlocale(LC_ALL, "en_US.UTF-8");
        }
    }

    // Register neoclaw-specific node types + conditions with NeoGraph's
    // process-wide registries. Idempotent and cheap. Lets bundled
    // topologies reference `llm_with_prompt`, `score_extract`,
    // `score_below_0_2`, etc. without per-engine setup.
    neoclaw::register_nodes();

    CliArgs cli = parse_cli(argc, argv);

    // -----------------------------------------------------------------
    // 1. Load configuration.
    // -----------------------------------------------------------------
    neoclaw::Config cfg;
    try {
        cfg = cli.config_path.empty()
            ? neoclaw::load_config_from_discovery(cli.project_root)
            : neoclaw::load_config(cli.config_path);
    } catch (const std::exception& e) {
        std::cerr << "neoclaw: " << e.what() << "\n";
        return 2;
    }

    if (!cli.project_root.empty()) {
        std::error_code ec;
        cfg.session.project_root = fs::weakly_canonical(cli.project_root, ec);
        if (ec) {
            std::cerr << "neoclaw: bad --project-root: " << ec.message() << "\n";
            return 2;
        }
    }
    if (cli.no_sandbox) cfg.tools.bash.sandbox = "none";

    if (!fs::exists(cfg.session.project_root)) {
        std::cerr << "neoclaw: project root does not exist: "
                  << cfg.session.project_root << "\n";
        return 2;
    }

    // -----------------------------------------------------------------
    // 1b. Interactive mode picker (FTXUI radiobox).
    //
    // When `topology:` is unset in YAML AND we're attached to a real
    // TTY, pop a one-screen picker so a fresh user doesn't have to know
    // about `topology:` / file paths to choose a persona. Esc / Ctrl-C
    // / the explicit "Agent default" row all leave cfg.topology empty,
    // preserving the v0.5 default behaviour for everyone who skips.
    //
    // CI / pipes / `--config foo.yaml` with topology pre-set: no UI.
    // -----------------------------------------------------------------
#if NEOCLAW_HAVE_TUI
    if (cfg.topology.empty() && neoclaw::is_picker_runnable()) {
        const auto dir = neoclaw::default_topologies_dir();
        if (!dir.empty()) {
            auto picked = neoclaw::pick_mode_tui(dir);
            if (!picked.skipped) {
                cfg.topology = picked.topology_path;
                std::cerr << "[neoclaw] picked: " << picked.topology_name << "\n";
            }
        }
    }
#endif

    print_banner(cfg);

    // -----------------------------------------------------------------
    // 2. Build tools per config.
    //
    // Wrapped in a closure because `/mode <name>` (later in the REPL)
    // needs to recreate tools when swapping topologies — the engine
    // takes ownership of the tool unique_ptrs at compile() time, so a
    // fresh batch is required for each engine instance.
    // -----------------------------------------------------------------
    auto make_tools = [&]() -> std::vector<std::unique_ptr<neograph::Tool>> {
        std::vector<std::unique_ptr<neograph::Tool>> ts;
        if (cfg.tools.read_file) {
            ts.push_back(std::make_unique<neoclaw::ReadFileTool>(
                cfg.session.project_root));
        }
        if (cfg.tools.write_file) {
            auto confirm = [](const std::string& rel, const std::string& content) {
                std::cout << "\n[neoclaw] assistant wants to write "
                          << content.size() << " bytes to " << rel << "\n";
                std::istringstream is(content);
                std::string ln;
                int shown = 0;
                while (std::getline(is, ln) && shown < 8) {
                    std::cout << "  | " << ln << "\n";
                    ++shown;
                }
                if (shown == 8) std::cout << "  | ...\n";
                return confirm_yn("[neoclaw] write this file?");
            };
            ts.push_back(std::make_unique<neoclaw::WriteFileTool>(
                cfg.session.project_root, confirm));
        }
        if (cfg.tools.grep) {
            ts.push_back(std::make_unique<neoclaw::GrepTool>(
                cfg.session.project_root));
        }
        if (cfg.tools.glob) {
            ts.push_back(std::make_unique<neoclaw::GlobTool>(
                cfg.session.project_root));
        }
        if (cfg.tools.bash.enabled) {
            neoclaw::SandboxOptions sbx;
            sbx.project_root  = cfg.session.project_root;
            sbx.mode          = neoclaw::parse_sandbox_mode(cfg.tools.bash.sandbox);
            sbx.timeout_sec   = cfg.tools.bash.timeout_sec;
            sbx.allow_network = cfg.tools.bash.allow_network;
            auto confirm = [](const std::string& cmd) {
                std::cout << "\n[neoclaw] assistant wants to run: " << cmd << "\n";
                return confirm_yn("[neoclaw] run this command?");
            };
            ts.push_back(std::make_unique<neoclaw::BashTool>(
                std::move(sbx), confirm));
        }
        return ts;
    };

    auto tools = make_tools();
    std::cout << "[neoclaw] " << tools.size() << " tool"
              << (tools.size() == 1 ? "" : "s") << " active: ";
    for (size_t i = 0; i < tools.size(); ++i) {
        std::cout << tools[i]->get_name()
                  << (i + 1 < tools.size() ? ", " : "\n\n");
    }
    if (tools.empty()) std::cout << "(none)\n\n";

    // -----------------------------------------------------------------
    // 3. Provider + Agent.
    //    Local backend: resolve + download GGUF on first run, then
    //    load the model into this process.
    //    Http backend:  point a light adapter at an external server.
    // -----------------------------------------------------------------
    std::shared_ptr<neograph::Provider> provider;
    try {
        if (cfg.backend == neoclaw::BackendKind::Local) {
            std::string path = neoclaw::resolve_model(
                cfg.model.id, cfg.model.filename);
            auto model = neoclaw::load_model(path);
            neoclaw::LocalProvider::Config lcfg;
            provider = std::make_shared<neoclaw::LocalProvider>(model, lcfg);
        } else {
            neoclaw::GemmaProvider::Config pcfg;
            pcfg.endpoint = cfg.server.endpoint;
            provider = std::make_shared<neoclaw::GemmaProvider>(pcfg);
        }
    } catch (const std::exception& e) {
        std::cerr << "[neoclaw] provider init failed: " << e.what() << "\n";
        return 3;
    }

    // Two execution paths share this REPL:
    //
    //   `agent`   — the v0.4 default. NeoGraph's pre-baked Agent ReAct
    //               loop. Conversation lives in `agent_conv`.
    //   `engine`  — JSON topology path (v0.5+). Same model + same tools,
    //               different orchestration per `cfg.topology` file.
    //               Conversation lives in `topology_conv` (json array).
    //
    // Exactly one of {agent, engine} is constructed per process.
    std::unique_ptr<neograph::llm::Agent>          agent;
    std::unique_ptr<neograph::graph::GraphEngine>  engine;
    std::vector<neograph::ChatMessage>             agent_conv;
    neograph::json                                  topology_conv = neograph::json::array();

    try {
        if (cfg.topology.empty()) {
            agent = std::make_unique<neograph::llm::Agent>(
                provider, std::move(tools), cfg.agent.system_prompt);
        } else {
            const auto path = neoclaw::resolve_topology_path(cfg.topology);
            std::cerr << "[neoclaw] topology: " << path.string() << "\n";
            engine = neoclaw::compile_topology(
                path, provider, std::move(tools), cfg.agent.system_prompt);
        }
    } catch (const std::exception& e) {
        std::cerr << "[neoclaw] orchestration init failed: " << e.what() << "\n";
        return 4;
    }

    // -----------------------------------------------------------------
    // 4. REPL.
    // -----------------------------------------------------------------
    std::string line;
    while (true) {
        std::cout << neoclaw::ui::fg_green() << neoclaw::ui::bold()
                  << "❯ " << neoclaw::ui::reset() << std::flush;
        if (!std::getline(std::cin, line)) { std::cout << "\n"; break; }
        if (line.empty()) continue;

        if (line == "/quit" || line == "/exit") break;
        if (line == "/help") {
            std::cout
              << "  /help              Show this help\n"
              << "  /quit              Exit (same as Ctrl-D)\n"
              << "  /reset             Clear conversation history\n"
              << "  /run <shell cmd>   Run a shell command directly (bypass LLM)\n"
              << "  /paste             Multi-line input mode — paste, then `/end` on its own line\n"
              << "  /mode <name>       Swap topology at runtime (e.g. `/mode pair`,\n"
              << "                     `/mode code-review-3pass`). `/mode default` returns to\n"
              << "                     the v0.4 hardcoded Agent ReAct loop. /reset is implied.\n"
              << "\n";
            continue;
        }
        if (line == "/paste") {
            std::cout << "[neoclaw] paste mode — end with `/end` on its own line\n";
            std::ostringstream multi;
            std::string ml;
            bool first = true;
            while (std::getline(std::cin, ml)) {
                if (ml == "/end") break;
                if (!first) multi << "\n";
                multi << ml;
                first = false;
            }
            line = multi.str();
            if (line.empty()) continue;
            // fall through to "send turn" logic below
        }
        if (line.rfind("/mode ", 0) == 0) {
            std::string spec = line.substr(6);
            // Trim leading/trailing whitespace.
            while (!spec.empty() && std::isspace(static_cast<unsigned char>(spec.front()))) spec.erase(0, 1);
            while (!spec.empty() && std::isspace(static_cast<unsigned char>(spec.back())))  spec.pop_back();
            if (spec.empty()) {
                std::cerr << "[neoclaw] /mode needs a topology name (try /help)\n";
                continue;
            }
            try {
                if (spec == "default" || spec == "agent") {
                    // Drop engine, build a fresh Agent with new tools.
                    auto fresh_tools = make_tools();
                    agent = std::make_unique<neograph::llm::Agent>(
                        provider, std::move(fresh_tools), cfg.agent.system_prompt);
                    engine.reset();
                    cfg.topology.clear();
                    agent_conv.clear();
                    topology_conv = neograph::json::array();
                    std::cerr << "[neoclaw] mode: agent (default ReAct)\n";
                } else {
                    // Allow `/mode pair` as shorthand for `/mode pair.json`.
                    // Only auto-append when the spec has no extension AND
                    // no explicit path separator — `/mode foo/bar.json`
                    // and `/mode /abs/path.json` are passed through as-is.
                    std::string lookup = spec;
                    if (lookup.find('.')  == std::string::npos
                     && lookup.find('/')  == std::string::npos
                     && lookup.find('\\') == std::string::npos) {
                        lookup += ".json";
                    }
                    const auto path = neoclaw::resolve_topology_path(lookup);
                    auto fresh_tools = make_tools();
                    auto fresh_engine = neoclaw::compile_topology(
                        path, provider, std::move(fresh_tools), cfg.agent.system_prompt);
                    // Only commit the swap after compile succeeds —
                    // a typo'd topology name keeps the previous engine alive.
                    engine = std::move(fresh_engine);
                    agent.reset();
                    cfg.topology = path.string();
                    agent_conv.clear();
                    topology_conv = neograph::json::array();
                    std::cerr << "[neoclaw] mode: " << path.filename().string() << "\n";
                }
            } catch (const std::exception& e) {
                std::cerr << "[neoclaw] /mode failed: " << e.what()
                          << " — keeping previous mode\n";
            }
            continue;
        }
        if (line == "/reset") {
            agent_conv.clear();
            topology_conv = neograph::json::array();
            std::cout << "[neoclaw] conversation cleared\n";
            continue;
        }
        if (line.rfind("/run ", 0) == 0) {
            std::string cmd = line.substr(5);
            neoclaw::SandboxOptions sbx;
            sbx.project_root  = cfg.session.project_root;
            sbx.mode          = neoclaw::parse_sandbox_mode(cfg.tools.bash.sandbox);
            sbx.timeout_sec   = cfg.tools.bash.timeout_sec;
            sbx.allow_network = cfg.tools.bash.allow_network;
            auto r = neoclaw::run_sandboxed(cmd, sbx);
            std::cout << r.output;
            if (!r.output.empty() && r.output.back() != '\n') std::cout << "\n";
            std::cout << "[exit=" << r.exit_code
                      << (r.timed_out ? " timed-out" : "")
                      << (r.truncated ? " truncated" : "")
                      << "]\n";
            continue;
        }

        auto on_token = [](const std::string& tok) {
            std::cout << tok << std::flush;
        };

        try {
            if (agent) {
                agent_conv.push_back({"user", line});
                agent->run_stream(agent_conv, on_token, cfg.agent.max_iterations);
            } else {
                topology_conv.push_back(
                    neograph::json{{"role", "user"}, {"content", line}});
                topology_conv = neoclaw::run_topology_turn(
                    *engine, std::move(topology_conv), on_token);
            }
        } catch (const std::exception& e) {
            std::cerr << "\n[neoclaw] agent error: " << e.what() << "\n";
            // Drop the trailing user message so the user can retry.
            if (agent) {
                if (!agent_conv.empty() && agent_conv.back().role == "user") {
                    agent_conv.pop_back();
                }
            } else if (!topology_conv.empty()) {
                // neograph::json (yyjson wrapper) has no back()/erase(idx) —
                // copy all but the last when the trailing entry is the user
                // turn we just appended.
                const size_t n = topology_conv.size();
                if (topology_conv[n - 1].value("role", "") == "user") {
                    auto trimmed = neograph::json::array();
                    for (size_t i = 0; i + 1 < n; ++i) trimmed.push_back(topology_conv[i]);
                    topology_conv = std::move(trimmed);
                }
            }
            continue;
        }
        std::cout << "\n";
    }

    std::cout << "[neoclaw] bye.\n";
    return 0;
}
