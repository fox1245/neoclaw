// neoclaw/src/main.cpp — REPL loop.
//
// Reads one user turn at a time from stdin, drives the NeoGraph Agent,
// streams tokens back, shows tool invocations inline. Ctrl-D (EOF) or
// `/quit` exits. `/help` prints commands, `/run <cmd>` short-circuits
// the LLM and runs a shell command under the sandbox directly.

#include "config.h"
#include "gemma_provider.h"
#include "sandbox.h"
#include "tools.h"

#include <neograph/llm/agent.h>
#include <neograph/tool.h>

#include <algorithm>
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

std::string shorten(const std::string& s, size_t n) {
    if (s.size() <= n) return s;
    return s.substr(0, n) + "…";
}

void print_banner(const neoclaw::Config& cfg) {
    std::cout
      << "\n"
      << "  ╭───────────────────────────────────────────────────────╮\n"
      << "  │  neoclaw v0.1 — local C++ coding agent                │\n"
      << "  │                                                       │\n"
      << "  │  model    : " << shorten(cfg.model.id, 40)
      << std::string(std::max<int>(0, 40 - (int)shorten(cfg.model.id, 40).size()), ' ')
      << "  │\n"
      << "  │  endpoint : " << shorten(cfg.server.endpoint, 40)
      << std::string(std::max<int>(0, 40 - (int)shorten(cfg.server.endpoint, 40).size()), ' ')
      << "  │\n"
      << "  │  project  : " << shorten(cfg.session.project_root.string(), 40)
      << std::string(std::max<int>(0, 40 - (int)shorten(cfg.session.project_root.string(), 40).size()), ' ')
      << "  │\n"
      << "  │  bash     : " << (cfg.tools.bash.enabled ? "enabled" : "disabled")
      << " (sandbox=" << cfg.tools.bash.sandbox << ")"
      << std::string(std::max<int>(0, 18 - (int)cfg.tools.bash.sandbox.size()), ' ')
      << "  │\n"
      << "  │                                                       │\n"
      << "  │  type /help for commands, Ctrl-D or /quit to exit     │\n"
      << "  ╰───────────────────────────────────────────────────────╯\n\n";
}

} // namespace

int main(int argc, char** argv) {
    CliArgs cli = parse_cli(argc, argv);

    // -----------------------------------------------------------------
    // 1. Load configuration.
    // -----------------------------------------------------------------
    neoclaw::Config cfg;
    try {
        cfg = cli.config_path.empty()
            ? neoclaw::load_config_from_discovery()
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

    print_banner(cfg);

    // -----------------------------------------------------------------
    // 2. Build tools per config.
    // -----------------------------------------------------------------
    std::vector<std::unique_ptr<neograph::Tool>> tools;
    if (cfg.tools.read_file) {
        tools.push_back(std::make_unique<neoclaw::ReadFileTool>(
            cfg.session.project_root));
    }
    if (cfg.tools.write_file) {
        auto confirm = [&](const std::string& rel, const std::string& content) {
            std::cout << "\n[neoclaw] assistant wants to write "
                      << content.size() << " bytes to " << rel << "\n";
            // Preview first ~8 lines for context.
            std::istringstream is(content);
            std::string line;
            int shown = 0;
            while (std::getline(is, line) && shown < 8) {
                std::cout << "  | " << line << "\n";
                ++shown;
            }
            if (shown == 8) std::cout << "  | ...\n";
            return confirm_yn("[neoclaw] write this file?");
        };
        tools.push_back(std::make_unique<neoclaw::WriteFileTool>(
            cfg.session.project_root, confirm));
    }
    if (cfg.tools.grep) {
        tools.push_back(std::make_unique<neoclaw::GrepTool>(
            cfg.session.project_root));
    }
    if (cfg.tools.glob) {
        tools.push_back(std::make_unique<neoclaw::GlobTool>(
            cfg.session.project_root));
    }
    if (cfg.tools.bash.enabled) {
        neoclaw::SandboxOptions sbx;
        sbx.project_root  = cfg.session.project_root;
        sbx.mode          = neoclaw::parse_sandbox_mode(cfg.tools.bash.sandbox);
        sbx.timeout_sec   = cfg.tools.bash.timeout_sec;
        sbx.allow_network = cfg.tools.bash.allow_network;
        auto confirm = [&](const std::string& cmd) {
            std::cout << "\n[neoclaw] assistant wants to run: " << cmd << "\n";
            return confirm_yn("[neoclaw] run this command?");
        };
        tools.push_back(std::make_unique<neoclaw::BashTool>(
            std::move(sbx), confirm));
    }
    std::cout << "[neoclaw] " << tools.size() << " tool"
              << (tools.size() == 1 ? "" : "s") << " active: ";
    for (size_t i = 0; i < tools.size(); ++i) {
        std::cout << tools[i]->get_name()
                  << (i + 1 < tools.size() ? ", " : "\n\n");
    }
    if (tools.empty()) std::cout << "(none)\n\n";

    // -----------------------------------------------------------------
    // 3. Provider + Agent.
    // -----------------------------------------------------------------
    neoclaw::GemmaProvider::Config pcfg;
    pcfg.endpoint = cfg.server.endpoint;
    auto provider = std::make_shared<neoclaw::GemmaProvider>(pcfg);

    neograph::llm::Agent agent(
        provider,
        std::move(tools),
        cfg.agent.system_prompt);

    // Conversation is kept across turns so the agent has context.
    std::vector<neograph::ChatMessage> conversation;

    // -----------------------------------------------------------------
    // 4. REPL.
    // -----------------------------------------------------------------
    std::string line;
    while (true) {
        std::cout << "\033[1;32m❯ \033[0m" << std::flush;
        if (!std::getline(std::cin, line)) { std::cout << "\n"; break; }
        if (line.empty()) continue;

        if (line == "/quit" || line == "/exit") break;
        if (line == "/help") {
            std::cout
              << "  /help              Show this help\n"
              << "  /quit              Exit (same as Ctrl-D)\n"
              << "  /reset             Clear conversation history\n"
              << "  /run <shell cmd>   Run a shell command directly (bypass LLM)\n"
              << "\n";
            continue;
        }
        if (line == "/reset") {
            conversation.clear();
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

        conversation.push_back({"user", line});

        try {
            agent.run_stream(conversation,
                [](const std::string& tok) {
                    std::cout << tok << std::flush;
                },
                cfg.agent.max_iterations);
        } catch (const std::exception& e) {
            std::cerr << "\n[neoclaw] agent error: " << e.what() << "\n";
            // Drop the trailing user message so the user can retry.
            if (!conversation.empty() && conversation.back().role == "user") {
                conversation.pop_back();
            }
            continue;
        }
        std::cout << "\n";
    }

    std::cout << "[neoclaw] bye.\n";
    return 0;
}
