// neoclaw/src/config.h — YAML-driven runtime configuration.
//
// One struct tree mirroring the example config in config/neoclaw.example.yaml.
// `load_config(path)` parses a YAML file; `default_config()` returns a
// sensible baseline (Gemma-4 E4B, localhost:8090, safe tools only).
#pragma once

#include <filesystem>
#include <string>

namespace neoclaw {

struct ModelConfig {
    std::string id        = "unsloth/gemma-4-E4B-it-GGUF";
    std::string filename  = "gemma-4-E4B-it-Q4_K_M.gguf"; ///< "" = auto-pick best
};

/// Inference backend selector.
/// * Local  — load the GGUF in-process via llama.cpp directly. No
///            external server needed. The "download and run" default.
/// * Http   — talk to a separate OpenAI-compatible server (llama.cpp
///            server, vLLM, ollama, text-generation-inference, …).
///            Useful when the model is pre-loaded in a persistent
///            process or lives on another host.
enum class BackendKind { Local, Http };

struct ServerConfig {
    int         port       = 8090;
    std::string endpoint   = "http://localhost:8090";  ///< used when backend=http
};

struct AgentConfig {
    std::string system_prompt =
        "You are a concise coding assistant. You work on the project in "
        "the current working directory. Prefer small diffs, explain what "
        "you are about to do, and ask for confirmation before any "
        "destructive change.";
    int max_iterations = 12;
};

struct BashConfig {
    bool        enabled        = false;
    std::string sandbox        = "bwrap";  ///< bwrap | none | opensandbox
    int         timeout_sec    = 30;
    bool        allow_network  = false;
};

struct ToolsConfig {
    bool       read_file  = true;
    bool       write_file = true;
    bool       grep       = true;
    bool       glob       = true;
    BashConfig bash{};
};

struct SessionConfig {
    std::filesystem::path project_root = std::filesystem::current_path();
};

struct Config {
    BackendKind   backend = BackendKind::Local;
    ModelConfig   model;
    ServerConfig  server;
    AgentConfig   agent;
    ToolsConfig   tools;
    SessionConfig session;
};

/// Return a baseline Config. Never throws.
Config default_config();

/// Load configuration from a YAML file. Fields absent in the YAML keep
/// their defaults from `default_config()`. Relative `session.project_root`
/// resolves against the CWD of the caller (not the YAML file path).
/// Throws std::runtime_error on YAML parse errors.
Config load_config(const std::filesystem::path& yaml_path);

/// Search for a config file in the conventional order and load whichever
/// exists first. Returns `default_config()` if none found.
///
/// Discovery order:
///   1. $NEOCLAW_CONFIG (if set)
///   2. <project_root>/neoclaw.yaml  (when `project_root_hint` non-empty)
///   3. ./neoclaw.yaml (in CWD)
///   4. $XDG_CONFIG_HOME/neoclaw/config.yaml
///   5. ~/.config/neoclaw/config.yaml
///
/// Step 2 matters when the user runs neoclaw from a directory other
/// than the project root (e.g. invoking from their home with
/// `neoclaw --project-root /path/to/proj`). Without it the project-local
/// config gets silently ignored.
Config load_config_from_discovery(
    const std::filesystem::path& project_root_hint = {});

} // namespace neoclaw
