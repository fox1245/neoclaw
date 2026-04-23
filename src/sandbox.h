// neoclaw/src/sandbox.h — minimal sandboxed subprocess runner.
//
// v0.1 backend is bubblewrap (`bwrap`). Project root is bind-mounted
// read-write, the rest of the filesystem is read-only, /proc and /dev
// are fresh. Network is unshared by default. Wall-clock timeout is
// enforced by wrapping the invocation in coreutils `timeout`.
//
// OpenSandbox (Alibaba) integration is a future-work hook: add another
// SandboxMode enum value + dispatch in sandbox.cpp. The interface here
// is mode-agnostic.
#pragma once

#include <filesystem>
#include <string>

namespace neoclaw {

enum class SandboxMode {
    None,     ///< No sandbox. Absolutely do not use for untrusted input.
    Bwrap,    ///< Linux-only, default. Requires `bwrap` in PATH.
    // OpenSandbox, // reserved
};

struct SandboxOptions {
    std::filesystem::path project_root;          ///< Writable root for the child.
    SandboxMode           mode = SandboxMode::Bwrap;
    int                   timeout_sec    = 30;
    bool                  allow_network  = false;
    size_t                max_output_bytes = 64 * 1024; ///< Truncate past this.
};

struct SandboxResult {
    int         exit_code   = -1;
    bool        timed_out   = false;
    bool        truncated   = false;
    std::string output;      ///< Combined stdout+stderr.
    std::string error_message; ///< Set when we couldn't even start the child.
};

/// Run a shell command (interpreted by `/bin/bash -c`) inside the
/// configured sandbox. Combines stdout+stderr. Captures up to
/// `opts.max_output_bytes` then drops further bytes (`truncated=true`).
SandboxResult run_sandboxed(const std::string& command,
                             const SandboxOptions& opts);

/// Parse the YAML-level `sandbox` string ("bwrap" | "none" |
/// "opensandbox") into the enum. Unknown values become `Bwrap` with a
/// stderr warning.
SandboxMode parse_sandbox_mode(const std::string& name);

} // namespace neoclaw
