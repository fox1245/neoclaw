// neoclaw/src/ui.h — terminal rendering helpers.
//
// Intentionally line-oriented: we do NOT take over the terminal (ncurses
// initscr / FTXUI style), because that conflicts with std::getline +
// a token-streaming Provider. All UI primitives are just "emit ANSI +
// count UTF-8 columns properly" — which is what ncurses would do under
// the hood via `wcwidth()` anyway.
//
// Colour is auto-disabled when stdout is not a tty, or when the
// `NO_COLOR` environment variable is set (https://no-color.org/).
#pragma once

#include <string>
#include <string_view>

namespace neoclaw::ui {

/// Visible column count of a UTF-8 string, honouring combining marks
/// and East-Asian wide characters. Falls back to byte count on
/// mb→wc decode errors.
size_t visual_width(std::string_view utf8);

/// Return true on first call after stdout is a tty and NO_COLOR is
/// unset. Cached after first call.
bool colour_enabled();

/// Convenience escape-sequence builders. Return empty string when
/// colour is disabled.
std::string dim();
std::string bold();
std::string reset();
std::string fg_cyan();
std::string fg_green();
std::string fg_yellow();
std::string fg_magenta();
std::string fg_red();

/// Print the startup banner. Pads every inner row to the same visual
/// column width so the box edges line up even on rows containing
/// multi-byte UTF-8 (em-dash, CJK, …).
struct BannerLines {
    std::string title;      ///< e.g. "neoclaw v0.1 — local C++ coding agent"
    std::string model;      ///< e.g. "unsloth/gemma-4-E4B-it-GGUF"
    std::string endpoint;   ///< e.g. "http://localhost:8090"
    std::string project;    ///< e.g. "/home/me/proj"
    std::string bash_line;  ///< e.g. "enabled (sandbox=bwrap)"
    std::string hint;       ///< e.g. "type /help for commands, Ctrl-D to exit"
};
void print_banner(const BannerLines& b);

/// Inline tool-invocation trace rendered to stderr between the user
/// turn and the final assistant response, so the user sees the agent's
/// work-in-progress.
void print_tool_start(const std::string& name, const std::string& args_preview);
void print_tool_result(const std::string& summary);

/// Remove chat-template control tokens that leak through gguf
/// streaming output (Gemma `<end_of_turn>`, ChatML `<|im_end|>`, …).
/// Operates on a streaming fragment and a carry buffer so a token
/// split across two chunks is still caught.
void strip_chat_artifacts(std::string& fragment, std::string& carry);

} // namespace neoclaw::ui
