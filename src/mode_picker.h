// neoclaw/src/mode_picker.h — FTXUI radiobox mode picker.
//
// When neoclaw boots on a TTY with no `topology:` set in YAML, this
// module pops a small clickable radiobox listing every bundled
// topology (plus an "Agent default" skip option), waits for the user
// to pick one, and returns the selected topology spec. The caller
// then feeds that spec into the same `compile_topology()` path the
// YAML field would have driven.
//
// Mouse + arrow-keys + Enter to select. Esc falls back to the v0.5
// Agent default. Non-TTY callers (CI, pipe-redirected stdin/stdout)
// should `is_picker_runnable()` first and skip the call entirely —
// FTXUI on a non-TTY behaves badly.
#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace neoclaw {

struct PickerResult {
    /// True if the user pressed Esc / Ctrl-C / picked the explicit
    /// "Agent default" row. Caller treats this as `topology` empty.
    bool        skipped = false;
    /// Absolute path to the chosen .json file (empty when skipped).
    std::string topology_path;
    /// Display name (e.g. "pair" / "code-review"), useful for log lines.
    std::string topology_name;
};

/// True only when stdin AND stdout are TTYs. FTXUI's ScreenInteractive
/// reads raw input from the terminal; on a pipe/redirect that hangs.
bool is_picker_runnable();

/// Scan `topologies_dir` for *.json, present a radiobox, return the
/// user's pick. `topologies_dir` is typically the `topologies/` dir
/// next to the binary (or the FHS share dir). If the directory is
/// missing or empty, returns `{skipped=true}` with no UI shown.
PickerResult pick_mode_tui(const std::filesystem::path& topologies_dir);

} // namespace neoclaw
