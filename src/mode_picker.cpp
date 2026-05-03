#include "mode_picker.h"

#include <neograph/json.h>

#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <unistd.h>

namespace fs = std::filesystem;

namespace neoclaw {

namespace {

// Prefer the first non-empty line of the topology JSON's `_comment`
// field as the human-readable description. Falls back to a one-line
// summary built from node count when no comment is present.
std::string describe_topology(const fs::path& json_path) {
    std::ifstream in(json_path);
    if (!in) return std::string{};
    std::ostringstream os;
    os << in.rdbuf();
    try {
        auto j = neograph::json::parse(os.str());
        if (j.contains("_comment")) {
            const auto& c = j["_comment"];
            if (c.is_string()) return c.template get<std::string>();
            if (c.is_array()) {
                for (const auto& line : c) {
                    if (!line.is_string()) continue;
                    auto s = line.template get<std::string>();
                    // First non-empty line is the title.
                    if (!s.empty()) return s;
                }
            }
        }
        // Fallback: count nodes for a useful tag.
        if (j.contains("nodes") && j["nodes"].is_object()) {
            return std::to_string(j["nodes"].size())
                 + "-node graph";
        }
    } catch (...) { /* malformed JSON — show no description */ }
    return std::string{};
}

// `pair.json` → `pair`. Used as the displayed row title.
std::string stem_of(const fs::path& p) { return p.stem().string(); }

} // namespace

bool is_picker_runnable() {
    return ::isatty(STDIN_FILENO) && ::isatty(STDOUT_FILENO);
}

PickerResult pick_mode_tui(const fs::path& topologies_dir) {
    PickerResult out;

    std::error_code ec;
    if (!fs::exists(topologies_dir, ec) || !fs::is_directory(topologies_dir, ec)) {
        out.skipped = true;
        return out;
    }

    std::vector<fs::path> jsons;
    for (const auto& entry : fs::directory_iterator(topologies_dir, ec)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() == ".json")
            jsons.push_back(entry.path());
    }
    // Sort alphabetically, but pin `pair.json` (the default ReAct
    // pair-programmer) to the top so a fresh user's first cursor lands
    // on the safe + recommended choice.
    std::sort(jsons.begin(), jsons.end(),
        [](const fs::path& a, const fs::path& b) {
            const bool a_pair = a.stem() == "pair";
            const bool b_pair = b.stem() == "pair";
            if (a_pair != b_pair) return a_pair;  // pair first
            return a < b;
        });

    if (jsons.empty()) {
        out.skipped = true;
        return out;
    }

    // Build the radiobox rows. Last row is the explicit "Agent default"
    // skip option so the user has a positive way to opt out (Esc also
    // works, but a visible row is friendlier).
    std::vector<std::string> labels;
    std::vector<std::string> details;
    labels.reserve(jsons.size() + 1);
    details.reserve(jsons.size() + 1);
    for (const auto& p : jsons) {
        labels.push_back(stem_of(p));
        details.push_back(describe_topology(p));
    }
    labels.push_back("(Agent default — no topology)");
    details.push_back(
        "Skip the JSON topology layer. Use the v0.4 hardcoded ReAct "
        "Agent loop. Same as leaving `topology:` empty in neoclaw.yaml.");

    int selected = 0;

    auto screen = ftxui::ScreenInteractive::TerminalOutput();

    using namespace ftxui;
    auto radio = Radiobox(&labels, &selected);

    // Renderer wraps the radiobox in a titled border + per-row detail
    // panel under the list.
    auto layout = Renderer(radio, [&] {
        std::string detail = (selected >= 0 && selected < static_cast<int>(details.size()))
                             ? details[static_cast<size_t>(selected)]
                             : std::string{};
        return vbox({
            text("neoclaw — pick a mode") | bold | center,
            separator(),
            radio->Render(),
            separator(),
            paragraph(detail) | dim,
            filler(),
            separator(),
            hbox({
                text("[Enter] start  "),
                text("[Esc] cancel  "),
                text("[click] pick row") | dim,
            }) | center,
        }) | border | size(WIDTH, GREATER_THAN, 60)
           | size(HEIGHT, GREATER_THAN, 14);
    });

    bool cancelled = false;
    auto wrapped = CatchEvent(layout, [&](Event e) {
        if (e == Event::Return) { screen.Exit(); return true; }
        if (e == Event::Escape) { cancelled = true; screen.Exit(); return true; }
        // Ctrl-C handled by ScreenInteractive's default — also exits.
        return false;
    });

    screen.Loop(wrapped);

    // Map result.
    const bool agent_default_row = (selected == static_cast<int>(jsons.size()));
    if (cancelled || agent_default_row) {
        out.skipped = true;
        return out;
    }

    out.skipped       = false;
    out.topology_path = jsons[static_cast<size_t>(selected)].string();
    out.topology_name = stem_of(jsons[static_cast<size_t>(selected)]);
    return out;
}

} // namespace neoclaw
