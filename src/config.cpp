#include "config.h"

#include <yaml-cpp/yaml.h>

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace neoclaw {

namespace {

// Convenience: only overwrite if the YAML node exists.
template <typename T>
void maybe(const YAML::Node& n, T& dst) {
    if (n && !n.IsNull()) dst = n.as<T>();
}

// Expand ~ to $HOME for path-like strings, leave the rest unchanged.
fs::path expand_tilde(const std::string& s) {
    if (s.empty() || s[0] != '~') return fs::path(s);
    const char* home = std::getenv("HOME");
    if (!home || !*home) return fs::path(s);
    return fs::path(home) / s.substr(s.size() > 1 && s[1] == '/' ? 2 : 1);
}

} // namespace

Config default_config() {
    return Config{};
}

Config load_config(const fs::path& yaml_path) {
    Config cfg = default_config();

    YAML::Node root;
    try {
        root = YAML::LoadFile(yaml_path.string());
    } catch (const YAML::Exception& e) {
        throw std::runtime_error(
            "neoclaw: failed to parse YAML at " + yaml_path.string() + ": "
            + e.what());
    }

    if (auto m = root["model"]) {
        maybe(m["id"],       cfg.model.id);
        maybe(m["filename"], cfg.model.filename);
    }
    if (auto s = root["server"]) {
        maybe(s["auto_spawn"], cfg.server.auto_spawn);
        maybe(s["port"],       cfg.server.port);
        maybe(s["endpoint"],   cfg.server.endpoint);
    }
    if (auto a = root["agent"]) {
        maybe(a["system_prompt"],  cfg.agent.system_prompt);
        maybe(a["max_iterations"], cfg.agent.max_iterations);
    }
    if (auto t = root["tools"]) {
        maybe(t["read_file"],  cfg.tools.read_file);
        maybe(t["write_file"], cfg.tools.write_file);
        maybe(t["grep"],       cfg.tools.grep);
        maybe(t["glob"],       cfg.tools.glob);
        if (auto b = t["bash"]) {
            maybe(b["enabled"],       cfg.tools.bash.enabled);
            maybe(b["sandbox"],       cfg.tools.bash.sandbox);
            maybe(b["timeout_sec"],   cfg.tools.bash.timeout_sec);
            maybe(b["allow_network"], cfg.tools.bash.allow_network);
        }
    }
    if (auto se = root["session"]) {
        if (auto pr = se["project_root"]) {
            const std::string raw = pr.as<std::string>();
            cfg.session.project_root = fs::weakly_canonical(
                fs::absolute(expand_tilde(raw)));
        }
    }
    return cfg;
}

Config load_config_from_discovery() {
    std::vector<fs::path> candidates;

    if (const char* env = std::getenv("NEOCLAW_CONFIG"); env && *env) {
        candidates.emplace_back(env);
    }
    candidates.emplace_back(fs::current_path() / "neoclaw.yaml");
    if (const char* xdg = std::getenv("XDG_CONFIG_HOME"); xdg && *xdg) {
        candidates.emplace_back(fs::path(xdg) / "neoclaw" / "config.yaml");
    }
    if (const char* home = std::getenv("HOME"); home && *home) {
        candidates.emplace_back(
            fs::path(home) / ".config" / "neoclaw" / "config.yaml");
    }

    for (const auto& p : candidates) {
        std::error_code ec;
        if (fs::exists(p, ec) && !ec) {
            return load_config(p);
        }
    }
    return default_config();
}

} // namespace neoclaw
