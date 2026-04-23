#include "tools.h"

#include <neograph/json.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <system_error>

namespace fs = std::filesystem;

namespace neoclaw {

namespace {

// Resolve `input` against `root`, rejecting anything that escapes.
// Returns std::nullopt for escape attempts or canonicalise failures.
std::optional<fs::path> resolve_in_root(const fs::path& root,
                                         const std::string& input,
                                         bool must_exist) {
    if (input.empty()) return std::nullopt;
    fs::path p = fs::path(input).is_absolute() ? fs::path(input) : (root / input);
    std::error_code ec;
    fs::path canon = must_exist
        ? fs::canonical(p, ec)
        : fs::weakly_canonical(p, ec);
    if (ec) return std::nullopt;
    fs::path canon_root = fs::canonical(root, ec);
    if (ec) return std::nullopt;
    // Check canon is equal to or descendant of canon_root.
    auto rel = fs::relative(canon, canon_root, ec);
    if (ec) return std::nullopt;
    if (rel.empty() || rel.string().rfind("..", 0) == 0) return std::nullopt;
    return canon;
}

std::string read_text_file(const fs::path& p, size_t max_bytes,
                            bool& truncated) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return {};
    std::ostringstream os;
    std::array<char, 8192> buf;
    size_t total = 0;
    while (f) {
        f.read(buf.data(), buf.size());
        const auto got = static_cast<size_t>(f.gcount());
        if (got == 0) break;
        if (total + got > max_bytes) {
            os.write(buf.data(), max_bytes - total);
            truncated = true;
            total = max_bytes;
            break;
        }
        os.write(buf.data(), got);
        total += got;
    }
    return os.str();
}

std::string err_json(const std::string& msg) {
    neograph::json j; j["error"] = msg;
    return j.dump();
}

} // namespace

// ---------------------------------------------------------------------
// ReadFile
// ---------------------------------------------------------------------
ReadFileTool::ReadFileTool(fs::path root) : root_(std::move(root)) {}

neograph::ChatTool ReadFileTool::get_definition() const {
    return {
        "read_file",
        "Read a text file from the project and return its contents. "
        "Paths are resolved relative to the project root. Binary files "
        "and files outside the project are rejected. Large files are "
        "truncated (128 KiB cap).",
        neograph::json{
            {"type", "object"},
            {"properties", {
                {"path", {
                    {"type", "string"},
                    {"description", "Relative path from the project root."}}}
            }},
            {"required", neograph::json::array({"path"})}
        }
    };
}

std::string ReadFileTool::execute(const neograph::json& args) {
    const std::string path = args.value("path", "");
    auto abs = resolve_in_root(root_, path, /*must_exist=*/true);
    if (!abs) return err_json("path is outside the project or does not exist: " + path);
    if (!fs::is_regular_file(*abs))
        return err_json("not a regular file: " + path);

    bool truncated = false;
    std::string content = read_text_file(*abs, 128 * 1024, truncated);
    neograph::json j;
    j["path"]      = path;
    j["content"]   = content;
    j["bytes"]     = content.size();
    j["truncated"] = truncated;
    return j.dump();
}

// ---------------------------------------------------------------------
// WriteFile
// ---------------------------------------------------------------------
WriteFileTool::WriteFileTool(fs::path root, Confirm confirm)
    : root_(std::move(root)), confirm_(std::move(confirm)) {}

neograph::ChatTool WriteFileTool::get_definition() const {
    return {
        "write_file",
        "Write text content to a file inside the project root, creating "
        "parent directories if needed. Overwrites any existing file at "
        "that path. The user is prompted to confirm before any change is "
        "persisted.",
        neograph::json{
            {"type", "object"},
            {"properties", {
                {"path",    {{"type","string"},
                              {"description","Relative path from project root."}}},
                {"content", {{"type","string"},
                              {"description","Full text content to write."}}}
            }},
            {"required", neograph::json::array({"path", "content"})}
        }
    };
}

std::string WriteFileTool::execute(const neograph::json& args) {
    const std::string path    = args.value("path", "");
    const std::string content = args.value("content", "");
    auto abs = resolve_in_root(root_, path, /*must_exist=*/false);
    if (!abs) return err_json("path is outside the project: " + path);

    if (confirm_ && !confirm_(path, content)) {
        return err_json("write cancelled by user");
    }

    std::error_code ec;
    fs::create_directories(abs->parent_path(), ec);
    if (ec) return err_json("mkdir parent failed: " + ec.message());

    std::ofstream f(*abs, std::ios::binary | std::ios::trunc);
    if (!f) return err_json("open for write failed");
    f.write(content.data(), static_cast<std::streamsize>(content.size()));
    if (!f) return err_json("write failed");

    neograph::json j;
    j["path"]    = path;
    j["bytes"]   = content.size();
    j["status"]  = "ok";
    return j.dump();
}

// ---------------------------------------------------------------------
// Grep — simple std::regex search across the project tree.
// ---------------------------------------------------------------------
GrepTool::GrepTool(fs::path root) : root_(std::move(root)) {}

neograph::ChatTool GrepTool::get_definition() const {
    return {
        "grep",
        "Search for a regular expression across text files in the "
        "project. Returns at most 100 matches with file path and line "
        "number. `path` narrows the search to a subdirectory or a "
        "single file (optional).",
        neograph::json{
            {"type", "object"},
            {"properties", {
                {"pattern", {{"type","string"},
                              {"description","POSIX ECMAScript regex."}}},
                {"path",    {{"type","string"},
                              {"description","Optional subdir or file."}}}
            }},
            {"required", neograph::json::array({"pattern"})}
        }
    };
}

std::string GrepTool::execute(const neograph::json& args) {
    const std::string pat_str = args.value("pattern", "");
    const std::string sub     = args.value("path", "");
    if (pat_str.empty()) return err_json("pattern is required");

    std::regex re;
    try { re = std::regex(pat_str); }
    catch (const std::regex_error& e) { return err_json(std::string("bad regex: ") + e.what()); }

    fs::path search_root = root_;
    if (!sub.empty()) {
        auto abs = resolve_in_root(root_, sub, /*must_exist=*/true);
        if (!abs) return err_json("search path outside project: " + sub);
        search_root = *abs;
    }

    neograph::json matches = neograph::json::array();
    int count = 0;
    const int cap = 100;

    auto inspect_file = [&](const fs::path& p) {
        std::ifstream f(p);
        if (!f) return;
        std::string line;
        int lineno = 0;
        while (std::getline(f, line) && count < cap) {
            ++lineno;
            if (std::regex_search(line, re)) {
                neograph::json m;
                m["file"] = fs::relative(p, root_).string();
                m["line"] = lineno;
                m["text"] = line.substr(0, 300);
                matches.push_back(m);
                ++count;
            }
        }
    };

    if (fs::is_regular_file(search_root)) {
        inspect_file(search_root);
    } else {
        std::error_code ec;
        for (auto it = fs::recursive_directory_iterator(
                search_root, fs::directory_options::skip_permission_denied, ec);
             it != fs::recursive_directory_iterator(); it.increment(ec)) {
            if (ec) { ec.clear(); continue; }
            if (!it->is_regular_file(ec)) continue;
            // Skip hidden / build dirs by convention.
            const auto& name = it->path().filename().string();
            if (!name.empty() && name[0] == '.') { it.disable_recursion_pending(); continue; }
            if (name == "build" || name == "node_modules" || name == "target") {
                it.disable_recursion_pending(); continue;
            }
            inspect_file(it->path());
            if (count >= cap) break;
        }
    }

    neograph::json j;
    j["matches"] = matches;
    j["count"]   = count;
    j["capped"]  = (count >= cap);
    return j.dump();
}

// ---------------------------------------------------------------------
// Glob — std::filesystem recursive match with shell-style wildcards.
// ---------------------------------------------------------------------
GlobTool::GlobTool(fs::path root) : root_(std::move(root)) {}

neograph::ChatTool GlobTool::get_definition() const {
    return {
        "glob",
        "List files matching a shell-style pattern (e.g. `src/**/*.cpp`) "
        "relative to the project root. Returns at most 500 paths.",
        neograph::json{
            {"type", "object"},
            {"properties", {
                {"pattern", {{"type","string"},
                              {"description","Glob pattern with ** support."}}}
            }},
            {"required", neograph::json::array({"pattern"})}
        }
    };
}

static std::string glob_to_regex(const std::string& glob) {
    std::string re;
    re.reserve(glob.size() * 2);
    for (size_t i = 0; i < glob.size(); ++i) {
        char c = glob[i];
        if (c == '*') {
            if (i + 1 < glob.size() && glob[i+1] == '*') {
                re += ".*";
                ++i;
                if (i + 1 < glob.size() && glob[i+1] == '/') ++i;
            } else {
                re += "[^/]*";
            }
        } else if (c == '?') {
            re += "[^/]";
        } else if (c == '.' || c == '+' || c == '(' || c == ')' ||
                   c == '^' || c == '$' || c == '|' || c == '{' ||
                   c == '}' || c == '\\') {
            re += '\\'; re += c;
        } else {
            re += c;
        }
    }
    return re;
}

std::string GlobTool::execute(const neograph::json& args) {
    const std::string pat = args.value("pattern", "");
    if (pat.empty()) return err_json("pattern is required");

    std::regex re;
    try { re = std::regex("^" + glob_to_regex(pat) + "$"); }
    catch (const std::regex_error& e) { return err_json(std::string("bad pattern: ") + e.what()); }

    neograph::json files = neograph::json::array();
    int count = 0;
    const int cap = 500;

    std::error_code ec;
    for (auto it = fs::recursive_directory_iterator(
            root_, fs::directory_options::skip_permission_denied, ec);
         it != fs::recursive_directory_iterator(); it.increment(ec)) {
        if (ec) { ec.clear(); continue; }
        if (!it->is_regular_file(ec)) continue;
        const auto& name = it->path().filename().string();
        if (!name.empty() && name[0] == '.') { it.disable_recursion_pending(); continue; }
        if (name == "build" || name == "node_modules" || name == "target") {
            it.disable_recursion_pending(); continue;
        }
        auto rel = fs::relative(it->path(), root_).string();
        if (std::regex_match(rel, re)) {
            files.push_back(rel);
            ++count;
            if (count >= cap) break;
        }
    }

    neograph::json j;
    j["files"]  = files;
    j["count"]  = count;
    j["capped"] = (count >= cap);
    return j.dump();
}

// ---------------------------------------------------------------------
// Bash — sandboxed shell exec.
// ---------------------------------------------------------------------
BashTool::BashTool(SandboxOptions opts, Confirm confirm)
    : opts_(std::move(opts)), confirm_(std::move(confirm)) {}

neograph::ChatTool BashTool::get_definition() const {
    return {
        "bash",
        "Run a shell command inside a sandboxed environment. The project "
        "directory is writable; everything else is read-only; network is "
        "unshared unless explicitly enabled. Combined stdout+stderr is "
        "returned. Commands that exceed the timeout are killed.",
        neograph::json{
            {"type", "object"},
            {"properties", {
                {"command", {{"type","string"},
                              {"description","Shell command (/bin/bash -lc)."}}}
            }},
            {"required", neograph::json::array({"command"})}
        }
    };
}

std::string BashTool::execute(const neograph::json& args) {
    const std::string cmd = args.value("command", "");
    if (cmd.empty()) return err_json("command is required");

    if (confirm_ && !confirm_(cmd)) {
        return err_json("command cancelled by user");
    }

    auto r = run_sandboxed(cmd, opts_);
    neograph::json j;
    if (!r.error_message.empty()) {
        j["error"] = r.error_message;
        return j.dump();
    }
    j["exit_code"] = r.exit_code;
    j["timed_out"] = r.timed_out;
    j["truncated"] = r.truncated;
    j["output"]    = r.output;
    return j.dump();
}

} // namespace neoclaw
