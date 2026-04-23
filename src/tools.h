// neoclaw/src/tools.h — file + shell tools for the agent.
//
// Every tool enforces that paths stay inside the configured project
// root. Relative paths are resolved against the project root, absolute
// paths are rejected unless they canonicalise *into* the project root.
// This is a safety-by-default stance: even a badly-prompted model
// cannot read /etc/passwd through these tools.
#pragma once

#include <neograph/tool.h>

#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "sandbox.h"

namespace neoclaw {

class ReadFileTool : public neograph::Tool {
public:
    explicit ReadFileTool(std::filesystem::path project_root);
    neograph::ChatTool get_definition() const override;
    std::string        execute(const neograph::json& args) override;
    std::string        get_name() const override { return "read_file"; }
private:
    std::filesystem::path root_;
};

class WriteFileTool : public neograph::Tool {
public:
    /// If `confirm` is set, it's invoked with the relative path + the
    /// full new content. Return true to allow the write, false to abort.
    /// nullptr disables confirmation (CI / non-interactive runs).
    using Confirm = std::function<bool(const std::string& rel_path,
                                        const std::string& content)>;

    WriteFileTool(std::filesystem::path project_root, Confirm confirm);
    neograph::ChatTool get_definition() const override;
    std::string        execute(const neograph::json& args) override;
    std::string        get_name() const override { return "write_file"; }
private:
    std::filesystem::path root_;
    Confirm               confirm_;
};

class GrepTool : public neograph::Tool {
public:
    explicit GrepTool(std::filesystem::path project_root);
    neograph::ChatTool get_definition() const override;
    std::string        execute(const neograph::json& args) override;
    std::string        get_name() const override { return "grep"; }
private:
    std::filesystem::path root_;
};

class GlobTool : public neograph::Tool {
public:
    explicit GlobTool(std::filesystem::path project_root);
    neograph::ChatTool get_definition() const override;
    std::string        execute(const neograph::json& args) override;
    std::string        get_name() const override { return "glob"; }
private:
    std::filesystem::path root_;
};

class BashTool : public neograph::Tool {
public:
    using Confirm = std::function<bool(const std::string& command)>;
    BashTool(SandboxOptions opts, Confirm confirm);
    neograph::ChatTool get_definition() const override;
    std::string        execute(const neograph::json& args) override;
    std::string        get_name() const override { return "bash"; }
private:
    SandboxOptions opts_;
    Confirm        confirm_;
};

} // namespace neoclaw
