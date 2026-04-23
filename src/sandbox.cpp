#include "sandbox.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;

namespace neoclaw {

SandboxMode parse_sandbox_mode(const std::string& name) {
    if (name == "none")  return SandboxMode::None;
    if (name == "bwrap") return SandboxMode::Bwrap;
    // Future: "opensandbox" -> SandboxMode::OpenSandbox
    std::cerr << "[neoclaw] unknown sandbox mode '" << name
              << "', falling back to bwrap\n";
    return SandboxMode::Bwrap;
}

namespace {

// Assemble the argv vector for the subprocess. The first tier is
// `timeout <N>s` so the child is force-killed if it overruns. Then the
// sandbox wrapper (bwrap) with its mount-namespace flags. Finally the
// actual shell invocation.
std::vector<std::string> build_argv(const std::string& command,
                                     const SandboxOptions& opts) {
    std::vector<std::string> argv;
    argv.reserve(64);

    // 1. timeout
    argv.push_back("timeout");
    argv.push_back("--kill-after=2s");
    argv.push_back(std::to_string(opts.timeout_sec) + "s");

    // 2. sandbox wrapper
    if (opts.mode == SandboxMode::Bwrap) {
        argv.push_back("bwrap");
        // Everything readonly by default…
        argv.push_back("--ro-bind"); argv.push_back("/");    argv.push_back("/");
        // …except the project root, which is the only writable tree.
        const std::string pr = opts.project_root.string();
        argv.push_back("--bind");    argv.push_back(pr);     argv.push_back(pr);
        // Fresh /proc, /dev, /tmp so the child can't see sibling processes
        // or host device nodes beyond the standard pseudo-set.
        argv.push_back("--proc");    argv.push_back("/proc");
        argv.push_back("--dev");     argv.push_back("/dev");
        argv.push_back("--tmpfs");   argv.push_back("/tmp");
        argv.push_back("--chdir");   argv.push_back(pr);
        argv.push_back("--die-with-parent");
        argv.push_back("--unshare-pid");
        argv.push_back("--unshare-uts");
        argv.push_back("--unshare-ipc");
        if (!opts.allow_network) argv.push_back("--unshare-net");
        argv.push_back("--");
    }
    // SandboxMode::None: no wrapper — the shell runs directly under
    // timeout. The caller has opted out of isolation.

    // 3. the command
    argv.push_back("/bin/bash");
    argv.push_back("-lc");
    argv.push_back(command);

    return argv;
}

// Exit-code semantics of GNU `timeout`:
//   124 = child exceeded the wall-clock timeout.
//   125 = timeout itself errored (bad option, …).
//   126/127 = child couldn't execute.
//   otherwise = the child's real exit code.
bool was_timeout_exit(int status) {
    if (!WIFEXITED(status)) return false;
    return WEXITSTATUS(status) == 124;
}

} // namespace

SandboxResult run_sandboxed(const std::string& command,
                             const SandboxOptions& opts) {
    SandboxResult r;

    auto argv_str = build_argv(command, opts);
    std::vector<char*> argv;
    argv.reserve(argv_str.size() + 1);
    for (auto& s : argv_str) argv.push_back(s.data());
    argv.push_back(nullptr);

    // Pipe for combined stdout+stderr capture.
    int pipe_fd[2];
    if (::pipe(pipe_fd) < 0) {
        r.error_message = std::string("pipe: ") + std::strerror(errno);
        return r;
    }

    pid_t pid = ::fork();
    if (pid < 0) {
        ::close(pipe_fd[0]); ::close(pipe_fd[1]);
        r.error_message = std::string("fork: ") + std::strerror(errno);
        return r;
    }

    if (pid == 0) {
        // Child: wire stdout+stderr to the pipe, drop stdin.
        ::close(pipe_fd[0]);
        if (::dup2(pipe_fd[1], STDOUT_FILENO) < 0) _exit(127);
        if (::dup2(pipe_fd[1], STDERR_FILENO) < 0) _exit(127);
        ::close(pipe_fd[1]);
        // Close stdin by redirecting from /dev/null — some tools read
        // stdin and otherwise block forever.
        int devnull = ::open("/dev/null", O_RDONLY);
        if (devnull >= 0) { ::dup2(devnull, STDIN_FILENO); ::close(devnull); }

        ::execvp(argv[0], argv.data());
        // execvp returned — the first arg wasn't found. Surface a
        // helpful message on the pipe so the parent can show it.
        const char* msg = "neoclaw: failed to exec sandbox wrapper\n";
        (void)::write(STDERR_FILENO, msg, std::strlen(msg));
        _exit(127);
    }

    // Parent: read until EOF, respecting the byte cap.
    ::close(pipe_fd[1]);

    std::array<char, 4096> buf;
    while (true) {
        ssize_t n = ::read(pipe_fd[0], buf.data(), buf.size());
        if (n < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (n == 0) break;
        if (r.output.size() + static_cast<size_t>(n) <= opts.max_output_bytes) {
            r.output.append(buf.data(), static_cast<size_t>(n));
        } else {
            const size_t remaining = opts.max_output_bytes > r.output.size()
                ? opts.max_output_bytes - r.output.size() : 0;
            if (remaining > 0) r.output.append(buf.data(), remaining);
            r.truncated = true;
            // Keep draining so the child isn't blocked on a full pipe,
            // but throw the bytes away.
        }
    }
    ::close(pipe_fd[0]);

    int status = 0;
    while (::waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) break;
    }

    r.timed_out = was_timeout_exit(status);
    if (WIFEXITED(status)) {
        r.exit_code = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        r.exit_code = 128 + WTERMSIG(status);
    }
    return r;
}

} // namespace neoclaw
