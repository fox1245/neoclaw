# neoclaw

**100% free, 100% local, 100% C++ — a mini Claude-Code-style coding
agent that runs entirely on your own hardware.**

One binary. First run downloads the model (Gemma-4 E4B, ~4.6 GB) to
`~/.cache/transformercpp/`. Subsequent runs are instant. No Python, no
Docker, no API keys, no data leaving the machine.

```
  +---------------------------------------------------------+
  |  neoclaw v0.2 - local C++ coding agent                  |
  |                                                         |
  |  model    : unsloth/gemma-4-E4B-it-GGUF                 |
  |  endpoint : in-process (TransformerCPP)                 |
  |  project  : /home/me/myproject                          |
  |  bash     : disabled (sandbox=bwrap)                    |
  |                                                         |
  |  type /help for commands, Ctrl-D or /quit to exit       |
  +---------------------------------------------------------+

[neoclaw] 4 tools active: read_file, write_file, grep, glob
[neoclaw] loading model....... done (4519 ms)

❯ what files are in this project?
  -> glob({"pattern":"**/*"})
The files present are README.md and main.cpp.

❯ add a comment to main.cpp explaining what it does
  -> read_file({"path":"main.cpp"})
  -> write_file({"path":"main.cpp", "content":"..."})
  [neoclaw] assistant wants to write 142 bytes to main.cpp [y/N] y
Done — added a single-line banner above `int main(...)`.
```

## Download and run

Pre-built Linux x86_64 bundle (~3 MB — model downloads on first run):

```bash
# Download + extract
curl -L https://github.com/fox1245/neoclaw/releases/latest/download/neoclaw-linux-x86_64.tar.gz \
  | tar -xz

# Run against your project
./neoclaw-*/bin/neoclaw --project-root /path/to/your/project
```

First run pulls ~4.6 GB of Gemma-4 GGUF to `~/.cache/transformercpp/`.
Roughly 5-15 minutes depending on your connection. Shown as a
progress bar.

### Prerequisites

- Linux x86_64 (the bundle).
- `libssl3`, `libstdc++6`, `glibc ≥ 2.34` — installed by default on
  Ubuntu 22.04+, Fedora 37+, Debian 12+, Arch.
- Optional: `bubblewrap` if you enable the bash tool (see below).
- Optional: NVIDIA GPU + CUDA drivers. Without a GPU, Gemma-4 E4B runs
  on CPU at roughly 8-15 tokens/sec; with an RTX 4070 Ti Q4_K_M
  reaches ~130 tokens/sec.

## Three ingredients

- **[NeoGraph](https://github.com/fox1245/NeoGraph)** — C++ agent
  orchestration (ReAct loop, tool dispatch, token streaming). The
  whole engine's hot path fits in <300 KB of L3 cache.
- **[TransformerCPP](https://github.com/fox1245/TransformerCPP)** —
  C++ transformer inference with Flash-Attention. Loads GGUF models
  (Gemma 4, Llama 3, Qwen 2.5, DeepSeek-Coder, …) and ships with a
  HuggingFace Hub client that caches and resumes partial downloads.
- **bubblewrap** — Linux kernel-namespace sandbox. Runs shell commands
  with a read-only filesystem, a bind-mount over your project
  directory, and network unshared by default.

## Build from source

Prereqs: CMake 3.20+, GCC 13+ / Clang 15+, Linux (bwrap). TransformerCPP
currently gitignores its `extern/` tree, so the build needs a sibling
`../TransformerCPP` clone with externals populated (single upstream
setup — see TransformerCPP's README).

```bash
git clone https://github.com/fox1245/TransformerCPP.git ../TransformerCPP
# …follow TransformerCPP's bootstrap to populate extern/ …

git clone https://github.com/fox1245/neoclaw.git
cd neoclaw
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Portable tarball:
./scripts/package.sh   # → dist/neoclaw-<ver>-linux-x86_64.tar.gz
```

CMake auto-detects the sibling `../TransformerCPP`; override with
`-DFETCHCONTENT_SOURCE_DIR_TRANSFORMERCPP=/abs/path` if it lives
elsewhere. NeoGraph and yaml-cpp are fetched by `FetchContent` on
first configure.

## REPL commands

| Command             | Action |
|---------------------|--------|
| `/help`             | Show commands |
| `/quit` / Ctrl-D    | Exit |
| `/reset`            | Clear conversation history |
| `/run <shell cmd>`  | Run a shell command in the sandbox directly (bypass the LLM) |

Every destructive action (file write, bash command) shows a `y/N`
confirmation on stdin before anything touches disk.

## Configuration

Create `./neoclaw.yaml` in your project root — or
`~/.config/neoclaw/config.yaml` globally. Minimum viable:

```yaml
backend: local            # or "http" for external OpenAI-compat server
model:
  id: unsloth/gemma-4-E4B-it-GGUF
agent:
  system_prompt: |
    You are a concise coding assistant.
tools:
  read_file: true
  write_file: true
  grep: true
  glob: true
  bash:
    enabled: true         # opt-in
    sandbox: bwrap
```

Config discovery order (first hit wins):

1. `$NEOCLAW_CONFIG`
2. `./neoclaw.yaml` (CWD)
3. `$XDG_CONFIG_HOME/neoclaw/config.yaml`
4. `~/.config/neoclaw/config.yaml`

### External server mode (`backend: http`)

Point neoclaw at any OpenAI-compatible endpoint — llama.cpp server,
vLLM, TransformerCPP's `http_server_demo`, … — instead of loading the
model in-process. Useful when the model is already loaded somewhere
and shared across sessions, or when it lives on another machine.

```yaml
backend: http
server:
  endpoint: http://localhost:8090
```

## Model choice

Model-swap is a config field. Any instruction-tuned GGUF on the Hub
with decent JSON discipline works. Sanity-tested:

| Model                                    | Params | Tool discipline |
|------------------------------------------|-------:|:---------------:|
| `unsloth/gemma-4-E4B-it-GGUF`            |   8 B  | **good** — 6/7 clean on discipline sweep |
| `unsloth/gemma-4-E2B-it-GGUF`            | 4.6 B  | marginal — mixes prose + JSON sometimes |
| `Qwen/Qwen2.5-Coder-7B-Instruct-GGUF`    |   7 B  | *untested*, likely better for code |
| `TheBloke/deepseek-coder-6.7B-instruct-GGUF` | 6.7 B | *untested* |

Swap by editing `model.id` in the YAML. TransformerCPP's HubClient
handles download + cache.

Set `NEOCLAW_LLAMA_VERBOSE=1` to restore llama.cpp's full per-load
tensor dump if you're debugging a model that won't load.

## Security

- Tool paths are canonicalised and **must** resolve inside
  `session.project_root`. `../../etc/passwd` is rejected before the
  tool body ever runs.
- `bash` runs under `bwrap` with a read-only root bind mount, the
  project directory bind-mounted read-write, and network namespace
  unshared by default. 30-second wall-clock timeout, truncated output
  at 64 KiB.
- Every destructive action (file write, bash command) fires a `y/N`
  stdin prompt. `--no-sandbox` only disables bwrap; confirmations
  always fire.

Nothing here is bullet-proof — a determined adversarial prompt can
still chain reads of files *inside* the project root. Treat neoclaw
like running a pilot tool over your own dotfiles: fine for personal
projects, not fine for shared multi-tenant systems.

## Status (v0.2)

- ✅ Single binary, auto-download, self-contained bundle
- ✅ Local in-process inference + remote HTTP fallback
- ✅ Read / Write / Grep / Glob / Bash tools
- ✅ Path containment, bwrap sandbox, y/N prompts

Coming in v0.3+:

- Diff preview before `write_file` commits
- `linenoise` / readline for arrow keys + history
- macOS `sandbox-exec` backend for the bash tool
- Plan mode for multi-step refactors
- GGUF chat template auto-detection (drop the hard-coded Gemma shape)

## License

MIT — see `LICENSE`.
