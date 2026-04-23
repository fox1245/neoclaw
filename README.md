# neoclaw

**100% free, 100% local, 100% C++ — a mini Claude-Code-style coding agent
that runs entirely on your own hardware.**

neoclaw is a one-binary agent that wires three ingredients together:

- **[NeoGraph](https://github.com/fox1245/NeoGraph)** — C++ agent
  orchestration (ReAct loop, tool dispatch, streaming). 0.9 MB stripped
  runtime; the whole engine fits in L3 cache.
- **[TransformerCPP](https://github.com/fox1245/TransformerCPP)** —
  C++ transformer inference with Flash-Attention. Loads GGUF models
  (Gemma 4, Llama 3, Qwen 2.5, DeepSeek-Coder, …) from the HuggingFace
  Hub on first run.
- **bubblewrap** — Linux kernel-namespace sandbox. Runs shell commands
  with a read-only filesystem, a bind-mount over your project directory,
  and network unshared by default.

No Python, no Docker, no API keys, no data leaving the machine.

```
  ╭───────────────────────────────────────────────────────╮
  │  neoclaw v0.1 — local C++ coding agent                │
  │                                                       │
  │  model    : unsloth/gemma-4-E4B-it-GGUF               │
  │  endpoint : http://localhost:8090                     │
  │  project  : /home/you/myproject                       │
  │  bash     : enabled (sandbox=bwrap)                   │
  │                                                       │
  │  type /help for commands, Ctrl-D or /quit to exit     │
  ╰───────────────────────────────────────────────────────╯

[neoclaw] 5 tools active: read_file, write_file, grep, glob, bash

❯ find the function that handles parsing JSON arguments
  I'll grep for it.
  [grep "parse.*json|json.*parse" src/]
  [→ 7 matches across 3 files]
  The main JSON arg parser is in src/gemma_provider.cpp:59 …

❯ run the tests
  [bash "cmake --build build && ctest --output-on-failure"]
  [neoclaw] assistant wants to run: cmake --build build && ctest --output-on-failure
  [neoclaw] run this command? [y/N] y
  ...
  100% tests passed, 0 tests failed out of 341
```

## Status

**v0.1 — proof of concept.** One-turn and multi-turn agent loops work
with Gemma 4 E4B. Read / Write / Grep / Glob / Bash tools implemented
with project-root path containment and bwrap sandbox. This repository
is meant as a reference implementation showing that a full Claude-Code-
class local coding agent fits in a few hundred lines of C++ on top of
existing C++ building blocks.

Not yet: diff previews, plan mode, multi-file edit transactions,
persistent sessions across restarts, linenoise / arrow-key history.
Those are v0.2+.

## Build

Prerequisites:

- Linux (bwrap is Linux-only). macOS sandbox support is v0.3+.
- CMake 3.20+, a C++20 compiler (GCC 13+ or Clang 15+).
- bubblewrap (`apt install bubblewrap`).
- CUDA-capable GPU *optional* — TransformerCPP falls back to CPU.

```bash
# 1. Clone TransformerCPP next to neoclaw and populate its extern/ tree.
#    (Upstream currently gitignores extern/{safetensors-cpp,sentencepiece,
#    onnxruntime,libtorch}; until that's fixed we stage them ourselves.)
git clone https://github.com/fox1245/TransformerCPP.git ../TransformerCPP
(cd ../TransformerCPP && scripts/bootstrap-extern.sh)   # or populate manually

# 2. Clone + build neoclaw. CMake auto-detects ../TransformerCPP and
#    reuses it; NeoGraph and yaml-cpp are fetched via FetchContent.
git clone https://github.com/fox1245/neoclaw.git
cd neoclaw
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

If `TransformerCPP` lives elsewhere, point CMake at it explicitly:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DFETCHCONTENT_SOURCE_DIR_TRANSFORMERCPP=/abs/path/to/TransformerCPP
```

The initial build takes ~10-20 minutes because TransformerCPP brings
its own vendored llama.cpp + ggml.

## Run

Start the inference server (terminal 1):

```bash
# Gemma 4 E4B Q4_K_M (~4.6 GB GGUF, auto-downloaded on first run):
./build/_deps/transformercpp-build/http_server_demo \
    unsloth/gemma-4-E4B-it-GGUF 8090
```

Start neoclaw (terminal 2):

```bash
cp config/neoclaw.example.yaml ./neoclaw.yaml
./build/neoclaw --project-root /path/to/your/project
```

First run downloads the model (4.6 GB for Gemma 4 E4B) to
`~/.cache/transformercpp/`. Subsequent runs re-use the cache.

## Configuration

See `config/neoclaw.example.yaml` for every knob. Minimum viable:

```yaml
model:
  id: unsloth/gemma-4-E4B-it-GGUF
server:
  endpoint: http://localhost:8090
agent:
  system_prompt: |
    You are a concise coding assistant.
tools:
  read_file: true
  write_file: true
  grep: true
  glob: true
  bash:
    enabled: true
    sandbox: bwrap
```

Config discovery (first hit wins):

1. `$NEOCLAW_CONFIG`
2. `./neoclaw.yaml` (CWD)
3. `$XDG_CONFIG_HOME/neoclaw/config.yaml`
4. `~/.config/neoclaw/config.yaml`

## Model choice

neoclaw passes `tools` via a system-prompt protocol (the server doesn't
translate OpenAI `tools` → prompt yet), so any instruction-tuned model
with decent JSON discipline works. Tested:

| Model                              | Params | Tool discipline |
|------------------------------------|-------:|:---------------:|
| `unsloth/gemma-4-E4B-it-GGUF`      |   8 B  | **good** (6/7 clean on sweep) |
| `unsloth/gemma-4-E2B-it-GGUF`      | 4.6 B  | marginal — works but mixes prose + JSON sometimes |
| `Qwen/Qwen2.5-Coder-7B-Instruct-GGUF` | 7 B | *untested yet*, likely better for code |

Swap by editing `model.id` in the YAML config — TransformerCPP's HubClient
handles the download + cache.

## Security

- Tool-accessible paths are canonicalised and **must** resolve inside
  `session.project_root`. The sibling-traversal trick (`../../etc/passwd`)
  is rejected upstream of the tool bodies.
- `bash` runs under bwrap with a read-only root bind and the project
  directory bind-mounted read-write. Network namespace is unshared by
  default. Wall-clock timeout defaults to 30 s.
- Every destructive action (file write, bash command) goes through a
  `y/N` confirmation prompt on stdin. `--no-sandbox` disables bwrap
  only; confirmation prompts always fire.

Nothing about this is bullet-proof — a sufficiently motivated attacker
who controls the user prompt can still get the model to chain reads of
files *inside* the project root. Treat neoclaw like you'd treat running
an AI-pilot tool over your personal dotfiles repo: fine for personal
projects, not fine for shared multi-tenant systems.

## License

MIT. See `LICENSE`.
