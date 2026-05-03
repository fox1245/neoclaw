# neoclaw

**100% free, 100% local, 100% C++ — a mini Claude-Code-style coding
agent that runs entirely on your own hardware.**

One binary. First run downloads the model (Gemma-4 E4B, ~4.6 GB) to
`~/.cache/neoclaw/`. Subsequent runs are instant. No Python, no
Docker, no API keys, no data leaving the machine.

```
  +---------------------------------------------------------+
  |  neoclaw v0.4 - local C++ coding agent                  |
  |                                                         |
  |  model    : unsloth/gemma-4-E4B-it-GGUF                 |
  |  endpoint : in-process (llama.cpp)                      |
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

First run pulls ~4.6 GB of Gemma-4 GGUF to `~/.cache/neoclaw/`.
Roughly 5-15 minutes depending on your connection. Shown as a
progress bar.

### Prerequisites

- Linux x86_64 (the bundle).
- `libssl3`, `libstdc++6`, `glibc ≥ 2.34` — installed by default on
  Ubuntu 22.04+, Fedora 37+, Debian 12+, Arch.
- Optional: `bubblewrap` if you enable the bash tool (see below).
- Optional: NVIDIA GPU + CUDA ≥ 12.0 driver. The release tarball ships
  with the llama.cpp CUDA backend compiled in; llama.cpp picks it at
  runtime only if a compatible GPU is present and falls back to CPU
  otherwise. Same bundle, same binary, both paths — **no separate
  CPU / GPU downloads**.

  Indicative throughput on Gemma-4 E4B Q4_K_M (4.6 GB file):

  | Hardware                     | Tokens/sec | % of theoretical max |
  |------------------------------|-----------:|---------------------:|
  | Ryzen 7 5800X (CPU only)     | ~8         | ~60% of RAM-bw cap |
  | Apple M-series (CPU only)    | ~15-25     | varies |
  | RTX 4070 Ti (CUDA)           | **~100**   | **~89%** of VRAM-bw cap (504 GB/s ÷ 4.5 GB ≈ 112 tok/s) |
  | RTX 4090                     | ~150-180   | ~85% of 1 TB/s cap |

  Token generation at batch 1 is memory-bandwidth bound — every token
  reads the full weight tensor out of VRAM once. The 4070 Ti's 100 tok/s
  is within a hair of its physical ceiling; no software change short of
  smaller weights (a tinier quant) or wider bandwidth silicon can push
  past it. Where there *is* headroom is in **not having to re-read**
  tokens you've already seen — i.e. KV-cache reuse across turns. That's
  v0.4 territory.

## Three ingredients

- **[NeoGraph](https://github.com/fox1245/NeoGraph)** — C++ agent
  orchestration (ReAct loop, tool dispatch, token streaming). The
  whole engine's hot path fits in <300 KB of L3 cache. **This is where
  neoclaw's value lives** — the harness, not the runtime.
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** — battle-tested
  C/C++ inference for GGUF models (Gemma, Llama, Qwen, DeepSeek-Coder,
  …). Memory-bandwidth-bound at batch 1, and llama.cpp gets within ~89%
  of the physical ceiling on common hardware — there is nothing left to
  optimise here that isn't already done. Consumed as a commodity dep.
- **bubblewrap** — Linux kernel-namespace sandbox. Runs shell commands
  with a read-only filesystem, a bind-mount over your project
  directory, and network unshared by default.

The HuggingFace Hub downloader is a ~250-LOC libcurl module that lives
in this repo (`src/hub.cpp`); we don't depend on a separate hub library.

## Build from source

Prereqs: CMake 3.20+, GCC 13+ / Clang 15+, Linux (bwrap), libcurl4-openssl-dev.

```bash
git clone https://github.com/fox1245/neoclaw.git
cd neoclaw

# CPU-only build (no CUDA needed):
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# CUDA build (requires nvcc — ~15-20 min, 4 GPU archs by default):
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNEOCLAW_BUILD_CUDA=ON
cmake --build build -j$(nproc)

# Portable tarball (bundles libggml-cuda.so automatically when present):
./scripts/package.sh   # → dist/neoclaw-<ver>-linux-x86_64.tar.gz
```

Scope the CUDA build to just your GPU for a faster compile:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNEOCLAW_BUILD_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=89   # Ada Lovelace (RTX 40xx) only
```

NeoGraph, llama.cpp, and yaml-cpp are all fetched by `FetchContent` on
first configure — no sibling clones, no extern/ tree to populate.

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

### Topology swap (`topology: <file>.json`)

NeoGraph compiles agent orchestration from a JSON definition — nodes,
edges, channels, conditional routing — and `GraphEngine::compile(json,
context)` is the entry point. neoclaw exposes that switch two ways:
the YAML field below, or — when neoclaw boots on a TTY without
`topology:` set — a clickable [FTXUI](https://github.com/ArthurSonzogni/FTXUI)
mode picker that lists every bundled topology, lets you pick with
arrows or mouse, and Esc / "Agent default" to fall back to the v0.5
hardcoded ReAct loop. CI / pipe-redirected stdin auto-skips the
picker, no flag needed.

```yaml
topology: pair.json   # default ReAct pair-programmer (the v0.4 behaviour)
# topology: code-review.json   # single-LLM diff reviewer, no write tools
```

Bundled topologies (under `topologies/`, also installed next to the
binary so `pair.json` resolves without a path prefix):

| File                       | Wiring                                                                                       | Use for                                                                                  |
|----------------------------|----------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| `pair.json`                | `__start__ → llm ⇄ tools → __end__`                                                           | Default. Pair-programmer ReAct loop.                                                     |
| `vibe.json`                | same as pair                                                                                 | Same wiring, different *vibe* — pair with a high-autonomy `system_prompt` and `max_iterations`.|
| `plan-then-act.json`       | `__start__ → planner → executor(subgraph: pair) → __end__`                                    | Two-phase: planner LLM emits a numbered plan, executor subgraph runs it with tools.       |
| `code-review.json`         | `__start__ → llm → __end__`                                                                   | One-shot reviewer. Pair with a review prompt.                                            |
| `code-review-3pass.json`   | `__start__ → mechanical → semantic → consensus → __end__`                                     | Ouroboros-inspired 3-stage gate. Each pass uses `llm_with_prompt` with a different angle.|
| `spec-first.json`          | `__start__ → interviewer → score_extract → (score_below_0_2 ? seed_writer : __end__) → __end__` | Socratic clarification loop with a math-gate (ambiguity ≤ 0.2 → crystallize spec).        |

The latter three rely on neoclaw-side custom registrations:

- **`llm_with_prompt`** — `llm_call` variant that reads `config.prompt`
  (and optional `config.model`) from the JSON node, so one topology
  can run multiple LLM stages with different reviewer angles or model
  pins. Foundation for ouroboros-style 1×/10×/30× tiered routing once
  neoclaw grows multi-Provider support.
- **`score_extract`** — parses `{"score": float}` (with prose
  tolerance: pure JSON, fenced markdown, or buried in narrative) from
  the last assistant message and writes the float to the `score`
  channel.
- **`score_below_0_2`** / **`score_below_0_5`** — conditions that
  read `score` and route `"ready"` (≤ threshold) vs `"more"` (above).
  Mirrors ouroboros's canonical Ambiguity gate.

All three are registered process-wide at `main()` entry via
`src/neoclaw_nodes.cpp` — see that file for the code, and
[ouroboros](https://github.com/Q00/ouroboros) for the design lineage.

Author your own with the same JSON shape used by NeoGraph examples
(`llm_call`, `tool_dispatch`, `intent_classifier`, `subgraph` are
built-in node types; `has_tool_calls`, `route_channel` are built-in
conditions). Drop the file in `./topologies/`, point `topology:` at it,
relaunch — same binary, same model, different agent persona.

Set `NEOCLAW_TRACE_GRAPH=1` to dump the per-turn execution trace
(`__start__ llm tools llm __end__ ...`) — handy when authoring a new
topology and the wiring isn't doing what you expected.

Streaming on the topology path requires NeoGraph at or beyond commit
`7bcf41e` (the `LLMCallNode::execute_stream_async` override that closes
the async-path streaming gap). neoclaw's `CMakeLists.txt` tracks that
via the `NeoGraph` `FetchContent` pin.

### External server mode (`backend: http`)

Point neoclaw at any OpenAI-compatible endpoint — llama.cpp server,
vLLM, ollama, text-generation-inference, … — instead of loading the
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

Swap by editing `model.id` in the YAML. neoclaw's built-in HF Hub
client downloads + caches into `~/.cache/neoclaw/` (override with
`NEOCLAW_CACHE_DIR=/path`).

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

## Status (v0.5)

- ✅ Single binary, auto-download, self-contained bundle
- ✅ Local in-process inference (llama.cpp, direct) + remote HTTP fallback
- ✅ Read / Write / Grep / Glob / Bash tools
- ✅ Path containment, bwrap sandbox, y/N prompts
- ✅ Built-in HF Hub downloader (no external hub dep)
- ✅ JSON-defined topology swap (`pair.json`, `code-review.json`, …)
- ✅ Clickable FTXUI mode picker on TTY launch (auto-skipped on pipes / CI)

v0.4 was the **TransformerCPP cutover** (inference is now llama.cpp,
consumed directly). v0.5 leans into the harness positioning: agent
orchestration is now a JSON file you swap, not a code path. The same
binary + same model becomes a different persona per topology — exactly
the "drop a file, change the agent" workflow NeoGraph was built for.

Coming next:

- Token-by-token streaming on the topology path (push the
  `LLMCallNode::execute_stream_async` override into NeoGraph upstream)
- More bundled topologies: `planner-executor.json`, `debate.json`,
  `deep-research.json`
- Diff preview before `write_file` commits
- `linenoise` / readline for arrow keys + history
- macOS `sandbox-exec` backend for the bash tool
- GGUF chat template auto-detection (drop the hard-coded Gemma shape)
- KV-cache reuse across turns (the only remaining inference-side win
  at batch=1 — shaves prefill on multi-turn conversations)

## License

MIT — see `LICENSE`.
