// neoclaw/src/local_provider.h — in-process NeoGraph Provider backed by
// llama.cpp directly. Drives `LlamaRunner::generate` with a streamer
// callback, no HTTP round trip. The sibling file gemma_provider.{h,cpp}
// implements the same Provider surface over HTTP for users who run a
// separate inference server (vLLM, text-generation-inference, ollama, …).
//
// This is the default backend for the "download and run" build: one
// binary, no companion server process, model weights pulled from
// HuggingFace Hub on first run via neoclaw::hub.
//
// v0.4 cut: previously this wrapped transformercpp::Model. TransformerCPP
// pivoted to a training-framework focus, so neoclaw consumes llama.cpp
// directly now — see src/llama_runner.h for the bridge.
#pragma once

#include <neograph/provider.h>

#include <memory>
#include <string>

namespace neoclaw {

class LlamaRunner;

class LocalProvider : public neograph::Provider {
public:
    struct Config {
        float temperature = 0.3f;
        // Bumped from 1024 → 8192 because write_file tool calls often
        // ship a multi-kilobyte file body inside the JSON `arguments.content`
        // field (CUDA kernels, C++ headers, full README bodies). Hitting
        // the cap mid-JSON breaks the balanced-brace parser and leaves
        // the agent with unusable output. 8192 covers ~6 KB of source
        // per turn, which is plenty for a single file.
        int   max_tokens  = 8192;
    };

    /// Takes ownership of a loaded LlamaRunner. Construct via
    /// `load_or_download(...)` below to handle cache + HF download.
    LocalProvider(std::shared_ptr<LlamaRunner> runner, Config cfg);

    neograph::ChatCompletion complete(
        const neograph::CompletionParams& params) override;

    neograph::ChatCompletion complete_stream(
        const neograph::CompletionParams& params,
        const neograph::StreamCallback&  on_chunk) override;

    std::string get_name() const override { return "neoclaw-local"; }

private:
    std::shared_ptr<LlamaRunner> runner_;
    Config                       cfg_;
};

/// Resolve a GGUF path: if `model_id` looks like a HuggingFace repo
/// ("org/name"), hand off to neoclaw::hub which caches to
/// ~/.cache/neoclaw/ by default. If it's a local path and the file
/// exists, use it as-is. Prints a single progress line to stderr during
/// download.
///
/// `filename` narrows to a specific GGUF inside the repo (e.g.
/// "gemma-4-E4B-it-Q4_K_M.gguf"). Empty → auto-pick via
/// hub::download_best_gguf (prefers Q4_K_M).
///
/// Throws std::runtime_error on resolve/download failure.
std::string resolve_model(const std::string& model_id,
                          const std::string& filename);

/// Load a GGUF from disk into an in-process LlamaRunner. Wraps
/// LlamaRunner::load with a friendlier stderr surface (load timing,
/// silenced llama.cpp tensor-manifest dump).
std::shared_ptr<LlamaRunner> load_model(const std::string& path);

} // namespace neoclaw
