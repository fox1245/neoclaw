// neoclaw/src/llama_runner.h — minimal C++ wrapper around the llama.cpp C
// API. Loads a GGUF, runs prefill + token-by-token generation with a
// streamer callback, and detects stop sequences post-detokenization.
//
// Replaces the v0.3 TransformerCPP::Model abstraction. neoclaw v0.4 talks
// to llama.cpp directly so we can track upstream releases on our own
// cadence — inference is a commodity, the value lives in the harness
// (Provider / Agent / tool loop / sandbox), not in re-wrapping the runtime.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

// Forward-declare llama.cpp's opaque handles so callers don't drag in
// llama.h transitively. local_provider.cpp is the only consumer and it
// already includes <llama.h> for log silencing.
struct llama_model;
struct llama_context;
struct llama_vocab;

namespace neoclaw {

struct LlamaRunnerConfig {
    int      n_ctx        = 8192;   // KV cache window (tokens)
    int      n_batch      = 512;    // prefill batch size
    int      n_gpu_layers = 999;    // -1 / 999 = offload everything; 0 = CPU only
    int      n_threads    = 0;      // 0 = let llama.cpp pick (hwconcurrency)
    uint32_t seed         = 0xC0FFEE;
};

struct GenerateConfig {
    int   max_new_tokens = 8192;
    float temperature    = 0.3f;
    float top_p          = 0.95f;
    int   top_k          = 64;

    // Stop sequences are matched against the rolling detokenized output
    // (not raw tokens). When matched, generation halts and the matched
    // substring is trimmed from the returned text.
    std::vector<std::string> stop_sequences;

    // Called once per newly generated token piece (post-detokenization).
    // The piece is the raw model output — caller does any UI filtering.
    // May be empty (some control tokens detokenize to "").
    std::function<void(const std::string& piece)> on_piece;
};

class LlamaRunner {
public:
    // Load a GGUF from disk. Throws std::runtime_error on failure.
    // The first call also initializes the llama.cpp backend (idempotent).
    static std::shared_ptr<LlamaRunner> load(
        const std::string& gguf_path,
        LlamaRunnerConfig  cfg = {});

    ~LlamaRunner();

    LlamaRunner(const LlamaRunner&)            = delete;
    LlamaRunner& operator=(const LlamaRunner&) = delete;

    // Generate from a fully rendered prompt string. Returns the complete
    // generated text (may be empty if the model emits only EOG immediately).
    // Stop sequences in `gc.stop_sequences` halt generation early.
    //
    // KV cache is reset between calls — neoclaw v0.4 does not yet reuse
    // prefix cache across turns. When that ships, this becomes a per-runner
    // sequence-state method instead.
    std::string generate(const std::string& prompt, const GenerateConfig& gc);

private:
    LlamaRunner() = default;

    llama_model*       model_ = nullptr;
    llama_context*     ctx_   = nullptr;
    const llama_vocab* vocab_ = nullptr;
    LlamaRunnerConfig  cfg_{};
};

} // namespace neoclaw
