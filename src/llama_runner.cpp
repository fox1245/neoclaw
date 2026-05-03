#include "llama_runner.h"

#include <llama.h>

#include <atomic>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

namespace neoclaw {

namespace {

// llama.cpp's backend has process-global state (ggml backends registry,
// CUDA handles). Initialise once per process; tear down at exit. atexit
// keeps the lifetime tied to process exit rather than the last runner —
// constructing a second runner after destroying the first would otherwise
// crash on backend re-init.
std::atomic<bool> g_backend_inited{false};
void ensure_backend_inited() {
    bool expected = false;
    if (g_backend_inited.compare_exchange_strong(expected, true)) {
        llama_backend_init();
        std::atexit([] { llama_backend_free(); });
    }
}

// Tokenize `text` against `vocab`, returning the token vector. Mirrors the
// two-call llama_tokenize pattern (probe size, then allocate). `add_bos`
// is left to the caller because the chat template often emits its own
// `<bos>` literal which we do NOT want duplicated by the tokenizer.
std::vector<llama_token> tokenize(const llama_vocab* vocab,
                                  const std::string& text,
                                  bool add_bos,
                                  bool parse_special) {
    if (text.empty()) return {};
    int32_t need = -llama_tokenize(vocab,
                                    text.data(),
                                    static_cast<int32_t>(text.size()),
                                    nullptr, 0,
                                    add_bos, parse_special);
    if (need <= 0) return {};
    std::vector<llama_token> out(static_cast<size_t>(need));
    int32_t got = llama_tokenize(vocab,
                                  text.data(),
                                  static_cast<int32_t>(text.size()),
                                  out.data(),
                                  static_cast<int32_t>(out.size()),
                                  add_bos, parse_special);
    if (got < 0)
        throw std::runtime_error("llama_tokenize failed");
    out.resize(static_cast<size_t>(got));
    return out;
}

// Detokenize a single token to its piece. Special tokens are rendered
// (special=true) so callers see e.g. `<end_of_turn>` literally — that's
// how stop-sequence matching catches the model's turn-end signal.
std::string token_to_piece(const llama_vocab* vocab, llama_token tok) {
    char buf[256];
    int32_t n = llama_token_to_piece(vocab, tok, buf, sizeof(buf),
                                      /*lstrip=*/0, /*special=*/true);
    if (n < 0) {
        // Buffer too small — re-allocate and retry. 256 covers virtually
        // every BPE/SentencePiece piece we've seen in practice.
        std::vector<char> big(static_cast<size_t>(-n));
        n = llama_token_to_piece(vocab, tok, big.data(),
                                  static_cast<int32_t>(big.size()),
                                  /*lstrip=*/0, /*special=*/true);
        if (n < 0) return {};
        return std::string(big.data(), static_cast<size_t>(n));
    }
    return std::string(buf, static_cast<size_t>(n));
}

} // namespace

// =======================================================================
// LlamaRunner
// =======================================================================

std::shared_ptr<LlamaRunner> LlamaRunner::load(
    const std::string& gguf_path,
    LlamaRunnerConfig  cfg) {

    ensure_backend_inited();

    auto self = std::shared_ptr<LlamaRunner>(new LlamaRunner());
    self->cfg_ = cfg;

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = cfg.n_gpu_layers;
    self->model_ = llama_model_load_from_file(gguf_path.c_str(), mparams);
    if (!self->model_)
        throw std::runtime_error("llama_model_load_from_file failed for " + gguf_path);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx       = static_cast<uint32_t>(cfg.n_ctx);
    cparams.n_batch     = static_cast<uint32_t>(cfg.n_batch);
    cparams.n_threads   = cfg.n_threads > 0
                            ? cfg.n_threads
                            : static_cast<int>(std::thread::hardware_concurrency());
    cparams.n_threads_batch = cparams.n_threads;
    self->ctx_ = llama_init_from_model(self->model_, cparams);
    if (!self->ctx_) {
        llama_model_free(self->model_);
        self->model_ = nullptr;
        throw std::runtime_error("llama_init_from_model failed");
    }

    self->vocab_ = llama_model_get_vocab(self->model_);
    return self;
}

LlamaRunner::~LlamaRunner() {
    if (ctx_)   llama_free(ctx_);
    if (model_) llama_model_free(model_);
}

std::string LlamaRunner::generate(const std::string& prompt,
                                   const GenerateConfig& gc) {
    // Reset KV cache — each call is a fresh sequence. Sequence-id 0 is
    // the only one we ever use; keeping multi-sequence support is left
    // for the future prefix-cache work (see header note).
    llama_memory_clear(llama_get_memory(ctx_), /*data=*/true);

    // Tokenize the prompt. The Gemma chat template our caller renders
    // already includes `<bos>` literally, so add_bos=false; parse_special
    // lets `<start_of_turn>` etc. tokenize as control tokens rather than
    // as plain ASCII.
    auto prompt_tokens = tokenize(vocab_, prompt,
                                   /*add_bos=*/false,
                                   /*parse_special=*/true);
    if (prompt_tokens.empty())
        throw std::runtime_error("empty prompt after tokenization");

    const int n_ctx = static_cast<int>(llama_n_ctx(ctx_));
    if (static_cast<int>(prompt_tokens.size()) >= n_ctx)
        throw std::runtime_error(
            "prompt (" + std::to_string(prompt_tokens.size()) +
            " tokens) exceeds context window (" + std::to_string(n_ctx) + ")");

    // Prefill: feed the prompt tokens through llama_decode in one batch
    // (or split into n_batch-sized chunks if the prompt is large).
    {
        const int n_batch = static_cast<int>(llama_n_batch(ctx_));
        for (size_t i = 0; i < prompt_tokens.size(); i += static_cast<size_t>(n_batch)) {
            const int32_t n_take = std::min(
                static_cast<int32_t>(prompt_tokens.size() - i),
                static_cast<int32_t>(n_batch));
            llama_batch batch = llama_batch_get_one(
                prompt_tokens.data() + i, n_take);
            const int rc = llama_decode(ctx_, batch);
            if (rc != 0)
                throw std::runtime_error(
                    "llama_decode (prefill) failed with rc=" + std::to_string(rc));
        }
    }

    // Build the sampler chain: top_k → top_p → temp → dist (rng).
    // Order matters — temp must come after the truncation samplers so
    // the softmax operates on the already-pruned distribution.
    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    llama_sampler* smpl = llama_sampler_chain_init(sparams);
    if (gc.top_k > 0)
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(gc.top_k));
    if (gc.top_p > 0.0f && gc.top_p < 1.0f)
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(gc.top_p, /*min_keep=*/1));
    if (gc.temperature > 0.0f)
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(gc.temperature));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(cfg_.seed));

    std::string full_text;
    full_text.reserve(static_cast<size_t>(gc.max_new_tokens) * 4);

    // Stop-sequence detection. We keep a `tail` window equal to the
    // longest stop-string and scan it after each new piece. Using the
    // full `full_text` would be O(n²); the tail bounds it to O(stop_len).
    size_t max_stop_len = 0;
    for (const auto& s : gc.stop_sequences)
        if (s.size() > max_stop_len) max_stop_len = s.size();

    // Generation loop. Each iteration: sample next token, decode it,
    // detokenize to a piece, fire the streamer, check stop conditions.
    llama_token next = 0;
    bool stopped_by_seq = false;
    std::string matched_stop;

    for (int i = 0; i < gc.max_new_tokens; ++i) {
        next = llama_sampler_sample(smpl, ctx_, /*idx=*/-1);

        // EOG = end-of-generation. Includes EOS, EOT, and any model-
        // specific terminators. Halt before emitting the token (it's a
        // control marker, not content).
        if (llama_vocab_is_eog(vocab_, next)) break;

        std::string piece = token_to_piece(vocab_, next);
        full_text += piece;
        if (gc.on_piece) gc.on_piece(piece);

        // Stop-sequence check on the rolling tail.
        if (max_stop_len > 0) {
            const size_t scan_from = full_text.size() > max_stop_len + piece.size()
                ? full_text.size() - (max_stop_len + piece.size())
                : 0;
            for (const auto& s : gc.stop_sequences) {
                const auto pos = full_text.find(s, scan_from);
                if (pos != std::string::npos) {
                    full_text.resize(pos);
                    matched_stop = s;
                    stopped_by_seq = true;
                    break;
                }
            }
            if (stopped_by_seq) break;
        }

        // Feed the just-sampled token back so the next iteration's
        // logits reflect it.
        llama_batch batch = llama_batch_get_one(&next, 1);
        const int rc = llama_decode(ctx_, batch);
        if (rc != 0) {
            llama_sampler_free(smpl);
            throw std::runtime_error(
                "llama_decode (gen) failed with rc=" + std::to_string(rc));
        }
    }

    llama_sampler_free(smpl);
    return full_text;
}

} // namespace neoclaw
