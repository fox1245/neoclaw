// neoclaw/src/local_provider.h — in-process NeoGraph Provider backed by
// TransformerCPP. Drives `transformercpp::Model::generate` directly
// with a streamer callback, no HTTP round trip. The sibling file
// gemma_provider.{h,cpp} implements the same surface over HTTP for
// users who run a separate inference server.
//
// This is the default backend for the "download and run" build: one
// binary, no companion server process, model weights pulled from
// HuggingFace Hub on first run via TransformerCPP's HubClient.
#pragma once

#include <neograph/provider.h>

#include <memory>
#include <string>

namespace transformercpp { class Model; }

namespace neoclaw {

class LocalProvider : public neograph::Provider {
public:
    struct Config {
        float temperature = 0.3f;
        int   max_tokens  = 1024;
    };

    /// Takes ownership of the loaded model. Construct via
    /// `load_or_download(...)` below to handle cache + HF download.
    LocalProvider(std::shared_ptr<transformercpp::Model> model,
                  Config cfg);

    neograph::ChatCompletion complete(
        const neograph::CompletionParams& params) override;

    neograph::ChatCompletion complete_stream(
        const neograph::CompletionParams& params,
        const neograph::StreamCallback&  on_chunk) override;

    std::string get_name() const override { return "neoclaw-local"; }

private:
    std::shared_ptr<transformercpp::Model> model_;
    Config                                 cfg_;
};

/// Resolve a GGUF path: if `model_id` looks like a HuggingFace repo
/// ("org/name"), hand off to TransformerCPP's HubClient which caches
/// to ~/.cache/transformercpp/ by default. If it's a local path and
/// the file exists, use it as-is. Prints a single progress line to
/// stderr during download.
///
/// `filename` narrows to a specific GGUF inside the repo (e.g.
/// "gemma-4-E4B-it-Q4_K_M.gguf"). Empty → auto-pick via
/// HubClient::download_best_gguf (prefers Q4_K_M).
///
/// Throws std::runtime_error on resolve/download failure.
std::string resolve_model(const std::string& model_id,
                          const std::string& filename);

/// Load a GGUF from disk into an in-process TransformerCPP Model.
/// Wraps AutoModel::from_pretrained with a friendlier error surface.
std::shared_ptr<transformercpp::Model> load_model(const std::string& path);

} // namespace neoclaw
