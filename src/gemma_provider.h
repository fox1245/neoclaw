// neoclaw/src/gemma_provider.h — NeoGraph Provider adapter for any
// instruction-tuned LLM served over an OpenAI-compatible endpoint.
//
// The underlying server (llama.cpp server, vLLM, ollama, text-generation-
// inference, …) usually doesn't wire OpenAI's `tools` JSON into the
// prompt — the model never sees the tools. We inject a tool-protocol
// system prompt on the way in, and parse the model's `{"tool_call": {}}`
// JSON block on the way out into NeoGraph's native `tool_calls` shape.
//
// Delegates the actual HTTP round trip to NeoGraph's OpenAIProvider
// (tested and streaming-capable) rather than reimplementing it.
#pragma once

#include <neograph/llm/openai_provider.h>
#include <neograph/provider.h>

#include <memory>
#include <string>

namespace neoclaw {

class GemmaProvider : public neograph::Provider {
public:
    struct Config {
        std::string endpoint   = "http://localhost:8090"; ///< Base URL (no /v1 suffix).
        std::string model_name = "gemma";                 ///< Informational; server ignores.
        float       temperature = 0.2f;                   ///< Default temperature.
        // See the note on LocalProvider::Config::max_tokens — tool
        // calls carrying file bodies can easily exceed 2k tokens.
        int         max_tokens  = 8192;

    };

    explicit GemmaProvider(Config cfg);

    neograph::ChatCompletion complete(
        const neograph::CompletionParams& params) override;

    neograph::ChatCompletion complete_stream(
        const neograph::CompletionParams& params,
        const neograph::StreamCallback& on_chunk) override;

    std::string get_name() const override { return "neoclaw-gemma"; }

private:
    Config                                          cfg_;
    std::unique_ptr<neograph::llm::OpenAIProvider>  delegate_;
};

} // namespace neoclaw
