// neoclaw/src/hub.h — minimal HuggingFace Hub model downloader.
//
// v0.3 used TransformerCPP::HubClient for this. v0.4 cuts the dependency
// entirely and ships a ~250-LOC libcurl downloader because: (a) the API
// surface we need is two functions, (b) we want to be free to evolve the
// cache layout (model namespacing, partial-file resume) without round-
// tripping through an upstream maintainer, (c) HuggingFace's HTTP layout
// is stable enough that this code rarely changes.
//
// Cache layout:
//   ~/.cache/neoclaw/<repo_owner>--<repo_name>/<filename>
//
// Override with NEOCLAW_CACHE_DIR=/path. Atomic on success (download to
// `.partial`, rename when complete) so a Ctrl-C mid-download never leaves
// a half-baked GGUF that load_from_file would try to mmap.
#pragma once

#include <cstddef>
#include <functional>
#include <string>

namespace neoclaw::hub {

using ProgressCallback = std::function<void(std::size_t got, std::size_t total)>;

// List the files in `repo_id` (e.g. "unsloth/gemma-4-E4B-it-GGUF"), pick
// the highest-priority GGUF using the same heuristic as TransformerCPP's
// HubClient (Q4_K_M > Q4_K_S > Q5_K_M > Q4_0 > Q8_0 > F16 > first .gguf),
// download to the cache, return the local path.
//
// Throws std::runtime_error on network / parse / disk failure.
std::string download_best_gguf(const std::string& repo_id,
                                ProgressCallback   progress);

// Direct download of a specific file from a HF repo. Same caching;
// idempotent — returns the cached path immediately if the file is
// already present at the expected size.
std::string download(const std::string& repo_id,
                     const std::string& filename,
                     ProgressCallback   progress);

} // namespace neoclaw::hub
