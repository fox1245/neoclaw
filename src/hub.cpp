#include "hub.h"

#include <neograph/json.h>

#include <curl/curl.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace neoclaw::hub {

namespace {

using neograph::json;

// ----- libcurl lifecycle ------------------------------------------------
//
// curl_global_init() is documented as not-thread-safe and "should be
// called once per program". We call it lazily on first use and rely on
// atexit cleanup. The CURL handles themselves are per-call (one per
// request) — the cost is negligible against multi-MB-per-second HF
// downloads, and the simpler control flow is worth more than handle
// reuse here.

void ensure_curl_global() {
    static const bool inited = [] {
        if (curl_global_init(CURL_GLOBAL_DEFAULT) != CURLE_OK)
            throw std::runtime_error("curl_global_init failed");
        std::atexit([] { curl_global_cleanup(); });
        return true;
    }();
    (void)inited;
}

// libcurl write-callback flavours: one accumulates into std::string (for
// the JSON manifest fetch), the other streams to a FILE* (for the actual
// model download — the GGUF can be 4+ GB so buffering in memory is out).
size_t write_to_string(void* ptr, size_t size, size_t nmemb, void* user) {
    auto* s = static_cast<std::string*>(user);
    s->append(static_cast<const char*>(ptr), size * nmemb);
    return size * nmemb;
}

size_t write_to_file(void* ptr, size_t size, size_t nmemb, void* user) {
    return std::fwrite(ptr, size, nmemb, static_cast<std::FILE*>(user));
}

// XFERINFOFUNCTION: libcurl-modern progress callback. Fires periodically
// during transfer; we forward to the user-supplied ProgressCallback when
// it exists.
struct ProgressBridge {
    ProgressCallback*  cb;
};
int xfer_progress(void* clientp,
                   curl_off_t dltotal, curl_off_t dlnow,
                   curl_off_t /*ultotal*/, curl_off_t /*ulnow*/) {
    auto* bridge = static_cast<ProgressBridge*>(clientp);
    if (bridge && bridge->cb && *bridge->cb) {
        (*bridge->cb)(static_cast<std::size_t>(dlnow),
                       static_cast<std::size_t>(dltotal));
    }
    return 0; // non-zero would abort the transfer
}

// ----- cache resolution -------------------------------------------------

fs::path cache_root() {
    if (const char* env = std::getenv("NEOCLAW_CACHE_DIR"); env && *env)
        return fs::path(env);
    if (const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg && *xdg)
        return fs::path(xdg) / "neoclaw";
    if (const char* home = std::getenv("HOME"); home && *home)
        return fs::path(home) / ".cache" / "neoclaw";
    return fs::temp_directory_path() / "neoclaw-cache";
}

// Repo IDs contain a `/` (org/name). We replace it with `--` so the
// cache directory is one segment deep — easier to enumerate, and avoids
// the sub-directory accumulating zillions of unrelated repos.
std::string sanitize_repo(const std::string& repo) {
    std::string out = repo;
    for (char& c : out) if (c == '/') c = '-';
    // Defensive: also flatten whitespace and shell metacharacters that
    // might wander in via misconfigured YAML. Leave alphanumerics,
    // dashes, underscores, dots untouched.
    for (char& c : out) {
        if (!(std::isalnum(static_cast<unsigned char>(c)) ||
              c == '-' || c == '_' || c == '.'))
            c = '_';
    }
    return out;
}

fs::path cache_file_path(const std::string& repo, const std::string& filename) {
    return cache_root() / sanitize_repo(repo) / filename;
}

// ----- HTTP plumbing ----------------------------------------------------

// GET <url> → body string. Follows redirects (HF API routes to the
// right datacenter) and verifies TLS.
std::string http_get_string(const std::string& url) {
    ensure_curl_global();
    CURL* curl = curl_easy_init();
    if (!curl) throw std::runtime_error("curl_easy_init failed");

    std::string body;
    char errbuf[CURL_ERROR_SIZE] = {0};
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 10L);
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 1L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_to_string);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &body);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "neoclaw/0.4 (+libcurl)");
    curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, errbuf);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 30L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 120L);

    const CURLcode rc = curl_easy_perform(curl);
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    curl_easy_cleanup(curl);

    if (rc != CURLE_OK) {
        throw std::runtime_error(
            "GET " + url + " failed: " +
            (errbuf[0] ? errbuf : curl_easy_strerror(rc)));
    }
    if (http_code >= 400) {
        throw std::runtime_error(
            "GET " + url + " returned HTTP " + std::to_string(http_code));
    }
    return body;
}

// GET <url> → write to FILE*, with progress callback. Used for model
// downloads where buffering in memory would blow RSS.
void http_get_to_file(const std::string& url,
                      std::FILE* out,
                      ProgressCallback& progress) {
    ensure_curl_global();
    CURL* curl = curl_easy_init();
    if (!curl) throw std::runtime_error("curl_easy_init failed");

    char errbuf[CURL_ERROR_SIZE] = {0};
    ProgressBridge bridge{&progress};

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 10L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_to_file);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, out);
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, xfer_progress);
    curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &bridge);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "neoclaw/0.4 (+libcurl)");
    curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, errbuf);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 30L);
    // No CURLOPT_TIMEOUT for downloads — a slow link on a 4 GB file
    // would otherwise abort partway through.

    const CURLcode rc = curl_easy_perform(curl);
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    curl_easy_cleanup(curl);

    if (rc != CURLE_OK) {
        throw std::runtime_error(
            "download " + url + " failed: " +
            (errbuf[0] ? errbuf : curl_easy_strerror(rc)));
    }
    if (http_code >= 400) {
        throw std::runtime_error(
            "download " + url + " returned HTTP " + std::to_string(http_code));
    }
}

// ----- GGUF picking -----------------------------------------------------
//
// Higher score wins. The priority list mirrors what most quantizers ship
// for instruction-tuned chat models: Q4_K_M is the modern sweet spot,
// Q4_K_S a touch smaller, Q5_K_M nicer if the user has the VRAM, Q8_0
// near-lossless, F16 the fallback for tiny models. Anything we don't
// recognise gets a non-zero floor so a single-quant repo still resolves.
int score_quant(const std::string& fname) {
    auto contains_ci = [&](const char* needle) {
        // Case-insensitive substring — HF filenames mix Q4_K_M and q4_k_m
        // depending on the uploader.
        std::string hay = fname;
        std::transform(hay.begin(), hay.end(), hay.begin(),
                        [](unsigned char c){ return std::tolower(c); });
        std::string ndl = needle;
        std::transform(ndl.begin(), ndl.end(), ndl.begin(),
                        [](unsigned char c){ return std::tolower(c); });
        return hay.find(ndl) != std::string::npos;
    };
    if (contains_ci("Q4_K_M")) return 100;
    if (contains_ci("Q4_K_S")) return  90;
    if (contains_ci("Q5_K_M")) return  85;
    if (contains_ci("Q5_K_S")) return  80;
    if (contains_ci("Q4_0"))   return  70;
    if (contains_ci("Q8_0"))   return  60;
    if (contains_ci("F16"))    return  40;
    if (contains_ci("BF16"))   return  35;
    return 1;
}

// Parse the HF tree-listing JSON. Endpoint:
//   https://huggingface.co/api/models/<repo>/tree/main
// Returns the array of `{path: "...", type: "file", size: N}` entries.
std::vector<std::string> list_repo_ggufs(const std::string& repo) {
    const std::string url =
        "https://huggingface.co/api/models/" + repo + "/tree/main";
    const std::string body = http_get_string(url);

    json j;
    try {
        j = json::parse(body);
    } catch (const std::exception& e) {
        throw std::runtime_error(
            std::string("HF tree listing parse failed: ") + e.what());
    }
    if (!j.is_array())
        throw std::runtime_error(
            "HF tree listing was not an array (repo missing or private?)");

    std::vector<std::string> ggufs;
    for (const auto& entry : j) {
        if (!entry.is_object() || !entry.contains("path")) continue;
        const auto path = entry["path"].get<std::string>();
        if (path.size() >= 5 &&
            path.compare(path.size() - 5, 5, ".gguf") == 0) {
            ggufs.push_back(path);
        }
    }
    return ggufs;
}

} // namespace

// =======================================================================
// Public API
// =======================================================================

std::string download(const std::string& repo,
                     const std::string& filename,
                     ProgressCallback   progress) {
    const fs::path target = cache_file_path(repo, filename);

    // Cache hit — the file is fully written. We don't validate the
    // content (no checksum from HF for the listing endpoint without
    // extra round-trips); a corrupt cache will surface as a llama.cpp
    // load error which the caller bubbles up clearly enough.
    if (fs::exists(target) && fs::is_regular_file(target)) {
        return target.string();
    }

    fs::create_directories(target.parent_path());

    const fs::path partial = target.string() + ".partial";
    std::FILE* fp = std::fopen(partial.string().c_str(), "wb");
    if (!fp)
        throw std::runtime_error("cannot open " + partial.string() + " for writing");

    const std::string url =
        "https://huggingface.co/" + repo + "/resolve/main/" + filename;

    try {
        http_get_to_file(url, fp, progress);
    } catch (...) {
        std::fclose(fp);
        std::error_code ec;
        fs::remove(partial, ec); // best-effort
        throw;
    }
    std::fclose(fp);

    // Atomic rename — point of no return. If the rename fails we leave
    // the .partial in place so the next run can still see it for forensics.
    std::error_code ec;
    fs::rename(partial, target, ec);
    if (ec)
        throw std::runtime_error(
            "rename " + partial.string() + " → " + target.string()
            + " failed: " + ec.message());

    return target.string();
}

std::string download_best_gguf(const std::string& repo,
                                ProgressCallback   progress) {
    const auto ggufs = list_repo_ggufs(repo);
    if (ggufs.empty())
        throw std::runtime_error("no .gguf files found in repo " + repo);

    // Pick highest-scoring filename. Ties broken by lexicographic order
    // for determinism — useful when comparing two machines' caches.
    std::string best = ggufs.front();
    int best_score = score_quant(best);
    for (size_t i = 1; i < ggufs.size(); ++i) {
        const int s = score_quant(ggufs[i]);
        if (s > best_score || (s == best_score && ggufs[i] < best)) {
            best = ggufs[i];
            best_score = s;
        }
    }
    return download(repo, best, progress);
}

} // namespace neoclaw::hub
