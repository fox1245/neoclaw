#include "ui.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <iostream>
#include <locale>
#include <string>
#include <unistd.h>

namespace neoclaw::ui {

// ---------------------------------------------------------------------
// Width via POSIX wcwidth. mbtowc over the UTF-8 bytes then ask libc
// how many columns the resulting wide char occupies. Assumes an
// UTF-8 locale is live — caller installs one once at startup.
// ---------------------------------------------------------------------
size_t visual_width(std::string_view s) {
    size_t cols = 0;
    const char*  p   = s.data();
    const char*  end = p + s.size();
    std::mbstate_t st{};
    while (p < end) {
        wchar_t wc = 0;
        size_t  n  = std::mbrtowc(&wc, p, static_cast<size_t>(end - p), &st);
        if (n == 0) break;
        if (n == static_cast<size_t>(-1) || n == static_cast<size_t>(-2)) {
            // Invalid / incomplete sequence — count one byte as one
            // column so the loop still terminates.
            ++cols; ++p;
            st = std::mbstate_t{};
            continue;
        }
        int w = wcwidth(wc);
        if (w < 0) w = 0;        // non-printable / combining
        cols += static_cast<size_t>(w);
        p += n;
    }
    return cols;
}

// ---------------------------------------------------------------------
// Colour gating — isatty(STDOUT) + NO_COLOR env. Cached.
// ---------------------------------------------------------------------
bool colour_enabled() {
    static const bool cached = [] {
        if (const char* nc = std::getenv("NO_COLOR"); nc && *nc) return false;
        return ::isatty(STDOUT_FILENO) != 0;
    }();
    return cached;
}

static std::string esc(const char* s) {
    return colour_enabled() ? std::string(s) : std::string();
}

std::string dim()       { return esc("\x1b[2m");  }
std::string bold()      { return esc("\x1b[1m");  }
std::string reset()     { return esc("\x1b[0m");  }
std::string fg_cyan()   { return esc("\x1b[36m"); }
std::string fg_green()  { return esc("\x1b[32m"); }
std::string fg_yellow() { return esc("\x1b[33m"); }
std::string fg_magenta(){ return esc("\x1b[35m"); }
std::string fg_red()    { return esc("\x1b[31m"); }

// ---------------------------------------------------------------------
// Banner. Inner width = 55 visual columns. Row layout:
//
//     " key    : value<pad>"
//       ^col 2           col 57
//
// Every row is padded to exactly INNER columns regardless of the
// UTF-8 contents of `value`.
// ---------------------------------------------------------------------
namespace {
constexpr size_t INNER = 55;

std::string pad_to(std::string_view s, size_t target) {
    const size_t w = visual_width(s);
    if (w >= target) return std::string(s);
    return std::string(s) + std::string(target - w, ' ');
}

// Unicode box-drawing (╭─╮│╰╯) renders unevenly on terminals with a
// CJK font fallback — `─` ends up wider than `│` and the right edge
// drifts, breaking the box. ASCII (`+-|`) renders 1 cell per char in
// every monospace font on the planet.
void row(std::string_view content) {
    std::cout << "  " << fg_cyan() << "| " << reset()
              << pad_to(content, INNER)
              << fg_cyan() << " |" << reset() << "\n";
}

void bar(const char* left, const char* fill, const char* right) {
    std::cout << "  " << fg_cyan() << left;
    for (size_t i = 0; i < INNER + 2; ++i) std::cout << fill;
    std::cout << right << reset() << "\n";
}

// Truncate with an ellipsis if longer than `max` visual cols.
std::string fit(std::string_view s, size_t max) {
    if (visual_width(s) <= max) return std::string(s);
    // Byte-walk and stop at max-1 cols, then append '…'.
    std::mbstate_t st{};
    const char* p   = s.data();
    const char* end = p + s.size();
    size_t cols     = 0;
    while (p < end) {
        wchar_t wc = 0;
        size_t  n  = std::mbrtowc(&wc, p, static_cast<size_t>(end - p), &st);
        if (n == 0 || n == static_cast<size_t>(-1) || n == static_cast<size_t>(-2)) break;
        int w = wcwidth(wc);
        if (w < 0) w = 0;
        if (cols + static_cast<size_t>(w) + 1 > max) break;
        cols += static_cast<size_t>(w);
        p += n;
    }
    return std::string(s.data(), p - s.data()) + "…";
}

std::string field(const char* key, std::string_view value) {
    const size_t key_width = 10;
    std::string out;
    out.reserve(80);
    out = std::string(" ") + key;
    // Pad key to fixed column.
    const size_t kw = visual_width(out);
    if (kw < key_width) out += std::string(key_width - kw, ' ');
    out += ": ";
    // Remaining space for the value.
    const size_t used = visual_width(out);
    const size_t avail = INNER > used ? INNER - used : 0;
    out += fit(value, avail);
    return out;
}
} // namespace

void print_banner(const BannerLines& b) {
    std::cout << "\n";
    bar("+", "-", "+");
    row(" " + b.title);
    row("");
    row(field("model",    b.model));
    row(field("endpoint", b.endpoint));
    row(field("project",  b.project));
    row(field("bash",     b.bash_line));
    row("");
    row(dim() + " " + b.hint + reset());
    bar("+", "-", "+");
    std::cout << "\n";
}

// ---------------------------------------------------------------------
// Tool trace — emitted to stderr so it doesn't interleave with the
// streamed assistant output on stdout.
// ---------------------------------------------------------------------
void print_tool_start(const std::string& name,
                       const std::string& args_preview) {
    std::cerr << fg_yellow() << "  ⟶ " << reset()
              << bold() << name << reset()
              << dim() << "(" << fit(args_preview, 80) << ")" << reset()
              << "\n";
}

void print_tool_result(const std::string& summary) {
    std::cerr << fg_green() << "  ← " << reset()
              << fit(summary, 96) << "\n";
}

// ---------------------------------------------------------------------
// Strip chat-template artifacts as they stream through. Carry the
// unmatched tail between chunks so a token split across the boundary
// still gets caught on the next call.
// ---------------------------------------------------------------------
namespace {
constexpr std::array<std::string_view, 9> kControlTokens = {
    "<|im_end|>",
    "<|im_start|>",
    "<end_of_turn>",
    "<start_of_turn>",
    // Some Gemma checkpoints occasionally emit slash-variant turn
    // markers in generated text. Strip them too so the user doesn't
    // see raw `</start_of_turn>` bleeding through after the real
    // turn ended.
    "</end_of_turn>",
    "</start_of_turn>",
    "<|end_of_text|>",
    "<|eot_id|>",
    "<eos>",
};

// Longest control-token length, used to size the carry buffer.
constexpr size_t kMaxTokenLen = 16;
} // namespace

void strip_chat_artifacts(std::string& fragment, std::string& carry) {
    // Combine carry + fragment, scan from 0, push clean bytes to out.
    std::string combined;
    combined.reserve(carry.size() + fragment.size());
    combined.append(carry);
    combined.append(fragment);

    std::string out;
    out.reserve(combined.size());

    size_t i = 0;
    while (i < combined.size()) {
        bool matched = false;
        if (combined[i] == '<') {
            // Check against each token.
            for (const auto& t : kControlTokens) {
                if (combined.size() - i >= t.size()
                    && combined.compare(i, t.size(), t) == 0) {
                    i += t.size();
                    matched = true;
                    break;
                }
            }
            // If an incomplete `<` prefix might still grow into a
            // control token (we don't have enough bytes yet), stop
            // here and push the remainder into carry.
            if (!matched) {
                const size_t remaining = combined.size() - i;
                if (remaining < kMaxTokenLen) {
                    // Could still be the start of a token split across
                    // the next chunk — defer.
                    break;
                }
            }
        }
        if (!matched) {
            out.push_back(combined[i]);
            ++i;
        }
    }

    carry.assign(combined, i, std::string::npos);
    fragment = std::move(out);
}

} // namespace neoclaw::ui
