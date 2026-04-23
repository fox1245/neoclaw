#!/usr/bin/env bash
# Package neoclaw + its runtime shared libs into a self-contained tarball
# that users can download, untar, and run. No system llama.cpp install
# needed — everything ships inside the bundle.
#
#   scripts/package.sh            → dist/neoclaw-<version>-linux-x86_64.tar.gz
#   scripts/package.sh --name X   → use X as the bundle stem
#
# Assumes you've already run cmake + cmake --build build.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/.." && pwd)"
build="$root/build"

if [[ ! -x "$build/neoclaw" ]]; then
    echo "[package] $build/neoclaw not found — build first:" >&2
    echo "           cmake -S $root -B $build -DCMAKE_BUILD_TYPE=Release" >&2
    echo "           cmake --build $build -j\$(nproc)"         >&2
    exit 1
fi

# Resolve version from CMakeLists (neoclaw VERSION 0.1.0 style).
version="$(grep -oP 'project\(neoclaw[^)]*VERSION \K[0-9.]+' "$root/CMakeLists.txt" | head -1)"
version="${version:-0.0.0}"

arch="$(uname -m)"
os="$(uname -s | tr '[:upper:]' '[:lower:]')"
stem="neoclaw-${version}-${os}-${arch}"
for opt in "$@"; do
    case "$opt" in
        --name) stem="$2"; shift 2 ;;
    esac
done

stage="$(mktemp -d)"
pkg="$stage/$stem"
mkdir -p "$pkg/bin" "$pkg/lib" "$pkg/config"

# 1. neoclaw binary — re-linked with $ORIGIN/../lib RPATH so the
# bundled libs are found at runtime without LD_LIBRARY_PATH gymnastics.
cp "$build/neoclaw" "$pkg/bin/neoclaw"
patchelf --set-rpath '$ORIGIN/../lib' "$pkg/bin/neoclaw" 2>/dev/null || {
    echo "[package] warning: patchelf not available — binary RPATH unchanged" >&2
    echo "           install patchelf or run from build tree with LD_LIBRARY_PATH" >&2
}

# 2. Runtime shared libs. Follow symlinks to the real versioned file
# AND keep the short SONAME so ld.so finds them.
for soname in libllama.so.0 libggml.so.0 libggml-base.so.0 libggml-cpu.so.0; do
    src="$build/bin/$soname"
    [[ -f "$src" ]] || { echo "[package] missing: $src" >&2; exit 2; }
    real="$(readlink -f "$src")"
    cp "$real" "$pkg/lib/$(basename "$real")"
    ln -sf "$(basename "$real")" "$pkg/lib/$soname"
done

# 3. Example config.
cp "$root/config/neoclaw.example.yaml" "$pkg/config/neoclaw.example.yaml"
cp "$root/README.md"                    "$pkg/README.md"
cp "$root/LICENSE"                      "$pkg/LICENSE"

# 4. Small shim README inside the bundle so new users have a one-page
# getting-started at the top of the archive.
cat > "$pkg/QUICKSTART.md" <<EOF
# neoclaw ${version}

Run:

    ./bin/neoclaw --project-root /path/to/your/project

On first run the model (~4.6 GB Gemma-4 E4B Q4_K_M) downloads to
\`~/.cache/transformercpp/\`. Subsequent runs re-use the cache.

Configuration: copy \`config/neoclaw.example.yaml\` to \`./neoclaw.yaml\`
in your project root (or \`~/.config/neoclaw/config.yaml\`) and edit.

See the full README.md for tools, sandbox, and model-swap notes.
EOF

# 5. Tarball.
mkdir -p "$root/dist"
tar -C "$stage" -czf "$root/dist/$stem.tar.gz" "$stem"
echo "[package] wrote $root/dist/$stem.tar.gz"
ls -l "$root/dist/$stem.tar.gz" | awk '{printf "          size: %.1f MB\n", $5/1024/1024}'

rm -rf "$stage"
