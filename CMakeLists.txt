cmake_minimum_required(VERSION 3.20)

# yaml-cpp 0.8.0 still declares cmake_minimum_required(VERSION 3.4), which
# CMake 4.x refuses by default. Honour the legacy floor through
# FetchContent. Harmless on older CMake; required on 4.0+.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)

# ---------------------------------------------------------------------------
# GPU support (opt-in)
#
# When NEOCLAW_BUILD_CUDA=ON, propagate `-DGGML_CUDA=ON` through FetchContent
# so llama.cpp builds its CUDA backend. At runtime llama discovers the CUDA
# backend dynamically — if it's there, every tensor offloads to GPU; if it's
# not (or NVIDIA driver is missing), the CPU kernels handle it. Same binary
# either way.
#
# CUDA toolkit (nvcc) must be available at configure time. Default OFF for
# maximum portability of the source build; `scripts/package.sh` flips it ON
# for release tarballs because the perf delta is ~15-20×.
# ---------------------------------------------------------------------------
option(NEOCLAW_BUILD_CUDA
    "Build llama.cpp with CUDA backend. Requires nvcc."
    OFF)
if(NEOCLAW_BUILD_CUDA)
    enable_language(CUDA)
    set(GGML_CUDA           ON  CACHE BOOL "" FORCE)
    set(GGML_CUDA_F16       ON  CACHE BOOL "" FORCE)   # half-precision where safe
    set(CMAKE_CUDA_STANDARD 17  CACHE STRING "" FORCE)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        # Covers Turing (75) through Ada Lovelace (89). Users with newer
        # silicon (Blackwell, 120) can append via -DCMAKE_CUDA_ARCHITECTURES.
        set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89" CACHE STRING "" FORCE)
    endif()
    message(STATUS "neoclaw: CUDA backend enabled (archs=${CMAKE_CUDA_ARCHITECTURES})")
endif()

project(neoclaw LANGUAGES CXX C VERSION 0.5.0)

# ---------------------------------------------------------------------------
# Project-wide C++ settings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Dependencies
#
# v0.4 architecture: neoclaw is a *harness* application. Inference is a
# commodity dependency that we consume from llama.cpp directly — no wrapper
# layer. This frees us to track upstream llama.cpp releases on our own
# cadence (new quants, KV-cache reuse, speculative decoding) without waiting
# on a transitive maintainer.
# ---------------------------------------------------------------------------
include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

# ----- NeoGraph (orchestration / Provider interface / Agent loop) -----
FetchContent_Declare(
    NeoGraph
    GIT_REPOSITORY https://github.com/fox1245/NeoGraph.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE
)

# Turn off the pieces neoclaw doesn't link — faster build, smaller binary.
set(NEOGRAPH_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
set(NEOGRAPH_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
set(NEOGRAPH_BUILD_MCP       OFF CACHE BOOL "" FORCE)
set(NEOGRAPH_BUILD_POSTGRES  OFF CACHE BOOL "" FORCE)
set(NEOGRAPH_BUILD_SQLITE    OFF CACHE BOOL "" FORCE)
set(NEOGRAPH_BUILD_UTIL      OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(NeoGraph)

# ----- llama.cpp (inference runtime, direct) -----
#
# Pinned to b8902 — the build verified against Gemma-4 E4B Q4_K_M during
# the v0.4 TransformerCPP-removal cutover. Bump the tag to follow upstream;
# the C API surface neoclaw uses (load_from_file / init_from_model /
# tokenize / decode / sampler chain / token_to_piece) is stable across
# recent releases.
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL           OFF CACHE BOOL "" FORCE)  # we do HF download ourselves
FetchContent_Declare(
    llama
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG        b8902
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(llama)

# ----- libcurl (HuggingFace Hub downloader) -----
#
# Use the system libcurl. Every Linux distro that the release tarball
# already requires (glibc ≥ 2.34, libssl3) ships libcurl as a system
# package, and following HF's cross-host CDN redirects (huggingface.co →
# cdn-lfs.huggingface.co) wants a mature HTTP client — not worth
# rewriting.
find_package(CURL REQUIRED)

# ----- FTXUI (clickable TUI for the mode picker) -----
#
# When neoclaw is launched on a TTY without `topology:` set in YAML, we
# pop a small radiobox listing the bundled topologies (pair / code-
# review / …) and the user picks one with arrows-or-mouse → Enter.
# Esc falls back to the v0.5 Agent default. Pure C++17, no external
# deps, ~1 MB statically-linked. We keep it OFF for headless builds
# (e.g. cibuildwheel) where the picker is never reachable anyway.
option(NEOCLAW_BUILD_TUI
    "Build the FTXUI mode picker. OFF strips the picker — neoclaw still runs but topology must be set in YAML or via env."
    ON)
if(NEOCLAW_BUILD_TUI)
    set(FTXUI_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
    set(FTXUI_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(FTXUI_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    # Build FTXUI static so the release tarball stays a single-binary
    # drop. We already PIC-ify everything (CMAKE_POSITION_INDEPENDENT_CODE
    # above) so this links cleanly into our PIE executable. Saves us
    # adding three .so paths to the install RPATH.
    set(FTXUI_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS       OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        ftxui
        GIT_REPOSITORY https://github.com/ArthurSonzogni/FTXUI.git
        # Pinned to the verified build (mode picker layout + mouse SGR-1006
        # tracking + Radiobox detail panel verified against this commit).
        # Bump the tag to follow upstream — FTXUI's component API is
        # stable across minor releases.
        GIT_TAG        03224dcaf7f9a059dbb13cd2977d3d06729107ed
        GIT_SHALLOW    FALSE
    )
    FetchContent_MakeAvailable(ftxui)
endif()

# ----- yaml-cpp (tiny YAML parser) -----
FetchContent_Declare(
    yaml-cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
    GIT_TAG        0.8.0
    GIT_SHALLOW    TRUE
)
set(YAML_CPP_BUILD_TESTS  OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS  OFF CACHE BOOL "" FORCE)
set(YAML_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(yaml-cpp)

# ---------------------------------------------------------------------------
# neoclaw binary
# ---------------------------------------------------------------------------
add_executable(neoclaw
    src/main.cpp
    src/config.cpp
    src/gemma_provider.cpp
    src/hub.cpp
    src/llama_runner.cpp
    src/local_provider.cpp
    src/sandbox.cpp
    src/tools.cpp
    src/topology.cpp
    src/ui.cpp
    $<$<BOOL:${NEOCLAW_BUILD_TUI}>:${CMAKE_CURRENT_SOURCE_DIR}/src/mode_picker.cpp>
)

target_include_directories(neoclaw PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Link against the NeoGraph orchestration layer + llama.cpp inference
# runtime + libcurl for HF downloads. neograph::llm pulls in the HTTP
# OpenAIProvider that the gemma_provider.cpp HTTP backend reuses.
target_link_libraries(neoclaw PRIVATE
    neograph::core
    neograph::llm
    llama
    ggml
    ggml-cpu
    ggml-base
    CURL::libcurl
    yaml-cpp::yaml-cpp)

if(NEOCLAW_BUILD_TUI)
    target_link_libraries(neoclaw PRIVATE
        ftxui::screen
        ftxui::dom
        ftxui::component)
    target_compile_definitions(neoclaw PRIVATE NEOCLAW_HAVE_TUI=1)
endif()

# BUILD_RPATH points at wherever the shared libs actually landed in this
# build. Under FetchContent + llama.cpp's defaults that's CMAKE_BINARY_DIR/bin.
set_target_properties(neoclaw PROPERTIES
    # Embed both the build-tree lib dir AND the install-tree relative
    # path into the binary's RPATH so it runs correctly from either
    # location — no patchelf repacking required when packaging.
    #
    # Three build-tree dirs to cover:
    #   bin/                   — llama / ggml shared libs
    #   _deps/neograph-build/  — neograph_core / _llm / _async (FetchContent
    #                            output dir, not bin/ by default)
    #   $ORIGIN/../lib         — install-tree fallback for shipped tarball
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${CMAKE_BINARY_DIR}/bin:${neograph_BINARY_DIR}:$ORIGIN/../lib")

if(NEOCLAW_BUILD_CUDA)
    set(_neoclaw_has_cuda TRUE CACHE INTERNAL "")
endif()

# ---------------------------------------------------------------------------
# Bundled JSON topologies
#
# Copy `topologies/*.json` next to the binary at build time so a user can
# write `topology: pair.json` in neoclaw.yaml and have it resolve via the
# `<exe-dir>/topologies/<name>` lookup path implemented in topology.cpp.
# Same idea at install time: place them in `share/neoclaw/topologies/` so
# the FHS-style `<exe>/../share/...` fallback finds them.
# ---------------------------------------------------------------------------
file(GLOB _neoclaw_topology_files
    "${CMAKE_CURRENT_SOURCE_DIR}/topologies/*.json")
add_custom_target(neoclaw_topologies ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory
        "$<TARGET_FILE_DIR:neoclaw>/topologies"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${_neoclaw_topology_files}
        "$<TARGET_FILE_DIR:neoclaw>/topologies/"
    DEPENDS ${_neoclaw_topology_files}
    COMMENT "Staging neoclaw topologies next to binary"
    VERBATIM)
add_dependencies(neoclaw neoclaw_topologies)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "neoclaw v${PROJECT_VERSION} configured:")
message(STATUS "  build type       : ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA backend     : ${NEOCLAW_BUILD_CUDA}")
message(STATUS "  NeoGraph         : ${neograph_SOURCE_DIR}")
message(STATUS "  llama.cpp        : ${llama_SOURCE_DIR}")
message(STATUS "  libcurl          : ${CURL_VERSION_STRING}")
message(STATUS "  yaml-cpp         : ${yaml-cpp_SOURCE_DIR}")
message(STATUS "")
