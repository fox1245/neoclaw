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
    "Build llama.cpp with CUDA backend (via TransformerCPP). Requires nvcc."
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

project(neoclaw LANGUAGES CXX C VERSION 0.3.0)

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
# Dependencies via FetchContent
#
# Everything is fetched from upstream. `neoclaw` is a clone-and-build repo;
# there is no system-wide NeoGraph/TransformerCPP install step. Pin tags
# when stabilising a release — `main`/`master` are fine during the v0.x
# prototype phase.
# ---------------------------------------------------------------------------
include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

# ----- NeoGraph -----
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

# ----- TransformerCPP -----
#
# TransformerCPP currently expects a fully populated `extern/` tree
# (safetensors-cpp, sentencepiece, libtorch, onnxruntime) that upstream
# does NOT commit — those paths are in its .gitignore. A plain
# FetchContent clone therefore can't build alone.
#
# Until TransformerCPP exposes its own FetchContent for those externals,
# we support two paths:
#
#   1. You have a built sibling clone — point CMake at it:
#        -DFETCHCONTENT_SOURCE_DIR_TRANSFORMERCPP=/path/to/TransformerCPP
#
#   2. Nothing set — we auto-detect `../TransformerCPP` next to neoclaw.
#      If present, we reuse it in place. If missing, we fall back to a
#      plain clone and emit a warning explaining the extern/ gap.
if(NOT DEFINED FETCHCONTENT_SOURCE_DIR_TRANSFORMERCPP)
    set(_neoclaw_sibling "${CMAKE_CURRENT_SOURCE_DIR}/../TransformerCPP")
    if(EXISTS "${_neoclaw_sibling}/extern/safetensors-cpp/safetensors.hh")
        set(FETCHCONTENT_SOURCE_DIR_TRANSFORMERCPP
            "${_neoclaw_sibling}" CACHE PATH "" FORCE)
        message(STATUS
            "TransformerCPP: reusing sibling clone at ${_neoclaw_sibling}")
    else()
        message(WARNING
            "TransformerCPP's vendored extern/ tree (safetensors-cpp, "
            "sentencepiece, …) is gitignored upstream. A fresh "
            "FetchContent clone will fail to build. Clone TransformerCPP "
            "separately, populate its extern/ (see its README), and pass "
            "-DFETCHCONTENT_SOURCE_DIR_TRANSFORMERCPP=/path/to/clone to "
            "neoclaw's cmake invocation.")
    endif()
endif()

FetchContent_Declare(
    TransformerCPP
    GIT_REPOSITORY https://github.com/fox1245/TransformerCPP.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(TransformerCPP)

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
    src/local_provider.cpp
    src/sandbox.cpp
    src/tools.cpp
    src/ui.cpp
)

target_include_directories(neoclaw PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    # TransformerCPP headers are consumed directly by local_provider.cpp
    # to drive Model::generate in-process. The `transformercpp` target
    # doesn't export its include dirs publicly, so list them here.
    ${transformercpp_SOURCE_DIR}/include
    # llama.h is needed to register our quiet log callback that mutes
    # llama.cpp's per-load tensor manifest dump.
    ${transformercpp_SOURCE_DIR}/extern/llama.cpp/include
    ${transformercpp_SOURCE_DIR}/extern/llama.cpp/ggml/include)

# Link against the NeoGraph orchestration layer + the TransformerCPP
# inference layer. The HTTP backend needs neograph::llm's OpenAIProvider;
# the local backend needs the transformercpp + llama + ggml chain.
target_link_libraries(neoclaw PRIVATE
    neograph::core
    neograph::llm
    transformercpp
    yaml-cpp::yaml-cpp)

# llama.cpp + ggml targets are exposed by the add_subdirectory call
# inside TransformerCPP. Link against the targets directly so CMake
# resolves dependencies and RPATHs without us guessing output paths.
# `llama` pulls in `ggml` transitively through llama.cpp's own CMake,
# but we keep the ggml variants explicit so the single-library split
# (base / cpu / cuda) is visible at link time.
target_link_libraries(neoclaw PRIVATE
    llama
    ggml
    ggml-cpu
    ggml-base)
# BUILD_RPATH points at wherever the shared libs actually landed in this
# build. Under FetchContent + llama.cpp's defaults that's CMAKE_BINARY_DIR/bin.
set_target_properties(neoclaw PROPERTIES
    # Embed both the build-tree lib dir AND the install-tree relative
    # path into the binary's RPATH so it runs correctly from either
    # location — no patchelf repacking required when packaging.
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${CMAKE_BINARY_DIR}/bin:$ORIGIN/../lib")

# When CUDA is compiled in, llama.cpp emits an extra backend shared lib
# (`libggml-cuda.so`) that must end up next to the CPU-side libs at
# runtime. llama.cpp already links it transitively into ggml, but the
# file also needs to be discoverable from the install-tree RPATH; the
# rpath already covers it since it sits in the same bin/ directory.
if(NEOCLAW_BUILD_CUDA)
    set(_neoclaw_has_cuda TRUE CACHE INTERNAL "")
    # CUDAToolkit::cudart is pulled in by llama.cpp's CMake; no extra
    # target_link_libraries here — just make sure nvcc ran.
endif()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "neoclaw v${PROJECT_VERSION} configured:")
message(STATUS "  build type       : ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA backend     : ${NEOCLAW_BUILD_CUDA}")
message(STATUS "  NeoGraph         : ${neograph_SOURCE_DIR}")
message(STATUS "  TransformerCPP   : ${transformercpp_SOURCE_DIR}")
message(STATUS "  yaml-cpp         : ${yaml-cpp_SOURCE_DIR}")
message(STATUS "")
