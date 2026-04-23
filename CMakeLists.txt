cmake_minimum_required(VERSION 3.20)

# yaml-cpp 0.8.0 still declares cmake_minimum_required(VERSION 3.4), which
# CMake 4.x refuses by default. Honour the legacy floor through
# FetchContent. Harmless on older CMake; required on 4.0+.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)

project(neoclaw LANGUAGES CXX C VERSION 0.1.0)

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
    src/sandbox.cpp
    src/tools.cpp
    src/ui.cpp
)

target_include_directories(neoclaw PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Link against the NeoGraph orchestration layer + the TransformerCPP
# inference layer. neoclaw uses the LLM surface of NeoGraph only — not
# MCP, not checkpoints — so we stay on the minimal core+llm+async split.
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
    BUILD_RPATH   "${CMAKE_BINARY_DIR}/bin"
    INSTALL_RPATH "$ORIGIN/../lib")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "neoclaw v${PROJECT_VERSION} configured:")
message(STATUS "  build type       : ${CMAKE_BUILD_TYPE}")
message(STATUS "  NeoGraph         : ${neograph_SOURCE_DIR}")
message(STATUS "  TransformerCPP   : ${transformercpp_SOURCE_DIR}")
message(STATUS "  yaml-cpp         : ${yaml-cpp_SOURCE_DIR}")
message(STATUS "")
