cmake_minimum_required(VERSION 3.18)
project(feelmri LANGUAGES CXX)

# ------------------------------------------------------------
# Optional GPU backend
# ------------------------------------------------------------
# Default is OFF: standard builds compile identically to the CPU-only
# baseline. Turn on with `-DFEELMRI_ENABLE_GPU=ON` to add the CUDA
# (M1, NVIDIA) or HIP (M2, AMD) compute path. The backend selector
# below switches between CUDA and HIP sources; the kernel TUs share the
# same source files because they go through the vendor-neutral shim at
# cpp/feelmri/runtime/device.hpp.
option(FEELMRI_ENABLE_GPU "Compile GPU compute path (CUDA / HIP)" OFF)
set(FEELMRI_GPU_BACKEND "cuda" CACHE STRING
    "GPU backend when FEELMRI_ENABLE_GPU=ON: 'cuda' (NVIDIA via CUDA) or "
    "'hip' (AMD via ROCm, or NVIDIA via __HIP_PLATFORM_NVIDIA__).")

# Globally cache downloaded dependencies across all builds
set(FETCHCONTENT_BASE_DIR "$ENV{HOME}/.cache/cmake" CACHE PATH "Directory for downloaded FetchContent dependencies")

# =========================================================
# Compiler Caching (ccache) integration
# =========================================================
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    message(STATUS "✅ Found ccache: ${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
else()
    message(STATUS "ℹ️ ccache not found. Install ccache to speed up recompilations.")
endif()

# =========================================================
# Dependency Detection
# =========================================================
message(STATUS "Checking for required system dependencies...")

# Core C++ Mathematical & Parallel Libraries
find_package(MPI)
if(MPI_FOUND)
    message(STATUS "✅ Found MPI")
endif()

find_package(BLAS REQUIRED)
if(BLAS_FOUND)
    message(STATUS "✅ Found BLAS")
endif()

find_package(LAPACK REQUIRED)
if(LAPACK_FOUND)
    message(STATUS "✅ Found LAPACK")
endif()


# Python Integration
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# ------------------------------------------------------------
# General build configuration
# ------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)  # Enable LTO if supported

# ------------------------------------------------------------
# Conditional optimization flags (architecture and LTO)
# ------------------------------------------------------------
include(CheckCXXCompilerFlag)

function(add_supported_flag flag)
    string(REPLACE "-" "_" flag_var "${flag}")
    string(REPLACE "=" "_" flag_var "${flag_var}")
    check_cxx_compiler_flag("${flag}" HAS_${flag_var})
    if (HAS_${flag_var})
        add_compile_options(${flag})
    endif()
endfunction()

set(CPU_TUNING_FLAG_PAIRS
    "-march=raptorlake;-mtune=raptorlake"
    "-march=native;-mtune=native"
)

set(USED_CPU_FLAGS "")

foreach(pair ${CPU_TUNING_FLAG_PAIRS})
    string(REPLACE ";" " " candidate_flags "${pair}")
    list(GET pair 0 march_flag)
    check_cxx_compiler_flag("${march_flag}" HAS_MARCH)
    if (HAS_MARCH)
        message(STATUS "Using CPU tuning flags: ${candidate_flags}")
        set(USED_CPU_FLAGS ${candidate_flags})
        break()
    endif()
endforeach()

if (USED_CPU_FLAGS)
    # Apply CPU tuning flags only to host C++ TUs. nvcc forwards unknown
    # flags to its default host compiler, which on Pop!_OS 22.04 is gcc-11
    # and does not recognize newer `-march` targets such as raptorlake;
    # the project's CXX build uses gcc-13 where those are valid.
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:${USED_CPU_FLAGS}>)
else()
    message(WARNING "⚠️ No supported CPU tuning flags found — using generic tuning.")
endif()

# ------------------------------------------------------------
# FEniCS Basix (C++) automatic compilation via FetchContent
# ------------------------------------------------------------
# Check if basix is already installed on the system
find_package(basix QUIET)

if(basix_FOUND)
    message(STATUS "✅ Found system basix.")
else()
    # Force Basix to compile statically and embed directly into feelmri extensions
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Force static libraries" FORCE)

    include(FetchContent)
    FetchContent_Declare(
        basix
        GIT_REPOSITORY https://github.com/FEniCS/basix.git
        GIT_TAG v0.10.0
        SOURCE_SUBDIR cpp
    )

    FetchContent_GetProperties(basix)
    if(NOT basix_POPULATED)
        message(STATUS "Fetching and building FEniCS Basix C++ library from GitHub...")
        FetchContent_Populate(basix)
        add_subdirectory(${basix_SOURCE_DIR}/cpp ${basix_BINARY_DIR})
    endif()
endif()

# ------------------------------------------------------------
# Eigen3 automatic installation (cached and robust)
# ------------------------------------------------------------
find_package(Eigen3 5.0 QUIET)

if(Eigen3_FOUND)
    message(STATUS "✅ Found system Eigen3: ${EIGEN3_INCLUDE_DIRS}")
    set(EIGEN_INCLUDE_DIR "${EIGEN3_INCLUDE_DIRS}")
    set(EIGEN_UNSUPPORTED_DIR "${EIGEN3_INCLUDE_DIRS}/unsupported")
else()
    include(FetchContent)
    FetchContent_Declare(
        Eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 5.0.1
    )

    FetchContent_GetProperties(Eigen)
    if (NOT eigen_POPULATED)
        message(STATUS "Fetching Eigen3 (5.0.1) from GitLab...")
        FetchContent_Populate(Eigen)
    endif()

    set(EIGEN_INCLUDE_DIR "${eigen_SOURCE_DIR}")
    set(EIGEN_UNSUPPORTED_DIR "${eigen_SOURCE_DIR}/unsupported")
endif()

include_directories(${EIGEN_INCLUDE_DIR} ${EIGEN_UNSUPPORTED_DIR})

if (EXISTS "${EIGEN_UNSUPPORTED_DIR}/Eigen/CXX11/Tensor")
    message(STATUS "✅ Eigen unsupported module found at: ${EIGEN_UNSUPPORTED_DIR}")
else()
    message(WARNING "⚠️ Eigen unsupported module NOT found — Tensor functionality may be limited.")
endif()


# ------------------------------------------------------------
# Source setup (core modules)
# ------------------------------------------------------------
set(SRC_DIR cpp/feelmri)
set(MODULES MRIAssemble Assemble BlochSimulator PODHelper)

# ------------------------------------------------------------
# GPU backend (CUDA at M1, HIP target at M2). All GPU-only sources
# live under cpp/feelmri/runtime and cpp/feelmri/kernels; they are
# compiled into a single 'feelmri_runtime' shared library that every
# pybind11 module links against. The shared library is the single
# point of ownership for the device runtime singleton, which avoids
# the per-module initialisation pitfall when multiple pybind11
# extensions each pull in their own copy of the CUDA runtime state.
# ------------------------------------------------------------
if(FEELMRI_ENABLE_GPU)
    if(FEELMRI_GPU_BACKEND STREQUAL "hip")
        # HIP build path. Requires `find_package(hip)` and that the .cu
        # sources be treated as HIP (either renamed to .hip, or compiled
        # with hipcc via LANGUAGE HIP). The shim at runtime/device.hpp
        # routes feelmri::gpu::* to hip*; the kernel arithmetic is
        # identical between backends.
        find_package(hip REQUIRED CONFIG)
        enable_language(HIP)
        add_compile_definitions(FEELMRI_GPU_BACKEND_HIP=1)
        message(STATUS "GPU backend enabled (HIP)")
    else()
        # CUDA build path (default).
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
        message(STATUS "GPU backend enabled (CUDA ${CUDAToolkit_VERSION})")
    endif()

    set(FEELMRI_RUNTIME_SOURCES
        ${SRC_DIR}/runtime/device_init.cpp
        ${SRC_DIR}/kernels/BlochSimulator_gpu.cu
        ${SRC_DIR}/kernels/MRIAssemble_gpu.cu
    )
    if(FEELMRI_GPU_BACKEND STREQUAL "hip")
        # When compiling for HIP, the .cu files are treated as HIP TUs.
        set_source_files_properties(
            ${SRC_DIR}/kernels/BlochSimulator_gpu.cu
            ${SRC_DIR}/kernels/MRIAssemble_gpu.cu
            PROPERTIES LANGUAGE HIP)
    endif()
    add_library(feelmri_runtime SHARED ${FEELMRI_RUNTIME_SOURCES})
    target_include_directories(feelmri_runtime PRIVATE ${SRC_DIR})
    set_target_properties(feelmri_runtime PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_STANDARD 17
        CUDA_SEPARABLE_COMPILATION ON
        OUTPUT_NAME feelmri_runtime
        INSTALL_RPATH "$ORIGIN"
        BUILD_WITH_INSTALL_RPATH OFF
    )
    # nvcc rejects -ffast-math on device passes; -fvisibility=hidden on
    # the runtime TU would also hide its extern "C" exports. Apply just
    # PIC + -O3 to the host TU and pass CUDA-specific flags via the
    # COMPILE_LANGUAGE generator expression.
    set_source_files_properties(${SRC_DIR}/runtime/device_init.cpp
        PROPERTIES COMPILE_OPTIONS "-O3;-fPIC")
    target_compile_options(feelmri_runtime PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math --expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:HIP>:-O3 -ffp-contract=fast>)
    if(FEELMRI_GPU_BACKEND STREQUAL "hip")
        target_link_libraries(feelmri_runtime PUBLIC hip::host hip::device)
    else()
        target_link_libraries(feelmri_runtime PUBLIC CUDA::cudart)
    endif()
    install(TARGETS feelmri_runtime LIBRARY DESTINATION feelmri)
endif()

foreach(mod ${MODULES})
    message(STATUS "Building module: ${mod}")
    pybind11_add_module(${mod} ${SRC_DIR}/${mod}.cpp)

    target_include_directories(${mod}
        PRIVATE
            ${SRC_DIR}
            ${EIGEN_INCLUDE_DIR}
            ${EIGEN_UNSUPPORTED_DIR}
            ${pybind11_INCLUDE_DIRS}
            ${Python_INCLUDE_DIRS}
    )

    target_link_libraries(${mod} PRIVATE basix)

    # Apply high-performance compile flags
    target_compile_options(${mod} PRIVATE
        -Ofast
        -ffast-math
        -fno-math-errno
        -fno-trapping-math
        -ffp-contract=fast
        -fvisibility=hidden
        -fPIC
        -funroll-loops
        -fopenmp-simd
        -mavx2
        -mfma
        -DNDEBUG
        -DEIGEN_NO_DEBUG
        -DEIGEN_FAST_MATH
        -DEIGEN_DONT_PARALLELIZE
    )

    if(FEELMRI_ENABLE_GPU)
        target_compile_definitions(${mod} PRIVATE FEELMRI_GPU=1)
        target_link_libraries(${mod} PRIVATE feelmri_runtime)
        set_target_properties(${mod} PROPERTIES
            INSTALL_RPATH "$ORIGIN"
            BUILD_WITH_INSTALL_RPATH OFF
        )
    endif()

    set_target_properties(${mod} PROPERTIES OUTPUT_NAME ${mod})
    install(TARGETS ${mod} DESTINATION feelmri)
    set_target_properties(${mod} PROPERTIES SKIP_INSTALL_ALL_DEPENDENCIES ON)
endforeach()


# ------------------------------------------------------------
# Summary output
# ------------------------------------------------------------
message(STATUS "------------------------------------------------------------")
message(STATUS "FeelMRI build configuration summary:")
message(STATUS "  C++ Standard:        ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build type:          ${CMAKE_BUILD_TYPE}")
message(STATUS "  Python:              ${Python_EXECUTABLE}")
message(STATUS "  Pybind11 includes:   ${pybind11_INCLUDE_DIRS}")
message(STATUS "  Eigen include dir:   ${EIGEN_INCLUDE_DIR}")
message(STATUS "  Eigen unsupported:   ${EIGEN_UNSUPPORTED_DIR}")
message(STATUS "  LTO enabled:         ${CMAKE_INTERPROCEDURAL_OPTIMIZATION}")
message(STATUS "  ccache enabled:      ${CCACHE_PROGRAM}")
message(STATUS "------------------------------------------------------------")