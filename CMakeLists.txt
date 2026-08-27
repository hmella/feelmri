cmake_minimum_required(VERSION 3.18)
project(feelmri LANGUAGES CXX)

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
# Optimization flags (architecture, fast math, LTO)
#
# Every flag is probed with check_cxx_compiler_flag before use, so this
# configures on x86-64 with or without AVX2 and on ARM64. Nothing here is
# hardcoded to the development machine: -march=native already enables whatever
# SIMD the host supports, so naming -mavx2/-mfma explicitly would only break
# hosts that lack them.
# ------------------------------------------------------------
include(CheckCXXCompilerFlag)

option(FEELMRI_NATIVE_ARCH
       "Tune for the building machine (-march=native). Produces binaries that \
are NOT redistributable -- turn OFF for wheels/conda packages." ON)
option(FEELMRI_FAST_MATH
       "Enable -Ofast/-ffast-math. Turn OFF for IEEE semantics when \
reproducing published numbers across compilers." ON)

# Collect the flags this compiler actually accepts.
set(FEELMRI_OPT_FLAGS "")
function(feelmri_try_flag flag)
    string(MAKE_C_IDENTIFIER "FEELMRI_HAS_${flag}" flag_var)
    check_cxx_compiler_flag("${flag}" ${flag_var})
    if (${flag_var})
        set(FEELMRI_OPT_FLAGS ${FEELMRI_OPT_FLAGS} ${flag} PARENT_SCOPE)
    endif()
endfunction()

# ---- CPU tuning ----
if (FEELMRI_NATIVE_ARCH)
    # Each candidate needs its OWN cache variable: check_cxx_compiler_flag
    # caches by variable name and silently skips the test when the variable is
    # already set, so reusing one name here made the fallback unreachable on
    # every machine that rejects the first candidate.
    # "|" as the intra-pair separator: CMake flattens semicolons inside list
    # items, which would iterate flag-by-flag and drop the -mtune half.
    set(CPU_TUNING_CANDIDATES "-march=raptorlake|-mtune=raptorlake"
                              "-march=native|-mtune=native"
                              "-mtune=native")
    set(USED_CPU_FLAGS "")
    foreach(pair ${CPU_TUNING_CANDIDATES})
        string(REPLACE "|" ";" pair_list "${pair}")
        list(GET pair_list 0 probe_flag)
        string(MAKE_C_IDENTIFIER "FEELMRI_HAS_${probe_flag}" probe_var)
        check_cxx_compiler_flag("${probe_flag}" ${probe_var})
        if (${probe_var})
            string(REPLACE "|" " " candidate_flags "${pair}")
            message(STATUS "Using CPU tuning flags: ${candidate_flags}")
            set(USED_CPU_FLAGS ${pair_list})
            break()
        endif()
    endforeach()
    if (USED_CPU_FLAGS)
        list(APPEND FEELMRI_OPT_FLAGS ${USED_CPU_FLAGS})
    else()
        message(WARNING "No supported CPU tuning flags found - using generic tuning.")
    endif()
else()
    message(STATUS "FEELMRI_NATIVE_ARCH=OFF - portable build, no -march/-mtune.")
endif()

# ---- Fast math ----
if (FEELMRI_FAST_MATH)
    foreach(flag -Ofast -ffast-math -fno-math-errno -fno-trapping-math -ffp-contract=fast)
        feelmri_try_flag(${flag})
    endforeach()
    list(APPEND FEELMRI_OPT_FLAGS -DEIGEN_FAST_MATH)
else()
    feelmri_try_flag(-O3)
    message(STATUS "FEELMRI_FAST_MATH=OFF - IEEE semantics, results will differ "
                   "slightly from the fast-math build.")
endif()

# ---- Always-on, still probed ----
foreach(flag -fvisibility=hidden -fPIC -funroll-loops -fopenmp-simd)
    feelmri_try_flag(${flag})
endforeach()
list(APPEND FEELMRI_OPT_FLAGS -DNDEBUG -DEIGEN_NO_DEBUG -DEIGEN_DONT_PARALLELIZE)

string(REPLACE ";" " " _flags_pretty "${FEELMRI_OPT_FLAGS}")
message(STATUS "feelmri optimization flags: ${_flags_pretty}")

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

    # Apply the probed optimization flags (see the block above)
    target_compile_options(${mod} PRIVATE ${FEELMRI_OPT_FLAGS})

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
message(STATUS "  Native arch:         ${FEELMRI_NATIVE_ARCH}")
message(STATUS "  Fast math:           ${FEELMRI_FAST_MATH}")
message(STATUS "  ccache enabled:      ${CCACHE_PROGRAM}")
message(STATUS "------------------------------------------------------------")