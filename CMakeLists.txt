cmake_minimum_required(VERSION 3.18)
project(feelmri LANGUAGES CXX)

# =========================================================
# Dependency detection and optional installation
# =========================================================
option(INSTALL_SYSTEM_DEPS "Automatically install required system packages (Linux only)" OFF)

if(UNIX AND NOT APPLE)
    message(STATUS "Checking for essential system dependencies...")

    find_program(APT_EXEC apt-get)
    if(NOT APT_EXEC)
        message(WARNING "apt-get not found; please install dependencies manually if missing.")
    else()
        if(INSTALL_SYSTEM_DEPS)
            message(STATUS "Installing missing system dependencies with apt-get (requires sudo)...")
            execute_process(COMMAND sudo ${APT_EXEC} update)
            execute_process(
                COMMAND sudo ${APT_EXEC} install -y --no-install-recommends
                    build-essential
                    python3
                    python3-dev
                    python3-pip
                    python3-tk
                    cmake
                    ninja-build
                    git
                    libopenmpi-dev
                    screen
                    nano
                RESULT_VARIABLE APT_RESULT
            )
            if(NOT APT_RESULT EQUAL 0)
                message(FATAL_ERROR "System dependency installation failed. Please install manually.")
            endif()
        else()
            message(STATUS "System dependencies will not be installed automatically.")
            message(STATUS "To enable auto-install, rerun with: cmake -DINSTALL_SYSTEM_DEPS=ON ..")
        endif()
    endif()
endif()


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
    add_compile_options(${USED_CPU_FLAGS})
else()
    message(WARNING "⚠️ No supported CPU tuning flags found — using generic tuning.")
endif()


# ------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# ------------------------------------------------------------
# Eigen3 automatic installation (cached and robust)
# ------------------------------------------------------------
include(FetchContent)

FetchContent_Declare(
    Eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
		GIT_TAG nightly
)

FetchContent_GetProperties(Eigen)
if (NOT eigen_POPULATED)
    message(STATUS "Fetching Eigen3 (nightly) from GitLab...")
    FetchContent_Populate(Eigen)
    set(FETCHCONTENT_BASE_DIR "$ENV{HOME}/.cache/cmake")
endif()

set(EIGEN_INCLUDE_DIR "${eigen_SOURCE_DIR}")
set(EIGEN_UNSUPPORTED_DIR "${eigen_SOURCE_DIR}/unsupported")

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
set(MODULES Assemble BlochSimulator MRI POD)

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

    # Apply high-performance compile flags
    target_compile_options(${mod} PRIVATE
        -Ofast
        -ffast-math
        -fno-math-errno
        -fno-trapping-math
        -ffp-contract=fast
        -fvisibility=hidden
        -fPIC
        -DNDEBUG
        -DEIGEN_NO_DEBUG
        -DEIGEN_FAST_MATH
        -DEIGEN_DONT_PARALLELIZE
    )

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
message(STATUS "------------------------------------------------------------")
