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
            message(STATUS "If you want CMake to try installing them, rerun with:")
            message(STATUS "    cmake -DINSTALL_SYSTEM_DEPS=ON ..")
        endif()
    endif()
endif()


# ------------------------------------------------------------
# General build configuration
# ------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Use Release mode by default
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# ------------------------------------------------------------
# Eigen3 automatic installation (robust version)
# ------------------------------------------------------------
include(FetchContent)

# Declare and fetch Eigen manually
FetchContent_Declare(
    Eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
)

# Explicitly fetch the content to ensure Eigen_SOURCE_DIR is defined
FetchContent_GetProperties(Eigen)
if (NOT eigen_POPULATED)
    message(STATUS "Fetching Eigen3 from GitLab...")
    FetchContent_Populate(Eigen)
# 		set(FETCHCONTENT_BASE_DIR "${CMAKE_BINARY_DIR}/_deps")
		set(FETCHCONTENT_BASE_DIR "$ENV{HOME}/.cache/cmake")
endif()

# Now set include paths explicitly
set(EIGEN_INCLUDE_DIR "${eigen_SOURCE_DIR}")
set(EIGEN_UNSUPPORTED_DIR "${eigen_SOURCE_DIR}/unsupported")

message(STATUS "Eigen source directory: ${eigen_SOURCE_DIR}")
message(STATUS "Eigen include directory: ${EIGEN_INCLUDE_DIR}")

# Add include directories globally
include_directories(${EIGEN_INCLUDE_DIR} ${EIGEN_UNSUPPORTED_DIR})

# Warn if unsupported Tensor module is missing
if (EXISTS "${EIGEN_UNSUPPORTED_DIR}/Eigen/CXX11/Tensor")
    message(STATUS "✅ Eigen unsupported module found at: ${EIGEN_UNSUPPORTED_DIR}")
else()
    message(WARNING "⚠️  Eigen unsupported module NOT found — Tensor functionality may be limited.")
endif()


# ------------------------------------------------------------
# Source setup
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

    # Optimization flags
    target_compile_options(${mod} PRIVATE
        -Ofast -ffast-math -fno-math-errno -fno-trapping-math -ffp-contract=fast
    )

    set_target_properties(${mod} PROPERTIES OUTPUT_NAME ${mod})
    install(TARGETS ${mod} DESTINATION feelmri)

		# Avoid installing dependencies of each module separately
		set_target_properties(${mod} PROPERTIES SKIP_INSTALL_ALL_DEPENDENCIES ON)		

endforeach()

# ------------------------------------------------------------
# Optional OpenMP acceleration (multi-threaded Eigen)
# ------------------------------------------------------------
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "Enabling OpenMP parallelization")
    foreach(mod ${MODULES})
        target_link_libraries(${mod} PRIVATE OpenMP::OpenMP_CXX)
    endforeach()
endif()

# ------------------------------------------------------------
# Summary output
# ------------------------------------------------------------
message(STATUS "------------------------------------------------------------")
message(STATUS "FeelMRI build configuration summary:")
message(STATUS "  C++ Standard:        ${CMAKE_CXX_STANDARD}")
message(STATUS "  Python:              ${Python_EXECUTABLE}")
message(STATUS "  Pybind11 includes:   ${pybind11_INCLUDE_DIRS}")
message(STATUS "  Eigen include dir:   ${EIGEN_INCLUDE_DIR}")
message(STATUS "  Eigen unsupported:   ${EIGEN_UNSUPPORTED_DIR}")
message(STATUS "------------------------------------------------------------")
