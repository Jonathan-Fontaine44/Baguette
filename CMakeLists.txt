cmake_minimum_required(VERSION 3.20)
project(Baguette VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
endif()

# ── Library ────────────────────────────────────────────────────────────────────

add_library(Baguette
    src/core/LinearExpr.cpp
    src/model/Model.cpp
    src/lp/StandardForm.cpp
    src/lp/Tableau.cpp
    src/lp/LPSolver.cpp
)

target_include_directories(Baguette PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# ── Tests ──────────────────────────────────────────────────────────────────────

include(CTest)
enable_testing()

include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.5.4
)
FetchContent_MakeAvailable(Catch2)

add_executable(BaguetteTests
    tests/test_core.cpp
    tests/test_model.cpp
    tests/test_lp.cpp
)

target_link_libraries(BaguetteTests PRIVATE
    Baguette
    Catch2::Catch2WithMain
)

include(Catch)
catch_discover_tests(BaguetteTests DISCOVERY_MODE PRE_TEST)

# ── Internal (white-box) tests ─────────────────────────────────────────────────
# These tests have access to src/ headers and exercise internal components
# (StandardForm, Tableau) that are not part of the public API.

add_executable(BaguetteInternalTests
    tests/test_standard_form.cpp
)

target_link_libraries(BaguetteInternalTests PRIVATE
    Baguette
    Catch2::Catch2WithMain
)

target_include_directories(BaguetteInternalTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

catch_discover_tests(BaguetteInternalTests DISCOVERY_MODE PRE_TEST)
