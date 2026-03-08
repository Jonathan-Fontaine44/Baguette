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
)

target_link_libraries(BaguetteTests PRIVATE
    Baguette
    Catch2::Catch2WithMain
)

add_test(NAME BaguetteTests COMMAND BaguetteTests)
