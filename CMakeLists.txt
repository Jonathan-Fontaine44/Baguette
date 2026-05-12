cmake_minimum_required(VERSION 3.20)
project(Baguette VERSION 0.6.1 LANGUAGES CXX)

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
    src/model/Presolve.cpp
    src/lp/algorithms/StandardForm.cpp
    src/lp/algorithms/SimplexTableau.cpp
    src/lp/algorithms/Extractor.cpp
    src/lp/algorithms/PrimalSimplex.cpp
    src/lp/algorithms/SimplexTableauBV.cpp
    src/lp/algorithms/PrimalSimplexBV.cpp
    src/lp/algorithms/DualSimplexBV.cpp
    src/lp/algorithms/DualSimplex.cpp
    src/lp/algorithms/LUTableau.cpp
    src/lp/algorithms/RevisedSimplex.cpp
    src/lp/algorithms/RevisedSimplexBV.cpp
    src/lp/algorithms/NetworkSimplex.cpp
    src/lp/algorithms/IPMSolver.cpp
    src/lp/algorithms/MehrotraIPM.cpp
    src/lp/Presolve.cpp
    src/lp/LPSolver.cpp
    src/milp/BranchAndBound.cpp
    src/milp/Presolve.cpp
    src/milp/CuttingPlanes.cpp
    src/cp/CPConstraints.cpp
    src/cp/constraints/allDiff.cpp
    src/cp/constraints/cumulative.cpp
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
    tests/core/test_core.cpp
    tests/model/test_model.cpp
    tests/lp/test_lp_methods.cpp
    tests/lp/test_lp.cpp
    tests/lp/test_dual_simplex.cpp
    tests/lp/test_warm_start.cpp
    tests/lp/test_farkas.cpp
    tests/lp/test_sensitivity.cpp
    tests/lp/test_revised_simplex.cpp
    tests/lp/test_network_simplex.cpp
    tests/lp/test_presolve.cpp
    tests/lp/test_ipm.cpp
    tests/lp/test_mehrotra_ipm.cpp
    tests/milp/test_bb.cpp
    tests/milp/test_bnc.cpp
    tests/milp/test_stats.cpp
    tests/milp/test_milp_classic.cpp
    tests/cp/test_cp.cpp
    tests/cp/constraints/test_allDiff.cpp
    tests/cp/constraints/test_cumulative.cpp
)

target_include_directories(BaguetteTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tests
)

target_link_libraries(BaguetteTests PRIVATE
    Baguette
    Catch2::Catch2WithMain
)

include(Catch)
catch_discover_tests(BaguetteTests DISCOVERY_MODE PRE_TEST)

# ── Internal (white-box) tests ─────────────────────────────────────────────────
# These tests have access to src/ headers and exercise internal components
# (StandardForm, SimplexTableau) that are not part of the public API.

add_executable(BaguetteInternalTests
    tests/lp/test_standard_form.cpp
    tests/lp/test_dual_sf.cpp
    tests/lp/test_tableau_init.cpp
)

target_link_libraries(BaguetteInternalTests PRIVATE
    Baguette
    Catch2::Catch2WithMain
)

target_include_directories(BaguetteInternalTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

catch_discover_tests(BaguetteInternalTests DISCOVERY_MODE PRE_TEST)

# ── Benchmarks (opt-in, Release recommended) ───────────────────────────────────
option(BAGUETTE_BENCHMARKS "Build Google Benchmark suite" OFF)
if(BAGUETTE_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
