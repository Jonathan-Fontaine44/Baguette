include(FetchContent)
FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.9.1
)
set(BENCHMARK_ENABLE_TESTING      OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS  OFF CACHE BOOL "" FORCE)
set(BENCHMARK_INSTALL_DOCS        OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebenchmark)

add_executable(BaguetteBench
    bench_lp_methods.cpp
    bench_milp.cpp
    bench_milp_large.cpp
    bench_flow.cpp
    bench_network_simplex.cpp
    bench_presolve.cpp
    bench_presolve_levels.cpp
    bench_cp.cpp
)

target_link_libraries(BaguetteBench PRIVATE
    Baguette
    benchmark::benchmark_main
)

target_include_directories(BaguetteBench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../tests
)

add_executable(BaguetteTSPBench bench_tsp_formulations.cpp)
target_link_libraries(BaguetteTSPBench PRIVATE Baguette)
target_include_directories(BaguetteTSPBench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../tests
)
