#include <benchmark/benchmark.h>

#include "baguette/lp/LPSolver.hpp"
#include "baguette/milp/BranchAndBound.hpp"
#include "lp/MILP_problems.hpp"

using namespace baguette;

// ── Helper ────────────────────────────────────────────────────────────────────

static void runMILP(benchmark::State& state,
                    LPMethod lpMethod,
                    bool enableCuts,
                    std::function<Model()> build)
{
    for (auto _ : state) {
        BBOptions opts;
        opts.lpOpts.method = lpMethod;
        opts.enableCuts = enableCuts;
        MILPResult r    = solveMILP(build(), opts);
        benchmark::DoNotOptimize(r.objectiveValue);
    }
}

// ── Knapsack 10 items ─────────────────────────────────────────────────────────
// Small B&B tree (10 binary variables); isolates LP-solver overhead per node.

BENCHMARK_CAPTURE(runMILP, Knapsack10/BB/DualSimplex,
    LPMethod::DualSimplex,    false,
    []() { return baguette_test::makeKnapsack10(); });

BENCHMARK_CAPTURE(runMILP, Knapsack10/BnC/DualSimplex,
    LPMethod::DualSimplex,    true,
    []() { return baguette_test::makeKnapsack10(); });

BENCHMARK_CAPTURE(runMILP, Knapsack10/BB/RevisedSimplex,
    LPMethod::RevisedSimplex, false,
    []() { return baguette_test::makeKnapsack10(); });

BENCHMARK_CAPTURE(runMILP, Knapsack10/BB/MehrotraIPM,
    LPMethod::MehrotraIPM,    false,
    []() { return baguette_test::makeKnapsack10(); });

// ── TSP-10 (LP relaxation = MILP optimal → single-node B&B) ──────────────────
// Measures pure LP solve cost; B&B terminates at the root with no branching.

BENCHMARK_CAPTURE(runMILP, TSP10/BB/DualSimplex,
    LPMethod::DualSimplex,    false,
    []() { return baguette_test::makeTSP10(); });

BENCHMARK_CAPTURE(runMILP, TSP10/BB/RevisedSimplex,
    LPMethod::RevisedSimplex, false,
    []() { return baguette_test::makeTSP10(); });

BENCHMARK_CAPTURE(runMILP, TSP10/BB/MehrotraIPM,
    LPMethod::MehrotraIPM,    false,
    []() { return baguette_test::makeTSP10(); });
