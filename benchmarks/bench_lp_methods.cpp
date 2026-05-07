#include <benchmark/benchmark.h>

#include "baguette/lp/LPSolver.hpp"
#include "lp/lp_problems.hpp"
#include "lp/MILP_problems.hpp"

using namespace baguette;

// ── Helper ────────────────────────────────────────────────────────────────────

static void runLP(benchmark::State& state,
                  LPMethod method,
                  std::function<Model()> build)
{
    for (auto _ : state) {
        LPOptions opts;
        opts.method = method;
        LPResult r  = solveLP(build(), opts);
        benchmark::DoNotOptimize(r.objectiveValue);
    }
}

// ── Small LP (2 variables, 3 constraints) ─────────────────────────────────────

BENCHMARK_CAPTURE(runLP, SimpleMax/PrimalSimplex,   LPMethod::PrimalSimplex,   makeSimpleMax);
BENCHMARK_CAPTURE(runLP, SimpleMax/DualSimplex,     LPMethod::DualSimplex,     makeSimpleMax);
BENCHMARK_CAPTURE(runLP, SimpleMax/PrimalSimplexBV, LPMethod::PrimalSimplexBV, makeSimpleMax);
BENCHMARK_CAPTURE(runLP, SimpleMax/DualSimplexBV,   LPMethod::DualSimplexBV,   makeSimpleMax);
BENCHMARK_CAPTURE(runLP, SimpleMax/RevisedSimplex,  LPMethod::RevisedSimplex,  makeSimpleMax);
BENCHMARK_CAPTURE(runLP, SimpleMax/RevisedSimplexBV,LPMethod::RevisedSimplexBV,makeSimpleMax);
BENCHMARK_CAPTURE(runLP, SimpleMax/MehrotraIPM,     LPMethod::MehrotraIPM,     makeSimpleMax);

// ── Medium LP (2 variables, GEQ constraints) ──────────────────────────────────

BENCHMARK_CAPTURE(runLP, MinGEQ/PrimalSimplex,   LPMethod::PrimalSimplex,   makeMinWithGEQ);
BENCHMARK_CAPTURE(runLP, MinGEQ/DualSimplex,     LPMethod::DualSimplex,     makeMinWithGEQ);
BENCHMARK_CAPTURE(runLP, MinGEQ/PrimalSimplexBV, LPMethod::PrimalSimplexBV, makeMinWithGEQ);
BENCHMARK_CAPTURE(runLP, MinGEQ/DualSimplexBV,   LPMethod::DualSimplexBV,   makeMinWithGEQ);
BENCHMARK_CAPTURE(runLP, MinGEQ/RevisedSimplex,  LPMethod::RevisedSimplex,  makeMinWithGEQ);
BENCHMARK_CAPTURE(runLP, MinGEQ/RevisedSimplexBV,LPMethod::RevisedSimplexBV,makeMinWithGEQ);
BENCHMARK_CAPTURE(runLP, MinGEQ/MehrotraIPM,     LPMethod::MehrotraIPM,     makeMinWithGEQ);

// ── Knapsack LP relaxation (10 variables, 1 constraint + bound rows) ──────────
// BV methods eliminate the 10 explicit UB rows → smaller tableau (1 vs 11 rows).

BENCHMARK_CAPTURE(runLP, Knapsack10/PrimalSimplex,   LPMethod::PrimalSimplex,
    []() { return baguette_test::makeKnapsack10(); });
BENCHMARK_CAPTURE(runLP, Knapsack10/DualSimplex,     LPMethod::DualSimplex,
    []() { return baguette_test::makeKnapsack10(); });
BENCHMARK_CAPTURE(runLP, Knapsack10/PrimalSimplexBV, LPMethod::PrimalSimplexBV,
    []() { return baguette_test::makeKnapsack10(); });
BENCHMARK_CAPTURE(runLP, Knapsack10/DualSimplexBV,   LPMethod::DualSimplexBV,
    []() { return baguette_test::makeKnapsack10(); });
BENCHMARK_CAPTURE(runLP, Knapsack10/RevisedSimplex,  LPMethod::RevisedSimplex,
    []() { return baguette_test::makeKnapsack10(); });
BENCHMARK_CAPTURE(runLP, Knapsack10/RevisedSimplexBV,LPMethod::RevisedSimplexBV,
    []() { return baguette_test::makeKnapsack10(); });
BENCHMARK_CAPTURE(runLP, Knapsack10/ShortStepIPM,    LPMethod::ShortStepIPM,
    []() { return baguette_test::makeKnapsack10(); });
BENCHMARK_CAPTURE(runLP, Knapsack10/MehrotraIPM,     LPMethod::MehrotraIPM,
    []() { return baguette_test::makeKnapsack10(); });

// ── TSP-10 LP relaxation (99 variables, 91 constraints) ───────────────────────
// No variable UBs → BV methods behave identically to classic (no UB rows to save).

BENCHMARK_CAPTURE(runLP, TSP10/PrimalSimplex,   LPMethod::PrimalSimplex,
    []() { return baguette_test::makeTSP10(); });
BENCHMARK_CAPTURE(runLP, TSP10/DualSimplex,     LPMethod::DualSimplex,
    []() { return baguette_test::makeTSP10(); });
BENCHMARK_CAPTURE(runLP, TSP10/PrimalSimplexBV, LPMethod::PrimalSimplexBV,
    []() { return baguette_test::makeTSP10(); });
BENCHMARK_CAPTURE(runLP, TSP10/DualSimplexBV,   LPMethod::DualSimplexBV,
    []() { return baguette_test::makeTSP10(); });
BENCHMARK_CAPTURE(runLP, TSP10/RevisedSimplex,  LPMethod::RevisedSimplex,
    []() { return baguette_test::makeTSP10(); });
BENCHMARK_CAPTURE(runLP, TSP10/RevisedSimplexBV,LPMethod::RevisedSimplexBV,
    []() { return baguette_test::makeTSP10(); });
BENCHMARK_CAPTURE(runLP, TSP10/MehrotraIPM,     LPMethod::MehrotraIPM,
    []() { return baguette_test::makeTSP10(); });

// ── Job shop LP relaxation (111 variables, ~200 constraints) ──────────────────
// BV methods save ~111 UB rows (one per bounded S[j][m] / Cmax / y[j][k][m]).

BENCHMARK_CAPTURE(runLP, JobShop10x2/PrimalSimplex,   LPMethod::PrimalSimplex,
    []() { return baguette_test::makeJobShop10(); });
BENCHMARK_CAPTURE(runLP, JobShop10x2/DualSimplex,     LPMethod::DualSimplex,
    []() { return baguette_test::makeJobShop10(); });
BENCHMARK_CAPTURE(runLP, JobShop10x2/PrimalSimplexBV, LPMethod::PrimalSimplexBV,
    []() { return baguette_test::makeJobShop10(); });
BENCHMARK_CAPTURE(runLP, JobShop10x2/DualSimplexBV,   LPMethod::DualSimplexBV,
    []() { return baguette_test::makeJobShop10(); });
BENCHMARK_CAPTURE(runLP, JobShop10x2/RevisedSimplex,  LPMethod::RevisedSimplex,
    []() { return baguette_test::makeJobShop10(); });
BENCHMARK_CAPTURE(runLP, JobShop10x2/RevisedSimplexBV,LPMethod::RevisedSimplexBV,
    []() { return baguette_test::makeJobShop10(); });
BENCHMARK_CAPTURE(runLP, JobShop10x2/MehrotraIPM,     LPMethod::MehrotraIPM,
    []() { return baguette_test::makeJobShop10(); });
