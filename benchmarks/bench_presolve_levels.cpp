// bench_presolve_levels.cpp
//
// Calibration benchmark for BBOptions::presolveLevel (0-6).
//
// Two sections:
//
//   1. Cost   — presolveMILPInPlace only (no B&B), knapsack-10.
//               Answers: "how much does each level cost?"
//
//   2. Presolve + root LP — presolveMILPInPlace then one LP solve, knapsack-10.
//               Answers: "does a higher level tighten the root LP bound?"
//
//   3. Solve  — full solveMILP, knapsack-20 (harder, requires real B&B tree).
//               Answers: "do higher levels save nodes and wall time?"
//
// Run with (Release build recommended):
//   BaguetteBench --benchmark_filter=PresolveLevel
//   BaguetteBench --benchmark_filter=PresolveLevel --benchmark_format=json

#include <benchmark/benchmark.h>

#include "baguette/lp/LPSolver.hpp"
#include "baguette/milp/BranchAndBound.hpp"
#include "baguette/milp/Presolve.hpp"
#include "baguette/model/Model.hpp"
#include "baguette/model/ModelEnums.hpp"
#include "lp/MILP_problems.hpp"

using namespace baguette;

// ── Local helpers ─────────────────────────────────────────────────────────────

// 20-item binary knapsack: weights 1,2,3,… (cycling), values n,n-1,…,1,
// capacity = n/2. Requires a real B&B tree; no purely LP-optimal integer point.
static Model makeKnapsackMILP(int n) {
    Model m;
    std::vector<Variable> x;
    x.reserve(n);
    LinearExpr obj, weight;
    const double capacity = double(n) / 2.0;
    for (int i = 0; i < n; ++i) {
        x.push_back(m.addVar(0.0, 1.0, VarType::Binary));
        weight += (1.0 + double(i % 3)) * x[i];
        obj    += double(n - i) * x[i];
    }
    m.addLPConstraint(weight, Sense::LessEq, capacity);
    m.setObjective(obj, ObjSense::Maximize);
    return m;
}

// ── Section 1: Presolve-only cost (levels 0-6) ───────────────────────────────
//
// Model : knapsack-10 (realistic instance, 10 binary variables).
// Each iteration builds the model and runs presolveMILPInPlace at the given
// level. Counters report what each level achieves so results are interpretable.
//
// Expected pattern:
//   L0  – instant (no-op)
//   L1  – µs range (LP bound-tightening + integer rounding + PR1)
//   L2  – ≈ L1 (CP propagation adds little on a pure MILP knapsack)
//   L3  – ms range (weak probing: propagation × 10 binary vars)
//   L4  – ms range (+ one root LP solve)
//   L5  – ms range (+ implication rows from weak probing)
//   L6  – 10× L3 range (strong probing: one LP solve × 2 × 10 binary vars)

static void BM_PresolveLevelCost(benchmark::State& state) {
    const uint32_t level = static_cast<uint32_t>(state.range(0));
    MILPPresolveOpts opts;
    opts.level      = level;
    opts.timeLimitS = 30.0;

    for (auto _ : state) {
        state.PauseTiming();
        Model m = baguette_test::makeKnapsack10();
        state.ResumeTiming();

        MILPPresolveResult r = presolveMILPInPlace(m, opts);
        benchmark::DoNotOptimize(r.fixedVars);

        state.counters["tightened"]   = double(r.boundsTightened);
        state.counters["fixed"]       = double(r.fixedVars);
        state.counters["probed"]      = double(r.varsProbed);
        state.counters["probed_fixed"]= double(r.varsProbedFixed);
        state.counters["implied"]     = double(r.impliedRowsAdded);
    }
}
BENCHMARK(BM_PresolveLevelCost)
    ->DenseRange(0, 6)
    ->Unit(benchmark::kMicrosecond);

// ── Section 2: Presolve + root LP (levels 0-6) ────────────────────────────────
//
// After presolveMILPInPlace, solve the LP relaxation of the presolved model
// (no branching). Reports the root LP objective value.
// A tighter root LP (closer to the IP optimum 106) means better pruning and
// fewer nodes in B&B — even before branching begins.
//
// knapsack-10 IP optimum = 106; LP relaxation ≈ 107.5 (item 8 fractional).
// Higher presolve levels that fix variables may close this gap.

static void BM_PresolveAndRootLP(benchmark::State& state) {
    const uint32_t level = static_cast<uint32_t>(state.range(0));
    MILPPresolveOpts pOpts;
    pOpts.level      = level;
    pOpts.timeLimitS = 30.0;

    LPOptions lpOpts;
    lpOpts.enablePresolve = false;
    lpOpts.method         = LPMethod::DualSimplexBV;
    lpOpts.timeLimitS     = 30.0;

    for (auto _ : state) {
        state.PauseTiming();
        Model m = baguette_test::makeKnapsack10();
        state.ResumeTiming();

        MILPPresolveResult pr = presolveMILPInPlace(m, pOpts);
        double rootObj = 0.0;
        if (!pr.infeasible) {
            LPResult lp = solveLP(m, lpOpts);
            rootObj = lp.objectiveValue;
            benchmark::DoNotOptimize(rootObj);
        }

        state.counters["root_lp_obj"]  = rootObj;
        state.counters["fixed"]        = double(pr.fixedVars);
        state.counters["probed_fixed"] = double(pr.varsProbedFixed);
        state.counters["implied"]      = double(pr.impliedRowsAdded);
    }
}
BENCHMARK(BM_PresolveAndRootLP)
    ->DenseRange(0, 6)
    ->Unit(benchmark::kMicrosecond);

// ── Section 3: Full MILP solve (levels 0-6) ───────────────────────────────────
//
// Model : knapsack-20 (20 binary items, harder — requires a real B&B tree).
// Each level is run end-to-end with solveMILP.
// Counters show how many nodes and LP solves each level requires, quantifying
// the B&B gain purchased by the presolve investment in sections 1-2.
//
// Expected pattern:
//   L0  – most nodes (B&B from scratch, wide domains)
//   L1  – fewer nodes (bounds tightened before branching)
//   L3+ – potentially fewer nodes if probing fixes binary variables
//   L6  – possibly fewer nodes still, but presolve itself costs more

static void BM_PresolveLevelSolve(benchmark::State& state) {
    const uint32_t level = static_cast<uint32_t>(state.range(0));
    BBOptions opts;
    opts.presolveLevel = level;
    opts.collectStats  = true;
    opts.timeLimitS    = 60.0;

    for (auto _ : state) {
        state.PauseTiming();
        Model m = makeKnapsackMILP(20);
        state.ResumeTiming();

        MILPResult r = solveMILP(m, opts);
        benchmark::DoNotOptimize(r.objectiveValue);

        if (r.stats) {
            state.counters["nodes"]     = double(r.stats->nodesExplored);
            state.counters["lp_solves"] = double(r.stats->lpSolvesTotal);
        }
        state.counters["status"] = double(int(r.status));
    }
}
BENCHMARK(BM_PresolveLevelSolve)
    ->DenseRange(0, 6)
    ->Unit(benchmark::kMillisecond);
