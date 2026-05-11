#include <benchmark/benchmark.h>

#include "baguette/core/Sense.hpp"
#include "baguette/lp/LPSolver.hpp"
#include "baguette/lp/presolve.hpp"
#include "baguette/milp/BranchAndBound.hpp"
#include "baguette/model/Model.hpp"
#include "baguette/model/ModelEnums.hpp"
#include "lp/MILP_problems.hpp"

using namespace baguette;

// ── Helpers ───────────────────────────────────────────────────────────────────

// Build a chain-of-constraints LP where presolveInPlace can tighten every bound.
// N variables, N-1 constraints: x[i] + x[i+1] <= limit[i].
static baguette::Model makeChainLP(int n) {
    Model m;
    std::vector<Variable> x;
    x.reserve(n);
    for (int i = 0; i < n; ++i)
        x.push_back(m.addVar(0.0, double(n), "x" + std::to_string(i)));
    for (int i = 0; i + 1 < n; ++i)
        m.addLPConstraint(1.0*x[i] + 1.0*x[i+1], Sense::LessEq, double(n - i));
    LinearExpr obj;
    for (int i = 0; i < n; ++i) obj += 1.0 * x[i];
    m.setObjective(obj, ObjSense::Minimize);
    return m;
}

// Build a knapsack MILP that benefits from presolveInPlace bound tightening.
// N binary items, one weight constraint.
static baguette::Model makeKnapsackMILP(int n) {
    Model m;
    std::vector<Variable> x;
    x.reserve(n);
    LinearExpr obj, weight;
    const double capacity = double(n) / 2.0;
    for (int i = 0; i < n; ++i) {
        x.push_back(m.addVar(0.0, 1.0, VarType::Binary));
        double w = 1.0 + (i % 3);
        double v = double(n - i);
        weight += w * x[i];
        obj    += v * x[i];
    }
    m.addLPConstraint(weight, Sense::LessEq, capacity);
    m.setObjective(obj, ObjSense::Maximize);
    return m;
}

// ── presolveInPlace-only timing (not including the LP solve) ─────────────────────────

static void BM_PresolveOnly_Chain20(benchmark::State& state) {
    for (auto _ : state) {
        Model m = makeChainLP(20);
        PresolveResult pr = presolveInPlace(m);
        benchmark::DoNotOptimize(pr.boundsTightened);
    }
}
BENCHMARK(BM_PresolveOnly_Chain20);

static void BM_PresolveOnly_Chain100(benchmark::State& state) {
    for (auto _ : state) {
        Model m = makeChainLP(100);
        PresolveResult pr = presolveInPlace(m);
        benchmark::DoNotOptimize(pr.boundsTightened);
    }
}
BENCHMARK(BM_PresolveOnly_Chain100);

// ── LP solve: with vs without presolveInPlace ────────────────────────────────────────

static void BM_LP_Chain20_NoPresolve(benchmark::State& state) {
    for (auto _ : state) {
        LPOptions opts;
        opts.enablePresolve = false;
        LPResult r = solveLP(makeChainLP(20), opts);
        benchmark::DoNotOptimize(r.objectiveValue);
    }
}
BENCHMARK(BM_LP_Chain20_NoPresolve);

static void BM_LP_Chain20_Presolve(benchmark::State& state) {
    for (auto _ : state) {
        LPOptions opts;
        opts.enablePresolve = true;
        LPResult r = solveLP(makeChainLP(20), opts);
        benchmark::DoNotOptimize(r.objectiveValue);
    }
}
BENCHMARK(BM_LP_Chain20_Presolve);

// ── MILP solve: with vs without presolveInPlace (knapsack) ───────────────────────────

static void BM_MILP_Knapsack15_NoPresolve(benchmark::State& state) {
    for (auto _ : state) {
        BBOptions opts;
        opts.enablePresolve = false;
        MILPResult r = solveMILP(makeKnapsackMILP(15), opts);
        benchmark::DoNotOptimize(r.objectiveValue);
    }
}
BENCHMARK(BM_MILP_Knapsack15_NoPresolve);

static void BM_MILP_Knapsack15_Presolve(benchmark::State& state) {
    for (auto _ : state) {
        BBOptions opts;
        opts.enablePresolve = true;
        MILPResult r = solveMILP(makeKnapsackMILP(15), opts);
        benchmark::DoNotOptimize(r.objectiveValue);
    }
}
BENCHMARK(BM_MILP_Knapsack15_Presolve);

// ── MILP solve: with vs without presolveInPlace (TSP10) ──────────────────────────────

static void BM_MILP_TSP10_NoPresolve(benchmark::State& state) {
    for (auto _ : state) {
        BBOptions opts;
        opts.enablePresolve = false;
        MILPResult r = solveMILP(baguette_test::makeTSP10(), opts);
        benchmark::DoNotOptimize(r.objectiveValue);
    }
}
BENCHMARK(BM_MILP_TSP10_NoPresolve);

static void BM_MILP_TSP10_Presolve(benchmark::State& state) {
    for (auto _ : state) {
        BBOptions opts;
        opts.enablePresolve = true;
        MILPResult r = solveMILP(baguette_test::makeTSP10(), opts);
        benchmark::DoNotOptimize(r.objectiveValue);
    }
}
BENCHMARK(BM_MILP_TSP10_Presolve);
