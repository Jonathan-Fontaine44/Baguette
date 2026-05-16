#include <benchmark/benchmark.h>

#include "baguette/cp/constraints/AllDiff.hpp"
#include "baguette/cp/constraints/Cumulative.hpp"
#include "baguette/model/Model.hpp"
#include "baguette/model/ModelEnums.hpp"

using namespace baguette;

// ── AllDiff helpers ───────────────────────────────────────────────────────────

// K variables: first `nFixed` are fixed to distinct values [0, nFixed),
// remaining K-nFixed have domain [0, K-1] (wide, no Hall interval needed).
// Exercises fixedValueElimination: each fixed value eliminates itself from
// every free variable's domain via lb/ub advancement.
static AllDiffConstraint makeAllDiff(int K, int nFixed) {
    AllDiffConstraint con;
    con.vars.reserve(K);
    Model dummy; // only used to allocate var IDs — not part of benchmark loop
    for (int i = 0; i < K; ++i)
        con.vars.push_back(dummy.addVar(0.0, double(K - 1), VarType::Integer));
    return con;
}

// Run one propagation call from a fresh model each iteration.
// Each iteration reconstructs the model so bounds are not already tight.
static void BM_AllDiff_propagate(benchmark::State& state, int K, int nFixed) {
    for (auto _ : state) {
        Model m;
        AllDiffConstraint con;
        con.vars.reserve(K);
        for (int i = 0; i < K; ++i) {
            double lo = (i < nFixed) ? double(i) : 0.0;
            double hi = (i < nFixed) ? double(i) : double(K - 1);
            con.vars.push_back(m.addVar(lo, hi, VarType::Integer));
        }
        PropagationResult r = propagate(con, m);
        benchmark::DoNotOptimize(r.changedVarIds.size());
    }
}

// K=20: 0 fixed (overhead baseline — nothing to eliminate)
BENCHMARK_CAPTURE(BM_AllDiff_propagate, K20_0fixed,   20,  0);
// K=20: 10 fixed (half the domain is pinned)
BENCHMARK_CAPTURE(BM_AllDiff_propagate, K20_10fixed,  20, 10);
// K=50: 0 fixed
BENCHMARK_CAPTURE(BM_AllDiff_propagate, K50_0fixed,   50,  0);
// K=50: 25 fixed (exercises O(K log K) sort + binary search path)
BENCHMARK_CAPTURE(BM_AllDiff_propagate, K50_25fixed,  50, 25);
// K=100: 50 fixed (stress test for fixedValueElimination)
BENCHMARK_CAPTURE(BM_AllDiff_propagate, K100_50fixed, 100, 50);

// ── Cumulative helpers ────────────────────────────────────────────────────────

// N tasks on a resource of capacity N/2.
// Each task: duration=D/4, consumption=1.
// Start windows: task i ∈ [0, D - D/4] so lst < ect → every task has a
// compulsory region of length 0 (lst == ect), except the last few which are
// forced to have compulsory regions by narrowing their window.
// Actually, to get non-trivial propagation, we use:
//   task i: start ∈ [0, W], duration=dur, consumption=1.
//   W = D - dur so tasks have zero compulsory region — but together they
//   may overload if a window is too tight, forcing est to advance.
//
// Scenario: N tasks, each dur=dur_size, window [0, horizon].
// Tasks 0..N/2-1 have a narrow window [0, dur_size-1] → compulsory region
// [dur_size-1, dur_size) of width 1 each.
// Tasks N/2..N-1 have wide window [0, horizon] → propagation pushes their est.
static void BM_Cumulative_propagate(benchmark::State& state, int N, int dur, int horizon) {
    const int cap     = N / 2;           // capacity = half the tasks
    const int nComp   = N / 2;           // first half: narrow window → compulsory region
    for (auto _ : state) {
        Model m;
        CumulativeConstraint con;
        con.capacity = cap;
        con.tasks.reserve(N);
        for (int i = 0; i < N; ++i) {
            int lo = 0;
            int hi = (i < nComp) ? (dur - 1) : horizon;
            Variable v = m.addVar(double(lo), double(hi), VarType::Integer);
            con.tasks.push_back({v, dur, 1});
        }
        PropagationResult r = propagate(con, m);
        benchmark::DoNotOptimize(r.changedVarIds.size());
    }
}

// N=5, dur=4, horizon=20: small scheduling problem
BENCHMARK_CAPTURE(BM_Cumulative_propagate, N5_dur4_H20,    5,  4,  20);
// N=10, dur=4, horizon=30: medium — exercises sliding window max
BENCHMARK_CAPTURE(BM_Cumulative_propagate, N10_dur4_H30,  10,  4,  30);
// N=10, dur=8, horizon=60: wider windows, longer compulsory regions
BENCHMARK_CAPTURE(BM_Cumulative_propagate, N10_dur8_H60,  10,  8,  60);
// N=20, dur=4, horizon=50: larger task set — profiles dominate
BENCHMARK_CAPTURE(BM_Cumulative_propagate, N20_dur4_H50,  20,  4,  50);
// N=20, dur=8, horizon=100: stress test — D=100, 20 tasks
BENCHMARK_CAPTURE(BM_Cumulative_propagate, N20_dur8_H100, 20,  8, 100);
