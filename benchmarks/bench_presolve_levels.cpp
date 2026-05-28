// bench_presolve_levels.cpp
//
// Calibration benchmark for BBOptions::presolveLevel (0-6).
//
// Part A — Knapsack-10 / Knapsack-20
//   1. Cost      — presolveMILPInPlace only (no B&B), knapsack-10.
//   2. Root LP   — presolveMILPInPlace then one LP solve, knapsack-10.
//   3. Solve     — full solveMILP, knapsack-20.
//
// Part B — Hard asymmetric TSP-10 (MTZ formulation)
//   The cyclic TSP-10 has LP=IP at the root (no branching needed) and is
//   not useful for calibration.  This part uses a pseudo-random asymmetric
//   instance where the MTZ LP relaxation is fractional.
//
//   TSP is the ideal stress-test for probing: fixing arc x[i→j]=1 propagates
//   through the two degree-equality constraints to force all 2(n-2) other
//   arcs incident on i or j to 0 via LP bound-tightening.  One probe can
//   cascade into O(n) additional binary fixings — far more than a knapsack.
//
//   4. TSP Cost  — presolveMILPInPlace only, hard TSP-10 MTZ.
//   5. TSP Root  — presolveMILPInPlace then root LP, hard TSP-10 MTZ.
//   6. TSP Solve — full solveMILP, hard TSP-10 MTZ.
//
// Part C — Hard asymmetric TSP-10 (SCF formulation)
//   Same instance as Part B but with the Single Commodity Flow formulation,
//   whose LP relaxation equals the DFJ bound (strictly tighter than MTZ).
//   The tighter LP allows strong probing (L6) to detect subtour infeasibility
//   and fix arc variables — unlike MTZ where probed_fixed stays 0.
//
//   7. SCF Cost  — presolveMILPInPlace only, hard TSP-10 SCF.
//   8. SCF Root  — presolveMILPInPlace then root LP, hard TSP-10 SCF.
//   9. SCF Solve — full solveMILP, hard TSP-10 SCF.
//
// Part D — Uncapacitated Facility Location 5×10
//   5 facilities (fixed cost 20 each), 10 clients, LCG assignment costs [1,10].
//   55 binary variables (5 y[i] + 50 x[i][j]), 60 LP constraints.
//   LP optimal = 67, IP optimal = 69.
//
//   Probing cascade: fixing y[i]=0 forces all x[i][j]=0 via linking constraints
//   (nCli fixings per probe), then tightens coverage constraints for other facilities.
//   Unlike TSP (probed_fixed=0), FL has enough constraint coupling that strong
//   probing (L6) is expected to fix some facility variables.
//
//   10. FL Cost  — presolveMILPInPlace only, facility location 5×10.
//   11. FL Root  — presolveMILPInPlace then root LP, facility location 5×10.
//   12. FL Solve — full solveMILP, facility location 5×10.
//
// Part E — Uncapacitated Facility Location 15×30
//   15 facilities (fixed cost 20 each), 30 clients, LCG assignment costs [1,10].
//   465 binary variables (15 y[i] + 450 x[i][j]), 480 LP constraints.
//   3× larger than Part D — more B&B nodes expected, probed_fixed may emerge.
//
//   13. FL2 Cost  — presolveMILPInPlace only, facility location 15×30.
//   14. FL2 Root  — presolveMILPInPlace then root LP, facility location 15×30.
//   15. FL2 Solve — full solveMILP, facility location 15×30.
//
// Part F — Set Partitioning small (10 elements, 30 columns)
//   10 singletons + 20 compound columns (size 2-4), seed 0xC0FFEE42.
//   30 binary variables, 10 equality coverage constraints.
//   LP optimal = 16, IP optimal = 19 (gap = 18.75%).
//   L5 raises root LP from 16 to 19 (= IP optimal) → first LP bound improvement.
//
//   16. SP Cost  — presolveMILPInPlace only, SP small.
//   17. SP Root  — presolveMILPInPlace then root LP, SP small.
//   18. SP Solve — full solveMILP, SP small.
//
// Part G — Set Partitioning large (30 elements, 90 columns)
//   30 singletons + 60 compound columns (size 2-5), seed 0xDEADC0DE.
//   90 binary variables, 30 equality coverage constraints.
//   LP optimal = 81, IP optimal = 82 (gap = 1.2%).
//   L1 fixes 3 columns (first fixed>0 across all families).
//
//   19. SP2 Cost  — presolveMILPInPlace only, SP large.
//   20. SP2 Root  — presolveMILPInPlace then root LP, SP large.
//   21. SP2 Solve — full solveMILP, SP large.
//
// Part H — Hard asymmetric TSP-10 (MTZ + AllDiff CP formulation)
//   Same instance as Parts B and C (seed 0xC0FFEE42), with AllDiff posted on
//   the 9 position variables u[1..9].  LP optimal = 19.15 (unchanged; CP
//   constraints are not linearised).  IP optimal = 20.
//
//   The AllDiff propagator activates at level 2 (CP propagation at fixed point).
//   This is the only family where L2 is expected to differ from L1: AllDiff
//   can tighten position domains and propagate to arc fixings, unlike the pure
//   LP bound-tightening of L1.
//
//   22. MTZAD Cost  — presolveMILPInPlace only, hard TSP-10 MTZAD.
//   23. MTZAD Root  — presolveMILPInPlace then root LP, hard TSP-10 MTZAD.
//   24. MTZAD Solve — full solveMILP, hard TSP-10 MTZAD.
//
// Run with (Release build recommended):
//   BaguetteBench --benchmark_filter=BM_Presolve|BM_TSP|BM_SCF|BM_FL|BM_SP|BM_MTZAD
//   BaguetteBench --benchmark_filter=BM_Presolve|BM_TSP|BM_SCF|BM_FL|BM_SP|BM_MTZAD --benchmark_format=json

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

// ── Part B: Hard asymmetric TSP-10 ───────────────────────────────────────────
//
// Builder: n-city complete asymmetric TSP (MTZ formulation).
// Costs generated by a deterministic LCG, integer values in [1, 10].
// For n=10: 90 binary arc variables + 9 integer MTZ position variables.
//
// Why TSP is special for probing
// ─────────────────────────────
// Each city has a degree-out constraint  Σⱼ x[i][j] = 1  and
// a degree-in  constraint  Σᵢ x[i][j] = 1  (tight equalities).
//
// Weak probing (level 3), fix arc x[a→b] = 1:
//   LP bound-tightening on the degree-out of city a: all x[a][k]=0 for k≠b.
//   LP bound-tightening on the degree-in  of city b: all x[k][b]=0 for k≠a.
//   → 2(n-2) binary variables forced to 0 in one step.
//   Each of those zero fixings further tightens MTZ bounds u[i].
//   This cascade makes probing disproportionately effective on TSP.
//
// Strong probing (level 6) additionally runs one LP per fix, detecting
// subtour infeasibility that propagation alone cannot see.
//
// Expected pattern for TSP vs knapsack
// ─────────────────────────────────────
//   Knapsack: probed_fixed ≈ 0 (single loose weight constraint → no cascade)
//   TSP     : probed_fixed >> 0, probing fixes many arc variables per iteration,
//             root LP bound improves, B&B node count drops at levels ≥ 3.

static Model makeHardTSP(int n, unsigned seed = 0xC0FFEE42u) {
    std::vector<baguette_test::TspArc> arcs;
    arcs.reserve(n * (n - 1));
    unsigned s = seed;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            s = s * 1664525u + 1013904223u;   // Knuth LCG
            double dist = 1.0 + double(s % 10u); // [1, 10]
            arcs.push_back({i, j, dist});
        }
    }
    return baguette_test::makeTSP(n, arcs);
}

// ── Section 4: Presolve-only cost, hard TSP-10 (levels 0-6) ─────────────────

static void BM_TSP_PresolveLevelCost(benchmark::State& state) {
    const uint32_t level = static_cast<uint32_t>(state.range(0));
    MILPPresolveOpts opts;
    opts.level      = level;
    opts.timeLimitS = 30.0;

    for (auto _ : state) {
        state.PauseTiming();
        Model m = makeHardTSP(10);
        state.ResumeTiming();

        MILPPresolveResult r = presolveMILPInPlace(m, opts);
        benchmark::DoNotOptimize(r.fixedVars);

        state.counters["tightened"]    = double(r.boundsTightened);
        state.counters["fixed"]        = double(r.fixedVars);
        state.counters["probed"]       = double(r.varsProbed);
        state.counters["probed_fixed"] = double(r.varsProbedFixed);
        state.counters["implied"]      = double(r.impliedRowsAdded);
    }
}
BENCHMARK(BM_TSP_PresolveLevelCost)
    ->DenseRange(0, 6)
    ->Unit(benchmark::kMicrosecond);

// ── Section 5: Presolve + root LP, hard TSP-10 (levels 0-6) ─────────────────
//
// Measures whether higher levels tighten the MTZ LP relaxation.
// If probing fixes arc variables, the LP is solved on a smaller (tighter) model
// and may yield a higher lower bound → fewer B&B nodes needed.

static void BM_TSP_PresolveAndRootLP(benchmark::State& state) {
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
        Model m = makeHardTSP(10);
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
BENCHMARK(BM_TSP_PresolveAndRootLP)
    ->DenseRange(0, 6)
    ->Unit(benchmark::kMicrosecond);

// ── Section 6: Full MILP solve, hard TSP-10 (levels 0-6) ────────────────────
//
// End-to-end solveMILP on the hard TSP-10. Counters show nodes and LP solves.
// Unlike the knapsack, TSP structure lets probing fix arcs before branching,
// so the B&B tree is expected to shrink significantly at levels ≥ 3.

static void BM_TSP_PresolveLevelSolve(benchmark::State& state) {
    const uint32_t level = static_cast<uint32_t>(state.range(0));
    BBOptions opts;
    opts.presolveLevel = level;
    opts.collectStats  = true;
    opts.timeLimitS    = 60.0;

    for (auto _ : state) {
        state.PauseTiming();
        Model m = makeHardTSP(10);
        state.ResumeTiming();

        MILPResult r = solveMILP(m, opts);
        benchmark::DoNotOptimize(r.objectiveValue);

        if (r.stats) {
            state.counters["nodes"]     = double(r.stats->nodesExplored);
            state.counters["lp_solves"] = double(r.stats->lpSolvesTotal);
        }
        state.counters["obj"]    = r.objectiveValue;
        state.counters["status"] = double(int(r.status));
    }
}
BENCHMARK(BM_TSP_PresolveLevelSolve)
    ->DenseRange(0, 6)
    ->Unit(benchmark::kMillisecond);

// ── Part C: Hard asymmetric TSP-10 (SCF formulation) ─────────────────────────
//
// Same instance as Part B but built with makeTSPFlow() (Single Commodity Flow).
//
// Why SCF is different from MTZ for probing
// ──────────────────────────────────────────
// SCF LP bound = DFJ bound (strictly ≥ MTZ bound).  Fixing an arc x[i→j]=1
// creates a flow imbalance: the unit of commodity that must route through i→j
// constrains the remaining flow variables.  The LP can detect subtour
// infeasibility that MTZ propagation cannot see.
//
// Consequently:
//   Weak probing  (L3): propagation still limited (no LP per fix), but SCF
//     degree constraints propagate the same cascade as MTZ.
//   Strong probing (L6): one LP per fix — LP infeasibility detection should
//     produce probed_fixed > 0, unlike MTZ.  The higher LP quality makes each
//     probe more informative.
//
// Cost: SCF has O(n²) flow variables on top of arc binaries (100 extra for
// n=10), so the LP is larger → each probe at L6 costs more than MTZ.
//
// Expected contrast vs Part B
// ────────────────────────────
//   MTZ (Part B) L6: probed_fixed ≈ 0 (LP relaxation too weak)
//   SCF (Part C) L6: probed_fixed > 0  (tighter LP detects arc infeasibility)
//   SCF solve: fewer nodes than MTZ at L4+ (better root LP bound)

static Model makeHardTSPMtz(int n, unsigned seed = 0xC0FFEE42u) {
    std::vector<baguette_test::TspArc> arcs;
    arcs.reserve(n * (n - 1));
    unsigned s = seed;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            s = s * 1664525u + 1013904223u;
            arcs.push_back({i, j, 1.0 + double(s % 10u)});
        }
    return baguette_test::makeTSPMtz(n, arcs);
}

static Model makeHardTSPFlow(int n, unsigned seed = 0xC0FFEE42u) {
    std::vector<baguette_test::TspArc> arcs;
    arcs.reserve(n * (n - 1));
    unsigned s = seed;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            s = s * 1664525u + 1013904223u;
            double dist = 1.0 + double(s % 10u);
            arcs.push_back({i, j, dist});
        }
    }
    return baguette_test::makeTSPFlow(n, arcs);
}

// ── Section 7: Presolve-only cost, hard TSP-10 SCF (levels 0-6) ─────────────

static void BM_SCF_PresolveLevelCost(benchmark::State& state) {
    const uint32_t level = static_cast<uint32_t>(state.range(0));
    MILPPresolveOpts opts;
    opts.level      = level;
    opts.timeLimitS = 30.0;

    for (auto _ : state) {
        state.PauseTiming();
        Model m = makeHardTSPFlow(10);
        state.ResumeTiming();

        MILPPresolveResult r = presolveMILPInPlace(m, opts);
        benchmark::DoNotOptimize(r.fixedVars);

        state.counters["tightened"]    = double(r.boundsTightened);
        state.counters["fixed"]        = double(r.fixedVars);
        state.counters["probed"]       = double(r.varsProbed);
        state.counters["probed_fixed"] = double(r.varsProbedFixed);
        state.counters["implied"]      = double(r.impliedRowsAdded);
    }
}
BENCHMARK(BM_SCF_PresolveLevelCost)
    ->DenseRange(0, 6)
    ->Unit(benchmark::kMicrosecond);

// ── Section 8: Presolve + root LP, hard TSP-10 SCF (levels 0-6) ─────────────
//
// Root LP bound on SCF is tighter than MTZ by construction.  If probing also
// fixes arc variables, the LP at B&B root is solved on a smaller model and
// may yield a higher lower bound.

static void BM_SCF_PresolveAndRootLP(benchmark::State& state) {
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
        Model m = makeHardTSPFlow(10);
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
BENCHMARK(BM_SCF_PresolveAndRootLP)
    ->DenseRange(0, 6)
    ->Unit(benchmark::kMicrosecond);

// ── Section 9: Full MILP solve, hard TSP-10 SCF (levels 0-6) ────────────────
//
// End-to-end solveMILP on the hard SCF TSP-10.  Because the root LP is
// tighter, B&B should require fewer nodes than MTZ even at L0.
// Higher presolve levels may reduce nodes further if probing fixes arcs.

static void BM_SCF_PresolveLevelSolve(benchmark::State& state) {
    const uint32_t level = static_cast<uint32_t>(state.range(0));
    BBOptions opts;
    opts.presolveLevel = level;
    opts.collectStats  = true;
    opts.timeLimitS    = 60.0;

    for (auto _ : state) {
        state.PauseTiming();
        Model m = makeHardTSPFlow(10);
        state.ResumeTiming();

        MILPResult r = solveMILP(m, opts);
        benchmark::DoNotOptimize(r.objectiveValue);

        if (r.stats) {
            state.counters["nodes"]     = double(r.stats->nodesExplored);
            state.counters["lp_solves"] = double(r.stats->lpSolvesTotal);
        }
        state.counters["obj"]    = r.objectiveValue;
        state.counters["status"] = double(int(r.status));
    }
}
BENCHMARK(BM_SCF_PresolveLevelSolve)
    ->DenseRange(0, 6)
    ->Unit(benchmark::kMillisecond);

// ── Part D: Uncapacitated Facility Location 5×10 ─────────────────────────────
//
// 5 facilities (fixed cost 20), 10 clients, LCG costs in [1,10].
// LP optimal = 67, IP optimal = 69.
//
// Probing cascade analysis
// ────────────────────────
// Unlike TSP, FL has a probing-friendly structure:
//   Fix y[i]=0: linking x[i][j] ≤ y[i]=0 forces all 10 x[i][j]=0.
//   Each zeroed x[i][j] tightens the coverage Σ_k x[k][j] ≥ 1, potentially
//   forcing another facility open for that client.
//   → 1 probe on y[i] can cascade into O(nCli) + O(nFac) fixings.
//
//   Fix y[i]=1: no immediate cascade (linking only provides upper bounds).
//
// Expected contrast with TSP
// ───────────────────────────
//   TSP (MTZ/SCF): probed_fixed = 0 at all levels
//   FL           : probed_fixed > 0 at L3+ if any y[i] is forced open/closed

// ── Section 10: Presolve-only cost, facility location 5×10 (levels 0-6) ──────

static void BM_FL_PresolveLevelCost(benchmark::State& state) {
    const uint32_t level = static_cast<uint32_t>(state.range(0));
    MILPPresolveOpts opts;
    opts.level      = level;
    opts.timeLimitS = 30.0;

    for (auto _ : state) {
        state.PauseTiming();
        Model m = baguette_test::makeFacilityLocation5x10();
        state.ResumeTiming();

        MILPPresolveResult r = presolveMILPInPlace(m, opts);
        benchmark::DoNotOptimize(r.fixedVars);

        state.counters["tightened"]    = double(r.boundsTightened);
        state.counters["fixed"]        = double(r.fixedVars);
        state.counters["probed"]       = double(r.varsProbed);
        state.counters["probed_fixed"] = double(r.varsProbedFixed);
        state.counters["implied"]      = double(r.impliedRowsAdded);
    }
}
BENCHMARK(BM_FL_PresolveLevelCost)
    ->DenseRange(0, 6)
    ->Unit(benchmark::kMicrosecond);

// ── Section 11: Presolve + root LP, facility location 5×10 (levels 0-6) ──────
//
// If probing fixes y[i]=1 or y[i]=0, the root LP is solved on a smaller model.
// A higher root LP bound means fewer B&B nodes.

static void BM_FL_PresolveAndRootLP(benchmark::State& state) {
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
        Model m = baguette_test::makeFacilityLocation5x10();
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
BENCHMARK(BM_FL_PresolveAndRootLP)
    ->DenseRange(0, 6)
    ->Unit(benchmark::kMicrosecond);

// ── Section 12: Full MILP solve, facility location 5×10 (levels 0-6) ─────────
//
// End-to-end solveMILP.  If probing fixes y[i], the B&B tree should shrink
// proportionally to the number of fixed facility decisions.

static void BM_FL_PresolveLevelSolve(benchmark::State& state) {
    const uint32_t level = static_cast<uint32_t>(state.range(0));
    BBOptions opts;
    opts.presolveLevel = level;
    opts.collectStats  = true;
    opts.timeLimitS    = 60.0;

    for (auto _ : state) {
        state.PauseTiming();
        Model m = baguette_test::makeFacilityLocation5x10();
        state.ResumeTiming();

        MILPResult r = solveMILP(m, opts);
        benchmark::DoNotOptimize(r.objectiveValue);

        if (r.stats) {
            state.counters["nodes"]     = double(r.stats->nodesExplored);
            state.counters["lp_solves"] = double(r.stats->lpSolvesTotal);
        }
        state.counters["obj"]    = r.objectiveValue;
        state.counters["status"] = double(int(r.status));
    }
}
BENCHMARK(BM_FL_PresolveLevelSolve)
    ->DenseRange(0, 6)
    ->Unit(benchmark::kMillisecond);

// ── Part E: Uncapacitated Facility Location 15×30 ────────────────────────────
//
// 15 facilities (fixed cost 20 each), 30 clients, LCG costs in [1,10].
// 465 binary variables (15 y[i] + 450 x[i][j]), 480 LP constraints.
//
// Scaling hypothesis: with 15 facilities and 30 clients, the coverage
// constraints are tighter per facility (2 clients/facility at optimum vs
// ~6 clients/facility for 5×10).  Probing y[i]=0 eliminates 30 x[i][j],
// forcing other facilities to absorb those clients — more likely to trigger
// LP infeasibility or bound propagation than in 5×10.
//
// Also tests whether L6 strong probing cost scales quadratically with nFac×nCli
// (100 LP solves at 5×10 vs potentially the same 50-cap probes but larger LPs).

// ── Section 13: Presolve-only cost, facility location 15×30 (levels 0-6) ─────

static void BM_FL2_PresolveLevelCost(benchmark::State& state) {
    const uint32_t level = static_cast<uint32_t>(state.range(0));
    MILPPresolveOpts opts;
    opts.level      = level;
    opts.timeLimitS = 30.0;

    for (auto _ : state) {
        state.PauseTiming();
        Model m = baguette_test::makeFacilityLocation15x30();
        state.ResumeTiming();

        MILPPresolveResult r = presolveMILPInPlace(m, opts);
        benchmark::DoNotOptimize(r.fixedVars);

        state.counters["tightened"]    = double(r.boundsTightened);
        state.counters["fixed"]        = double(r.fixedVars);
        state.counters["probed"]       = double(r.varsProbed);
        state.counters["probed_fixed"] = double(r.varsProbedFixed);
        state.counters["implied"]      = double(r.impliedRowsAdded);
    }
}
BENCHMARK(BM_FL2_PresolveLevelCost)
    ->DenseRange(0, 6)
    ->Unit(benchmark::kMicrosecond);

// ── Section 14: Presolve + root LP, facility location 15×30 (levels 0-6) ─────

static void BM_FL2_PresolveAndRootLP(benchmark::State& state) {
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
        Model m = baguette_test::makeFacilityLocation15x30();
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
BENCHMARK(BM_FL2_PresolveAndRootLP)
    ->DenseRange(0, 6)
    ->Unit(benchmark::kMicrosecond);

// ── Section 15: Full MILP solve, facility location 15×30 (levels 0-6) ────────

static void BM_FL2_PresolveLevelSolve(benchmark::State& state) {
    const uint32_t level = static_cast<uint32_t>(state.range(0));
    BBOptions opts;
    opts.presolveLevel = level;
    opts.collectStats  = true;
    opts.timeLimitS    = 60.0;

    for (auto _ : state) {
        state.PauseTiming();
        Model m = baguette_test::makeFacilityLocation15x30();
        state.ResumeTiming();

        MILPResult r = solveMILP(m, opts);
        benchmark::DoNotOptimize(r.objectiveValue);

        if (r.stats) {
            state.counters["nodes"]     = double(r.stats->nodesExplored);
            state.counters["lp_solves"] = double(r.stats->lpSolvesTotal);
        }
        state.counters["obj"]    = r.objectiveValue;
        state.counters["status"] = double(int(r.status));
    }
}
BENCHMARK(BM_FL2_PresolveLevelSolve)
    ->DenseRange(0, 6)
    ->Unit(benchmark::kMillisecond);

// ── Part F: Set Partitioning — small (10 elements, 30 columns) ───────────────
//
// 10 elements, 30 columns (10 singletons + 20 compound of size 2-4).
// LP optimal = 16, IP optimal = 19 (gap = 18.75%).
//
// Structure: equality coverage constraints (Σ x[i] = 1 per element).
// Probing hypothesis: fixing x[i]=0 removes coverage from its elements;
// any element left with a single remaining column forces that column to 1,
// which in turn removes it from other elements — cascade chain.

// ── Section 16: Presolve-only cost, SP small (levels 0-6) ────────────────────

static void BM_SP_PresolveLevelCost(benchmark::State& state) {
    const uint32_t level = static_cast<uint32_t>(state.range(0));
    MILPPresolveOpts opts;
    opts.level      = level;
    opts.timeLimitS = 30.0;

    for (auto _ : state) {
        state.PauseTiming();
        Model m = baguette_test::makeSetPartitioningSmall();
        state.ResumeTiming();

        MILPPresolveResult r = presolveMILPInPlace(m, opts);
        benchmark::DoNotOptimize(r.fixedVars);

        state.counters["tightened"]    = double(r.boundsTightened);
        state.counters["fixed"]        = double(r.fixedVars);
        state.counters["probed"]       = double(r.varsProbed);
        state.counters["probed_fixed"] = double(r.varsProbedFixed);
        state.counters["implied"]      = double(r.impliedRowsAdded);
    }
}
BENCHMARK(BM_SP_PresolveLevelCost)
    ->DenseRange(0, 6)
    ->Unit(benchmark::kMicrosecond);

// ── Section 17: Presolve + root LP, SP small (levels 0-6) ────────────────────

static void BM_SP_PresolveAndRootLP(benchmark::State& state) {
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
        Model m = baguette_test::makeSetPartitioningSmall();
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
BENCHMARK(BM_SP_PresolveAndRootLP)
    ->DenseRange(0, 6)
    ->Unit(benchmark::kMicrosecond);

// ── Section 18: Full MILP solve, SP small (levels 0-6) ───────────────────────

static void BM_SP_PresolveLevelSolve(benchmark::State& state) {
    const uint32_t level = static_cast<uint32_t>(state.range(0));
    BBOptions opts;
    opts.presolveLevel = level;
    opts.collectStats  = true;
    opts.timeLimitS    = 60.0;

    for (auto _ : state) {
        state.PauseTiming();
        Model m = baguette_test::makeSetPartitioningSmall();
        state.ResumeTiming();

        MILPResult r = solveMILP(m, opts);
        benchmark::DoNotOptimize(r.objectiveValue);

        if (r.stats) {
            state.counters["nodes"]     = double(r.stats->nodesExplored);
            state.counters["lp_solves"] = double(r.stats->lpSolvesTotal);
        }
        state.counters["obj"]    = r.objectiveValue;
        state.counters["status"] = double(int(r.status));
    }
}
BENCHMARK(BM_SP_PresolveLevelSolve)
    ->DenseRange(0, 6)
    ->Unit(benchmark::kMillisecond);

// ── Part G: Set Partitioning — large (30 elements, 90 columns) ───────────────
//
// 30 elements, 90 columns (30 singletons + 60 compound of size 2-5).
// LP optimal = 81, IP optimal = 82 (gap = 1.2%).
//
// Larger instance; lower gap means fewer B&B nodes expected.
// Probing may fix more variables given the larger column-element overlap.

// ── Section 19: Presolve-only cost, SP large (levels 0-6) ────────────────────

static void BM_SP2_PresolveLevelCost(benchmark::State& state) {
    const uint32_t level = static_cast<uint32_t>(state.range(0));
    MILPPresolveOpts opts;
    opts.level      = level;
    opts.timeLimitS = 30.0;

    for (auto _ : state) {
        state.PauseTiming();
        Model m = baguette_test::makeSetPartitioningLarge();
        state.ResumeTiming();

        MILPPresolveResult r = presolveMILPInPlace(m, opts);
        benchmark::DoNotOptimize(r.fixedVars);

        state.counters["tightened"]    = double(r.boundsTightened);
        state.counters["fixed"]        = double(r.fixedVars);
        state.counters["probed"]       = double(r.varsProbed);
        state.counters["probed_fixed"] = double(r.varsProbedFixed);
        state.counters["implied"]      = double(r.impliedRowsAdded);
    }
}
BENCHMARK(BM_SP2_PresolveLevelCost)
    ->DenseRange(0, 6)
    ->Unit(benchmark::kMicrosecond);

// ── Section 20: Presolve + root LP, SP large (levels 0-6) ────────────────────

static void BM_SP2_PresolveAndRootLP(benchmark::State& state) {
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
        Model m = baguette_test::makeSetPartitioningLarge();
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
BENCHMARK(BM_SP2_PresolveAndRootLP)
    ->DenseRange(0, 6)
    ->Unit(benchmark::kMicrosecond);

// ── Section 21: Full MILP solve, SP large (levels 0-6) ───────────────────────

static void BM_SP2_PresolveLevelSolve(benchmark::State& state) {
    const uint32_t level = static_cast<uint32_t>(state.range(0));
    BBOptions opts;
    opts.presolveLevel = level;
    opts.collectStats  = true;
    opts.timeLimitS    = 60.0;

    for (auto _ : state) {
        state.PauseTiming();
        Model m = baguette_test::makeSetPartitioningLarge();
        state.ResumeTiming();

        MILPResult r = solveMILP(m, opts);
        benchmark::DoNotOptimize(r.objectiveValue);

        if (r.stats) {
            state.counters["nodes"]     = double(r.stats->nodesExplored);
            state.counters["lp_solves"] = double(r.stats->lpSolvesTotal);
        }
        state.counters["obj"]    = r.objectiveValue;
        state.counters["status"] = double(int(r.status));
    }
}
BENCHMARK(BM_SP2_PresolveLevelSolve)
    ->DenseRange(0, 6)
    ->Unit(benchmark::kMillisecond);

// ── Part H: Hard asymmetric TSP-10 (MTZ + AllDiff CP) ────────────────────────
//
// Same instance as Part B (seed 0xC0FFEE42), with an AllDiff constraint on
// the 9 MTZ position variables u[1..9].  The AllDiff propagator activates at
// level 2 (CP propagation to fixed point), unlike all previous families where
// L2 ≈ L1.
//
// Hypothesis: at L2, AllDiff bounds-consistency can tighten position domains
// (u[i] ∈ [1,9] all distinct) without any arc being fixed.  At L3+, when an
// arc is probed (x[i→j]=0 or 1), the resulting position constraint propagates
// through AllDiff to tighten other position variables.

// ── Section 22: Presolve-only cost, hard TSP-10 MTZAD (levels 0-6) ───────────

static void BM_MTZAD_PresolveLevelCost(benchmark::State& state) {
    const uint32_t level = static_cast<uint32_t>(state.range(0));
    MILPPresolveOpts opts;
    opts.level      = level;
    opts.timeLimitS = 30.0;

    for (auto _ : state) {
        state.PauseTiming();
        Model m = makeHardTSPMtz(10);
        state.ResumeTiming();

        MILPPresolveResult r = presolveMILPInPlace(m, opts);
        benchmark::DoNotOptimize(r.fixedVars);

        state.counters["tightened"]    = double(r.boundsTightened);
        state.counters["fixed"]        = double(r.fixedVars);
        state.counters["probed"]       = double(r.varsProbed);
        state.counters["probed_fixed"] = double(r.varsProbedFixed);
        state.counters["implied"]      = double(r.impliedRowsAdded);
    }
}
BENCHMARK(BM_MTZAD_PresolveLevelCost)
    ->DenseRange(0, 6)
    ->Unit(benchmark::kMicrosecond);

// ── Section 23: Presolve + root LP, hard TSP-10 MTZAD (levels 0-6) ───────────

static void BM_MTZAD_PresolveAndRootLP(benchmark::State& state) {
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
        Model m = makeHardTSPMtz(10);
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
BENCHMARK(BM_MTZAD_PresolveAndRootLP)
    ->DenseRange(0, 6)
    ->Unit(benchmark::kMicrosecond);

// ── Section 24: Full MILP solve, hard TSP-10 MTZAD (levels 0-6) ──────────────

static void BM_MTZAD_PresolveLevelSolve(benchmark::State& state) {
    const uint32_t level = static_cast<uint32_t>(state.range(0));
    BBOptions opts;
    opts.presolveLevel = level;
    opts.collectStats  = true;
    opts.timeLimitS    = 60.0;

    for (auto _ : state) {
        state.PauseTiming();
        Model m = makeHardTSPMtz(10);
        state.ResumeTiming();

        MILPResult r = solveMILP(m, opts);
        benchmark::DoNotOptimize(r.objectiveValue);

        if (r.stats) {
            state.counters["nodes"]     = double(r.stats->nodesExplored);
            state.counters["lp_solves"] = double(r.stats->lpSolvesTotal);
        }
        state.counters["obj"]    = r.objectiveValue;
        state.counters["status"] = double(int(r.status));
    }
}
BENCHMARK(BM_MTZAD_PresolveLevelSolve)
    ->DenseRange(0, 6)
    ->Unit(benchmark::kMillisecond);
