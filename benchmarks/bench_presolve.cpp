#include <benchmark/benchmark.h>

#include "baguette/core/Sense.hpp"
#include "baguette/lp/LPSolver.hpp"
#include "baguette/lp/presolve.hpp"
#include "baguette/milp/BranchAndBound.hpp"
#include "baguette/model/Model.hpp"
#include "baguette/model/ModelEnums.hpp"
#include "lp/MILP_problems.hpp"

using namespace baguette;

// ── Model builders ────────────────────────────────────────────────────────────

// Chain LP: x[i] + x[i+1] <= n-i. presolveTB tightens every upper bound.
// No variable gets fully fixed → presolveElim removes 0 vars/rows on this model.
static Model makeChainLP(int n) {
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

// Knapsack MILP: n binary items, one weight constraint.
static Model makeKnapsackMILP(int n) {
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

// Elimination LP: n/2 variables pre-fixed (lb==ub==2), n/2 free in [0,5].
// nFixed/2 constraints involve only fixed variables → all redundant after elim.
// nFree/2 non-redundant GEQ constraints on the free variables.
// presolveElim removes nFixed vars and nFixed/2 rows; the LP on the reduced
// model is half the size.
static Model makeElimLP(int n) {
    Model m;
    int nFixed = n / 2;
    int nFree  = n - nFixed;
    std::vector<Variable> fixed_vars, free_vars;
    for (int i = 0; i < nFixed; ++i)
        fixed_vars.push_back(m.addVar(2.0, 2.0, "f" + std::to_string(i)));
    for (int i = 0; i < nFree; ++i)
        free_vars.push_back(m.addVar(0.0, 5.0, "y" + std::to_string(i)));

    // Constraints involving only fixed vars: 2+2=4 <= 100 → always redundant.
    for (int i = 0; i < nFixed / 2; ++i)
        m.addLPConstraint(1.0*fixed_vars[i] + 1.0*fixed_vars[(i+1) % nFixed],
                          Sense::LessEq, 100.0);

    // Non-redundant constraints on free vars: y[i]+y[i+1] >= 1.
    for (int i = 0; i < nFree / 2; ++i)
        m.addLPConstraint(1.0*free_vars[i] + 1.0*free_vars[(i+1) % nFree],
                          Sense::GreaterEq, 1.0);

    LinearExpr obj;
    for (auto& v : fixed_vars) obj += 1.0 * v;
    for (auto& v : free_vars)  obj += 1.0 * v;
    m.setObjective(obj, ObjSense::Minimize);
    return m;
}

// ── Helpers: LP option sets ───────────────────────────────────────────────────

static LPOptions lpNone()  { LPOptions o; o.enablePresolve = false; o.enableElimination = false; o.timeLimitS = 10.0; return o; }
static LPOptions lpTB()    { LPOptions o; o.enablePresolve = true;  o.enableElimination = false; o.timeLimitS = 10.0; return o; }
static LPOptions lpElim()  { LPOptions o; o.enablePresolve = true;  o.enableElimination = true;  o.timeLimitS = 10.0; return o; }

static BBOptions bbNone()  { BBOptions o; o.enablePresolve = false; o.enableElimination = false; o.timeLimitS = 10.0; return o; }
static BBOptions bbTB()    { BBOptions o; o.enablePresolve = true;  o.enableElimination = false; o.timeLimitS = 10.0; return o; }
static BBOptions bbElim()  { BBOptions o; o.enablePresolve = true;  o.enableElimination = true;  o.timeLimitS = 10.0; return o; }

// ── presolveTB-only timing ────────────────────────────────────────────────────

static void BM_PresolveOnly_TB_Chain20(benchmark::State& state) {
    for (auto _ : state) {
        Model m = makeChainLP(20);
        PresolveResult pr = presolveTBInPlace(m);
        benchmark::DoNotOptimize(pr.boundsTightened);
    }
}
BENCHMARK(BM_PresolveOnly_TB_Chain20);

static void BM_PresolveOnly_Elim_Chain20(benchmark::State& state) {
    for (auto _ : state) {
        Model m = makeChainLP(20);
        presolveTBInPlace(m);
        EliminationRecord rec;
        Model reduced = presolveElim(m, rec);
        benchmark::DoNotOptimize(rec.varsEliminated);
        benchmark::DoNotOptimize(rec.rowsEliminated);
    }
}
BENCHMARK(BM_PresolveOnly_Elim_Chain20);

static void BM_PresolveOnly_TB_Chain100(benchmark::State& state) {
    for (auto _ : state) {
        Model m = makeChainLP(100);
        PresolveResult pr = presolveTBInPlace(m);
        benchmark::DoNotOptimize(pr.boundsTightened);
    }
}
BENCHMARK(BM_PresolveOnly_TB_Chain100);

static void BM_PresolveOnly_Elim_Chain100(benchmark::State& state) {
    for (auto _ : state) {
        Model m = makeChainLP(100);
        presolveTBInPlace(m);
        EliminationRecord rec;
        Model reduced = presolveElim(m, rec);
        benchmark::DoNotOptimize(rec.varsEliminated);
        benchmark::DoNotOptimize(rec.rowsEliminated);
    }
}
BENCHMARK(BM_PresolveOnly_Elim_Chain100);

// ── LP solve: Chain 20 (NoPresolve / TB / Elim) ───────────────────────────────

static void BM_LP_Chain20_NoPresolve(benchmark::State& state) {
    for (auto _ : state) {
        LPResult r = solveLP(makeChainLP(20), lpNone());
        benchmark::DoNotOptimize(r.objectiveValue);
    }
}
BENCHMARK(BM_LP_Chain20_NoPresolve);

static void BM_LP_Chain20_TB(benchmark::State& state) {
    for (auto _ : state) {
        LPResult r = solveLP(makeChainLP(20), lpTB());
        benchmark::DoNotOptimize(r.objectiveValue);
    }
}
BENCHMARK(BM_LP_Chain20_TB);

static void BM_LP_Chain20_Elim(benchmark::State& state) {
    for (auto _ : state) {
        LPResult r = solveLP(makeChainLP(20), lpElim());
        benchmark::DoNotOptimize(r.objectiveValue);
    }
}
BENCHMARK(BM_LP_Chain20_Elim);

// ── LP solve: ElimLP 60 (NoPresolve / TB / Elim) ─────────────────────────────
// 30 fixed vars + 30 free vars, 15 redundant rows + 15 binding rows.
// Elim removes 30 vars and 15 rows → solves a 30-var / 15-row LP.

static void BM_LP_Elim60_NoPresolve(benchmark::State& state) {
    for (auto _ : state) {
        LPResult r = solveLP(makeElimLP(60), lpNone());
        benchmark::DoNotOptimize(r.objectiveValue);
    }
}
BENCHMARK(BM_LP_Elim60_NoPresolve);

static void BM_LP_Elim60_TB(benchmark::State& state) {
    for (auto _ : state) {
        LPResult r = solveLP(makeElimLP(60), lpTB());
        benchmark::DoNotOptimize(r.objectiveValue);
    }
}
BENCHMARK(BM_LP_Elim60_TB);

static void BM_LP_Elim60_Elim(benchmark::State& state) {
    for (auto _ : state) {
        LPResult r = solveLP(makeElimLP(60), lpElim());
        benchmark::DoNotOptimize(r.objectiveValue);
    }
}
BENCHMARK(BM_LP_Elim60_Elim);

// ── LP solve: ElimLP 200 (NoPresolve / TB / Elim) ────────────────────────────

static void BM_LP_Elim200_NoPresolve(benchmark::State& state) {
    for (auto _ : state) {
        LPResult r = solveLP(makeElimLP(200), lpNone());
        benchmark::DoNotOptimize(r.objectiveValue);
    }
}
BENCHMARK(BM_LP_Elim200_NoPresolve);

static void BM_LP_Elim200_TB(benchmark::State& state) {
    for (auto _ : state) {
        LPResult r = solveLP(makeElimLP(200), lpTB());
        benchmark::DoNotOptimize(r.objectiveValue);
    }
}
BENCHMARK(BM_LP_Elim200_TB);

static void BM_LP_Elim200_Elim(benchmark::State& state) {
    for (auto _ : state) {
        LPResult r = solveLP(makeElimLP(200), lpElim());
        benchmark::DoNotOptimize(r.objectiveValue);
    }
}
BENCHMARK(BM_LP_Elim200_Elim);

// ── MILP solve: Knapsack 15 (NoPresolve / TB / Elim) ─────────────────────────

static void BM_MILP_Knapsack15_NoPresolve(benchmark::State& state) {
    for (auto _ : state) {
        MILPResult r = solveMILP(makeKnapsackMILP(15), bbNone());
        benchmark::DoNotOptimize(r.objectiveValue);
    }
}
BENCHMARK(BM_MILP_Knapsack15_NoPresolve);

static void BM_MILP_Knapsack15_TB(benchmark::State& state) {
    for (auto _ : state) {
        MILPResult r = solveMILP(makeKnapsackMILP(15), bbTB());
        benchmark::DoNotOptimize(r.objectiveValue);
    }
}
BENCHMARK(BM_MILP_Knapsack15_TB);

static void BM_MILP_Knapsack15_Elim(benchmark::State& state) {
    for (auto _ : state) {
        MILPResult r = solveMILP(makeKnapsackMILP(15), bbElim());
        benchmark::DoNotOptimize(r.objectiveValue);
    }
}
BENCHMARK(BM_MILP_Knapsack15_Elim);

// ── MILP solve: TSP10 (NoPresolve / TB / Elim) ───────────────────────────────

static void BM_MILP_TSP10_NoPresolve(benchmark::State& state) {
    for (auto _ : state) {
        MILPResult r = solveMILP(baguette_test::makeTSP10(), bbNone());
        benchmark::DoNotOptimize(r.objectiveValue);
    }
}
BENCHMARK(BM_MILP_TSP10_NoPresolve);

static void BM_MILP_TSP10_TB(benchmark::State& state) {
    for (auto _ : state) {
        MILPResult r = solveMILP(baguette_test::makeTSP10(), bbTB());
        benchmark::DoNotOptimize(r.objectiveValue);
    }
}
BENCHMARK(BM_MILP_TSP10_TB);

static void BM_MILP_TSP10_Elim(benchmark::State& state) {
    for (auto _ : state) {
        MILPResult r = solveMILP(baguette_test::makeTSP10(), bbElim());
        benchmark::DoNotOptimize(r.objectiveValue);
    }
}
BENCHMARK(BM_MILP_TSP10_Elim);
