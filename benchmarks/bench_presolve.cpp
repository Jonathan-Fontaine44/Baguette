#include <benchmark/benchmark.h>

#include "baguette/core/Sense.hpp"
#include "baguette/lp/LPSolver.hpp"
#include "baguette/lp/presolve.hpp"
#include "baguette/milp/BranchAndBound.hpp"
#include "baguette/milp/Presolve.hpp"
#include "baguette/model/Model.hpp"
#include "baguette/model/ModelEnums.hpp"
#include "lp/MILP_problems.hpp"

using namespace baguette;

// â”€â”€ Model builders â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

// Chain LP: x[i] + x[i+1] <= n-i. presolveTB tightens every upper bound.
// No variable gets fully fixed â†’ presolveElim removes 0 vars/rows on this model.
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

// MILPCascade: n Integer variables in groups of 5.
// Each group: x[i] in [0,10], per-var LP constraint x[i] <= 3.9, group sum >= 13.5.
// presolveMILPInPlace runs two outer iterations:
//   iter 1: LP tightens ub to 3.9, MILP rounds to 3. (bt=n, br=n)
//   iter 2: With ub=3, LP tightens lb to 1.5, MILP rounds to 2. (bt=n, br=n)
// presolveTBInPlace alone reaches ub=3.9 but cannot round lb to 2.
// n must be a multiple of 5.
static Model makeMILPCascade(int n) {
    Model m;
    std::vector<Variable> x;
    x.reserve(n);
    for (int i = 0; i < n; ++i)
        x.push_back(m.addVar(0.0, 10.0, VarType::Integer, "x" + std::to_string(i)));
    for (int g = 0; g < n / 5; ++g) {
        LinearExpr sum;
        for (int k = 0; k < 5; ++k) {
            int vi = g * 5 + k;
            m.addLPConstraint(1.0 * x[vi], Sense::LessEq, 3.9);
            sum += 1.0 * x[vi];
        }
        m.addLPConstraint(sum, Sense::GreaterEq, 13.5);
    }
    LinearExpr obj;
    for (auto& v : x) obj += 1.0 * v;
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
// nFixed/2 constraints involve only fixed variables â†’ all redundant after elim.
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

    // Constraints involving only fixed vars: 2+2=4 <= 100 â†’ always redundant.
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

// â”€â”€ Helpers: LP option sets â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

static LPOptions lpNone()  { LPOptions o; o.enablePresolve = false; o.enableElimination = false; o.timeLimitS = 10.0; return o; }
static LPOptions lpTB()    { LPOptions o; o.enablePresolve = true;  o.enableElimination = false; o.timeLimitS = 10.0; return o; }
static LPOptions lpElim()  { LPOptions o; o.enablePresolve = true;  o.enableElimination = true;  o.timeLimitS = 10.0; return o; }

static BBOptions bbNone()  { BBOptions o; o.presolveLevel = 0; o.enableElimination = false; o.timeLimitS = 10.0; return o; }
static BBOptions bbTB()    { BBOptions o; o.presolveLevel = 1;  o.enableElimination = false; o.timeLimitS = 10.0; return o; }
static BBOptions bbElim()  { BBOptions o; o.presolveLevel = 1;  o.enableElimination = true;  o.timeLimitS = 10.0; return o; }

// â”€â”€ presolveTB-only timing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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

// â”€â”€ LP solve: Chain 20 (NoPresolve / TB / Elim) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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

// â”€â”€ LP solve: ElimLP 60 (NoPresolve / TB / Elim) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// 30 fixed vars + 30 free vars, 15 redundant rows + 15 binding rows.
// Elim removes 30 vars and 15 rows â†’ solves a 30-var / 15-row LP.

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

// â”€â”€ LP solve: ElimLP 200 (NoPresolve / TB / Elim) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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

// â”€â”€ MILP solve: Knapsack 15 (NoPresolve / TB / Elim) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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

// â”€â”€ MILP solve: TSP10 (NoPresolve / TB / Elim) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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

// â”€â”€ TSP10: presolve-only timing + variable/row change stats â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

static void BM_PresolveOnly_TB_TSP10(benchmark::State& state) {
    for (auto _ : state) {
        Model m = baguette_test::makeTSP10();
        PresolveResult pr = presolveTBInPlace(m);
        benchmark::DoNotOptimize(pr.boundsTightened);
        state.counters["bounds_tightened"] = pr.boundsTightened;
        state.counters["fixed_vars"]       = pr.fixedVars;
        state.counters["passes"]           = pr.passesRun;
    }
}
BENCHMARK(BM_PresolveOnly_TB_TSP10);

static void BM_PresolveOnly_Elim_TSP10(benchmark::State& state) {
    for (auto _ : state) {
        Model m = baguette_test::makeTSP10();
        presolveTBInPlace(m);
        EliminationRecord rec;
        Model reduced = presolveElim(m, rec);
        benchmark::DoNotOptimize(rec.varsEliminated);
        state.counters["vars_eliminated"] = rec.varsEliminated;
        state.counters["rows_eliminated"] = rec.rowsEliminated;
    }
}
BENCHMARK(BM_PresolveOnly_Elim_TSP10);

// â”€â”€ MILPCascade: LP-presolve vs MILP-presolve (presolve-only timing) â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Cascade20: 4 groups of 5 vars â€” two outer MILP iterations.
// Cascade50: 10 groups  â€” same cascade, scaled up.
// LP-only (presolveTBInPlace): reaches ub=3.9, cannot round lb â†’ misses iter 2.
// MILP     (presolveMILPInPlace): two rounds â†’ tighter [2,3] bounds.

static void BM_PresolveOnly_LP_Cascade20(benchmark::State& state) {
    for (auto _ : state) {
        Model m = makeMILPCascade(20);
        PresolveResult pr = presolveTBInPlace(m);
        benchmark::DoNotOptimize(pr.boundsTightened);
        state.counters["bounds_tightened"] = pr.boundsTightened;
        state.counters["passes"]           = pr.passesRun;
    }
}
BENCHMARK(BM_PresolveOnly_LP_Cascade20);

static void BM_PresolveOnly_MILP_Cascade20(benchmark::State& state) {
    for (auto _ : state) {
        Model m = makeMILPCascade(20);
        MILPPresolveResult pr = presolveMILPInPlace(m);
        benchmark::DoNotOptimize(pr.boundsTightened);
        state.counters["bounds_tightened"] = pr.boundsTightened;
        state.counters["bounds_rounded"]   = pr.boundsRounded;
        state.counters["passes"]           = pr.passesRun;
    }
}
BENCHMARK(BM_PresolveOnly_MILP_Cascade20);

static void BM_PresolveOnly_LP_Cascade50(benchmark::State& state) {
    for (auto _ : state) {
        Model m = makeMILPCascade(50);
        PresolveResult pr = presolveTBInPlace(m);
        benchmark::DoNotOptimize(pr.boundsTightened);
        state.counters["bounds_tightened"] = pr.boundsTightened;
        state.counters["passes"]           = pr.passesRun;
    }
}
BENCHMARK(BM_PresolveOnly_LP_Cascade50);

static void BM_PresolveOnly_MILP_Cascade50(benchmark::State& state) {
    for (auto _ : state) {
        Model m = makeMILPCascade(50);
        MILPPresolveResult pr = presolveMILPInPlace(m);
        benchmark::DoNotOptimize(pr.boundsTightened);
        state.counters["bounds_tightened"] = pr.boundsTightened;
        state.counters["bounds_rounded"]   = pr.boundsRounded;
        state.counters["passes"]           = pr.passesRun;
    }
}
BENCHMARK(BM_PresolveOnly_MILP_Cascade50);

// â”€â”€ MILPCascade: full MILP solve (NoPresolve / MILP-presolve) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Measures the end-to-end B&B benefit of MILP presolve on the cascade model.
// NoPresolve: B&B starts with x[i] âˆˆ [0,10] â€” wide domains, many branches.
// MILP-TB:    B&B starts with x[i] âˆˆ [2,3]  â€” domains reduced by 87.5%.

static void BM_MILP_Cascade20_NoPresolve(benchmark::State& state) {
    for (auto _ : state) {
        MILPResult r = solveMILP(makeMILPCascade(20), bbNone());
        benchmark::DoNotOptimize(r.objectiveValue);
    }
}
BENCHMARK(BM_MILP_Cascade20_NoPresolve);

static void BM_MILP_Cascade20_MILPPresolve(benchmark::State& state) {
    for (auto _ : state) {
        MILPResult r = solveMILP(makeMILPCascade(20), bbTB());
        benchmark::DoNotOptimize(r.objectiveValue);
    }
}
BENCHMARK(BM_MILP_Cascade20_MILPPresolve);

