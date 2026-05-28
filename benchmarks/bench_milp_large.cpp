#include <benchmark/benchmark.h>
#include <functional>
#include <string>
#include <vector>

#include "baguette/core/Sense.hpp"
#include "baguette/lp/LPSolver.hpp"
#include "baguette/milp/BranchAndBound.hpp"
#include "baguette/model/Model.hpp"
#include "baguette/model/ModelEnums.hpp"

using namespace baguette;

// ── Model builders ─────────────────────────────────────────────────────────

// 50-item 0/1 knapsack. w[i] = 1+(i%3), p[i] = 50-i, capacity = 25.
// LP relaxation is fractional → B&B required.
static Model makeKnapsack50()
{
    Model m;
    std::vector<Variable> x;
    x.reserve(50);
    LinearExpr obj, weight;
    for (int i = 0; i < 50; ++i) {
        x.push_back(m.addVar(0.0, 1.0, VarType::Binary));
        weight += (1.0 + (i % 3)) * x[i];
        obj    += double(50 - i)  * x[i];
    }
    m.addLPConstraint(weight, Sense::LessEq, 25.0);
    m.setObjective(obj, ObjSense::Maximize);
    return m;
}

// n-city TSP, MTZ formulation, adjacent-arc costs (cost 1 for adjacent cities,
// cost n otherwise). LP relaxation = MILP optimal = n: the cyclic tour
// 0→1→…→(n-1)→0 is an integer extreme point of the MTZ LP polytope.
// Single-node B&B: measures LP solve time on an (n(n-1)+(n-1))-variable LP.
static Model makeTSPn(int n)
{
    Model m;
    std::vector<std::vector<Variable>> x(n, std::vector<Variable>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (i != j) x[i][j] = m.addVar(0.0, 1.0, VarType::Binary);

    std::vector<Variable> u(n - 1);
    for (int k = 0; k < n - 1; ++k)
        u[k] = m.addVar(1.0, double(n - 1), VarType::Integer);

    LinearExpr obj;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (i != j) {
                double c = (j == (i + 1) % n || j == (i + n - 1) % n) ? 1.0 : double(n);
                obj += c * x[i][j];
            }
    m.setObjective(obj, ObjSense::Minimize);

    for (int i = 0; i < n; ++i) {
        LinearExpr out, in;
        for (int j = 0; j < n; ++j)
            if (j != i) { out += 1.0 * x[i][j]; in += 1.0 * x[j][i]; }
        if (i > 0) m.addLPConstraint(out, Sense::Equal, 1.0);
        m.addLPConstraint(in,  Sense::Equal, 1.0);
    }

    for (int i = 1; i < n; ++i)
        for (int j = 1; j < n; ++j)
            if (i != j) {
                LinearExpr mtz;
                mtz += 1.0 * u[i - 1] + -1.0 * u[j - 1] + double(n) * x[i][j];
                m.addLPConstraint(mtz, Sense::LessEq, double(n - 1));
            }
    return m;
}

// Cascade50: 10 groups × 5 integer vars ∈ [0,10].
// presolveMILP reduces domains to [2,3]; NoPresolve leaves [0,10] (huge tree).
static Model makeCascade50()
{
    const int n = 50;
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

// ── Benchmark runner ────────────────────────────────────────────────────────

static void runLarge(benchmark::State& state,
                     LPMethod          method,
                     bool              enablePresolve,
                     std::function<Model()> build)
{
    for (auto _ : state) {
        BBOptions opts;
        opts.lpOpts.method  = method;
        opts.presolveLevel  = enablePresolve ? 1u : 0u;
        opts.timeLimitS     = 300.0; // 5-minute wall-clock limit
        opts.collectStats   = true;
        MILPResult r = solveMILP(build(), opts);
        benchmark::DoNotOptimize(r);
        state.counters["status"]    = double(int(r.status));
        state.counters["obj"]       = r.objectiveValue;
        if (r.stats) {
            state.counters["nodes"]     = r.stats->nodesExplored;
            state.counters["lp_solves"] = r.stats->lpSolvesTotal;
        }
    }
}

// ── Knapsack50 — real B&B (LP relaxation is fractional) ────────────────────

BENCHMARK_CAPTURE(runLarge, Knapsack50/DualSimplex,
    LPMethod::DualSimplex,    true, makeKnapsack50);
BENCHMARK_CAPTURE(runLarge, Knapsack50/DualSimplexBV,
    LPMethod::DualSimplexBV,  true, makeKnapsack50);
BENCHMARK_CAPTURE(runLarge, Knapsack50/PrimalSimplexBV,
    LPMethod::PrimalSimplexBV, true, makeKnapsack50);
BENCHMARK_CAPTURE(runLarge, Knapsack50/RevisedSimplexBV,
    LPMethod::RevisedSimplexBV, true, makeKnapsack50);
BENCHMARK_CAPTURE(runLarge, Knapsack50/MehrotraIPM,
    LPMethod::MehrotraIPM,    true, makeKnapsack50);

// ── TSP50 — single-node B&B (LP = MILP optimal) ────────────────────────────
// 2 499 variables, 2 451 constraints. Measures LP solve time at scale.

BENCHMARK_CAPTURE(runLarge, TSP50/DualSimplex,
    LPMethod::DualSimplex,    true, []() { return makeTSPn(50); });
BENCHMARK_CAPTURE(runLarge, TSP50/DualSimplexBV,
    LPMethod::DualSimplexBV,  true, []() { return makeTSPn(50); });
BENCHMARK_CAPTURE(runLarge, TSP50/PrimalSimplexBV,
    LPMethod::PrimalSimplexBV, true, []() { return makeTSPn(50); });
BENCHMARK_CAPTURE(runLarge, TSP50/RevisedSimplexBV,
    LPMethod::RevisedSimplexBV, true, []() { return makeTSPn(50); });
BENCHMARK_CAPTURE(runLarge, TSP50/MehrotraIPM,
    LPMethod::MehrotraIPM,    true, []() { return makeTSPn(50); });

// ── Cascade50 NoPresolve — huge tree, measures nodes/s under time limit ─────

BENCHMARK_CAPTURE(runLarge, Cascade50/NoPresolve/DualSimplex,
    LPMethod::DualSimplex,    false, makeCascade50);
BENCHMARK_CAPTURE(runLarge, Cascade50/NoPresolve/DualSimplexBV,
    LPMethod::DualSimplexBV,  false, makeCascade50);
BENCHMARK_CAPTURE(runLarge, Cascade50/NoPresolve/PrimalSimplexBV,
    LPMethod::PrimalSimplexBV, false, makeCascade50);
BENCHMARK_CAPTURE(runLarge, Cascade50/NoPresolve/RevisedSimplexBV,
    LPMethod::RevisedSimplexBV, false, makeCascade50);
BENCHMARK_CAPTURE(runLarge, Cascade50/NoPresolve/MehrotraIPM,
    LPMethod::MehrotraIPM,    false, makeCascade50);

// ── Cascade50 MILPPresolve — trivially fast post-presolve ──────────────────
// Presolve reduces [0,10] → [2,3]; pre-LP bound prune eliminates the tree.

BENCHMARK_CAPTURE(runLarge, Cascade50/MILPPresolve/DualSimplex,
    LPMethod::DualSimplex,    true, makeCascade50);
BENCHMARK_CAPTURE(runLarge, Cascade50/MILPPresolve/DualSimplexBV,
    LPMethod::DualSimplexBV,  true, makeCascade50);
BENCHMARK_CAPTURE(runLarge, Cascade50/MILPPresolve/PrimalSimplexBV,
    LPMethod::PrimalSimplexBV, true, makeCascade50);
BENCHMARK_CAPTURE(runLarge, Cascade50/MILPPresolve/RevisedSimplexBV,
    LPMethod::RevisedSimplexBV, true, makeCascade50);
BENCHMARK_CAPTURE(runLarge, Cascade50/MILPPresolve/MehrotraIPM,
    LPMethod::MehrotraIPM,    true, makeCascade50);
