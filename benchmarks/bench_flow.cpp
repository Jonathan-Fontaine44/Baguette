#include <benchmark/benchmark.h>
#include <functional>

#include "baguette/lp/LPSolver.hpp"
#include "baguette/milp/BranchAndBound.hpp"
#include "graph_problems.hpp"

using namespace baguette;

// ── LP relaxation ─────────────────────────────────────────────────────────────
// DAG 100 nœuds, densité 12 % (~680 arcs, ~2100 lignes en SF).
// MehrotraIPM exclu ici : sa normale-équation (m×m×n) est trop coûteuse
// à ce rang ; voir benchmark dédié sur instance réduite ci-dessous.

static void BM_FlowLP(benchmark::State& state,
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

// 1 source, 1 puits — 100 nœuds
BENCHMARK_CAPTURE(BM_FlowLP, Flow1S1T_100/PrimalSimplex,  LPMethod::PrimalSimplex,
    []() { return baguette_test::makeFlowDAG(); });
BENCHMARK_CAPTURE(BM_FlowLP, Flow1S1T_100/DualSimplex,    LPMethod::DualSimplex,
    []() { return baguette_test::makeFlowDAG(); });
BENCHMARK_CAPTURE(BM_FlowLP, Flow1S1T_100/RevisedSimplex, LPMethod::RevisedSimplex,
    []() { return baguette_test::makeFlowDAG(); });

// 2 sources, 2 puits — 100 nœuds
BENCHMARK_CAPTURE(BM_FlowLP, Flow2S2T_100/PrimalSimplex,  LPMethod::PrimalSimplex,
    []() { return baguette_test::makeFlowDAG2S2T(); });
BENCHMARK_CAPTURE(BM_FlowLP, Flow2S2T_100/DualSimplex,    LPMethod::DualSimplex,
    []() { return baguette_test::makeFlowDAG2S2T(); });
BENCHMARK_CAPTURE(BM_FlowLP, Flow2S2T_100/RevisedSimplex, LPMethod::RevisedSimplex,
    []() { return baguette_test::makeFlowDAG2S2T(); });

// ── IPM sur instance réduite (30 nœuds, ~100 arcs, ~300 lignes SF) ────────────
// Taille adaptée à la complexité O(m³) de la factorisation LU de la normale.

BENCHMARK_CAPTURE(BM_FlowLP, Flow1S1T_30/DualSimplex,    LPMethod::DualSimplex,
    []() { return baguette_test::makeFlowDAG(30); });
BENCHMARK_CAPTURE(BM_FlowLP, Flow1S1T_30/MehrotraIPM,    LPMethod::MehrotraIPM,
    []() { return baguette_test::makeFlowDAG(30); });

BENCHMARK_CAPTURE(BM_FlowLP, Flow2S2T_30/DualSimplex,    LPMethod::DualSimplex,
    []() { return baguette_test::makeFlowDAG2S2T(30); });
BENCHMARK_CAPTURE(BM_FlowLP, Flow2S2T_30/MehrotraIPM,    LPMethod::MehrotraIPM,
    []() { return baguette_test::makeFlowDAG2S2T(30); });

// ── MILP : B&B et B&C sur instance 30 nœuds (MILP exponential, rester petit) ──

static void BM_FlowMILP(benchmark::State& state,
                         LPMethod lpMethod,
                         bool enableCuts,
                         std::function<Model()> build)
{
    for (auto _ : state) {
        BBOptions opts;
        opts.lpMethod   = lpMethod;
        opts.enableCuts = enableCuts;
        opts.maxNodes   = 200;   // borne le nombre de nœuds explorés
        MILPResult r    = solveMILP(build(), opts);
        benchmark::DoNotOptimize(r.objectiveValue);
    }
}

// 1 source, 1 puits — 30 nœuds
BENCHMARK_CAPTURE(BM_FlowMILP, Flow1S1T_30/BB/DualSimplex,
    LPMethod::DualSimplex,  false, []() { return baguette_test::makeFlowDAG(30); });
BENCHMARK_CAPTURE(BM_FlowMILP, Flow1S1T_30/BnC/DualSimplex,
    LPMethod::DualSimplex,  true,  []() { return baguette_test::makeFlowDAG(30); });

// 2 sources, 2 puits — 30 nœuds
BENCHMARK_CAPTURE(BM_FlowMILP, Flow2S2T_30/BB/DualSimplex,
    LPMethod::DualSimplex,  false, []() { return baguette_test::makeFlowDAG2S2T(30); });
BENCHMARK_CAPTURE(BM_FlowMILP, Flow2S2T_30/BnC/DualSimplex,
    LPMethod::DualSimplex,  true,  []() { return baguette_test::makeFlowDAG2S2T(30); });
