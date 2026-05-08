#include <benchmark/benchmark.h>

#include "baguette/lp/LPSolver.hpp"
#include "baguette/model/Model.hpp"

using namespace baguette;

// ── Network problem builders ──────────────────────────────────────────────────

// 3-arc: supply 4 at node 0, transit node 1, demand 4 at node 2.
// Arcs: (0→1) c=2, (0→2) c=7, (1→2) c=3.  Optimal: x01=4,x12=4, obj=20.
static Model makeNet3Feasible() {
    Model m;
    auto x01 = m.addVar(0.0, 10.0);
    auto x02 = m.addVar(0.0, 10.0);
    auto x12 = m.addVar(0.0, 10.0);
    m.addLPConstraint( 1.0*x01 + 1.0*x02,              Sense::Equal,  4.0);
    m.addLPConstraint(-1.0*x01              + 1.0*x12, Sense::Equal,  0.0);
    m.addLPConstraint(             -1.0*x02 - 1.0*x12, Sense::Equal, -4.0);
    m.setObjective(2.0*x01 + 7.0*x02 + 3.0*x12, ObjSense::Minimize);
    return m;
}

// 2-arc: unbalanced flow (supply 5, demand 3 only) → infeasible.
static Model makeNetInfeasible() {
    Model m;
    auto x = m.addVar(0.0, 10.0);
    m.addLPConstraint( 1.0*x, Sense::Equal,  5.0);
    m.addLPConstraint(-1.0*x, Sense::Equal, -3.0);
    m.setObjective(1.0*x, ObjSense::Minimize);
    return m;
}

// 10-arc: 6-node network with multiple parallel paths.
// Nodes: 0 (supply 10) → 5 (demand 10); transit nodes 1,2,3,4.
// Arcs (10 total):
//   x0:(0→1) c=1, x1:(0→2) c=3, x2:(0→3) c=2,
//   x3:(1→4) c=2, x4:(1→5) c=5,
//   x5:(2→4) c=1, x6:(2→5) c=4,
//   x7:(3→4) c=3, x8:(3→5) c=2,
//   x9:(4→5) c=1
// Shortest paths to node 5 (cost per unit):
//   0→1→4→5: 1+2+1=4,  0→3→5: 2+2=4  (both optimal, cost 4/unit)
// Optimal: split 10 units on cost-4 paths → obj=40.
static Model makeNet10Feasible() {
    Model m;
    auto x0 = m.addVar(0.0, 10.0);
    auto x1 = m.addVar(0.0, 10.0);
    auto x2 = m.addVar(0.0, 10.0);
    auto x3 = m.addVar(0.0, 10.0);
    auto x4 = m.addVar(0.0, 10.0);
    auto x5 = m.addVar(0.0, 10.0);
    auto x6 = m.addVar(0.0, 10.0);
    auto x7 = m.addVar(0.0, 10.0);
    auto x8 = m.addVar(0.0, 10.0);
    auto x9 = m.addVar(0.0, 10.0);

    // node 0: supply 10
    m.addLPConstraint( 1.0*x0 + 1.0*x1 + 1.0*x2,
                       Sense::Equal, 10.0);
    // node 1: transit
    m.addLPConstraint(-1.0*x0              + 1.0*x3 + 1.0*x4,
                       Sense::Equal,  0.0);
    // node 2: transit
    m.addLPConstraint(         -1.0*x1              + 1.0*x5 + 1.0*x6,
                       Sense::Equal,  0.0);
    // node 3: transit
    m.addLPConstraint(                  -1.0*x2                        + 1.0*x7 + 1.0*x8,
                       Sense::Equal,  0.0);
    // node 4: transit (receives from 1,2,3; sends to 5)
    m.addLPConstraint(-1.0*x3 - 1.0*x5 - 1.0*x7              + 1.0*x9,
                       Sense::Equal,  0.0);
    // node 5: demand 10
    m.addLPConstraint(-1.0*x4 - 1.0*x6 - 1.0*x8 - 1.0*x9,
                       Sense::Equal, -10.0);

    m.setObjective(1.0*x0 + 3.0*x1 + 2.0*x2
                 + 2.0*x3 + 5.0*x4
                 + 1.0*x5 + 4.0*x6
                 + 3.0*x7 + 2.0*x8
                 + 1.0*x9,
                 ObjSense::Minimize);
    return m;
}

// ── Benchmark helper ──────────────────────────────────────────────────────────

static void runNet(benchmark::State& state,
                   LPMethod method,
                   Model(*build)())
{
    for (auto _ : state) {
        LPOptions opts;
        opts.method = method;
        LPResult r  = solveLP(build(), opts);
        benchmark::DoNotOptimize(r.objectiveValue);
    }
}

// ── Net3Feasible: 3 arcs, 3 nodes ────────────────────────────────────────────

BENCHMARK_CAPTURE(runNet, Net3Feasible/NetworkSimplex,   LPMethod::NetworkSimplex,   makeNet3Feasible);
BENCHMARK_CAPTURE(runNet, Net3Feasible/PrimalSimplex,    LPMethod::PrimalSimplex,    makeNet3Feasible);
BENCHMARK_CAPTURE(runNet, Net3Feasible/DualSimplex,      LPMethod::DualSimplex,      makeNet3Feasible);
BENCHMARK_CAPTURE(runNet, Net3Feasible/PrimalSimplexBV,  LPMethod::PrimalSimplexBV,  makeNet3Feasible);
BENCHMARK_CAPTURE(runNet, Net3Feasible/DualSimplexBV,    LPMethod::DualSimplexBV,    makeNet3Feasible);
BENCHMARK_CAPTURE(runNet, Net3Feasible/RevisedSimplex,   LPMethod::RevisedSimplex,   makeNet3Feasible);
BENCHMARK_CAPTURE(runNet, Net3Feasible/RevisedSimplexBV, LPMethod::RevisedSimplexBV, makeNet3Feasible);
BENCHMARK_CAPTURE(runNet, Net3Feasible/MehrotraIPM,      LPMethod::MehrotraIPM,      makeNet3Feasible);

// ── NetInfeasible: unbalanced flow ────────────────────────────────────────────

BENCHMARK_CAPTURE(runNet, NetInfeasible/NetworkSimplex,   LPMethod::NetworkSimplex,   makeNetInfeasible);
BENCHMARK_CAPTURE(runNet, NetInfeasible/PrimalSimplex,    LPMethod::PrimalSimplex,    makeNetInfeasible);
BENCHMARK_CAPTURE(runNet, NetInfeasible/DualSimplex,      LPMethod::DualSimplex,      makeNetInfeasible);
BENCHMARK_CAPTURE(runNet, NetInfeasible/PrimalSimplexBV,  LPMethod::PrimalSimplexBV,  makeNetInfeasible);
BENCHMARK_CAPTURE(runNet, NetInfeasible/DualSimplexBV,    LPMethod::DualSimplexBV,    makeNetInfeasible);
BENCHMARK_CAPTURE(runNet, NetInfeasible/RevisedSimplex,   LPMethod::RevisedSimplex,   makeNetInfeasible);
BENCHMARK_CAPTURE(runNet, NetInfeasible/RevisedSimplexBV, LPMethod::RevisedSimplexBV, makeNetInfeasible);
BENCHMARK_CAPTURE(runNet, NetInfeasible/MehrotraIPM,      LPMethod::MehrotraIPM,      makeNetInfeasible);

// ── Net10Feasible: 10 arcs, 6 nodes ──────────────────────────────────────────

BENCHMARK_CAPTURE(runNet, Net10Feasible/NetworkSimplex,   LPMethod::NetworkSimplex,   makeNet10Feasible);
BENCHMARK_CAPTURE(runNet, Net10Feasible/PrimalSimplex,    LPMethod::PrimalSimplex,    makeNet10Feasible);
BENCHMARK_CAPTURE(runNet, Net10Feasible/DualSimplex,      LPMethod::DualSimplex,      makeNet10Feasible);
BENCHMARK_CAPTURE(runNet, Net10Feasible/PrimalSimplexBV,  LPMethod::PrimalSimplexBV,  makeNet10Feasible);
BENCHMARK_CAPTURE(runNet, Net10Feasible/DualSimplexBV,    LPMethod::DualSimplexBV,    makeNet10Feasible);
BENCHMARK_CAPTURE(runNet, Net10Feasible/RevisedSimplex,   LPMethod::RevisedSimplex,   makeNet10Feasible);
BENCHMARK_CAPTURE(runNet, Net10Feasible/RevisedSimplexBV, LPMethod::RevisedSimplexBV, makeNet10Feasible);
BENCHMARK_CAPTURE(runNet, Net10Feasible/MehrotraIPM,      LPMethod::MehrotraIPM,      makeNet10Feasible);
