#pragma once
// Fixed-charge network design on a random DAG.
//
// Nodes 0..nNodes-1 in topological order (i < j ⟹ arc goes forward only).
// Supports multiple sources and sinks, each with their own supply/demand.
//
// Variables:
//   x[a] ∈ [0, cap[a]]  — flow on arc a (Continuous)
//   y[a] ∈ {0,1}        — 1 if arc a is opened (Binary)
//
// Constraints:
//   x[a] ≤ cap[a] · y[a]                     ∀ a   (capacity if open)
//   Σ_{a: from=v} x[a] − Σ_{a: to=v} x[a] = b[v]  ∀ v   (flow conservation)
//   b[source_k] = supply_k > 0
//   b[sink_k]   = −demand_k < 0
//   b[others]   = 0
//   Σ supply = Σ demand  (feasibility)
//
// Objective:
//   min  Σ_a  fixedCost[a] · y[a]  +  routingCost[a] · x[a]

#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "baguette/core/Sense.hpp"
#include "baguette/model/Model.hpp"

namespace baguette_test {

struct FlowArc {
    int    from, to;
    double cap;
    double fixedCost;
    double routingCost;
};

/// Build the arc list for a random DAG (backbone + random long-range arcs).
/// Backbone 0→1→…→(n-1) guarantees source-to-sink connectivity.
/// Each additional pair (i,j), i+2≤j, is added with probability extraProb.
inline std::vector<FlowArc> makeDAGArcs(int    nNodes    = 100,
                                        double extraProb = 0.12,
                                        unsigned seed    = 42)
{
    std::mt19937 rng(seed);
    auto randi = [&](int lo, int hi) {
        return std::uniform_int_distribution<int>(lo, hi)(rng);
    };
    auto randd = [&](double lo, double hi) {
        return std::uniform_real_distribution<double>(lo, hi)(rng);
    };

    std::vector<FlowArc> arcs;
    arcs.reserve(nNodes + static_cast<int>(nNodes * nNodes * extraProb / 2));

    // Backbone: guarantees every node is reachable from 0 and can reach n-1
    for (int i = 0; i < nNodes - 1; ++i)
        arcs.push_back({i, i + 1,
                        randd(5.0, 20.0),
                        randd(2.0, 15.0),
                        randd(0.1, 2.0)});

    // Additional long-range arcs (i→j, j ≥ i+2)
    const int thresh = static_cast<int>(extraProb * 100.0);
    for (int i = 0; i < nNodes - 2; ++i)
        for (int j = i + 2; j < nNodes; ++j)
            if (randi(0, 99) < thresh)
                arcs.push_back({i, j,
                                randd(5.0, 20.0),
                                randd(2.0, 15.0),
                                randd(0.1, 2.0)});

    return arcs;
}

/// Build the MILP model from a pre-computed arc list.
///
/// @param arcs    Arc list (must respect DAG order: from < to).
/// @param nNodes  Number of nodes.
/// @param supply  supply[v] > 0: source,  supply[v] < 0: sink,  0: transshipment.
///                Must satisfy Σ supply[v] = 0.
inline baguette::Model makeFlowDAGModel(const std::vector<FlowArc>& arcs,
                                        int nNodes,
                                        const std::vector<double>& supply)
{
    using namespace baguette;

    const int nArcs = static_cast<int>(arcs.size());

    Model m;
    std::vector<Variable> x(nArcs), y(nArcs);
    for (int a = 0; a < nArcs; ++a) {
        x[a] = m.addVar(0.0, arcs[a].cap);
        y[a] = m.addVar(0.0, 1.0, VarType::Binary);
    }

    // x[a] ≤ cap[a] · y[a]
    for (int a = 0; a < nArcs; ++a) {
        LinearExpr e;
        e += 1.0 * x[a];
        e += -arcs[a].cap * y[a];
        m.addLPConstraint(e, Sense::LessEq, 0.0);
    }

    // Flow conservation: outflow − inflow = supply[v]
    for (int v = 0; v < nNodes; ++v) {
        LinearExpr flow;
        for (int a = 0; a < nArcs; ++a) {
            if (arcs[a].from == v) flow += 1.0 * x[a];
            if (arcs[a].to   == v) flow -= 1.0 * x[a];
        }
        m.addLPConstraint(flow, Sense::Equal, supply[v]);
    }

    // Objective: min Σ fixedCost·y + routingCost·x
    LinearExpr obj;
    for (int a = 0; a < nArcs; ++a) {
        obj += arcs[a].fixedCost   * y[a];
        obj += arcs[a].routingCost * x[a];
    }
    m.setObjective(obj, ObjSense::Minimize);

    return m;
}

/// Single source / single sink convenience builder.
///
/// @param nNodes    Number of nodes (source=0, sink=nNodes-1).
/// @param extraProb Probability of adding each non-backbone arc.
/// @param seed      RNG seed for reproducibility.
/// @param demand    Flow required from source to sink.
inline baguette::Model makeFlowDAG(int      nNodes    = 100,
                                   double   extraProb = 0.12,
                                   unsigned seed      = 42,
                                   double   demand    = 10.0)
{
    const auto arcs = makeDAGArcs(nNodes, extraProb, seed);
    std::vector<double> supply(nNodes, 0.0);
    supply[0]          =  demand;
    supply[nNodes - 1] = -demand;
    return makeFlowDAGModel(arcs, nNodes, supply);
}

/// Two sources (nodes 0, 1) and two sinks (nodes n-2, n-1).
///
/// Each source supplies demand/2; each sink absorbs demand/2.
/// The backbone 0→1→…→n-1 ensures all four nodes are connected.
///
/// @param nNodes    Number of nodes (≥ 4).
/// @param extraProb Probability of adding each non-backbone arc.
/// @param seed      RNG seed for reproducibility.
/// @param demand    Total flow routed (split equally across sources/sinks).
inline baguette::Model makeFlowDAG2S2T(int      nNodes    = 100,
                                       double   extraProb = 0.12,
                                       unsigned seed      = 42,
                                       double   demand    = 10.0)
{
    const auto arcs = makeDAGArcs(nNodes, extraProb, seed);
    std::vector<double> supply(nNodes, 0.0);
    supply[0]          =  demand / 2.0;   // source 1
    supply[1]          =  demand / 2.0;   // source 2
    supply[nNodes - 2] = -demand / 2.0;   // sink 1
    supply[nNodes - 1] = -demand / 2.0;   // sink 2
    return makeFlowDAGModel(arcs, nNodes, supply);
}

} // namespace baguette_test
