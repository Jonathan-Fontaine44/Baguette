#pragma once
// Shared MILP problem definitions (solved as LP relaxations by the test suite).
// Each model uses proper integer/binary variable types; the LP solver ignores
// integrality and solves the continuous relaxation.

#include <algorithm>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "baguette/core/Sense.hpp"
#include "baguette/lp/LPResult.hpp"
#include "baguette/model/Model.hpp"
#include "lp_problems.hpp"

namespace baguette_test {

/// One arc in a sparse distance matrix.
struct TspArc { int from, to; double dist; };

/// Build an n-city TSP model (MTZ formulation) from a sparse distance matrix.
///
/// Variables
///   x[i][j] ∈ {0,1} (Binary)      — arc i→j  (n*(n-1) arc vars)
///   u[i]    ∈ [1,n-1] (Integer)   — MTZ position, i = 1..n-1  (n-1 vars)
///
/// Constraints
///   Degree out = 1 for every city i > 0   (n-1 constraints; city 0 dropped to
///   Degree in  = 1 for every city i        avoid rank deficiency)
///   MTZ: u[i] − u[j] + n·x[i][j] ≤ n−1   for i,j ∈ {1,…,n−1}, i ≠ j
///
/// Absent arcs receive cost bigM = n × max_dist + 1 so they are never chosen
/// in an optimal integer tour while keeping the model always feasible.
///
/// @param n     Number of cities (0..n-1).
/// @param arcs  Sparse distance entries; self-loops are ignored.
///
/// @note Complexity
///   O(n²) variables, O(n²) constraints (MTZ), O(|arcs|) cost setup.
inline baguette::Model makeTSP(int n, const std::vector<TspArc>& arcs) {
    using namespace baguette;

    double maxDist = 0.0;
    for (const auto& a : arcs)
        if (a.from != a.to) maxDist = std::max(maxDist, a.dist);
    const double bigM = double(n) * maxDist;

    // Dense cost matrix: absent arcs default to bigM.
    std::vector<std::vector<double>> c(n, std::vector<double>(n, bigM));
    for (const auto& a : arcs)
        if (a.from != a.to) c[a.from][a.to] = a.dist;

    Model m;

    // Arc variables x[i][j], i ≠ j.
    std::vector<std::vector<Variable>> x(n, std::vector<Variable>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (i != j) x[i][j] = m.addVar(0.0, 1.0, VarType::Binary);

    // MTZ position variables u[1..n-1] ∈ [1, n-1].
    std::vector<Variable> u(n);
    for (int i = 1; i < n; ++i)
        u[i] = m.addVar(1.0, double(n - 1), VarType::Integer);

    // Objective: min Σ c[i][j] · x[i][j].
    LinearExpr obj;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (i != j) obj += c[i][j] * x[i][j];
    m.setObjective(obj, ObjSense::Minimize);

    // Degree constraints. City 0 degree-out dropped (rank deficiency).
    for (int i = 0; i < n; ++i) {
        LinearExpr out, in;
        for (int j = 0; j < n; ++j)
            if (j != i) { out += 1.0 * x[i][j]; in += 1.0 * x[j][i]; }
        if (i > 0) m.addLPConstraint(out, Sense::Equal, 1.0);
        m.addLPConstraint(in, Sense::Equal, 1.0);
    }

    // MTZ subtour elimination: u[i] − u[j] + n·x[i][j] ≤ n−1.
    for (int i = 1; i < n; ++i)
        for (int j = 1; j < n; ++j)
            if (i != j) {
                LinearExpr mtz;
                mtz += 1.0 * u[i];
                mtz += -1.0 * u[j];
                mtz += double(n) * x[i][j];
                m.addLPConstraint(mtz, Sense::LessEq, double(n - 1));
            }

    return m;
}

/// Sparse distance matrix for the 10-city cyclic TSP.
///
/// Only the 20 cycle-adjacent arcs (i → i±1 mod 10) are listed, each with
/// cost 1.  Absent arcs receive bigM = 10 × 1 + 1 = 11 inside makeTSP().
/// LP optimal = TSP optimal = 10  (the cyclic tour 0→1→…→9→0 is an integer
/// extreme point of the MTZ LP polytope, so LP and IP coincide here).
inline std::vector<TspArc> makeTSPArcs(int n) {
    std::vector<TspArc> arcs;
    arcs.reserve(2 * n);
    for (int i = 0; i < n; ++i) {
        arcs.push_back({i, (i + 1) % n, 1.0});
        arcs.push_back({i, (i + n - 1) % n, 1.0});
    }
    return arcs;
}

inline baguette::Model makeTSP10() {
    return makeTSP(10, makeTSPArcs(10));
}

/// One item in a 0/1 knapsack instance.
struct KnapsackItem { double weight, profit; };

/// Build a 0/1 knapsack model from a list of items.
///
/// Variables:  x[i] ∈ {0,1} (Binary)
/// Constraint: Σ w[i]·x[i] ≤ capacity   (and ≥ minLoad when minLoad > 0)
/// Objective:  max Σ p[i]·x[i]
///
/// @par Complexity
///   O(|items|) variables and constraints.
inline baguette::Model makeKnapsack(const std::vector<KnapsackItem>& items,
                                    double capacity, double minLoad = 0.0) {
    using namespace baguette;
    Model m;

    std::vector<Variable> x;
    x.reserve(items.size());
    for (std::size_t i = 0; i < items.size(); ++i)
        x.push_back(m.addVar(0.0, 1.0, VarType::Binary));

    LinearExpr cap;
    for (std::size_t i = 0; i < items.size(); ++i) cap += items[i].weight * x[i];
    m.addLPConstraint(cap, Sense::LessEq, capacity);
    if (minLoad > 0.0)
        m.addLPConstraint(cap, Sense::GreaterEq, minLoad);

    LinearExpr obj;
    for (std::size_t i = 0; i < items.size(); ++i) obj += items[i].profit * x[i];
    m.setObjective(obj, ObjSense::Maximize);

    return m;
}

/// Items for the 10-item knapsack instance.
///
/// Sorted by decreasing p/w ratio [5.0, 4.5, 4.0, …, 1.0, 1.0].
/// capacity=50 → LP optimal = 110 (item 9 taken at fraction ½).
/// capacity=5, minLoad=6 → infeasible (5 < 6, no x ∈ [0,1]^10 satisfies both).
inline std::vector<KnapsackItem> makeKnapsack10Items() {
    return {
        {1, 5}, {2, 9}, {3, 12}, {4, 14}, {5, 15},
        {6,15}, {7,14}, {8, 12}, {9,  9}, {10,10},
    };
}

inline baguette::Model makeKnapsack10(double capacity = 50.0, double minLoad = 0.0) {
    return makeKnapsack(makeKnapsack10Items(), capacity, minLoad);
}

/// One job in a 2-machine flow shop: processing times on machine 0 then machine 1.
struct JobShopJob { double p0, p1; };

/// Build a 2-machine flow shop model (M0 → M1 for all jobs).
///
/// Variables:
///   S[j][m] ∈ [0, M]     — start time of job j on machine m (continuous)
///   y[j][k][m] ∈ {0,1}   — 1 if job j precedes job k on machine m (binary), j < k
///   C_max ∈ [0, cmaxUb]  — makespan (continuous)
///
/// Constraints:
///   Precedence:  S[j][1] − S[j][0]                       ≥ p[j][0]       ∀ j
///   Disj. (A):   S[k][m] − S[j][m] − M·y[j][k][m]       ≥ p[j][m] − M   ∀ j<k, m
///   Disj. (B):   S[j][m] − S[k][m] + M·y[j][k][m]       ≥ p[k][m]       ∀ j<k, m
///   Makespan:    C_max   − S[j][1]                        ≥ p[j][1]       ∀ j
///
/// Big-M is computed as Σ(p[j][0] + p[j][1]) — the tightest valid upper bound.
///
/// @par Complexity
///   O(n) start-time vars, O(n²) sequencing vars and disjunctive constraints.
inline baguette::Model makeJobShop(const std::vector<JobShopJob>& jobs, double cmaxUb) {
    using namespace baguette;
    const int n = static_cast<int>(jobs.size());

    double bigM = 0.0;
    for (const auto& j : jobs) bigM += j.p0 + j.p1;

    Model m;

    // Start-time variables S[j][m] ∈ [0, bigM].
    std::vector<std::array<Variable, 2>> S(n);
    for (int j = 0; j < n; ++j)
        for (int mm = 0; mm < 2; ++mm)
            S[j][mm] = m.addVar(0.0, bigM);

    // Sequencing variables y[j][k][m] ∈ {0,1}, j < k.
    std::vector<std::vector<std::array<Variable, 2>>> y(
        n, std::vector<std::array<Variable, 2>>(n));
    for (int j = 0; j < n; ++j)
        for (int k = j + 1; k < n; ++k)
            for (int mm = 0; mm < 2; ++mm)
                y[j][k][mm] = m.addVar(0.0, 1.0, VarType::Binary);

    Variable Cmax = m.addVar(0.0, cmaxUb);

    // Precedence: S[j][1] - S[j][0] >= p[j][0].
    for (int j = 0; j < n; ++j) {
        LinearExpr e;
        e += 1.0 * S[j][1];
        e += -1.0 * S[j][0];
        m.addLPConstraint(e, Sense::GreaterEq, jobs[j].p0);
    }

    // Disjunctive constraints.
    for (int j = 0; j < n; ++j)
        for (int k = j + 1; k < n; ++k)
            for (int mm = 0; mm < 2; ++mm) {
                const double pj = mm == 0 ? jobs[j].p0 : jobs[j].p1;
                const double pk = mm == 0 ? jobs[k].p0 : jobs[k].p1;
                // (A) j before k
                LinearExpr a;
                a += 1.0 * S[k][mm];
                a += -1.0 * S[j][mm];
                a += -bigM * y[j][k][mm];
                m.addLPConstraint(a, Sense::GreaterEq, pj - bigM);
                // (B) k before j
                LinearExpr b;
                b += 1.0 * S[j][mm];
                b += -1.0 * S[k][mm];
                b += bigM * y[j][k][mm];
                m.addLPConstraint(b, Sense::GreaterEq, pk);
            }

    // Makespan: C_max - S[j][1] >= p[j][1].
    for (int j = 0; j < n; ++j) {
        LinearExpr e;
        e += 1.0 * Cmax;
        e += -1.0 * S[j][1];
        m.addLPConstraint(e, Sense::GreaterEq, jobs[j].p1);
    }

    m.setObjective(1.0 * Cmax, ObjSense::Minimize);
    return m;
}

/// Jobs for the 10-job flow shop instance (p[j][0]+p[j][1]=5 for every job).
///
/// cmaxUb=40 → LP optimal = 5 (y=½ makes big-M constraints trivially slack).
/// cmaxUb=4  → infeasible (C_max ≥ 5 from precedence contradicts C_max ≤ 4).
inline std::vector<JobShopJob> makeJobShop10Jobs() {
    return {
        {3,2},{2,3},{4,1},{1,4},{3,2},{2,3},{4,1},{1,4},{3,2},{2,3}
    };
}

inline baguette::Model makeJobShop10(double cmaxUb = 40.0) {
    return makeJobShop(makeJobShop10Jobs(), cmaxUb);
}

} // namespace baguette_test

/// LP relaxations of classic MILP problems.
inline std::vector<LPTestCase> makeRelaxedMILPTestSuite() {
    using namespace baguette;

    return {
        // ── TSP 10 cities ────────────────────────────────────────────────────
        // See makeTSP10() for the full formulation description.
        // LP optimal = 10 (cyclic tour 0→1→…→9→0, all unit-cost adjacent arcs).
        {"tsp_10_feasible", LPStatus::Optimal, 10.0,
            []() { return baguette_test::makeTSP10(); }},

        // ── 0/1 Knapsack 10 items ────────────────────────────────────────────
        // LP optimal = 110: fractional fill of item 9 (ratio 1.0) at ½.
        {"knapsack_10", LPStatus::Optimal, 110.0,
            []() { return baguette_test::makeKnapsack10(); }},

        // Infeasible: capacity=5 < minLoad=6, no x ∈ [0,1]^10 satisfies both.
        {"knapsack_10_infeasible", LPStatus::Infeasible, 0.0,
            []() { return baguette_test::makeKnapsack10(5.0, 6.0); }},

        // ── Job shop 10 jobs × 2 machines ───────────────────────────────────
        // LP optimal = 5: big-M disjunctive constraints trivially slack at y=½.
        {"jobshop_10x2", LPStatus::Optimal, 5.0,
            []() { return baguette_test::makeJobShop10(); }},

        // Infeasible: C_max ≤ 4 (ub) contradicts C_max ≥ p[j][0]+p[j][1]=5.
        {"jobshop_10x2_infeasible", LPStatus::Infeasible, 0.0,
            []() { return baguette_test::makeJobShop10(4.0); }},
    };
}
