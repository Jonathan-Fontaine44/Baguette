#pragma once
// Shared MILP problem definitions (solved as LP relaxations by the test suite).
// Each model uses proper integer/binary variable types; the LP solver ignores
// integrality and solves the continuous relaxation.

#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "baguette/core/Sense.hpp"
#include "baguette/lp/LPResult.hpp"
#include "baguette/model/Model.hpp"
#include "lp_problems.hpp"

namespace baguette_test {

/// LP relaxation of the 10-city TSP (MTZ formulation).
///
/// Variables
///   x[i][j] ∈ {0,1} (Binary)   — arc i→j  (90 arc vars)
///   u[i]    ∈ [1,9] (Integer)  — position in tour, i = 1..9 (9 vars)
///
/// Constraints
///   Degree in/out = 1 for every city           (19 constraints — depot outflow dropped)
///   MTZ: u[i] − u[j] + n·x[i][j] ≤ n−1        (72 constraints)
///
/// Costs: 1 for the two cycle-adjacent arcs of each city, n=10 otherwise.
/// LP optimal = TSP optimal = 10  (the cyclic tour 0→1→…→9→0 is an integer
/// extreme point of the MTZ LP polytope, so LP and IP coincide here).
inline baguette::Model makeTSP10() {
    using namespace baguette;
    const int n = 10;
    Model m;

    // Arc variables x[i][j], i ≠ j: binary, relaxed to [0,1] by the LP solver.
    Variable xv[10][10] = {};
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (i != j) xv[i][j] = m.addVar(0.0, 1.0, VarType::Binary);

    // Position variables u[1]..u[9] in [1, n-1]: integer.
    // uv[k] corresponds to city k+1.
    Variable uv[9] = {};
    for (int k = 0; k < n - 1; ++k)
        uv[k] = m.addVar(1.0, double(n - 1), VarType::Integer);

    // Objective: min Σ c[i][j] · x[i][j].
    // Cost = 1 for cycle-adjacent arcs (j == i±1 mod n), n otherwise.
    LinearExpr obj;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (i != j) {
                const double c =
                    (j == (i + 1) % n || j == (i + n - 1) % n) ? 1.0 : double(n);
                obj += c * xv[i][j];
            }
    m.setObjective(obj, ObjSense::Minimize);

    // Degree constraints: exactly one outgoing and one incoming arc per city.
    // The degree-out constraint for city 0 is dropped: it is implied by the
    // other 9 degree-out constraints + all 10 degree-in constraints
    // (Σ outflows = Σ inflows = n), so including it makes the equality
    // system rank-deficient.
    for (int i = 0; i < n; ++i) {
        LinearExpr out, in;
        for (int j = 0; j < n; ++j)
            if (j != i) { out += 1.0 * xv[i][j]; in += 1.0 * xv[j][i]; }
        if (i > 0) m.addLPConstraint(out, Sense::Equal, 1.0);
        m.addLPConstraint(in, Sense::Equal, 1.0);
    }

    // MTZ subtour elimination: u[i] − u[j] + n·x[i][j] ≤ n−1
    // for i, j ∈ {1,…,n−1}, i ≠ j.  City 0 is the depot (no u variable).
    for (int i = 1; i < n; ++i)
        for (int j = 1; j < n; ++j)
            if (i != j) {
                LinearExpr mtz;
                mtz += 1.0 * uv[i - 1];
                mtz += -1.0 * uv[j - 1];
                mtz += double(n) * xv[i][j];
                m.addLPConstraint(mtz, Sense::LessEq, double(n - 1));
            }

    return m;
}

/// LP relaxation of a 10-item 0/1 knapsack.
///
/// Variables: x[i] ∈ {0,1} (Binary), relaxed to [0,1] by the LP solver.
/// Constraint: Σ w[i]·x[i] ≤ capacity.
/// Objective: max Σ p[i]·x[i].
///
/// Items sorted by decreasing p/w ratio [5.0, 4.5, 4.0, …, 1.0, 1.0].
/// capacity=50 → LP optimal = 110 (item 9 taken at fraction ½).
/// capacity=5, minLoad=6 → infeasible (5 < 6, no x ∈ [0,1]^10 satisfies both).
inline baguette::Model makeKnapsack10(double capacity = 50.0, double minLoad = 0.0) {
    using namespace baguette;
    constexpr double weights[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    constexpr double profits[10] = {5, 9, 12, 14, 15, 15, 14, 12, 9, 10};

    Model m;
    Variable x[10];
    for (int i = 0; i < 10; ++i)
        x[i] = m.addVar(0.0, 1.0, VarType::Binary);

    LinearExpr cap;
    for (int i = 0; i < 10; ++i) cap += weights[i] * x[i];
    m.addLPConstraint(cap, Sense::LessEq, capacity);
    if (minLoad > 0.0)
        m.addLPConstraint(cap, Sense::GreaterEq, minLoad);

    LinearExpr obj;
    for (int i = 0; i < 10; ++i) obj += profits[i] * x[i];
    m.setObjective(obj, ObjSense::Maximize);

    return m;
}

/// LP relaxation of a 10-job 2-machine job shop (flow shop: M0 → M1 for all jobs).
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
/// Processing times chosen so p[j][0]+p[j][1]=5 for every job.
/// cmaxUb=40 → LP optimal = 5 (y=½ makes big-M constraints trivially slack).
/// cmaxUb=4  → infeasible (C_max ≥ 5 from precedence contradicts C_max ≤ 4).
inline baguette::Model makeJobShop10(double cmaxUb = 40.0) {
    using namespace baguette;
    constexpr int     nJobs      = 10;
    constexpr double  p[nJobs][2] = {
        {3,2},{2,3},{4,1},{1,4},{3,2},{2,3},{4,1},{1,4},{3,2},{2,3}
    };
    // M = Σ p[j][0] + Σ p[j][1] = 40: conservative big-M.
    constexpr double M = 40.0;

    Model m;

    // Start-time variables S[j][m] ∈ [0, M].
    Variable S[nJobs][2];
    for (int j = 0; j < nJobs; ++j)
        for (int mm = 0; mm < 2; ++mm)
            S[j][mm] = m.addVar(0.0, M);

    // Sequencing variables y[j][k][m] ∈ {0,1}, for j < k only.
    Variable y[nJobs][nJobs][2] = {};
    for (int j = 0; j < nJobs; ++j)
        for (int k = j + 1; k < nJobs; ++k)
            for (int mm = 0; mm < 2; ++mm)
                y[j][k][mm] = m.addVar(0.0, 1.0, VarType::Binary);

    // Makespan variable — upper bound is the parameter (40 = feasible, 4 = infeasible).
    Variable Cmax = m.addVar(0.0, cmaxUb);

    // Precedence: S[j][1] - S[j][0] >= p[j][0].
    for (int j = 0; j < nJobs; ++j) {
        LinearExpr e;
        e += 1.0 * S[j][1];
        e += -1.0 * S[j][0];
        m.addLPConstraint(e, Sense::GreaterEq, p[j][0]);
    }

    // Disjunctive: no two jobs overlap on the same machine.
    for (int j = 0; j < nJobs; ++j)
        for (int k = j + 1; k < nJobs; ++k)
            for (int mm = 0; mm < 2; ++mm) {
                // (A) j before k: S[k][m] - S[j][m] - M*y >= p[j][m] - M
                LinearExpr a;
                a += 1.0 * S[k][mm];
                a += -1.0 * S[j][mm];
                a += -M * y[j][k][mm];
                m.addLPConstraint(a, Sense::GreaterEq, p[j][mm] - M);

                // (B) k before j: S[j][m] - S[k][m] + M*y >= p[k][m]
                LinearExpr b;
                b += 1.0 * S[j][mm];
                b += -1.0 * S[k][mm];
                b += M * y[j][k][mm];
                m.addLPConstraint(b, Sense::GreaterEq, p[k][mm]);
            }

    // Makespan: C_max - S[j][1] >= p[j][1].
    for (int j = 0; j < nJobs; ++j) {
        LinearExpr e;
        e += 1.0 * Cmax;
        e += -1.0 * S[j][1];
        m.addLPConstraint(e, Sense::GreaterEq, p[j][1]);
    }

    m.setObjective(1.0 * Cmax, ObjSense::Minimize);
    return m;
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
