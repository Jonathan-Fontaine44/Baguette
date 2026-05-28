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
#include "baguette/cp/constraints/AllDiff.hpp"
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

/// Build an n-city TSP model (Single-Commodity Flow formulation).
///
/// Variables
///   x[i][j] ∈ {0,1} (Binary)       — arc i→j  (n*(n-1) arc vars)
///   g[i][j] ∈ [0,1]  (Continuous)  — normalised flow g = f/(n-1)  (n*(n-1) vars)
///
/// Constraints
///   Degree out = 1 for every city i > 0   (city 0 dropped; implied by in-degree)
///   Degree in  = 1 for every city i
///   Flow capacity:     g[i][j] ≤ x[i][j]                       for i ≠ j
///   Flow conservation: Σⱼ g[i][j] − Σⱼ g[j][i] = −1/(n−1)    for i = 1..n-1
///     (city 0 is the source; its balance is implied by the remaining n-1 equations)
///
/// Normalising f by (n-1) keeps all LP matrix coefficients in {−1, 0, 1}, which
/// avoids large-coefficient degeneracy that affects some simplex pivoting strategies.
///
/// The LP relaxation of SCF has the same bound as DFJ (exponential SEC), which
/// is strictly tighter than MTZ for most instances.  Both give LP = IP = 10
/// on the 10-city cyclic instance.
///
/// @param n     Number of cities (0..n-1).
/// @param arcs  Sparse distance entries; self-loops are ignored.
///
/// @note Complexity
///   O(n²) binary variables, O(n²) flow variables, O(n²) constraints.
inline baguette::Model makeTSPFlow(int n, const std::vector<TspArc>& arcs) {
    using namespace baguette;

    double maxDist = 0.0;
    for (const auto& a : arcs)
        if (a.from != a.to) maxDist = std::max(maxDist, a.dist);
    const double bigM = double(n) * maxDist;

    std::vector<std::vector<double>> c(n, std::vector<double>(n, bigM));
    for (const auto& a : arcs)
        if (a.from != a.to) c[a.from][a.to] = a.dist;

    Model m;

    // Arc variables x[i][j], i ≠ j.
    std::vector<std::vector<Variable>> x(n, std::vector<Variable>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (i != j) x[i][j] = m.addVar(0.0, 1.0, VarType::Binary);

    // Normalised flow variables g[i][j] = f[i][j]/(n-1) ∈ [0,1], i ≠ j.
    std::vector<std::vector<Variable>> g(n, std::vector<Variable>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (i != j) g[i][j] = m.addVar(0.0, 1.0);

    // Objective: min Σ c[i][j] · x[i][j].
    LinearExpr obj;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (i != j) obj += c[i][j] * x[i][j];
    m.setObjective(obj, ObjSense::Minimize);

    // Degree constraints (city-0 degree-out dropped, same convention as MTZ).
    for (int i = 0; i < n; ++i) {
        LinearExpr out, in;
        for (int j = 0; j < n; ++j)
            if (j != i) { out += 1.0 * x[i][j]; in += 1.0 * x[j][i]; }
        if (i > 0) m.addLPConstraint(out, Sense::Equal, 1.0);
        m.addLPConstraint(in, Sense::Equal, 1.0);
    }

    // Flow capacity: g[i][j] ≤ x[i][j]  (all coefficients ±1).
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (i != j) {
                LinearExpr e;
                e += 1.0 * g[i][j];
                e += -1.0 * x[i][j];
                m.addLPConstraint(e, Sense::LessEq, 0.0);
            }

    // Flow conservation for cities 1..n-1: Σⱼ g[i][j] − Σⱼ g[j][i] = −1/(n−1).
    // City 0 (source) is implied by the remaining n-1 equations.
    const double rhs = -1.0 / double(n - 1);
    for (int i = 1; i < n; ++i) {
        LinearExpr e;
        for (int j = 0; j < n; ++j)
            if (j != i) { e += 1.0 * g[i][j]; e += -1.0 * g[j][i]; }
        m.addLPConstraint(e, Sense::Equal, rhs);
    }

    return m;
}

inline baguette::Model makeTSP10Flow() {
    return makeTSPFlow(10, makeTSPArcs(10));
}

/// Build an n-city TSP model (MTZ + AllDiff CP formulation).
///
/// Identical to makeTSP() with one addition: a CP AllDiff constraint is posted
/// on the n-1 position variables u[1..n-1].
///
/// In any valid MTZ solution the positions are already distinct by construction,
/// so AllDiff is redundant for integer feasibility.  It is useful during B&B:
/// the AllDiff propagator (bounds consistency + Hall intervals) can tighten
/// position domains early and prune subtour-prone branches before the LP solve.
///
/// The LP relaxation is unchanged (CP constraints are not linearised).
/// LP optimal = 10 for the 10-city cyclic instance.
///
/// @param n     Number of cities (0..n-1).
/// @param arcs  Sparse distance entries; self-loops are ignored.
inline baguette::Model makeTSPMtz(int n, const std::vector<TspArc>& arcs) {
    using namespace baguette;

    double maxDist = 0.0;
    for (const auto& a : arcs)
        if (a.from != a.to) maxDist = std::max(maxDist, a.dist);
    const double bigM = double(n) * maxDist;

    std::vector<std::vector<double>> c(n, std::vector<double>(n, bigM));
    for (const auto& a : arcs)
        if (a.from != a.to) c[a.from][a.to] = a.dist;

    Model m;

    std::vector<std::vector<Variable>> x(n, std::vector<Variable>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (i != j) x[i][j] = m.addVar(0.0, 1.0, VarType::Binary);

    std::vector<Variable> u(n);
    for (int i = 1; i < n; ++i)
        u[i] = m.addVar(1.0, double(n - 1), VarType::Integer);

    LinearExpr obj;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (i != j) obj += c[i][j] * x[i][j];
    m.setObjective(obj, ObjSense::Minimize);

    for (int i = 0; i < n; ++i) {
        LinearExpr out, in;
        for (int j = 0; j < n; ++j)
            if (j != i) { out += 1.0 * x[i][j]; in += 1.0 * x[j][i]; }
        if (i > 0) m.addLPConstraint(out, Sense::Equal, 1.0);
        m.addLPConstraint(in, Sense::Equal, 1.0);
    }

    for (int i = 1; i < n; ++i)
        for (int j = 1; j < n; ++j)
            if (i != j) {
                LinearExpr mtz;
                mtz += 1.0 * u[i];
                mtz += -1.0 * u[j];
                mtz += double(n) * x[i][j];
                m.addLPConstraint(mtz, Sense::LessEq, double(n - 1));
            }

    // AllDiff on position variables u[1..n-1]: all positions are distinct.
    m.addCPConstraint(
        AllDiffConstraint(std::vector<Variable>(u.begin() + 1, u.end())));

    return m;
}

inline baguette::Model makeTSP10Mtz() {
    return makeTSPMtz(10, makeTSPArcs(10));
}

// ── Lifted MTZ (Desrochers-Laporte, 1991) ─────────────────────────────────────

/// Build an n-city TSP model (Lifted MTZ / Desrochers-Laporte formulation).
///
/// Same variables as makeTSP() (arc binaries + integer position vars u[1..n-1]).
/// Subtour elimination is replaced by the DL strengthened constraint:
///
///   u[i] − u[j] + (n−1)·x[i][j] + (n−3)·x[j][i] ≤ n−2   ∀ i,j ∈ {1..n-1}, i≠j
///
/// This is derived by exploiting that x[i][j]=1 forces u[j]=u[i]+1 (immediate
/// successor), so the coefficient of x[j][i] can be tightened from 0 to n-3
/// compared to the pair of standard MTZ constraints.
///
/// LP bound: MTZ < LMTZ < SCF = DFJ.
/// LP optimal = 10 on the 10-city cyclic instance (same as MTZ and SCF).
///
/// @param n     Number of cities (0..n-1).
/// @param arcs  Sparse distance entries; self-loops are ignored.
///
/// @note Complexity O(n²) variables and constraints — identical to MTZ.
inline baguette::Model makeTSPLifted(int n, const std::vector<TspArc>& arcs) {
    using namespace baguette;

    double maxDist = 0.0;
    for (const auto& a : arcs)
        if (a.from != a.to) maxDist = std::max(maxDist, a.dist);
    const double bigM = double(n) * maxDist;

    std::vector<std::vector<double>> c(n, std::vector<double>(n, bigM));
    for (const auto& a : arcs)
        if (a.from != a.to) c[a.from][a.to] = a.dist;

    Model m;

    std::vector<std::vector<Variable>> x(n, std::vector<Variable>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (i != j) x[i][j] = m.addVar(0.0, 1.0, VarType::Binary);

    std::vector<Variable> u(n);
    for (int i = 1; i < n; ++i)
        u[i] = m.addVar(1.0, double(n - 1), VarType::Integer);

    LinearExpr obj;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (i != j) obj += c[i][j] * x[i][j];
    m.setObjective(obj, ObjSense::Minimize);

    for (int i = 0; i < n; ++i) {
        LinearExpr out, in;
        for (int j = 0; j < n; ++j)
            if (j != i) { out += 1.0 * x[i][j]; in += 1.0 * x[j][i]; }
        if (i > 0) m.addLPConstraint(out, Sense::Equal, 1.0);
        m.addLPConstraint(in, Sense::Equal, 1.0);
    }

    // DL subtour elimination: u[i]−u[j]+(n−1)·x[i][j]+(n−3)·x[j][i] ≤ n−2.
    // For n=3 the (n-3)=0 term vanishes and DL coincides with MTZ.
    for (int i = 1; i < n; ++i)
        for (int j = 1; j < n; ++j)
            if (i != j) {
                LinearExpr e;
                e += 1.0 * u[i];
                e += -1.0 * u[j];
                e += double(n - 1) * x[i][j];
                if (n >= 4) e += double(n - 3) * x[j][i];
                m.addLPConstraint(e, Sense::LessEq, double(n - 2));
            }

    return m;
}

inline baguette::Model makeTSP10Lifted() {
    return makeTSPLifted(10, makeTSPArcs(10));
}

// ── Multi-Commodity Flow (MCF) ─────────────────────────────────────────────────

/// Build an n-city TSP model (Multi-Commodity Flow formulation).
///
/// For each destination city k ∈ {1..n-1}, city 0 routes one unit of
/// commodity k through the tour network to city k.
///
/// Variables
///   x[i][j]  ∈ {0,1} (Binary)     — arc i→j   (n*(n-1) arc vars)
///   h[k][i][j] ∈ [0,1] (Continuous) — flow of commodity k on arc i→j
///                                      (n*(n-1)*(n-1) flow vars)
///
/// Constraints
///   Degree out = 1, degree in = 1  (same as MTZ, city-0 out dropped)
///   Capacity:     h[k][i][j] ≤ x[i][j]                       ∀ k, (i,j)
///   Conservation: Σⱼ h[k][i][j] − Σⱼ h[k][j][i] = b[k][i]  ∀ k, i≠0
///     where b[k][i] = −1 if i=k (sink), 0 otherwise (transit)
///     (source city 0 balance is implied by the remaining n-1 equations)
///
/// LP bound = DFJ bound (strictly tighter than SCF for general instances).
/// LP optimal = 10 on the 10-city cyclic instance.
///
/// @param n     Number of cities (0..n-1).
/// @param arcs  Sparse distance entries; self-loops are ignored.
///
/// @note Complexity O(n³) flow variables and O(n³) capacity constraints.
///   Practical for n ≤ ~15; the cyclic 10-city instance has 810 flow vars.
inline baguette::Model makeTSPMCF(int n, const std::vector<TspArc>& arcs) {
    using namespace baguette;

    double maxDist = 0.0;
    for (const auto& a : arcs)
        if (a.from != a.to) maxDist = std::max(maxDist, a.dist);
    const double bigM = double(n) * maxDist;

    std::vector<std::vector<double>> c(n, std::vector<double>(n, bigM));
    for (const auto& a : arcs)
        if (a.from != a.to) c[a.from][a.to] = a.dist;

    Model m;

    // Arc variables x[i][j], i ≠ j.
    std::vector<std::vector<Variable>> x(n, std::vector<Variable>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (i != j) x[i][j] = m.addVar(0.0, 1.0, VarType::Binary);

    // Flow variables h[k][i][j] ∈ [0,1] for commodity k=1..n-1, arc (i,j), i≠j.
    // Indexed as h[k-1][i][j] (k shifted to 0-based).
    std::vector<std::vector<std::vector<Variable>>> h(
        n - 1, std::vector<std::vector<Variable>>(n, std::vector<Variable>(n)));
    for (int k = 1; k < n; ++k)
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                if (i != j) h[k - 1][i][j] = m.addVar(0.0, 1.0);

    // Objective.
    LinearExpr obj;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (i != j) obj += c[i][j] * x[i][j];
    m.setObjective(obj, ObjSense::Minimize);

    // Degree constraints (city-0 degree-out dropped).
    for (int i = 0; i < n; ++i) {
        LinearExpr out, in;
        for (int j = 0; j < n; ++j)
            if (j != i) { out += 1.0 * x[i][j]; in += 1.0 * x[j][i]; }
        if (i > 0) m.addLPConstraint(out, Sense::Equal, 1.0);
        m.addLPConstraint(in, Sense::Equal, 1.0);
    }

    // Capacity: h[k][i][j] ≤ x[i][j] for each commodity k and arc (i,j).
    for (int k = 1; k < n; ++k)
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                if (i != j) {
                    LinearExpr e;
                    e += 1.0 * h[k - 1][i][j];
                    e += -1.0 * x[i][j];
                    m.addLPConstraint(e, Sense::LessEq, 0.0);
                }

    // Flow conservation for cities i=1..n-1 (source city 0 is implied).
    // b[k][i] = -1 if i=k (sink), 0 otherwise (transit).
    for (int k = 1; k < n; ++k)
        for (int i = 1; i < n; ++i) {
            LinearExpr e;
            for (int j = 0; j < n; ++j)
                if (j != i) { e += 1.0 * h[k - 1][i][j]; e += -1.0 * h[k - 1][j][i]; }
            m.addLPConstraint(e, Sense::Equal, (i == k) ? -1.0 : 0.0);
        }

    return m;
}

inline baguette::Model makeTSP10MCF() {
    return makeTSPMCF(10, makeTSPArcs(10));
}

// ── DFJ with explicit SEC (Dantzig-Fulkerson-Johnson, 1954) ───────────────────

/// Build an n-city TSP model (DFJ formulation with all SEC enumerated).
///
/// All Subtour Elimination Constraints are added explicitly by iterating over
/// every proper subset S ⊆ {0..n-1} with 2 ≤ |S| ≤ n-2:
///
///   Σᵢ∈S Σⱼ∈S x[i][j]  ≤  |S| − 1
///
/// This gives the tightest possible LP relaxation (DFJ LP bound = convex hull
/// of Hamiltonian tours for complete graphs).  It is impractical for large n
/// but provides a clean reference formulation for small instances.
///
/// LP optimal = 10 on the 10-city cyclic instance (~1000 SEC constraints).
///
/// @param n     Number of cities (0..n-1). Must satisfy n ≤ 20 (2^n subsets).
/// @param arcs  Sparse distance entries; self-loops are ignored.
///
/// @note Complexity  O(2^n × n) constraint generation; O(2^n × n²) LP size.
///   Practical for n ≤ 18. For n=10: ~1000 SEC rows; LP is still fast.
inline baguette::Model makeTSPDFJ(int n, const std::vector<TspArc>& arcs) {
    using namespace baguette;

    double maxDist = 0.0;
    for (const auto& a : arcs)
        if (a.from != a.to) maxDist = std::max(maxDist, a.dist);
    const double bigM = double(n) * maxDist;

    std::vector<std::vector<double>> c(n, std::vector<double>(n, bigM));
    for (const auto& a : arcs)
        if (a.from != a.to) c[a.from][a.to] = a.dist;

    Model m;

    std::vector<std::vector<Variable>> x(n, std::vector<Variable>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (i != j) x[i][j] = m.addVar(0.0, 1.0, VarType::Binary);

    LinearExpr obj;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (i != j) obj += c[i][j] * x[i][j];
    m.setObjective(obj, ObjSense::Minimize);

    // Degree constraints (city-0 degree-out dropped).
    for (int i = 0; i < n; ++i) {
        LinearExpr out, in;
        for (int j = 0; j < n; ++j)
            if (j != i) { out += 1.0 * x[i][j]; in += 1.0 * x[j][i]; }
        if (i > 0) m.addLPConstraint(out, Sense::Equal, 1.0);
        m.addLPConstraint(in, Sense::Equal, 1.0);
    }

    // Explicit SEC for every proper subset S with 2 ≤ |S| ≤ n-2.
    // Enumerate using bitmask; bit k = 1 means city k ∈ S.
    const int full = (1 << n) - 1;
    for (int mask = 3; mask <= full - 1; ++mask) {
        // Compute |S| via Kernighan bit count.
        int sz = 0;
        for (int t = mask; t; t &= t - 1) ++sz;
        if (sz < 2 || sz > n - 2) continue;

        // By symmetry S and V\S yield the same cut; only enumerate one half.
        if (mask > (full ^ mask)) continue;

        LinearExpr sec;
        for (int i = 0; i < n; ++i) {
            if (!(mask >> i & 1)) continue;
            for (int j = 0; j < n; ++j) {
                if (j == i || !(mask >> j & 1)) continue;
                sec += 1.0 * x[i][j];
            }
        }
        m.addLPConstraint(sec, Sense::LessEq, double(sz - 1));
    }

    return m;
}

inline baguette::Model makeTSP10DFJ() {
    return makeTSPDFJ(10, makeTSPArcs(10));
}

/// One item in a 0/1 knapsack instance.
struct KnapsackItem { double weight, profit; };

/// Build a 0/1 knapsack model from a list of items.
///
/// Variables:  x[i] ∈ {0,1} (Binary)
/// Constraint: Σ w[i]·x[i] ≤ capacity   (and ≥ minLoad when minLoad > 0)
/// Objective:  max Σ p[i]·x[i]
///
/// @note Complexity
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
/// @note Complexity
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

// ── Uncapacitated Facility Location (UFL) ─────────────────────────────────────

/// Build an Uncapacitated Facility Location model.
///
/// Variables:
///   y[i]    ∈ {0,1} (Binary)   — open facility i             (nFac vars)
///   x[i][j] ∈ {0,1} (Binary)   — assign client j to i        (nFac × nCli vars)
///
/// Constraints:
///   Coverage: Σᵢ x[i][j] = 1            ∀ j ∈ {0,…,nCli-1}  (nCli equalities)
///   Linking:  x[i][j] − y[i]   ≤ 0      ∀ i,j               (nFac×nCli constraints)
///
/// Objective: min Σᵢ f[i]·y[i] + Σᵢⱼ c[i][j]·x[i][j]
///
/// Probing: fixing y[i]=0 forces all nCli linking constraints to set x[i][j]=0,
/// which in turn tightens the nCli coverage equalities for the remaining facilities.
/// One probe on y[i] can cascade into O(nCli) binary fixings — unlike TSP where
/// arc probing never produces LP infeasibility in the MTZ or SCF formulations.
///
/// @param fixedCosts   fixedCosts[i] — cost to open facility i  (size nFac)
/// @param assignCosts  assignCosts[i][j] — cost to serve client j from i  (nFac × nCli)
///
/// @note Complexity
///   O(nFac × nCli) variables and constraints.
inline baguette::Model makeFacilityLocation(
        const std::vector<double>& fixedCosts,
        const std::vector<std::vector<double>>& assignCosts) {
    using namespace baguette;
    const int nFac = static_cast<int>(fixedCosts.size());
    const int nCli = static_cast<int>(assignCosts[0].size());

    Model m;

    // y[i] ∈ {0,1}: open facility i.
    std::vector<Variable> y(nFac);
    LinearExpr obj;
    for (int i = 0; i < nFac; ++i) {
        y[i] = m.addVar(0.0, 1.0, VarType::Binary);
        obj += fixedCosts[i] * y[i];
    }

    // x[i][j] ∈ {0,1}: assign client j to facility i.
    std::vector<std::vector<Variable>> x(nFac, std::vector<Variable>(nCli));
    for (int i = 0; i < nFac; ++i)
        for (int j = 0; j < nCli; ++j) {
            x[i][j] = m.addVar(0.0, 1.0, VarType::Binary);
            obj += assignCosts[i][j] * x[i][j];
        }

    // Coverage: Σᵢ x[i][j] = 1 ∀ j.
    for (int j = 0; j < nCli; ++j) {
        LinearExpr cov;
        for (int i = 0; i < nFac; ++i) cov += 1.0 * x[i][j];
        m.addLPConstraint(cov, Sense::Equal, 1.0);
    }

    // Linking: x[i][j] - y[i] ≤ 0 ∀ i,j.
    for (int i = 0; i < nFac; ++i)
        for (int j = 0; j < nCli; ++j) {
            LinearExpr lnk;
            lnk += 1.0 * x[i][j];
            lnk += -1.0 * y[i];
            m.addLPConstraint(lnk, Sense::LessEq, 0.0);
        }

    m.setObjective(obj, ObjSense::Minimize);
    return m;
}

/// Build a 5-facility × 10-client UFL instance.
///
/// Fixed costs:      f[i] = 20 for all i (uniform).
/// Assignment costs: integers in [1, 10] from a deterministic LCG (seed 0xDEADBEEF).
///
/// Model size: 55 binary variables (5 + 50), 60 constraints (10 coverage + 50 linking).
/// LP optimal = 67, IP optimal = 69.
///
/// @note Complexity
///   O(nFac × nCli) variables and constraints.
inline baguette::Model makeFacilityLocation5x10(unsigned seed = 0xDEADBEEFu) {
    const int nFac = 5, nCli = 10;
    std::vector<double> fixedCosts(nFac, 20.0);
    std::vector<std::vector<double>> assignCosts(nFac, std::vector<double>(nCli));
    unsigned s = seed;
    for (int i = 0; i < nFac; ++i)
        for (int j = 0; j < nCli; ++j) {
            s = s * 1664525u + 1013904223u;
            assignCosts[i][j] = 1.0 + double(s % 10u);
        }
    return makeFacilityLocation(fixedCosts, assignCosts);
}

/// Build a 15-facility × 30-client UFL instance.
///
/// Fixed costs:      f[i] = 20 for all i (uniform, same as 5×10).
/// Assignment costs: integers in [1, 10] from a deterministic LCG (seed 0xCAFEBABE).
///
/// Model size: 465 binary variables (15 + 450), 480 constraints (30 coverage + 450 linking).
/// Instance is 3× larger than 5×10 and expected to require more B&B nodes.
///
/// @note Complexity
///   O(nFac × nCli) variables and constraints.
inline baguette::Model makeFacilityLocation15x30(unsigned seed = 0xCAFEBABEu) {
    const int nFac = 15, nCli = 30;
    std::vector<double> fixedCosts(nFac, 20.0);
    std::vector<std::vector<double>> assignCosts(nFac, std::vector<double>(nCli));
    unsigned s = seed;
    for (int i = 0; i < nFac; ++i)
        for (int j = 0; j < nCli; ++j) {
            s = s * 1664525u + 1013904223u;
            assignCosts[i][j] = 1.0 + double(s % 10u);
        }
    return makeFacilityLocation(fixedCosts, assignCosts);
}

// ── Set Partitioning ──────────────────────────────────────────────────────────

/// Build a Set Partitioning model from an explicit column list.
///
/// Each column (subset) covers a list of elements; exactly one selected column
/// must cover each element.  Elements may appear in several columns (redundancy).
///
/// Variables:  x[i] ∈ {0,1} (Binary) — select column i
///
/// Constraints:
///   Σ_{i : e ∈ subsets[i]} x[i] = 1   ∀ element e   (partitioning equalities)
///
/// Objective:  min Σᵢ costs[i] · x[i]
///
/// Probing notes
///   Setting x[i]=0 removes column i from the coverage of every element it
///   covers.  For elements now covered by a single remaining column j,
///   x[j] is forced to 1 (otherwise the equality is violated).  That forcing
///   can in turn remove other columns from their elements, producing a cascade.
///   The cascade depth depends on the overlap structure of the column matrix.
///
/// @param nElements  Total number of elements (universe size).
/// @param subsets    subsets[i] = sorted list of elements covered by column i.
/// @param costs      costs[i]   = objective coefficient of x[i].
///
/// @note Complexity O(Σ|subsets[i]|) variables and constraint entries.
inline baguette::Model makeSetPartitioning(
        int nElements,
        const std::vector<std::vector<int>>& subsets,
        const std::vector<double>& costs) {
    using namespace baguette;
    const int nCols = static_cast<int>(subsets.size());

    Model m;

    std::vector<Variable> x(nCols);
    LinearExpr obj;
    for (int i = 0; i < nCols; ++i) {
        x[i] = m.addVar(0.0, 1.0, VarType::Binary);
        obj += costs[i] * x[i];
    }
    m.setObjective(obj, ObjSense::Minimize);

    // Invert subset lists: cover[e] = indices of columns that cover element e.
    std::vector<std::vector<int>> cover(nElements);
    for (int i = 0; i < nCols; ++i)
        for (int e : subsets[i])
            cover[e].push_back(i);

    // Partitioning: each element covered by exactly one selected column.
    for (int e = 0; e < nElements; ++e) {
        LinearExpr row;
        for (int i : cover[e]) row += 1.0 * x[i];
        m.addLPConstraint(row, Sense::Equal, 1.0);
    }

    return m;
}

} // namespace baguette_test

/// LP relaxations of classic MILP problems.
inline std::vector<LPTestCase> makeRelaxedMILPTestSuite() {
    using namespace baguette;

    return {
        // ── TSP 10 cities — MTZ formulation ─────────────────────────────────
        // See makeTSP10() for the full formulation description.
        // LP optimal = 10 (cyclic tour 0→1→…→9→0, all unit-cost adjacent arcs).
        {"tsp_10_mtz", LPStatus::Optimal, 10.0,
            []() { return baguette_test::makeTSP10(); }},

        // ── TSP 10 cities — Single-Commodity Flow formulation ────────────────
        // Same instance, stronger LP relaxation (SCF bound = DFJ bound ≥ MTZ bound).
        // LP optimal = 10 (cyclic tour is an integer extreme point for both).
        {"tsp_10_scf", LPStatus::Optimal, 10.0,
            []() { return baguette_test::makeTSP10Flow(); }},

        // ── TSP 10 cities — MTZ + AllDiff CP ────────────────────────────────
        // Same MTZ LP relaxation (CP constraints not linearised) + AllDiff on
        // position variables u[1..9].  LP optimal = 10.
        {"tsp_10_mtz_alldiff", LPStatus::Optimal, 10.0,
            []() { return baguette_test::makeTSP10Mtz(); }},

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

        // ── Uncapacitated Facility Location 5×10 ────────────────────────────
        // 5 facilities (fixed cost 20), 10 clients, LCG costs in [1,10].
        // LP optimal = 67 (fractional facility opening).
        // IP optimal = 69 (2-3 facilities opened).
        {"facility_location_5x10", LPStatus::Optimal, 67.0,
            []() { return baguette_test::makeFacilityLocation5x10(); }},
    };
}
