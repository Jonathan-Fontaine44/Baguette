#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "baguette/lp/LPResult.hpp"
#include "baguette/lp/LPSolver.hpp"
#include "baguette/model/Model.hpp"

using namespace baguette;
using Catch::Matchers::WithinAbs;

static constexpr double kTol = 1e-6;

// ── Reference LP ──────────────────────────────────────────────────────────────
//
//   min  -x1 - x2
//   s.t.  2*x1 +   x2  <=  4
//           x1 + 2*x2  <=  4
//         x1 in [0, 3],  x2 in [0, 3]
//
// Explicit finite upper bounds are required so that branching on either
// variable does not change the finiteness of any bound.  Changing a bound
// from finite to infinite (or vice versa) alters the standard-form structure
// (upper-bound rows / free-split columns) and makes the parent basis
// incompatible, triggering the automatic cold-start fallback.
//
// Standard form: nRows = 4  (2 model rows + 2 upper-bound rows for x1, x2).
// LP optimum:  x1 = x2 = 4/3,  obj = -8/3.

static Model makeFractionalLP() {
    Model m;
    auto x1 = m.addVar(0.0, 3.0, "x1");
    auto x2 = m.addVar(0.0, 3.0, "x2");

    LinearExpr obj;
    obj.addTerm(x1, -1.0);
    obj.addTerm(x2, -1.0);
    m.setObjective(obj);

    LinearExpr c1;
    c1.addTerm(x1, 2.0);
    c1.addTerm(x2, 1.0);
    m.addConstraint(c1, Sense::LessEq, 4.0);

    LinearExpr c2;
    c2.addTerm(x1, 1.0);
    c2.addTerm(x2, 2.0);
    m.addConstraint(c2, Sense::LessEq, 4.0);

    return m;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

TEST_CASE("Warm-start: identity returns same result", "[warm_start]") {
    Model m      = makeFractionalLP();
    auto  parent = solveDetailed(m);
    REQUIRE(parent.result.status == LPStatus::Optimal);

    auto warm = solveDualDetailed(m, 0, 0.0, SolverClock::now(), parent.basis);
    REQUIRE(warm.result.status == LPStatus::Optimal);
    REQUIRE_THAT(warm.result.objectiveValue,
                 WithinAbs(parent.result.objectiveValue, kTol));
    REQUIRE_THAT(warm.result.primalValues[0],
                 WithinAbs(parent.result.primalValues[0], kTol));
    REQUIRE_THAT(warm.result.primalValues[1],
                 WithinAbs(parent.result.primalValues[1], kTol));
}

TEST_CASE("Warm-start: left branch x1 <= 1", "[warm_start]") {
    // Branch: x1 <= floor(4/3) = 1.
    // Child optimum: x1=1, x2=1.5, obj=-2.5.
    Model    parent_m = makeFractionalLP();
    Variable x1{0};
    auto     parent   = solveDetailed(parent_m);
    REQUIRE(parent.result.status == LPStatus::Optimal);

    Model child_m = parent_m.withVarBounds(x1, 0.0, 1.0);

    auto warm = solveDualDetailed(child_m, 0, 0.0, SolverClock::now(), parent.basis);
    auto cold = solveDetailed(child_m);

    REQUIRE(warm.result.status == LPStatus::Optimal);
    REQUIRE(warm.result.primalValues[0] <= 1.0 + kTol);
    REQUIRE_THAT(warm.result.objectiveValue,
                 WithinAbs(cold.result.objectiveValue, kTol));
    REQUIRE_THAT(warm.result.objectiveValue, WithinAbs(-2.5, kTol));
}

TEST_CASE("Warm-start: right branch x1 >= 2", "[warm_start]") {
    // Branch: x1 >= ceil(4/3) = 2.
    // Child optimum: x1=2, x2=0, obj=-2.0.
    Model    parent_m = makeFractionalLP();
    Variable x1{0};
    auto     parent   = solveDetailed(parent_m);
    REQUIRE(parent.result.status == LPStatus::Optimal);

    Model child_m = parent_m.withVarBounds(x1, 2.0, 3.0);

    auto warm = solveDualDetailed(child_m, 0, 0.0, SolverClock::now(), parent.basis);
    auto cold = solveDetailed(child_m);

    REQUIRE(warm.result.status == LPStatus::Optimal);
    REQUIRE(warm.result.primalValues[0] >= 2.0 - kTol);
    REQUIRE_THAT(warm.result.objectiveValue,
                 WithinAbs(cold.result.objectiveValue, kTol));
    REQUIRE_THAT(warm.result.objectiveValue, WithinAbs(-2.0, kTol));
}

TEST_CASE("Warm-start: infeasible branch (empty domain lb > ub)", "[warm_start]") {
    Model    parent_m = makeFractionalLP();
    Variable x1{0};
    auto     parent   = solveDetailed(parent_m);
    REQUIRE(parent.result.status == LPStatus::Optimal);

    // Explicitly contradictory bounds: lb > ub.
    Model child_m = parent_m.withVarBounds(x1, 2.0, 1.0);

    auto warm = solveDualDetailed(child_m, 0, 0.0, SolverClock::now(), parent.basis);
    REQUIRE(warm.result.status == LPStatus::Infeasible);
}

TEST_CASE("Warm-start: infeasible by constraints after tight bounds", "[warm_start]") {
    // x1 >= 3 and x2 >= 3 violates  2*x1 + x2 <= 4  (gives 9 <= 4).
    Model    parent_m = makeFractionalLP();
    Variable x1{0};
    Variable x2{1};
    auto     parent   = solveDetailed(parent_m);
    REQUIRE(parent.result.status == LPStatus::Optimal);

    Model child_m = parent_m.withVarBounds(x1, 3.0, 3.0)
                             .withVarBounds(x2, 3.0, 3.0);

    auto warm = solveDualDetailed(child_m, 0, 0.0, SolverClock::now(), parent.basis);
    REQUIRE(warm.result.status == LPStatus::Infeasible);
}

TEST_CASE("Warm-start: incompatible basis falls back to cold solve", "[warm_start]") {
    // nRows = 4 for the reference LP (2 model rows + 2 upper-bound rows).
    // A basis with the wrong number of entries triggers the size mismatch check
    // and falls back transparently to a cold primal solve.
    Model m = makeFractionalLP();

    BasisRecord bad_basis;
    bad_basis.basicCols = {0, 1, 2}; // 3 entries ≠ nRows (4)
    bad_basis.colKind   = {ColumnKind::Original,
                           ColumnKind::Original,
                           ColumnKind::Original};

    auto result = solveDualDetailed(m, 0, 0.0, SolverClock::now(), bad_basis);
    REQUIRE(result.result.status == LPStatus::Optimal);
    REQUIRE_THAT(result.result.objectiveValue, WithinAbs(-8.0 / 3.0, kTol));
}

TEST_CASE("Warm-start: backtrack left then right via setVarBounds", "[warm_start]") {
    // Simulate a B&B node with in-place mutation and restore (no copies).
    //
    //   root (x1* = x2* = 4/3,  obj = -8/3)
    //   ├── left  x1 in [0, 1]  →  x1=1,  x2=1.5,  obj=-2.5
    //   └── right x1 in [2, 3]  →  x1=2,  x2=0,    obj=-2.0
    //
    // After each branch the bounds are restored with setVarBounds() and the
    // root solution is re-verified to confirm the backtrack was complete.
    Model    m  = makeFractionalLP();
    Variable x1{0};

    auto root = solveDetailed(m);
    REQUIRE(root.result.status == LPStatus::Optimal);

    const double savedLb = m.getHot().lb[x1.id]; // 0.0
    const double savedUb = m.getHot().ub[x1.id]; // 3.0

    // ── Left branch ──────────────────────────────────────────────────────────
    m.setVarBounds(x1, 0.0, 1.0);
    auto left = solveDualDetailed(m, 0, 0.0, SolverClock::now(), root.basis);
    REQUIRE(left.result.status == LPStatus::Optimal);
    REQUIRE_THAT(left.result.objectiveValue, WithinAbs(-2.5, kTol));
    REQUIRE(left.result.primalValues[0] <= 1.0 + kTol);

    // Backtrack
    m.setVarBounds(x1, savedLb, savedUb);
    REQUIRE(m.getHot().lb[x1.id] == savedLb);
    REQUIRE(m.getHot().ub[x1.id] == savedUb);

    // ── Right branch (same model instance, same root basis) ──────────────────
    m.setVarBounds(x1, 2.0, 3.0);
    auto right = solveDualDetailed(m, 0, 0.0, SolverClock::now(), root.basis);
    REQUIRE(right.result.status == LPStatus::Optimal);
    REQUIRE_THAT(right.result.objectiveValue, WithinAbs(-2.0, kTol));
    REQUIRE(right.result.primalValues[0] >= 2.0 - kTol);

    // Backtrack
    m.setVarBounds(x1, savedLb, savedUb);

    // Root model is fully restored: re-solving gives the original optimum.
    auto restored = solveDetailed(m);
    REQUIRE(restored.result.status == LPStatus::Optimal);
    REQUIRE_THAT(restored.result.objectiveValue, WithinAbs(-8.0 / 3.0, kTol));
}

TEST_CASE("Warm-start: two-level B&B tree gives consistent results", "[warm_start]") {
    // Root → left (x1<=1) → left again (x2<=1).
    // Final optimum: x1=1, x2=1, obj=-2.
    Model    root_m = makeFractionalLP();
    Variable x1{0};
    Variable x2{1};

    auto root = solveDetailed(root_m);
    REQUIRE(root.result.status == LPStatus::Optimal);

    Model child_m = root_m.withVarBounds(x1, 0.0, 1.0);
    auto  child   = solveDualDetailed(child_m, 0, 0.0, SolverClock::now(), root.basis);
    REQUIRE(child.result.status == LPStatus::Optimal);

    Model grand_m = child_m.withVarBounds(x2, 0.0, 1.0);
    auto  grand   = solveDualDetailed(grand_m, 0, 0.0, SolverClock::now(), child.basis);
    REQUIRE(grand.result.status == LPStatus::Optimal);
    REQUIRE_THAT(grand.result.objectiveValue, WithinAbs(-2.0, kTol));
    REQUIRE_THAT(grand.result.primalValues[0], WithinAbs(1.0, kTol));
    REQUIRE_THAT(grand.result.primalValues[1], WithinAbs(1.0, kTol));
}
