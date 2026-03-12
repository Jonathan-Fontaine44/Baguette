#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <limits>

#include "baguette/core/Config.hpp"
#include "baguette/lp/LPSolver.hpp"
#include "baguette/model/Model.hpp"

using namespace baguette;
using Catch::Matchers::WithinAbs;

static constexpr double kTol = 1e-6;
static const double kInf = std::numeric_limits<double>::infinity();

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Build a simple maximisation problem:
///   max  x + y
///   s.t. x + y <= 4
///        x     <= 3
///            y <= 3
///        x, y >= 0
/// Optimal: x=1, y=3 or x=3, y=1 - objective = 4.
static Model makeSimpleMax() {
    Model m;
    auto x = m.addVar(0.0, kInf, "x");
    auto y = m.addVar(0.0, kInf, "y");
    m.addConstraint(1.0 * x + 1.0 * y, Sense::LessEq, 4.0);
    m.addConstraint(1.0 * x,            Sense::LessEq, 3.0);
    m.addConstraint(1.0 * y,            Sense::LessEq, 3.0);
    m.setObjective(1.0 * x + 1.0 * y, ObjSense::Maximize);
    return m;
}

/// Build a minimisation problem with GreaterEq constraints:
///   min  2x + 3y
///   s.t.  x +  y >= 4
///        2x +  y >= 6
///        x, y >= 0
/// Optimal: corner (4, 0) satisfies both constraints, objective = 8.
static Model makeMinWithGEQ() {
    Model m;
    auto x = m.addVar(0.0, kInf, "x");
    auto y = m.addVar(0.0, kInf, "y");
    m.addConstraint(1.0 * x + 1.0 * y, Sense::GreaterEq, 4.0);
    m.addConstraint(2.0 * x + 1.0 * y, Sense::GreaterEq, 6.0);
    m.setObjective(2.0 * x + 3.0 * y, ObjSense::Minimize);
    return m;
}

/// Build a problem with an equality constraint:
///   min  x + y
///   s.t. x + y = 5
///        x >= 0, y >= 0
/// Optimal: any (x, y) with x+y=5, objective = 5.
static Model makeEqualityConstraint() {
    Model m;
    auto x = m.addVar(0.0, kInf, "x");
    auto y = m.addVar(0.0, kInf, "y");
    m.addConstraint(1.0 * x + 1.0 * y, Sense::Equal, 5.0);
    m.setObjective(1.0 * x + 1.0 * y, ObjSense::Minimize);
    return m;
}

/// Build an infeasible problem:
///   x >= 3  AND  x <= 2
static Model makeInfeasible() {
    Model m;
    auto x = m.addVar(0.0, kInf, "x");
    m.addConstraint(1.0 * x, Sense::GreaterEq, 3.0);
    m.addConstraint(1.0 * x, Sense::LessEq,    2.0);
    m.setObjective(1.0 * x, ObjSense::Minimize);
    return m;
}

/// Build an unbounded problem:
///   min  -x,  x >= 0  (no upper bound)
static Model makeUnbounded() {
    Model m;
    auto x = m.addVar(0.0, kInf, "x");
    m.addConstraint(1.0 * x, Sense::GreaterEq, 0.0);
    m.setObjective(-1.0 * x, ObjSense::Minimize);
    return m;
}

/// Build a problem with a finite upper bound handled via extra row:
///   max  x,  0 <= x <= 5
/// Optimal: x=5, objective=5.
static Model makeUpperBound() {
    Model m;
    auto x = m.addVar(0.0, 5.0, "x");
    m.setObjective(1.0 * x, ObjSense::Maximize);
    return m;
}

/// Build a problem with a non-zero lower bound:
///   min  x,  3 <= x <= 10
/// Optimal: x=3, objective=3.
static Model makeLowerBound() {
    Model m;
    auto x = m.addVar(3.0, 10.0, "x");
    m.setObjective(1.0 * x, ObjSense::Minimize);
    return m;
}

// ── Tests: LPResult (solve) ───────────────────────────────────────────────────

TEST_CASE("LP solve - simple maximisation", "[lp]") {
    auto res = solve(makeSimpleMax());
    REQUIRE(res.status == LPStatus::Optimal);
    CHECK_THAT(res.objectiveValue, WithinAbs(4.0, kTol));
    REQUIRE(res.primalValues.size() == 2);
    CHECK_THAT(res.primalValues[0] + res.primalValues[1], WithinAbs(4.0, kTol));
}

TEST_CASE("LP solve - minimisation with GEQ constraints", "[lp]") {
    auto res = solve(makeMinWithGEQ());
    REQUIRE(res.status == LPStatus::Optimal);
    // Optimal corner: (4, 0) with obj = 2*4 + 3*0 = 8
    CHECK_THAT(res.objectiveValue, WithinAbs(8.0, kTol));
    REQUIRE(res.primalValues.size() == 2);
    CHECK_THAT(res.primalValues[0], WithinAbs(4.0, kTol));
    CHECK_THAT(res.primalValues[1], WithinAbs(0.0, kTol));
}

TEST_CASE("LP solve - equality constraint", "[lp]") {
    auto res = solve(makeEqualityConstraint());
    REQUIRE(res.status == LPStatus::Optimal);
    CHECK_THAT(res.objectiveValue, WithinAbs(5.0, kTol));
    REQUIRE(res.primalValues.size() == 2);
    CHECK_THAT(res.primalValues[0] + res.primalValues[1], WithinAbs(5.0, kTol));
}

TEST_CASE("LP solve - infeasible problem", "[lp]") {
    auto res = solve(makeInfeasible());
    REQUIRE(res.status == LPStatus::Infeasible);
    CHECK(res.primalValues.empty());
}

TEST_CASE("LP solve - unbounded problem", "[lp]") {
    auto res = solve(makeUnbounded());
    REQUIRE(res.status == LPStatus::Unbounded);
    CHECK(res.primalValues.empty());
}

TEST_CASE("LP solve - finite upper bound", "[lp]") {
    auto res = solve(makeUpperBound());
    REQUIRE(res.status == LPStatus::Optimal);
    CHECK_THAT(res.objectiveValue, WithinAbs(5.0, kTol));
    REQUIRE(res.primalValues.size() == 1);
    CHECK_THAT(res.primalValues[0], WithinAbs(5.0, kTol));
}

TEST_CASE("LP solve - non-zero lower bound", "[lp]") {
    auto res = solve(makeLowerBound());
    REQUIRE(res.status == LPStatus::Optimal);
    CHECK_THAT(res.objectiveValue, WithinAbs(3.0, kTol));
    REQUIRE(res.primalValues.size() == 1);
    CHECK_THAT(res.primalValues[0], WithinAbs(3.0, kTol));
}

TEST_CASE("LP solve - maxIter limit stops early", "[lp]") {
    auto res = solve(makeMinWithGEQ(), /*maxIter=*/1);
    CHECK((res.status == LPStatus::MaxIter || res.status == LPStatus::Optimal));
}

TEST_CASE("LP solve - degenerate LP (multiple tie ratios) solves correctly", "[lp]") {
    // Highly degenerate problem: the optimal vertex is shared by 4 constraints,
    // forcing multiple ratio-test ties (ratio = 0) during pivoting. Exercises
    // the Bland tie-breaking in selectLeaving with a finite maxIter guard.
    //
    //   min  x + y
    //   s.t.  x + y >= 4       (binding at (4,0) and (0,4))
    //         x     <= 4       (binding at (4, *))
    //             y <= 4       (binding at (*, 4))
    //         x + y <= 4       (makes (4,0) and (0,4) the only optima — degenerate)
    //         x, y >= 0
    // Optimal: obj = 4, degenerate (infinitely many solutions on x+y=4).
    Model m;
    auto x = m.addVar(0.0, kInf, "x");
    auto y = m.addVar(0.0, kInf, "y");
    m.addConstraint(1.0 * x + 1.0 * y, Sense::GreaterEq, 4.0);
    m.addConstraint(1.0 * x,            Sense::LessEq,    4.0);
    m.addConstraint(1.0 * y,            Sense::LessEq,    4.0);
    m.addConstraint(1.0 * x + 1.0 * y, Sense::LessEq,    4.0);
    m.setObjective(1.0 * x + 1.0 * y, ObjSense::Minimize);

    auto res = solve(m, /*maxIter=*/200);

    REQUIRE(res.status == LPStatus::Optimal);
    CHECK_THAT(res.objectiveValue, WithinAbs(4.0, kTol));
    REQUIRE(res.primalValues.size() == 2);
    CHECK_THAT(res.primalValues[0] + res.primalValues[1], WithinAbs(4.0, kTol));
}

TEST_CASE("LP solve - reinversion every pivot yields identical solution", "[lp]") {
    // Force reinvert() after every single pivot (period = 1) and verify that
    // the primal solution and objective are numerically identical to the
    // default period (50). Uses makeSimpleMax() which requires several pivots.
    const uint32_t savedPeriod = baguette::reinversion_period;
    baguette::set_reinversion_period(1);
    auto resFreq = solve(makeSimpleMax());
    baguette::set_reinversion_period(savedPeriod);

    auto resRef = solve(makeSimpleMax());

    REQUIRE(resFreq.status == LPStatus::Optimal);
    REQUIRE(resRef.status  == LPStatus::Optimal);
    CHECK_THAT(resFreq.objectiveValue,
               WithinAbs(resRef.objectiveValue, kTol));
    REQUIRE(resFreq.primalValues.size() == resRef.primalValues.size());
    for (std::size_t j = 0; j < resRef.primalValues.size(); ++j)
        CHECK_THAT(resFreq.primalValues[j],
                   WithinAbs(resRef.primalValues[j], kTol));
}

// ── Tests: LPDetailedResult (solveDetailed) ───────────────────────────────────

TEST_CASE("LP solveDetailed - primal matches solve()", "[lp]") {
    Model m = makeSimpleMax();
    auto simple   = solve(m);
    auto detailed = solveDetailed(m);

    REQUIRE(detailed.result.status == LPStatus::Optimal);
    REQUIRE_THAT(detailed.result.objectiveValue,
                 WithinAbs(simple.objectiveValue, kTol));
    REQUIRE(detailed.result.primalValues.size() == simple.primalValues.size());
    for (std::size_t j = 0; j < simple.primalValues.size(); ++j)
        CHECK_THAT(detailed.result.primalValues[j],
                   WithinAbs(simple.primalValues[j], kTol));
}

TEST_CASE("LP solveDetailed - dual variables (simple max)", "[lp]") {
    auto det = solveDetailed(makeSimpleMax());
    REQUIRE(det.result.status == LPStatus::Optimal);
    // 3 LessEq constraints, max problem -> shadow prices >= 0
    REQUIRE(det.dualValues.size() == 3);
    for (double y : det.dualValues)
        CHECK(y >= -kTol);
}

TEST_CASE("LP solveDetailed - dual variables (GEQ, min)", "[lp]") {
    auto det = solveDetailed(makeMinWithGEQ());
    REQUIRE(det.result.status == LPStatus::Optimal);
    // 2 GEQ constraints, min problem -> shadow prices >= 0
    REQUIRE(det.dualValues.size() == 2);
    for (double y : det.dualValues)
        CHECK(y >= -kTol);
}

TEST_CASE("LP solveDetailed - reduced costs at optimum", "[lp]") {
    auto det = solveDetailed(makeSimpleMax());
    REQUIRE(det.result.status == LPStatus::Optimal);
    REQUIRE(det.reducedCosts.size() == 2);
    // At optimum, reduced cost of a basic (non-zero) variable must be 0
    for (std::size_t j = 0; j < det.reducedCosts.size(); ++j)
        if (det.result.primalValues[j] > kTol)
            CHECK_THAT(det.reducedCosts[j], WithinAbs(0.0, kTol));
}

TEST_CASE("LP solveDetailed - basis record populated", "[lp]") {
    auto det = solveDetailed(makeSimpleMax());
    REQUIRE(det.result.status == LPStatus::Optimal);
    // 3 constraints, no UB rows (inf upper bounds) -> 3 basis entries
    CHECK(det.basis.basicCols.size() == 3);
    CHECK(!det.basis.colKind.empty());
    CHECK(det.basis.colKind.size() == det.basis.colOrigin.size());
}

TEST_CASE("LP solveDetailed - result sub-object accessible", "[lp]") {
    LPDetailedResult det = solveDetailed(makeSimpleMax());
    const LPResult& res = det.result;
    CHECK(res.status == LPStatus::Optimal);
    CHECK_THAT(res.objectiveValue, WithinAbs(det.result.objectiveValue, kTol));
}

TEST_CASE("LP solveDetailed - infeasible returns no dual/rc", "[lp]") {
    auto det = solveDetailed(makeInfeasible());
    REQUIRE(det.result.status == LPStatus::Infeasible);
    CHECK(det.dualValues.empty());
    CHECK(det.reducedCosts.empty());
    CHECK(det.basis.basicCols.empty());
}

// ── Variable bound edge cases ──────────────────────────────────────────────────

TEST_CASE("LP semi-infinite lb - min x with x in [3, inf]", "[lp]") {
    // min x,  x >= 3  (no explicit constraint, bound alone drives the solution)
    // Optimal: x = 3,  obj = 3.
    Model m;
    auto x = m.addVar(3.0, kInf, "x");
    m.setObjective(1.0 * x, ObjSense::Minimize);

    auto res = solve(m);

    REQUIRE(res.status == LPStatus::Optimal);
    CHECK_THAT(res.objectiveValue,    WithinAbs(3.0, kTol));
    CHECK_THAT(res.primalValues[0],   WithinAbs(3.0, kTol));
}

TEST_CASE("LP fully free variable - max x in [-inf, +inf] with x <= 4", "[lp]") {
    // max x,  x <= 4,  x ∈ (−∞, +∞)
    // Solved via split x = x⁺ − x⁻. Optimal: x = 4, obj = 4.
    Model m;
    auto x = m.addVar(-kInf, kInf, "x");
    m.addConstraint(1.0 * x, Sense::LessEq, 4.0);
    m.setObjective(1.0 * x, ObjSense::Maximize);

    auto res = solve(m);

    REQUIRE(res.status == LPStatus::Optimal);
    CHECK_THAT(res.objectiveValue,  WithinAbs(4.0, kTol));
    REQUIRE(res.primalValues.size() == 1);
    CHECK_THAT(res.primalValues[0], WithinAbs(4.0, kTol));
}

TEST_CASE("LP fully free variable - unbounded (min x, no constraint)", "[lp]") {
    // min x,  x ∈ (−∞, +∞),  no constraint
    // x⁻ grows without bound → Unbounded.
    Model m;
    auto x = m.addVar(-kInf, kInf, "x");
    m.setObjective(1.0 * x, ObjSense::Minimize);

    auto res = solve(m);

    REQUIRE(res.status == LPStatus::Unbounded);
    CHECK(res.primalValues.empty());
}

TEST_CASE("LP fully free variable - min x with x >= -3 (GEQ constraint)", "[lp]") {
    // min x,  x ∈ (−∞, +∞),  x >= −3
    // Optimal: x = −3, obj = −3.
    Model m;
    auto x = m.addVar(-kInf, kInf, "x");
    m.addConstraint(1.0 * x, Sense::GreaterEq, -3.0);
    m.setObjective(1.0 * x, ObjSense::Minimize);

    auto res = solve(m);

    REQUIRE(res.status == LPStatus::Optimal);
    CHECK_THAT(res.objectiveValue,  WithinAbs(-3.0, kTol));
    REQUIRE(res.primalValues.size() == 1);
    CHECK_THAT(res.primalValues[0], WithinAbs(-3.0, kTol));
}

TEST_CASE("LP two free variables - min x+y with x+y >= 5", "[lp]") {
    // min x + y,  x,y ∈ (−∞, +∞),  x + y >= 5
    // Optimal: x + y = 5, obj = 5.
    Model m;
    auto x = m.addVar(-kInf, kInf, "x");
    auto y = m.addVar(-kInf, kInf, "y");
    m.addConstraint(1.0 * x + 1.0 * y, Sense::GreaterEq, 5.0);
    m.setObjective(1.0 * x + 1.0 * y, ObjSense::Minimize);

    auto res = solve(m);

    REQUIRE(res.status == LPStatus::Optimal);
    CHECK_THAT(res.objectiveValue, WithinAbs(5.0, kTol));
    REQUIRE(res.primalValues.size() == 2);
    CHECK_THAT(res.primalValues[0] + res.primalValues[1], WithinAbs(5.0, kTol));
}

TEST_CASE("LP fully free variable - Equal constraint", "[lp]") {
    // min x + y,  x ∈ (−∞, +∞),  y >= 0,  x + y = 3
    // Feasible for any x ≤ 3 (y = 3 − x ≥ 0). Objective = x + (3−x) = 3 always.
    // Optimal: obj = 3.
    Model m;
    auto x = m.addVar(-kInf, kInf, "x");
    auto y = m.addVar(0.0,   kInf, "y");
    m.addConstraint(1.0 * x + 1.0 * y, Sense::Equal, 3.0);
    m.setObjective(1.0 * x + 1.0 * y, ObjSense::Minimize);

    auto res = solve(m);

    REQUIRE(res.status == LPStatus::Optimal);
    CHECK_THAT(res.objectiveValue, WithinAbs(3.0, kTol));
    REQUIRE(res.primalValues.size() == 2);
    CHECK_THAT(res.primalValues[0] + res.primalValues[1], WithinAbs(3.0, kTol));
}

TEST_CASE("LP semi-infinite ub - max x with x in [-inf, 10]", "[lp]") {
    // max x,  x <= 10  (ub-shifted: x' = 10 - x, x' >= 0)
    // Optimal: x = 10, obj = 10.
    Model m;
    auto x = m.addVar(-kInf, 10.0, "x");
    m.setObjective(1.0 * x, ObjSense::Maximize);

    auto res = solve(m);

    REQUIRE(res.status == LPStatus::Optimal);
    CHECK_THAT(res.objectiveValue,    WithinAbs(10.0, kTol));
    CHECK_THAT(res.primalValues[0],   WithinAbs(10.0, kTol));
}

TEST_CASE("LP redundant Equal constraint - does not crash", "[lp]") {
    // min x + y,  x + y = 5,  2x + 2y = 10  (second is 2x first, redundant)
    // After phase I one artificial cannot be driven out (row is all-zeros in
    // non-artificial columns). preparePhaseTwo must not access sfOrig.c
    // out-of-bounds for that row.
    // Optimal: x + y = 5, obj = 5.
    Model m;
    auto x = m.addVar(0.0, kInf, "x");
    auto y = m.addVar(0.0, kInf, "y");
    m.addConstraint(1.0 * x + 1.0 * y, Sense::Equal, 5.0);
    m.addConstraint(2.0 * x + 2.0 * y, Sense::Equal, 10.0);
    m.setObjective(1.0 * x + 1.0 * y, ObjSense::Minimize);

    auto res = solve(m);

    REQUIRE(res.status == LPStatus::Optimal);
    CHECK_THAT(res.objectiveValue, WithinAbs(5.0, kTol));
    REQUIRE(res.primalValues.size() == 2);
    CHECK_THAT(res.primalValues[0] + res.primalValues[1], WithinAbs(5.0, kTol));
}

TEST_CASE("LP redundant GEQ constraint - does not crash", "[lp]") {
    // min x,  x >= 3,  2x >= 6  (second is 2x first, redundant)
    // Same structure: phase I leaves one artificial stuck in basis.
    // Optimal: x = 3, obj = 3.
    Model m;
    auto x = m.addVar(0.0, kInf, "x");
    m.addConstraint(1.0 * x, Sense::GreaterEq, 3.0);
    m.addConstraint(2.0 * x, Sense::GreaterEq, 6.0);
    m.setObjective(1.0 * x, ObjSense::Minimize);

    auto res = solve(m);

    REQUIRE(res.status == LPStatus::Optimal);
    CHECK_THAT(res.objectiveValue,  WithinAbs(3.0, kTol));
    CHECK_THAT(res.primalValues[0], WithinAbs(3.0, kTol));
}

// ── Objective constant ────────────────────────────────────────────────────────

TEST_CASE("LP objective constant - Minimize", "[lp]") {
    // min x + 100  s.t. x >= 3,  x >= 0
    // Optimal: x = 3, obj = 103.
    Model m;
    auto x = m.addVar(0.0, kInf, "x");
    m.addConstraint(1.0 * x, Sense::GreaterEq, 3.0);
    LinearExpr obj = 1.0 * x;
    obj.constant   = 100.0;
    m.setObjective(obj, ObjSense::Minimize);

    auto res = solve(m);

    REQUIRE(res.status == LPStatus::Optimal);
    CHECK_THAT(res.objectiveValue,  WithinAbs(103.0, kTol));
    CHECK_THAT(res.primalValues[0], WithinAbs(3.0,   kTol));
}

TEST_CASE("LP objective constant - Maximize", "[lp]") {
    // max x - 50  s.t. x <= 7,  x >= 0
    // Optimal: x = 7, obj = -43.
    Model m;
    auto x = m.addVar(0.0, kInf, "x");
    m.addConstraint(1.0 * x, Sense::LessEq, 7.0);
    LinearExpr obj = 1.0 * x;
    obj.constant   = -50.0;
    m.setObjective(obj, ObjSense::Maximize);

    auto res = solve(m);

    REQUIRE(res.status == LPStatus::Optimal);
    CHECK_THAT(res.objectiveValue,  WithinAbs(-43.0, kTol));
    CHECK_THAT(res.primalValues[0], WithinAbs(7.0,   kTol));
}

TEST_CASE("LP objective constant - does not affect optimal solution", "[lp]") {
    // Same problem as above with two different constants: solution must be identical.
    auto makeModel = [](double c) {
        Model m;
        auto x = m.addVar(0.0, kInf, "x");
        auto y = m.addVar(0.0, kInf, "y");
        m.addConstraint(1.0*x + 1.0*y, Sense::LessEq, 4.0);
        LinearExpr obj = 1.0*x + 2.0*y;
        obj.constant   = c;
        m.setObjective(obj, ObjSense::Minimize);
        return m;
    };

    auto r0 = solve(makeModel(0.0));
    auto r1 = solve(makeModel(99.0));

    REQUIRE(r0.status == LPStatus::Optimal);
    REQUIRE(r1.status == LPStatus::Optimal);
    // Optimal solution (x=0, y=0, obj=0 / 99) — same variable values.
    REQUIRE(r0.primalValues.size() == r1.primalValues.size());
    for (std::size_t j = 0; j < r0.primalValues.size(); ++j)
        CHECK_THAT(r1.primalValues[j], WithinAbs(r0.primalValues[j], kTol));
    // Objective values differ by exactly the constant.
    CHECK_THAT(r1.objectiveValue - r0.objectiveValue, WithinAbs(99.0, kTol));
}

// ── Redundant constraints ──────────────────────────────────────────────────────

TEST_CASE("Redundant Equal constraint: correct primal and objective", "[redundant]") {
    // Bug regression: repairRedundantRows used to assign an original-variable
    // column (non-zero cost, non-zero tableau entries) to the dummy row produced
    // by the linearly dependent constraint.  That variable would then enter the
    // basis at a different row during phase II, creating a duplicate in basicCols.
    // primalSolution iterates basicCols in order and overwrites x[j] with the
    // redundant row's value (0), producing a primal solution inconsistent with
    // the reported objective value.
    //
    //   min  x1 - 2*x2
    //   s.t. x1 + x2 = 4          (Equal)
    //        2*x1 + 2*x2 = 8      (Equal, linearly dependent on row 0)
    //        x1, x2 in [0, 10]
    //
    // Optimal: x1 = 0, x2 = 4, obj = -8.
    // Before the fix: primalValues = {0, 0}, objectiveValue = -8  (inconsistent).
    // After  the fix: primalValues = {0, 4}, objectiveValue = -8  (consistent).
    Model m;
    auto x1 = m.addVar(0.0, 10.0, "x1");
    auto x2 = m.addVar(0.0, 10.0, "x2");

    m.addConstraint(1.0 * x1 + 1.0 * x2, Sense::Equal, 4.0);
    m.addConstraint(2.0 * x1 + 2.0 * x2, Sense::Equal, 8.0);
    m.setObjective(1.0 * x1 + -2.0 * x2, ObjSense::Minimize);

    auto det = solveDetailed(m);
    REQUIRE(det.result.status == LPStatus::Optimal);
    REQUIRE_THAT(det.result.objectiveValue,    WithinAbs(-8.0, kTol));
    REQUIRE_THAT(det.result.primalValues[x1.id], WithinAbs(0.0,  kTol));
    REQUIRE_THAT(det.result.primalValues[x2.id], WithinAbs(4.0,  kTol));
    // Consistency check: obj must equal c^T x, not just the rc-row value.
    double recomputed = 1.0 * det.result.primalValues[x1.id]
                      - 2.0 * det.result.primalValues[x2.id];
    REQUIRE_THAT(recomputed, WithinAbs(det.result.objectiveValue, kTol));
}

TEST_CASE("Redundant GEQ constraint: correct primal and objective", "[redundant]") {
    // Same regression, GEQ variant. The second constraint (4x+4y >= 8) is
    // twice the first (2x+2y >= 4) and produces the same degenerate phase-I
    // exit with an artificial stuck in the basis.
    //
    //   min  3x + y
    //   s.t. 2x + 2y >= 4   (GreaterEq)
    //        4x + 4y >= 8   (GreaterEq, linearly dependent)
    //        x, y in [0, 10]
    //
    // Optimal on the line 2x+2y=4: minimise 3x+y = 3x+(2-x) = 2+2x → x=0,
    // y=2, obj=2.
    Model m;
    auto x = m.addVar(0.0, 10.0, "x");
    auto y = m.addVar(0.0, 10.0, "y");

    m.addConstraint(2.0 * x + 2.0 * y, Sense::GreaterEq, 4.0);
    m.addConstraint(4.0 * x + 4.0 * y, Sense::GreaterEq, 8.0);
    m.setObjective(3.0 * x + 1.0 * y, ObjSense::Minimize);

    auto det = solveDetailed(m);
    REQUIRE(det.result.status == LPStatus::Optimal);
    REQUIRE_THAT(det.result.objectiveValue,     WithinAbs(2.0,  kTol));
    REQUIRE_THAT(det.result.primalValues[x.id], WithinAbs(0.0,  kTol));
    REQUIRE_THAT(det.result.primalValues[y.id], WithinAbs(2.0,  kTol));
    double recomputed = 3.0 * det.result.primalValues[x.id]
                      + 1.0 * det.result.primalValues[y.id];
    REQUIRE_THAT(recomputed, WithinAbs(det.result.objectiveValue, kTol));
}
