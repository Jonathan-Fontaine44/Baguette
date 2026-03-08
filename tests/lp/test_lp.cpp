#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <limits>

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

TEST_CASE("LP solveDetailed - implicit conversion to LPResult", "[lp]") {
    LPDetailedResult det = solveDetailed(makeSimpleMax());
    const LPResult& res = det; // implicit conversion
    CHECK(res.status == det.result.status);
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
