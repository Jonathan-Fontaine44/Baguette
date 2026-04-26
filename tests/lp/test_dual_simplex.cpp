#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <limits>

#include "baguette/lp/LPSolver.hpp"
#include "baguette/model/Model.hpp"

using namespace baguette;
using Catch::Matchers::WithinAbs;

static constexpr double kTol = 1e-6;
static const double     kInf = std::numeric_limits<double>::infinity();

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Reuse model builders from the primal test suite (redeclared locally).

static Model makeSimpleMinLEQ() {
    // min 2x + 3y  s.t. x+y<=4, x<=3, y<=3,  x,y>=0
    // Optimal: x=0, y=0, obj=0  (unbounded below without >= constraint).
    // Use a meaningful problem instead:
    // min x + y  s.t. x+y>=4, x<=5, y<=5,  x,y>=0
    // Optimal: x+y=4, obj=4.
    Model m;
    auto x = m.addVar(0.0, kInf, "x");
    auto y = m.addVar(0.0, kInf, "y");
    m.addLPConstraint(1.0 * x + 1.0 * y, Sense::GreaterEq, 4.0);
    m.addLPConstraint(1.0 * x,            Sense::LessEq,    5.0);
    m.addLPConstraint(1.0 * y,            Sense::LessEq,    5.0);
    m.setObjective(1.0 * x + 1.0 * y, ObjSense::Minimize);
    return m;
}

static Model makeMinWithGEQ() {
    // min 2x + 3y  s.t. x+y>=4, 2x+y>=6,  x,y>=0
    // Optimal: x=4, y=0, obj=8
    Model m;
    auto x = m.addVar(0.0, kInf, "x");
    auto y = m.addVar(0.0, kInf, "y");
    m.addLPConstraint(1.0 * x + 1.0 * y, Sense::GreaterEq, 4.0);
    m.addLPConstraint(2.0 * x + 1.0 * y, Sense::GreaterEq, 6.0);
    m.setObjective(2.0 * x + 3.0 * y, ObjSense::Minimize);
    return m;
}

static Model makeInfeasible() {
    Model m;
    auto x = m.addVar(0.0, kInf, "x");
    m.addLPConstraint(1.0 * x, Sense::GreaterEq, 3.0);
    m.addLPConstraint(1.0 * x, Sense::LessEq,    2.0);
    m.setObjective(1.0 * x, ObjSense::Minimize);
    return m;
}

// ── Tests: solveDual matches solve() ─────────────────────────────────────────

TEST_CASE("dual simplex - GEQ min: matches primal solve", "[dual_simplex]") {
    auto primal = solve(makeMinWithGEQ());
    auto dual   = solveDual(makeMinWithGEQ());

    REQUIRE(primal.status == LPStatus::Optimal);
    REQUIRE(dual.status   == LPStatus::Optimal);
    CHECK_THAT(dual.objectiveValue, WithinAbs(primal.objectiveValue, kTol));
    REQUIRE(dual.primalValues.size() == primal.primalValues.size());
    for (std::size_t j = 0; j < primal.primalValues.size(); ++j)
        CHECK_THAT(dual.primalValues[j], WithinAbs(primal.primalValues[j], kTol));
}

TEST_CASE("dual simplex - mixed LEQ+GEQ: matches primal solve", "[dual_simplex]") {
    auto m1 = makeSimpleMinLEQ();
    auto primal = solve(m1);
    auto dual   = solveDual(m1);

    REQUIRE(primal.status == LPStatus::Optimal);
    REQUIRE(dual.status   == LPStatus::Optimal);
    CHECK_THAT(dual.objectiveValue, WithinAbs(primal.objectiveValue, kTol));
}

TEST_CASE("dual simplex - infeasible detected correctly", "[dual_simplex]") {
    // x >= 3 AND x <= 2: infeasible
    auto res = solveDual(makeInfeasible());
    REQUIRE(res.status == LPStatus::Infeasible);
    CHECK(res.primalValues.empty());
}

TEST_CASE("dual simplex - single variable bounded: min x in [3, inf]", "[dual_simplex]") {
    // min x,  x >= 3  (lb-shift: x' = x-3, min x' with x'>=0 → x'=0, x=3)
    Model m;
    auto x = m.addVar(3.0, kInf, "x");
    m.setObjective(1.0 * x, ObjSense::Minimize);

    auto res = solveDual(m);
    REQUIRE(res.status == LPStatus::Optimal);
    CHECK_THAT(res.objectiveValue, WithinAbs(3.0, kTol));
    CHECK_THAT(res.primalValues[0], WithinAbs(3.0, kTol));
}

TEST_CASE("dual simplex - LessEq only, non-negative costs", "[dual_simplex]") {
    // min 5x + 4y  s.t. x+y<=10, x<=7, y<=8,  x,y>=0
    // Minimum at (0,0): obj=0
    Model m;
    auto x = m.addVar(0.0, kInf, "x");
    auto y = m.addVar(0.0, kInf, "y");
    m.addLPConstraint(1.0 * x + 1.0 * y, Sense::LessEq, 10.0);
    m.addLPConstraint(1.0 * x,            Sense::LessEq,  7.0);
    m.addLPConstraint(1.0 * y,            Sense::LessEq,  8.0);
    m.setObjective(5.0 * x + 4.0 * y, ObjSense::Minimize);

    auto res = solveDual(m);
    REQUIRE(res.status == LPStatus::Optimal);
    CHECK_THAT(res.objectiveValue, WithinAbs(0.0, kTol));
}

TEST_CASE("dual simplex - objective value matches primal (5-var problem)", "[dual_simplex]") {
    // min 3x1 + 2x2 + x3 + 4x4 + 2x5
    // s.t. x1 + x2 + x3 >= 6
    //      x2 + x3 + x4 >= 4
    //      x3 + x4 + x5 >= 3
    //      xi >= 0
    Model m;
    auto x1 = m.addVar(0.0, kInf, "x1");
    auto x2 = m.addVar(0.0, kInf, "x2");
    auto x3 = m.addVar(0.0, kInf, "x3");
    auto x4 = m.addVar(0.0, kInf, "x4");
    auto x5 = m.addVar(0.0, kInf, "x5");
    m.addLPConstraint(1.0*x1 + 1.0*x2 + 1.0*x3,                    Sense::GreaterEq, 6.0);
    m.addLPConstraint(           1.0*x2 + 1.0*x3 + 1.0*x4,          Sense::GreaterEq, 4.0);
    m.addLPConstraint(                    1.0*x3 + 1.0*x4 + 1.0*x5, Sense::GreaterEq, 3.0);
    m.setObjective(3.0*x1 + 2.0*x2 + 1.0*x3 + 4.0*x4 + 2.0*x5, ObjSense::Minimize);

    auto primal = solve(m);
    auto dual   = solveDual(m);

    REQUIRE(primal.status == LPStatus::Optimal);
    REQUIRE(dual.status   == LPStatus::Optimal);
    CHECK_THAT(dual.objectiveValue, WithinAbs(primal.objectiveValue, kTol));
}

TEST_CASE("dual simplex - Upper-bound variable", "[dual_simplex]") {
    // min x,  0 <= x <= 5  → optimal x=0, obj=0
    Model m;
    auto x = m.addVar(0.0, 5.0, "x");
    m.setObjective(1.0 * x, ObjSense::Minimize);

    auto primal = solve(m);
    auto dual   = solveDual(m);

    REQUIRE(primal.status == LPStatus::Optimal);
    REQUIRE(dual.status   == LPStatus::Optimal);
    CHECK_THAT(dual.objectiveValue, WithinAbs(primal.objectiveValue, kTol));
    CHECK_THAT(dual.primalValues[0], WithinAbs(primal.primalValues[0], kTol));
}

// ── Tests: solveDualDetailed ──────────────────────────────────────────────────

TEST_CASE("dual simplex - solveDualDetailed primal matches solve", "[dual_simplex]") {
    auto detailed = solveDualDetailed(makeMinWithGEQ());
    auto simple   = solve(makeMinWithGEQ());

    REQUIRE(detailed.result.status == LPStatus::Optimal);
    CHECK_THAT(detailed.result.objectiveValue, WithinAbs(simple.objectiveValue, kTol));
    REQUIRE(detailed.result.primalValues.size() == simple.primalValues.size());
    for (std::size_t j = 0; j < simple.primalValues.size(); ++j)
        CHECK_THAT(detailed.result.primalValues[j],
                   WithinAbs(simple.primalValues[j], kTol));
}

TEST_CASE("dual simplex - solveDualDetailed dual values non-negative (GEQ min)", "[dual_simplex]") {
    auto det = solveDualDetailed(makeMinWithGEQ());
    REQUIRE(det.result.status == LPStatus::Optimal);
    REQUIRE(det.dualValues.size() == 2);
    // GEQ constraints, min problem → shadow prices >= 0
    for (double y : det.dualValues)
        CHECK(y >= -kTol);
}

TEST_CASE("dual simplex - solveDualDetailed reduced costs at optimum", "[dual_simplex]") {
    auto det = solveDualDetailed(makeMinWithGEQ());
    REQUIRE(det.result.status == LPStatus::Optimal);
    // Basic (non-zero) variables must have zero reduced cost
    for (std::size_t j = 0; j < det.result.primalValues.size(); ++j)
        if (det.result.primalValues[j] > kTol)
            CHECK_THAT(det.reducedCosts[j], WithinAbs(0.0, kTol));
}

TEST_CASE("dual simplex - fallback for Equal constraint gives correct result", "[dual_simplex]") {
    // Model with Equal constraint: dual simplex falls back to primal.
    // min x + y  s.t. x + y = 5,  x,y >= 0
    Model m;
    auto x = m.addVar(0.0, kInf, "x");
    auto y = m.addVar(0.0, kInf, "y");
    m.addLPConstraint(1.0 * x + 1.0 * y, Sense::Equal, 5.0);
    m.setObjective(1.0 * x + 1.0 * y, ObjSense::Minimize);

    auto res = solveDual(m);
    REQUIRE(res.status == LPStatus::Optimal);
    CHECK_THAT(res.objectiveValue, WithinAbs(5.0, kTol));
}

TEST_CASE("dual simplex - maxIter limit respected", "[dual_simplex]") {
    auto res = solveDual(makeMinWithGEQ(), /*maxIter=*/1);
    CHECK((res.status == LPStatus::MaxIter || res.status == LPStatus::Optimal));
}

TEST_CASE("dual simplex - solveDualDetailed result sub-object accessible", "[dual_simplex]") {
    LPDetailedResult det = solveDualDetailed(makeMinWithGEQ());
    const LPResult& res  = det.result;
    CHECK(res.status == LPStatus::Optimal);
    CHECK_THAT(res.objectiveValue, WithinAbs(det.result.objectiveValue, kTol));
}

TEST_CASE("dual simplex - strong duality: solveDual obj == solve obj", "[dual_simplex]") {
    // Verify strong duality through both solvers on a richer instance.
    // min 6x + 4y + 3z
    // s.t. 2x + y + z >= 4
    //      x + 2y + z >= 3
    //      x + y + 2z >= 2
    //      x,y,z >= 0
    Model m;
    auto x = m.addVar(0.0, kInf, "x");
    auto y = m.addVar(0.0, kInf, "y");
    auto z = m.addVar(0.0, kInf, "z");
    m.addLPConstraint(2.0*x + 1.0*y + 1.0*z, Sense::GreaterEq, 4.0);
    m.addLPConstraint(1.0*x + 2.0*y + 1.0*z, Sense::GreaterEq, 3.0);
    m.addLPConstraint(1.0*x + 1.0*y + 2.0*z, Sense::GreaterEq, 2.0);
    m.setObjective(6.0*x + 4.0*y + 3.0*z, ObjSense::Minimize);

    auto primal = solve(m);
    auto dual   = solveDual(m);

    REQUIRE(primal.status == LPStatus::Optimal);
    REQUIRE(dual.status   == LPStatus::Optimal);
    CHECK_THAT(dual.objectiveValue, WithinAbs(primal.objectiveValue, kTol));
}

TEST_CASE("dual simplex - timeLimitS = 0 returns TimeLimit", "[dual_simplex][timelimit]") {
    auto res = solveDual(makeMinWithGEQ(), 0, 0.0);
    REQUIRE(res.status == LPStatus::TimeLimit);
}

TEST_CASE("LP relaxation infeasible from B&B test", "[dual_simplex]") {
    Model m;
    Variable x = m.addVar(0.0, 10.0, VarType::Integer, "x");
    Variable y = m.addVar(0.0, 10.0, VarType::Integer, "y");
    m.addLPConstraint(1.0 * x + 1.0 * y, Sense::LessEq, -1.0);
    m.setObjective(1.0 * x + 1.0 * y, ObjSense::Minimize);

    LPResult lpResult = solveDual(m);

    REQUIRE(lpResult.status == LPStatus::Infeasible);
}