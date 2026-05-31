#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <limits>

#include "baguette/core/Sense.hpp"
#include "baguette/lp/LPResult.hpp"
#include "baguette/lp/LPSolver.hpp"
#include "baguette/model/Model.hpp"

using namespace baguette;
using Catch::Matchers::WithinAbs;

static constexpr double kTol = 1e-6;
static const double     kInf = std::numeric_limits<double>::infinity();

static LPOptions revisedOpts() {
    LPOptions o;
    o.method = LPMethod::RevisedSimplex;
    o.enablePresolve = false;
    return o;
}

// ── Optimal LPs ──────────────────────────────────────────────────────────────

TEST_CASE("RevisedSimplex: min x+y s.t. x+y>=4, x<=5, y<=5", "[revised]") {
    Model m;
    auto x = m.addVar(0.0, kInf, "x");
    auto y = m.addVar(0.0, kInf, "y");
    m.addLPConstraint(1.0 * x + 1.0 * y, Sense::GreaterEq, 4.0);
    m.addLPConstraint(1.0 * x,            Sense::LessEq,    5.0);
    m.addLPConstraint(1.0 * y,            Sense::LessEq,    5.0);
    m.setObjective(1.0 * x + 1.0 * y, ObjSense::Minimize);

    LPDetailedResult r = solveLPDetailed(m, revisedOpts());

    REQUIRE(r.result.status == LPStatus::Optimal);
    REQUIRE_THAT(r.result.objectiveValue, WithinAbs(4.0, kTol));
    // x+y = 4 with both >=0: many optimal vertices; just verify obj value
}

TEST_CASE("RevisedSimplex: max 5x+4y s.t. 6x+4y<=24, x+2y<=6, x,y>=0", "[revised]") {
    Model m;
    auto x = m.addVar(0.0, kInf, "x");
    auto y = m.addVar(0.0, kInf, "y");
    m.addLPConstraint(6.0 * x + 4.0 * y, Sense::LessEq, 24.0);
    m.addLPConstraint(1.0 * x + 2.0 * y, Sense::LessEq,  6.0);
    m.setObjective(5.0 * x + 4.0 * y, ObjSense::Maximize);

    LPDetailedResult r = solveLPDetailed(m, revisedOpts());

    REQUIRE(r.result.status == LPStatus::Optimal);
    REQUIRE_THAT(r.result.objectiveValue, WithinAbs(21.0, kTol));
    REQUIRE_THAT(r.result.primalValues[x.id], WithinAbs(3.0, kTol));
    REQUIRE_THAT(r.result.primalValues[y.id], WithinAbs(1.5, kTol));
}

TEST_CASE("RevisedSimplex: single variable min -x, x<=3", "[revised]") {
    Model m;
    auto x = m.addVar(0.0, 3.0, "x");
    m.setObjective(-1.0 * x, ObjSense::Minimize);

    LPResult r = solveLP(m, revisedOpts());

    REQUIRE(r.status == LPStatus::Optimal);
    REQUIRE_THAT(r.objectiveValue,    WithinAbs(-3.0, kTol));
    REQUIRE_THAT(r.primalValues[x.id], WithinAbs(3.0, kTol));
}

TEST_CASE("RevisedSimplex: three-variable LP", "[revised]") {
    // min -x - 2y - 3z  s.t. x+y+z<=10, x<=6, y<=6, z<=6
    // Optimal: z=6, y=4, x=0 (or similar), obj = -0-8-18 = -26
    Model m;
    auto x = m.addVar(0.0, 6.0, "x");
    auto y = m.addVar(0.0, 6.0, "y");
    auto z = m.addVar(0.0, 6.0, "z");
    m.addLPConstraint(1.0*x + 1.0*y + 1.0*z, Sense::LessEq, 10.0);
    m.setObjective(-1.0*x + -2.0*y + -3.0*z, ObjSense::Minimize);

    LPDetailedResult r = solveLPDetailed(m, revisedOpts());

    REQUIRE(r.result.status == LPStatus::Optimal);
    REQUIRE_THAT(r.result.objectiveValue, WithinAbs(-26.0, kTol));
    REQUIRE_THAT(r.result.primalValues[z.id], WithinAbs(6.0, kTol));
}

// ── Results match PrimalSimplex ───────────────────────────────────────────────

TEST_CASE("RevisedSimplex: agrees with PrimalSimplex on all results", "[revised]") {
    Model m;
    auto x = m.addVar(0.0, kInf, "x");
    auto y = m.addVar(0.0, kInf, "y");
    m.addLPConstraint(2.0*x + 1.0*y, Sense::LessEq, 14.0);
    m.addLPConstraint(1.0*x + 2.0*y, Sense::LessEq, 14.0);
    m.addLPConstraint(1.0*x + 1.0*y, Sense::GreaterEq, 4.0);
    m.setObjective(-3.0*x + -5.0*y, ObjSense::Minimize);

    LPOptions primalOpts;
    primalOpts.method = LPMethod::PrimalSimplex;
    primalOpts.enablePresolve = false;
    LPDetailedResult rP = solveLPDetailed(m, primalOpts);
    LPDetailedResult rR = solveLPDetailed(m, revisedOpts());

    REQUIRE(rP.result.status == LPStatus::Optimal);
    REQUIRE(rR.result.status == LPStatus::Optimal);
    REQUIRE_THAT(rR.result.objectiveValue,
                 WithinAbs(rP.result.objectiveValue, kTol));
    REQUIRE_THAT(rR.result.primalValues[x.id],
                 WithinAbs(rP.result.primalValues[x.id], kTol));
    REQUIRE_THAT(rR.result.primalValues[y.id],
                 WithinAbs(rP.result.primalValues[y.id], kTol));
    // Dual values
    REQUIRE_THAT(rR.dualValues[0], WithinAbs(rP.dualValues[0], kTol));
    REQUIRE_THAT(rR.dualValues[1], WithinAbs(rP.dualValues[1], kTol));
    REQUIRE_THAT(rR.dualValues[2], WithinAbs(rP.dualValues[2], kTol));
}

// ── Infeasible ────────────────────────────────────────────────────────────────

TEST_CASE("RevisedSimplex: LP infeasible by constraints", "[revised]") {
    Model m;
    auto x = m.addVar(0.0, kInf, "x");
    m.addLPConstraint(1.0 * x, Sense::GreaterEq, 5.0);
    m.addLPConstraint(1.0 * x, Sense::LessEq,    3.0);
    m.setObjective(1.0 * x, ObjSense::Minimize);

    LPResult r = solveLP(m, revisedOpts());

    REQUIRE(r.status == LPStatus::Infeasible);
    REQUIRE(r.primalValues.empty());
}

TEST_CASE("RevisedSimplex: LP infeasible by lb>ub", "[revised]") {
    Model root;
    auto x = root.addVar(0.0, 5.0, "x");
    root.setObjective(1.0 * x, ObjSense::Minimize);

    Model child = root.withVarBounds(x, 5.0, 3.0); // lb > ub

    LPDetailedResult r = solveLPDetailed(child, revisedOpts());

    REQUIRE(r.result.status == LPStatus::Infeasible);
    CHECK(r.farkas.infeasVarId == static_cast<int32_t>(x.id));
}

TEST_CASE("RevisedSimplex: Farkas certificate populated for infeasible LP", "[revised]") {
    // x+y >= 10, x<=3, y<=3 - infeasible (sum at most 6)
    Model m;
    auto x = m.addVar(0.0, kInf, "x");
    auto y = m.addVar(0.0, kInf, "y");
    m.addLPConstraint(1.0*x + 1.0*y, Sense::GreaterEq, 10.0);
    m.addLPConstraint(1.0*x,          Sense::LessEq,     3.0);
    m.addLPConstraint(1.0*y,          Sense::LessEq,     3.0);
    m.setObjective(1.0*x + 1.0*y, ObjSense::Minimize);

    LPDetailedResult r = solveLPDetailed(m, revisedOpts());

    REQUIRE(r.result.status == LPStatus::Infeasible);
    CHECK_FALSE(r.farkas.y.empty());
    CHECK(r.farkas.infeasVarId == -1);
}

// ── Unbounded ─────────────────────────────────────────────────────────────────

TEST_CASE("RevisedSimplex: LP unbounded", "[revised]") {
    Model m;
    auto x = m.addVar(0.0, kInf, "x");
    m.setObjective(-1.0 * x, ObjSense::Minimize); // min -x with x unbounded above

    LPResult r = solveLP(m, revisedOpts());

    REQUIRE(r.status == LPStatus::Unbounded);
}

// ── Sensitivity ───────────────────────────────────────────────────────────────

TEST_CASE("RevisedSimplex: sensitivity agrees with PrimalSimplex", "[revised]") {
    Model m;
    auto x = m.addVar(0.0, kInf, "x");
    auto y = m.addVar(0.0, kInf, "y");
    m.addLPConstraint(1.0*x + 2.0*y, Sense::LessEq, 8.0);
    m.addLPConstraint(2.0*x + 1.0*y, Sense::LessEq, 8.0);
    m.setObjective(-3.0*x + -5.0*y, ObjSense::Minimize);

    LPOptions primalOpts;
    primalOpts.method             = LPMethod::PrimalSimplex;
    primalOpts.computeSensitivity = true;
    primalOpts.enablePresolve     = false;

    LPOptions revOpts             = revisedOpts();
    revOpts.computeSensitivity    = true;

    LPDetailedResult rP = solveLPDetailed(m, primalOpts);
    LPDetailedResult rR = solveLPDetailed(m, revOpts);

    REQUIRE(rP.result.status == LPStatus::Optimal);
    REQUIRE(rR.result.status == LPStatus::Optimal);

    REQUIRE_THAT(rR.result.objectiveValue,
                 WithinAbs(rP.result.objectiveValue, kTol));

    REQUIRE(rR.sensitivity.rhsRange.size() == rP.sensitivity.rhsRange.size());
    for (std::size_t i = 0; i < rP.sensitivity.rhsRange.size(); ++i) {
        if (!std::isinf(rP.sensitivity.rhsRange[i][0]))
            REQUIRE_THAT(rR.sensitivity.rhsRange[i][0],
                         WithinAbs(rP.sensitivity.rhsRange[i][0], kTol));
        if (!std::isinf(rP.sensitivity.rhsRange[i][1]))
            REQUIRE_THAT(rR.sensitivity.rhsRange[i][1],
                         WithinAbs(rP.sensitivity.rhsRange[i][1], kTol));
    }
    REQUIRE(rR.sensitivity.objRange.size() == rP.sensitivity.objRange.size());
    for (std::size_t j = 0; j < rP.sensitivity.objRange.size(); ++j) {
        if (!std::isinf(rP.sensitivity.objRange[j][0]))
            REQUIRE_THAT(rR.sensitivity.objRange[j][0],
                         WithinAbs(rP.sensitivity.objRange[j][0], kTol));
        if (!std::isinf(rP.sensitivity.objRange[j][1]))
            REQUIRE_THAT(rR.sensitivity.objRange[j][1],
                         WithinAbs(rP.sensitivity.objRange[j][1], kTol));
    }
}

// ── Equal constraints ─────────────────────────────────────────────────────────

TEST_CASE("RevisedSimplex: Equal constraint", "[revised]") {
    // min x+y  s.t. x+y=5, x<=4, y<=4
    // Optimal: any (x,y) with x+y=5, 1<=x<=4, 1<=y<=4. Obj=5.
    Model m;
    auto x = m.addVar(0.0, 4.0, "x");
    auto y = m.addVar(0.0, 4.0, "y");
    m.addLPConstraint(1.0*x + 1.0*y, Sense::Equal,   5.0);
    m.setObjective(1.0*x + 1.0*y, ObjSense::Minimize);

    LPResult r = solveLP(m, revisedOpts());

    REQUIRE(r.status == LPStatus::Optimal);
    REQUIRE_THAT(r.objectiveValue, WithinAbs(5.0, kTol));
}
