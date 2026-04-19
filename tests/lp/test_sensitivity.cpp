#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <limits>
#include <vector>

#include "baguette/core/Sense.hpp"
#include "baguette/lp/LPResult.hpp"
#include "baguette/lp/LPSolver.hpp"
#include "baguette/model/Model.hpp"

using namespace baguette;
using Catch::Matchers::WithinAbs;

static constexpr double kTol = 1e-6;
static const double     kInf = std::numeric_limits<double>::infinity();

// ── Helpers ───────────────────────────────────────────────────────────────────

static bool isInfPos(double v) { return std::isinf(v) && v > 0.0; }
static bool isInfNeg(double v) { return std::isinf(v) && v < 0.0; }

// ── Test LPs ──────────────────────────────────────────────────────────────────

// LP-A (minimize, LessEq)
//   min  −x1 − 2·x2
//   s.t. x1 + x2 ≤ 4
//        x1 − x2 ≤ 2
//        x1, x2 ≥ 0
//
// Optimal: x1 = 0, x2 = 4, obj = −8.
// Basis: {x2 (row 0), s2 (row 1)}.
//
// Analytically derived sensitivity ranges:
//   rhsRange[0] = [0, +∞]     b0 can decrease to 0 (x2 → 0) or grow freely.
//   rhsRange[1] = [−4, +∞]    b1 can decrease to −4 or grow freely.
//   objRange[x1] = [−2, +∞]   rc[x1]=1; basis holds until c_x1 ≤ −2.
//   objRange[x2] = (−∞, −1]   basis holds until c_x2 ≥ −1 (then x1 enters).
static Model makeLPA() {
    Model m;
    auto x1 = m.addVar(0.0, kInf, "x1");
    auto x2 = m.addVar(0.0, kInf, "x2");
    m.addConstraint(1.0 * x1 + 1.0 * x2, Sense::LessEq, 4.0);
    m.addConstraint(1.0 * x1 - 1.0 * x2, Sense::LessEq, 2.0);
    m.setObjective(-1.0 * x1 - 2.0 * x2, ObjSense::Minimize);
    return m;
}

// LP-B (maximize, same constraints as LP-A)
//   max  x1 + 2·x2
//
// Same optimal point; factor for conversion flips the objective ranges:
//   objRange[x1] = (−∞, 2]    rc[x1]=1; basis holds until c_x1 ≥ 2.
//   objRange[x2] = [1, +∞]    basis holds until c_x2 ≤ 1.
static Model makeLPB() {
    Model m;
    auto x1 = m.addVar(0.0, kInf, "x1");
    auto x2 = m.addVar(0.0, kInf, "x2");
    m.addConstraint(1.0 * x1 + 1.0 * x2, Sense::LessEq, 4.0);
    m.addConstraint(1.0 * x1 - 1.0 * x2, Sense::LessEq, 2.0);
    m.setObjective(1.0 * x1 + 2.0 * x2, ObjSense::Maximize);
    return m;
}

// LP-C (minimize, GreaterEq + LessEq)
//   min  x1
//   s.t. x1 ≥ 3   (GreaterEq, not negated)
//        x1 ≤ 10  (LessEq)
//        x1 ≥ 0
//
// Optimal: x1 = 3, obj = 3.
// Basis: {x1 (row 1), s2 (row 1 — LessEq slack)}.
//
// Analytically derived sensitivity ranges:
//   rhsRange[0 (GEQ, b=3)] = [0, 10]   b can go down to 0 or up to 10 (ub).
//   rhsRange[1 (LE,  b=10)] = [3, +∞]  b can go down to 3 (= x1 value) or grow freely.
//   objRange[x1] = [0, +∞]             basis holds while c_x1 ≥ 0.
static Model makeLPC() {
    Model m;
    auto x1 = m.addVar(0.0, kInf, "x1");
    m.addConstraint(1.0 * x1, Sense::GreaterEq, 3.0);
    m.addConstraint(1.0 * x1, Sense::LessEq,   10.0);
    m.setObjective(1.0 * x1, ObjSense::Minimize);
    return m;
}

// LP-D (minimize, Equal constraint, primal phase-I path)
//   min  x1
//   s.t. x1 + x2 = 4   (Equal)
//        x1, x2 ≥ 0
//
// Optimal: x1 = 0, x2 = 4, obj = 0.
// Basis: {x2 (row 0)}.
//
// Analytically derived sensitivity ranges:
//   rhsRange[0 (EQ, b=4)] = [0, +∞]   RHS can decrease to 0 or grow freely.
//   objRange[x1] = [0, +∞]             rc[x1]=1; basis holds while c_x1 ≥ 0.
//   objRange[x2] = (−∞, 1]             basis holds while c_x2 ≤ 1.
static Model makeLPD() {
    Model m;
    auto x1 = m.addVar(0.0, kInf, "x1");
    auto x2 = m.addVar(0.0, kInf, "x2");
    m.addConstraint(1.0 * x1 + 1.0 * x2, Sense::Equal, 4.0);
    m.setObjective(1.0 * x1, ObjSense::Minimize);
    return m;
}

// LP-E (minimize, for dual-simplex path)
//   min  3·x1 + 2·x2
//   s.t. x1 + x2 ≥ 4   (GreaterEq — forces dual simplex to iterate)
//        x1 ≤ 5         (LessEq)
//        x2 ≤ 5         (LessEq)
//        x1, x2 ≥ 0
//
// All objective coefficients ≥ 0 → dual path works without fallback.
// Optimal: x1 = 0, x2 = 4, obj = 8.
// Basis: {x2 (row 0), s2 (row 1), s3 (row 2)}.
//
// Analytically derived sensitivity ranges:
//   rhsRange[0 (GEQ, b=4)] = [0, 5]    b can decrease to 0 or up to 5 (= x2 ub).
//   rhsRange[1 (LE,  b=5)] = [0, +∞]   s2 is basic at 5; can decrease to 0.
//   rhsRange[2 (LE,  b=5)] = [4, +∞]   s3 is basic at 1; can decrease to 4 (= x2).
//   objRange[x1] = [2, +∞]             rc[x1]=1; basis holds while c_x1 ≥ 2.
//   objRange[x2] = [0, 3]              dual ratio test gives [0, 3].
static Model makeLPE() {
    Model m;
    auto x1 = m.addVar(0.0, kInf, "x1");
    auto x2 = m.addVar(0.0, kInf, "x2");
    m.addConstraint(1.0 * x1 + 1.0 * x2, Sense::GreaterEq, 4.0);
    m.addConstraint(1.0 * x1,             Sense::LessEq,    5.0);
    m.addConstraint(1.0 * x2,             Sense::LessEq,    5.0);
    m.setObjective(3.0 * x1 + 2.0 * x2, ObjSense::Minimize);
    return m;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

// ── LP-A: primal path, LessEq constraints, minimize ──────────────────────────

TEST_CASE("Sensitivity LP-A primal: sizes and status") {
    Model m = makeLPA();
    LPDetailedResult res = solveDetailed(m, 0, kInf, SolverClock::now(), true);

    REQUIRE(res.result.status == LPStatus::Optimal);
    CHECK(res.sensitivity.rhsRange.size() == m.numConstraints());
    CHECK(res.sensitivity.objRange.size() == m.numVars());
}

TEST_CASE("Sensitivity LP-A primal: RHS ranging") {
    Model m = makeLPA();
    LPDetailedResult res = solveDetailed(m, 0, kInf, SolverClock::now(), true);
    REQUIRE(res.result.status == LPStatus::Optimal);

    const auto& rhs = res.sensitivity.rhsRange;

    // Constraint 0: x1+x2 ≤ 4.  Range [0, +∞].
    CHECK_THAT(rhs[0][0], WithinAbs(0.0, kTol));
    CHECK(isInfPos(rhs[0][1]));

    // Constraint 1: x1−x2 ≤ 2.  Range [−4, +∞].
    CHECK_THAT(rhs[1][0], WithinAbs(-4.0, kTol));
    CHECK(isInfPos(rhs[1][1]));
}

TEST_CASE("Sensitivity LP-A primal: objective ranging") {
    Model m = makeLPA();
    LPDetailedResult res = solveDetailed(m, 0, kInf, SolverClock::now(), true);
    REQUIRE(res.result.status == LPStatus::Optimal);

    const auto& obj = res.sensitivity.objRange;

    // x1 non-basic (rc = 1).  Range [−2, +∞].
    CHECK_THAT(obj[0][0], WithinAbs(-2.0, kTol));
    CHECK(isInfPos(obj[0][1]));

    // x2 basic (row 0).  Range (−∞, −1].
    CHECK(isInfNeg(obj[1][0]));
    CHECK_THAT(obj[1][1], WithinAbs(-1.0, kTol));
}

// ── LP-B: maximize, same geometry as LP-A ────────────────────────────────────

TEST_CASE("Sensitivity LP-B maximize: RHS ranging matches LP-A") {
    // Maximize flips the objective but leaves the basis and primal solution unchanged,
    // so RHS ranges are identical to LP-A.
    Model ma = makeLPA();
    Model mb = makeLPB();

    LPDetailedResult ra = solveDetailed(ma, 0, kInf, SolverClock::now(), true);
    LPDetailedResult rb = solveDetailed(mb, 0, kInf, SolverClock::now(), true);

    REQUIRE(ra.result.status == LPStatus::Optimal);
    REQUIRE(rb.result.status == LPStatus::Optimal);

    for (std::size_t i = 0; i < ma.numConstraints(); ++i) {
        // lo bounds
        if (std::isinf(ra.sensitivity.rhsRange[i][0]))
            CHECK(std::isinf(rb.sensitivity.rhsRange[i][0]));
        else
            CHECK_THAT(rb.sensitivity.rhsRange[i][0],
                       WithinAbs(ra.sensitivity.rhsRange[i][0], kTol));
        // hi bounds
        if (std::isinf(ra.sensitivity.rhsRange[i][1]))
            CHECK(std::isinf(rb.sensitivity.rhsRange[i][1]));
        else
            CHECK_THAT(rb.sensitivity.rhsRange[i][1],
                       WithinAbs(ra.sensitivity.rhsRange[i][1], kTol));
    }
}

TEST_CASE("Sensitivity LP-B maximize: objective ranging flipped") {
    Model m = makeLPB();
    LPDetailedResult res = solveDetailed(m, 0, kInf, SolverClock::now(), true);
    REQUIRE(res.result.status == LPStatus::Optimal);

    const auto& obj = res.sensitivity.objRange;

    // x1 non-basic.  Range (−∞, 2].
    CHECK(isInfNeg(obj[0][0]));
    CHECK_THAT(obj[0][1], WithinAbs(2.0, kTol));

    // x2 basic.  Range [1, +∞].
    CHECK_THAT(obj[1][0], WithinAbs(1.0, kTol));
    CHECK(isInfPos(obj[1][1]));
}

// ── LP-C: GreaterEq + LessEq constraints ─────────────────────────────────────

TEST_CASE("Sensitivity LP-C GreaterEq: RHS ranging") {
    Model m = makeLPC();
    LPDetailedResult res = solveDetailed(m, 0, kInf, SolverClock::now(), true);
    REQUIRE(res.result.status == LPStatus::Optimal);

    const auto& rhs = res.sensitivity.rhsRange;

    // Constraint 0: x1 ≥ 3.  Range [0, 10].
    CHECK_THAT(rhs[0][0], WithinAbs(0.0,  kTol));
    CHECK_THAT(rhs[0][1], WithinAbs(10.0, kTol));

    // Constraint 1: x1 ≤ 10.  Range [3, +∞].
    CHECK_THAT(rhs[1][0], WithinAbs(3.0, kTol));
    CHECK(isInfPos(rhs[1][1]));
}

TEST_CASE("Sensitivity LP-C GreaterEq: objective ranging") {
    Model m = makeLPC();
    LPDetailedResult res = solveDetailed(m, 0, kInf, SolverClock::now(), true);
    REQUIRE(res.result.status == LPStatus::Optimal);

    const auto& obj = res.sensitivity.objRange;

    // x1 basic.  Range [0, +∞].
    CHECK_THAT(obj[0][0], WithinAbs(0.0, kTol));
    CHECK(isInfPos(obj[0][1]));
}

// ── LP-D: Equal constraint (primal phase-I path) ──────────────────────────────

TEST_CASE("Sensitivity LP-D Equal: RHS ranging") {
    Model m = makeLPD();
    LPDetailedResult res = solveDetailed(m, 0, kInf, SolverClock::now(), true);
    REQUIRE(res.result.status == LPStatus::Optimal);

    const auto& rhs = res.sensitivity.rhsRange;
    REQUIRE(rhs.size() == 1);

    // Constraint 0: x1+x2 = 4.  Range [0, +∞].
    CHECK_THAT(rhs[0][0], WithinAbs(0.0, kTol));
    CHECK(isInfPos(rhs[0][1]));
}

TEST_CASE("Sensitivity LP-D Equal: objective ranging") {
    Model m = makeLPD();
    LPDetailedResult res = solveDetailed(m, 0, kInf, SolverClock::now(), true);
    REQUIRE(res.result.status == LPStatus::Optimal);

    const auto& obj = res.sensitivity.objRange;

    // x1 non-basic (rc = 1).  Range [0, +∞].
    CHECK_THAT(obj[0][0], WithinAbs(0.0, kTol));
    CHECK(isInfPos(obj[0][1]));

    // x2 basic.  Range (−∞, 1].
    CHECK(isInfNeg(obj[1][0]));
    CHECK_THAT(obj[1][1], WithinAbs(1.0, kTol));
}

// ── LP-E: dual-simplex path ───────────────────────────────────────────────────

TEST_CASE("Sensitivity LP-E dual path: RHS ranging") {
    Model m = makeLPE();
    LPDetailedResult res = solveDualDetailed(m, 0, kInf, SolverClock::now(), {}, true);
    REQUIRE(res.result.status == LPStatus::Optimal);

    const auto& rhs = res.sensitivity.rhsRange;
    REQUIRE(rhs.size() == 3);

    // Constraint 0: x1+x2 ≥ 4.  Range [0, 5].
    CHECK_THAT(rhs[0][0], WithinAbs(0.0, kTol));
    CHECK_THAT(rhs[0][1], WithinAbs(5.0, kTol));

    // Constraint 1: x1 ≤ 5.  Range [0, +∞].
    CHECK_THAT(rhs[1][0], WithinAbs(0.0, kTol));
    CHECK(isInfPos(rhs[1][1]));

    // Constraint 2: x2 ≤ 5.  Range [4, +∞].
    CHECK_THAT(rhs[2][0], WithinAbs(4.0, kTol));
    CHECK(isInfPos(rhs[2][1]));
}

TEST_CASE("Sensitivity LP-E dual path: objective ranging") {
    Model m = makeLPE();
    LPDetailedResult res = solveDualDetailed(m, 0, kInf, SolverClock::now(), {}, true);
    REQUIRE(res.result.status == LPStatus::Optimal);

    const auto& obj = res.sensitivity.objRange;

    // x1 non-basic (rc = 1).  Range [2, +∞].
    CHECK_THAT(obj[0][0], WithinAbs(2.0, kTol));
    CHECK(isInfPos(obj[0][1]));

    // x2 basic.  Range [0, 3].
    CHECK_THAT(obj[1][0], WithinAbs(0.0, kTol));
    CHECK_THAT(obj[1][1], WithinAbs(3.0, kTol));
}

// ── Consistency: solveDetailed and solveDualDetailed agree ───────────────────

TEST_CASE("Sensitivity LP-E: primal and dual paths agree") {
    Model m = makeLPE();
    LPDetailedResult rP = solveDetailed(m, 0, kInf, SolverClock::now(), true);
    LPDetailedResult rD = solveDualDetailed(m, 0, kInf, SolverClock::now(), {}, true);

    REQUIRE(rP.result.status == LPStatus::Optimal);
    REQUIRE(rD.result.status == LPStatus::Optimal);

    // Both paths must reach the same optimal solution (and thus the same basis).
    REQUIRE(rP.result.primalValues.size() == rD.result.primalValues.size());
    for (std::size_t j = 0; j < rP.result.primalValues.size(); ++j)
        CHECK_THAT(rP.result.primalValues[j],
                   WithinAbs(rD.result.primalValues[j], kTol));

    const auto& rhsP = rP.sensitivity.rhsRange;
    const auto& rhsD = rD.sensitivity.rhsRange;
    REQUIRE(rhsP.size() == rhsD.size());
    for (std::size_t i = 0; i < rhsP.size(); ++i) {
        for (int side = 0; side < 2; ++side) {
            if (std::isinf(rhsP[i][side]))
                CHECK(std::isinf(rhsD[i][side]));
            else
                CHECK_THAT(rhsD[i][side], WithinAbs(rhsP[i][side], kTol));
        }
    }

    const auto& objP = rP.sensitivity.objRange;
    const auto& objD = rD.sensitivity.objRange;
    REQUIRE(objP.size() == objD.size());
    for (std::size_t j = 0; j < objP.size(); ++j) {
        for (int side = 0; side < 2; ++side) {
            if (std::isinf(objP[j][side]))
                CHECK(std::isinf(objD[j][side]));
            else
                CHECK_THAT(objD[j][side], WithinAbs(objP[j][side], kTol));
        }
    }
}

// ── Non-optimal: sensitivity containers are empty ────────────────────────────

TEST_CASE("Sensitivity empty when status is Infeasible") {
    Model m;
    auto x = m.addVar(0.0, kInf, "x");
    m.addConstraint(1.0 * x, Sense::GreaterEq, 5.0);
    m.addConstraint(1.0 * x, Sense::LessEq,    3.0);
    m.setObjective(1.0 * x, ObjSense::Minimize);

    LPDetailedResult res = solveDualDetailed(m, 0, kInf, SolverClock::now(), {}, true);
    REQUIRE(res.result.status == LPStatus::Infeasible);
    CHECK(res.sensitivity.rhsRange.empty());
    CHECK(res.sensitivity.objRange.empty());
}

TEST_CASE("Sensitivity empty when status is Unbounded") {
    // min −x, x ≥ 0: unbounded below.
    Model m;
    auto x = m.addVar(0.0, kInf, "x");
    m.setObjective(-1.0 * x, ObjSense::Minimize);

    LPDetailedResult res = solveDetailed(m, 0, kInf, SolverClock::now(), true);
    REQUIRE(res.result.status == LPStatus::Unbounded);
    CHECK(res.sensitivity.rhsRange.empty());
    CHECK(res.sensitivity.objRange.empty());
}

// ── Range endpoints are consistent: lo ≤ current ≤ hi ───────────────────────

TEST_CASE("Sensitivity LP-A: current parameter values are inside their ranges") {
    Model m = makeLPA();
    LPDetailedResult res = solveDetailed(m, 0, kInf, SolverClock::now(), true);
    REQUIRE(res.result.status == LPStatus::Optimal);

    const auto& constraints = m.getConstraints();
    const auto& rhs         = res.sensitivity.rhsRange;

    for (std::size_t i = 0; i < constraints.size(); ++i) {
        CHECK(rhs[i][0] <= constraints[i].rhs + kTol);
        CHECK(rhs[i][1] >= constraints[i].rhs - kTol);
    }

    const auto& objCoeffs = m.getHot().obj;
    const auto& obj       = res.sensitivity.objRange;

    for (std::size_t j = 0; j < m.numVars(); ++j) {
        CHECK(obj[j][0] <= objCoeffs[j] + kTol);
        CHECK(obj[j][1] >= objCoeffs[j] - kTol);
    }
}

TEST_CASE("Sensitivity LP-E dual: current parameter values are inside their ranges") {
    Model m = makeLPE();
    LPDetailedResult res = solveDualDetailed(m, 0, kInf, SolverClock::now(), {}, true);
    REQUIRE(res.result.status == LPStatus::Optimal);

    const auto& constraints = m.getConstraints();
    const auto& rhs         = res.sensitivity.rhsRange;

    for (std::size_t i = 0; i < constraints.size(); ++i) {
        CHECK(rhs[i][0] <= constraints[i].rhs + kTol);
        CHECK(rhs[i][1] >= constraints[i].rhs - kTol);
    }

    const auto& objCoeffs = m.getHot().obj;
    const auto& obj       = res.sensitivity.objRange;

    for (std::size_t j = 0; j < m.numVars(); ++j) {
        CHECK(obj[j][0] <= objCoeffs[j] + kTol);
        CHECK(obj[j][1] >= objCoeffs[j] - kTol);
    }
}
