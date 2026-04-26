#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

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

// ── Helper ────────────────────────────────────────────────────────────────────

/// Verify the Farkas property for the original model:
///   A_model^T y >= -tol  (component-wise, one entry per variable)
///   b_model^T y < -tol   (strictly negative)
static bool checkFarkasProperty(const FarkasRay& ray, const Model& model,
                                 double tol = 1e-6) {
    if (ray.y.empty()) return false;
    const auto& constraints = model.getLPConstraints();
    if (ray.y.size() != constraints.size()) return false;

    const std::size_t nVars = model.numVars();
    const auto& lb = model.getHot().lb;

    // A^T y (one entry per variable)
    std::vector<double> ATy(nVars, 0.0);
    // b^T y using the lb-shifted RHS:  b_shifted[i] = rhs[i] - sum_j A[i,j]*lb[j]
    // This matches the standard-form b that the solver actually used.
    double bShiftedTy = 0.0;

    for (std::size_t i = 0; i < constraints.size(); ++i) {
        double yi = ray.y[i];
        double bShifted = constraints[i].rhs;
        for (std::size_t k = 0; k < constraints[i].lhs.varIds.size(); ++k) {
            uint32_t  j   = constraints[i].lhs.varIds[k];
            double    aij = constraints[i].lhs.coeffs[k];
            ATy[j]       += yi * aij;
            bShifted     -= aij * lb[j]; // subtract lb-shift contribution
        }
        bShiftedTy += yi * bShifted;
    }

    for (double v : ATy)
        if (v < -tol) return false;

    return bShiftedTy < -tol; // must be strictly negative
}

// ── Test models ───────────────────────────────────────────────────────────────

/// x >= 3, x <= 2  (infeasible by dual simplex)
static Model makeSimpleInfeasibleDual() {
    Model m;
    auto x = m.addVar(0.0, kInf, "x");
    m.addLPConstraint(1.0 * x, Sense::GreaterEq, 3.0);
    m.addLPConstraint(1.0 * x, Sense::LessEq,    2.0);
    m.setObjective(1.0 * x, ObjSense::Minimize);
    return m;
}

/// x + y >= 10, x <= 3, y <= 3  (infeasible — sum at most 6)
static Model makeMultiVarInfeasible() {
    Model m;
    auto x = m.addVar(0.0, kInf, "x");
    auto y = m.addVar(0.0, kInf, "y");
    m.addLPConstraint(1.0 * x + 1.0 * y, Sense::GreaterEq, 10.0);
    m.addLPConstraint(1.0 * x,            Sense::LessEq,     3.0);
    m.addLPConstraint(1.0 * y,            Sense::LessEq,     3.0);
    m.setObjective(1.0 * x + 1.0 * y, ObjSense::Minimize);
    return m;
}

/// x + y = 10, x <= 3, y <= 3  (infeasible — sum at most 6, forces primal phase-I)
static Model makeEqualConstraintInfeasible() {
    Model m;
    auto x = m.addVar(0.0, kInf, "x");
    auto y = m.addVar(0.0, kInf, "y");
    m.addLPConstraint(1.0 * x + 1.0 * y, Sense::Equal,   10.0);
    m.addLPConstraint(1.0 * x,            Sense::LessEq,   3.0);
    m.addLPConstraint(1.0 * y,            Sense::LessEq,   3.0);
    m.setObjective(1.0 * x, ObjSense::Minimize);
    return m;
}

/// Feasible reference LP for warm-start tests.
/// min -x1 - x2  s.t.  2x1+x2 <= 4, x1+2x2 <= 4,  x1 in [0,3], x2 in [0,3]
static Model makeFeasibleLP() {
    Model m;
    auto x1 = m.addVar(0.0, 3.0, "x1");
    auto x2 = m.addVar(0.0, 3.0, "x2");
    m.addLPConstraint(2.0 * x1 + 1.0 * x2, Sense::LessEq, 4.0);
    m.addLPConstraint(1.0 * x1 + 2.0 * x2, Sense::LessEq, 4.0);
    m.setObjective(-1.0 * x1 + -1.0 * x2, ObjSense::Minimize);
    return m;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

TEST_CASE("Farkas - dual simplex infeasible: y populated and property holds") {
    Model m = makeSimpleInfeasibleDual();
    LPDetailedResult res = solveDualDetailed(m);

    REQUIRE(res.result.status == LPStatus::Infeasible);
    // Tableau certificate: y non-empty, infeasVarId unset
    REQUIRE_FALSE(res.farkas.y.empty());
    REQUIRE(res.farkas.infeasVarId == -1);
    REQUIRE(res.farkas.y.size() == 2);

    // Mathematical property: A^T y >= 0, b^T y < 0
    CHECK(checkFarkasProperty(res.farkas, m));

    // Exact values for this simple problem (derived analytically):
    //   y[0] = -1 (GEQ x >= 3), y[1] = 1 (LessEq x <= 2)
    //   b^T y = 3*(-1) + 2*1 = -1 < 0
    double bTy = res.farkas.y[0] * 3.0 + res.farkas.y[1] * 2.0;
    CHECK(bTy < 0.0);
}

TEST_CASE("Farkas - dual simplex infeasible multi-var: property holds") {
    Model m = makeMultiVarInfeasible();
    LPDetailedResult res = solveDualDetailed(m);

    REQUIRE(res.result.status == LPStatus::Infeasible);
    REQUIRE_FALSE(res.farkas.y.empty());
    REQUIRE(res.farkas.infeasVarId == -1);
    REQUIRE(res.farkas.y.size() == 3);

    CHECK(checkFarkasProperty(res.farkas, m));
}

TEST_CASE("Farkas - primal phase-I infeasible (Equal constraint): property holds") {
    Model m = makeEqualConstraintInfeasible();
    LPDetailedResult res = solveDetailed(m);

    REQUIRE(res.result.status == LPStatus::Infeasible);
    REQUIRE_FALSE(res.farkas.y.empty());
    REQUIRE(res.farkas.infeasVarId == -1);
    REQUIRE(res.farkas.y.size() == 3);

    CHECK(checkFarkasProperty(res.farkas, m));
}

TEST_CASE("Farkas - no ray when status is Optimal") {
    // x + y >= 4, x <= 5, y <= 5 — feasible
    Model m;
    auto x = m.addVar(0.0, kInf, "x");
    auto y = m.addVar(0.0, kInf, "y");
    m.addLPConstraint(1.0 * x + 1.0 * y, Sense::GreaterEq, 4.0);
    m.addLPConstraint(1.0 * x,            Sense::LessEq,    5.0);
    m.addLPConstraint(1.0 * y,            Sense::LessEq,    5.0);
    m.setObjective(1.0 * x + 1.0 * y, ObjSense::Minimize);

    LPDetailedResult res = solveDualDetailed(m);

    REQUIRE(res.result.status == LPStatus::Optimal);
    CHECK(res.farkas.y.empty());
    CHECK(res.farkas.infeasVarId == -1);
}

TEST_CASE("Farkas - early lb > ub: infeasVarId set, y empty") {
    // addVar validates lb <= ub, so we start with valid bounds and
    // use withVarBounds (no validation) to create the infeasible child.
    Model root;
    auto x = root.addVar(0.0, 5.0, "x");
    root.setObjective(1.0 * x, ObjSense::Minimize);

    // Force lb > ub — both solvers detect this before building the tableau
    Model m = root.withVarBounds(x, 5.0, 3.0); // lb=5 > ub=3

    {
        LPDetailedResult res = solveDetailed(m);
        REQUIRE(res.result.status == LPStatus::Infeasible);
        CHECK(res.farkas.y.empty());
        CHECK(res.farkas.infeasVarId == static_cast<int32_t>(x.id));
    }
    {
        LPDetailedResult res = solveDualDetailed(m);
        REQUIRE(res.result.status == LPStatus::Infeasible);
        CHECK(res.farkas.y.empty());
        CHECK(res.farkas.infeasVarId == static_cast<int32_t>(x.id));
    }
}

TEST_CASE("Farkas - B&B warm-start infeasible child: tableau-based certificate") {
    // Root LP: min -x1-x2 s.t. 2x1+x2<=4, x1+2x2<=4, x1 in [0,3], x2 in [0,3]
    // Solve root to get a BasisRecord, then create an infeasible child:
    //   x1 in [2,3], x2 in [2,3] → 2*2+2=6 > 4 → constraints violated → infeasible
    Model root_m;
    auto x1 = root_m.addVar(0.0, 3.0, "x1");
    auto x2 = root_m.addVar(0.0, 3.0, "x2");
    root_m.addLPConstraint(2.0 * x1 + 1.0 * x2, Sense::LessEq, 4.0);
    root_m.addLPConstraint(1.0 * x1 + 2.0 * x2, Sense::LessEq, 4.0);
    root_m.setObjective(-1.0 * x1 + -1.0 * x2, ObjSense::Minimize);

    LPDetailedResult rootRes = solveDualDetailed(root_m);
    REQUIRE(rootRes.result.status == LPStatus::Optimal);

    // Child: force x1 >= 2 and x2 >= 2 (both have ub=3 already)
    Model child_m = root_m.withVarBounds(x1, 2.0, 3.0)
                          .withVarBounds(x2, 2.0, 3.0);
    LPDetailedResult childRes = solveDualDetailed(child_m, 0, kInf,
                                                   SolverClock::now(),
                                                   rootRes.basis);

    REQUIRE(childRes.result.status == LPStatus::Infeasible);
    // Must be a tableau certificate (no lb > ub violation, bounds are compatible)
    CHECK(childRes.farkas.infeasVarId == -1);
    REQUIRE_FALSE(childRes.farkas.y.empty());
    CHECK(childRes.farkas.y.size() == root_m.numConstraints());

    CHECK(checkFarkasProperty(childRes.farkas, child_m));
}

TEST_CASE("Farkas - solveDetailed and solveDualDetailed agree on dual-path problem") {
    // For a problem with only LessEq/GEQ constraints, both solvers return
    // Infeasible with a valid Farkas certificate.
    Model m = makeMultiVarInfeasible();

    LPDetailedResult resDual   = solveDualDetailed(m);
    LPDetailedResult resPrimal = solveDetailed(m);

    REQUIRE(resDual.result.status   == LPStatus::Infeasible);
    REQUIRE(resPrimal.result.status == LPStatus::Infeasible);

    CHECK(checkFarkasProperty(resDual.farkas,   m));
    CHECK(checkFarkasProperty(resPrimal.farkas, m));
}
