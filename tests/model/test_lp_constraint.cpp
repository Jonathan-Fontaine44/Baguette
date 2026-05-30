#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "baguette/model/Model.hpp"
#include "baguette/lp/LPSolver.hpp"

using namespace baguette;
using Catch::Approx;

// ── Variable arithmetic operators ─────────────────────────────────────────────

TEST_CASE("Variable + Variable builds two-term LinearExpr", "[LPConstraint]") {
    Model m;
    Variable x = m.addVar(0.0, 10.0);
    Variable y = m.addVar(0.0, 10.0);

    LinearExpr e = x + y;
    REQUIRE(e.size() == 2);
    REQUIRE(e.coeffs[0] == Approx(1.0));
    REQUIRE(e.coeffs[1] == Approx(1.0));
}

TEST_CASE("Variable - Variable builds two-term LinearExpr", "[LPConstraint]") {
    Model m;
    Variable x = m.addVar(0.0, 10.0);
    Variable y = m.addVar(0.0, 10.0);

    LinearExpr e = x - y;
    REQUIRE(e.size() == 2);
    // x has id=0 (coeff +1), y has id=1 (coeff -1)
    REQUIRE(e.coeffs[0] == Approx( 1.0));
    REQUIRE(e.coeffs[1] == Approx(-1.0));
}

TEST_CASE("LinearExpr + Variable appends unit term", "[LPConstraint]") {
    Model m;
    Variable x = m.addVar(0.0, 10.0);
    Variable y = m.addVar(0.0, 10.0);

    LinearExpr e = 2.0 * x + y;
    REQUIRE(e.size() == 2);
    REQUIRE(e.coeffs[0] == Approx(2.0));
    REQUIRE(e.coeffs[1] == Approx(1.0));
}

TEST_CASE("LinearExpr - Variable subtracts unit term", "[LPConstraint]") {
    Model m;
    Variable x = m.addVar(0.0, 10.0);
    Variable y = m.addVar(0.0, 10.0);

    LinearExpr e = 3.0 * x - y;
    REQUIRE(e.size() == 2);
    REQUIRE(e.coeffs[0] == Approx( 3.0));
    REQUIRE(e.coeffs[1] == Approx(-1.0));
}

TEST_CASE("Variable + LinearExpr prepends unit term", "[LPConstraint]") {
    Model m;
    Variable x = m.addVar(0.0, 10.0);
    Variable y = m.addVar(0.0, 10.0);

    LinearExpr e = x + 2.0 * y;
    REQUIRE(e.size() == 2);
    REQUIRE(e.coeffs[0] == Approx(1.0));
    REQUIRE(e.coeffs[1] == Approx(2.0));
}

TEST_CASE("Variable - LinearExpr negates and prepends", "[LPConstraint]") {
    Model m;
    Variable x = m.addVar(0.0, 10.0);
    Variable y = m.addVar(0.0, 10.0);

    LinearExpr e = x - 2.0 * y;
    REQUIRE(e.size() == 2);
    REQUIRE(e.coeffs[0] == Approx( 1.0));
    REQUIRE(e.coeffs[1] == Approx(-2.0));
}

TEST_CASE("LinearExpr += Variable appends unit term in-place", "[LPConstraint]") {
    Model m;
    Variable x = m.addVar(0.0, 10.0);
    Variable y = m.addVar(0.0, 10.0);
    Variable z = m.addVar(0.0, 10.0);

    LinearExpr row;
    row += x;
    row += y;
    row += z;
    REQUIRE(row.size() == 3);
    for (double c : row.coeffs) REQUIRE(c == Approx(1.0));
}

TEST_CASE("LinearExpr -= Variable subtracts unit term in-place", "[LPConstraint]") {
    Model m;
    Variable x = m.addVar(0.0, 10.0);
    Variable y = m.addVar(0.0, 10.0);

    LinearExpr row = 1.0 * x;
    row -= y;
    REQUIRE(row.size() == 2);
    REQUIRE(row.coeffs[0] == Approx( 1.0));
    REQUIRE(row.coeffs[1] == Approx(-1.0));
}

// ── Constraint operators (scalar RHS) ────────────────────────────────────────

TEST_CASE("operator<= builds normalized LPConstraint", "[LPConstraint]") {
    Model m;
    Variable x = m.addVar(0.0, 10.0);
    Variable y = m.addVar(0.0, 10.0);

    LPConstraint c = x + y <= 5.0;
    REQUIRE(c.sense == Sense::LessEq);
    REQUIRE(c.rhsConst == Approx(5.0));
    REQUIRE(c.isNormalized());
    REQUIRE(c.lhs.size() == 2);
}

TEST_CASE("operator>= builds normalized LPConstraint", "[LPConstraint]") {
    Model m;
    Variable x = m.addVar(0.0, 10.0);

    LPConstraint c = 3.0 * x >= 1.0;
    REQUIRE(c.sense == Sense::GreaterEq);
    REQUIRE(c.rhsConst == Approx(1.0));
    REQUIRE(c.isNormalized());
}

TEST_CASE("operator== builds normalized LPConstraint", "[LPConstraint]") {
    Model m;
    Variable x = m.addVar(0.0, 10.0);

    LPConstraint c = 1.0 * x == 7.0;
    REQUIRE(c.sense == Sense::Equal);
    REQUIRE(c.rhsConst == Approx(7.0));
    REQUIRE(c.isNormalized());
}

TEST_CASE("Variable operator<= scalar builds LPConstraint", "[LPConstraint]") {
    Model m;
    Variable x = m.addVar(0.0, 10.0);

    LPConstraint c = x <= 4.0;
    REQUIRE(c.sense == Sense::LessEq);
    REQUIRE(c.rhsConst == Approx(4.0));
    REQUIRE(c.lhs.size() == 1);
    REQUIRE(c.lhs.coeffs[0] == Approx(1.0));
}

TEST_CASE("Variable operator>= scalar builds LPConstraint", "[LPConstraint]") {
    Model m;
    Variable x = m.addVar(0.0, 10.0);

    LPConstraint c = x >= 2.0;
    REQUIRE(c.sense == Sense::GreaterEq);
    REQUIRE(c.rhsConst == Approx(2.0));
}

TEST_CASE("Variable operator== scalar builds LPConstraint", "[LPConstraint]") {
    Model m;
    Variable x = m.addVar(0.0, 10.0);

    LPConstraint c = x == 3.0;
    REQUIRE(c.sense == Sense::Equal);
    REQUIRE(c.rhsConst == Approx(3.0));
}

// ── Constraint operators (two-sided) ─────────────────────────────────────────

TEST_CASE("operator<= with two LinearExpr is not normalized", "[LPConstraint]") {
    Model m;
    Variable x = m.addVar(0.0, 10.0);
    Variable y = m.addVar(0.0, 10.0);

    LPConstraint c = 1.0 * x + 1.0 * y <= 2.0 * y;
    REQUIRE_FALSE(c.isNormalized());
    REQUIRE(c.lhs.size() == 2);
    REQUIRE(c.rhs.size() == 1);
}

TEST_CASE("operator>= with two LinearExpr preserves sides", "[LPConstraint]") {
    Model m;
    Variable x = m.addVar(0.0, 10.0);
    Variable y = m.addVar(0.0, 10.0);

    LPConstraint c = 2.0 * x >= 1.0 * y;
    REQUIRE_FALSE(c.isNormalized());
    REQUIRE(c.sense == Sense::GreaterEq);
}

// ── normalize() ──────────────────────────────────────────────────────────────

TEST_CASE("normalize() moves rhs variables to lhs", "[LPConstraint]") {
    Model m;
    Variable x = m.addVar(0.0, 10.0);
    Variable y = m.addVar(0.0, 10.0);

    // x + y >= z + 3  →  x + y - z >= 3
    Variable z = m.addVar(0.0, 10.0);
    LPConstraint c  = (x + y) >= (1.0 * z);
    c.rhsConst = 3.0;

    LPConstraint n = c.normalize();
    REQUIRE(n.isNormalized());
    REQUIRE(n.sense == Sense::GreaterEq);
    REQUIRE(n.rhsConst == Approx(3.0));
    REQUIRE(n.lhs.size() == 3); // x, y, z
    // z should appear with coefficient -1
    auto it = std::find(n.lhs.varIds.begin(), n.lhs.varIds.end(), z.id);
    REQUIRE(it != n.lhs.varIds.end());
    std::size_t idx = static_cast<std::size_t>(it - n.lhs.varIds.begin());
    REQUIRE(n.lhs.coeffs[idx] == Approx(-1.0));
}

TEST_CASE("normalize() on already-normalized constraint is identity", "[LPConstraint]") {
    Model m;
    Variable x = m.addVar(0.0, 10.0);

    LPConstraint c = 2.0 * x <= 8.0;
    LPConstraint n = c.normalize();
    REQUIRE(n.isNormalized());
    REQUIRE(n.rhsConst == Approx(8.0));
    REQUIRE(n.lhs.size() == 1);
    REQUIRE(n.lhs.coeffs[0] == Approx(2.0));
}

// ── addLPConstraint with LPConstraint argument ────────────────────────────────

TEST_CASE("addLPConstraint(LPConstraint) returns valid ConstraintId", "[LPConstraint]") {
    Model m;
    Variable x = m.addVar(0.0, 10.0);
    Variable y = m.addVar(0.0, 10.0);

    ConstraintId id = m.addLPConstraint(x + y <= 5.0);
    REQUIRE(id == 0);
    REQUIRE(m.numConstraints() == 1);

    ConstraintId id2 = m.addLPConstraint(x >= 1.0);
    REQUIRE(id2 == 1);
    REQUIRE(m.numConstraints() == 2);
}

TEST_CASE("getLPConstraints() returns normalized form", "[LPConstraint]") {
    Model m;
    Variable x = m.addVar(0.0, 10.0);
    Variable y = m.addVar(0.0, 10.0);

    // Two-sided: x + y >= 2*y  →  normalized: x - y >= 0
    m.addLPConstraint((x + y) >= 2.0 * y);

    const auto& norm = m.getLPConstraints();
    REQUIRE(norm.size() == 1);
    REQUIRE(norm[0].isNormalized());
    REQUIRE(norm[0].sense == Sense::GreaterEq);
    REQUIRE(norm[0].rhsConst == Approx(0.0));
    REQUIRE(norm[0].lhs.size() == 2); // x (coeff +1) and y (coeff -1)
}

TEST_CASE("getLPConstraint(id) returns original two-sided form", "[LPConstraint]") {
    Model m;
    Variable x = m.addVar(0.0, 10.0);
    Variable y = m.addVar(0.0, 10.0);

    // Two-sided: 2x >= y + 3
    LPConstraint original = 2.0 * x >= 1.0 * y;
    original.rhsConst = 3.0;
    ConstraintId id = m.addLPConstraint(original);

    LPConstraint retrieved = m.getLPConstraint(id);
    REQUIRE_FALSE(retrieved.isNormalized());        // still two-sided
    REQUIRE(retrieved.lhs.size() == 1);             // only x on lhs
    REQUIRE(retrieved.rhs.size() == 1);             // only y on rhs
    REQUIRE(retrieved.rhsConst == Approx(3.0));
    REQUIRE(retrieved.sense == Sense::GreaterEq);
}

TEST_CASE("getLPConstraint(id) for scalar-rhs constraint returns same form", "[LPConstraint]") {
    Model m;
    Variable x = m.addVar(0.0, 10.0);

    ConstraintId id = m.addLPConstraint(3.0 * x <= 9.0);
    LPConstraint c  = m.getLPConstraint(id);
    REQUIRE(c.isNormalized());
    REQUIRE(c.sense == Sense::LessEq);
    REQUIRE(c.rhsConst == Approx(9.0));
}

TEST_CASE("getLPConstraint(id) throws on invalid id", "[LPConstraint]") {
    Model m;
    REQUIRE_THROWS_AS(m.getLPConstraint(0), std::out_of_range);
}

// ── Natural modeling style with new operators ─────────────────────────────────

TEST_CASE("Model built with new operators solves correctly", "[LPConstraint]") {
    // max x + y  s.t. x + y <= 4, x <= 3, y <= 3
    Model m;
    Variable x = m.addVar(0.0, 5.0);
    Variable y = m.addVar(0.0, 5.0);

    m.addLPConstraint(x + y <= 4.0);
    m.addLPConstraint(x     <= 3.0);
    m.addLPConstraint(y     <= 3.0);
    m.setObjective(x + y, ObjSense::Maximize);

    auto result = solveLP(m);
    REQUIRE(result.status == LPStatus::Optimal);
    REQUIRE(result.objectiveValue == Approx(4.0));
}

TEST_CASE("Model built with loop += style solves correctly", "[LPConstraint]") {
    // min x0 + x1 + x2  s.t. x0 + x1 + x2 >= 1, each xi in [0,1]
    Model m;
    std::vector<Variable> x(3);
    for (auto& xi : x) xi = m.addVar(0.0, 1.0);

    LinearExpr sum;
    for (auto xi : x) sum += xi;
    m.addLPConstraint(sum >= 1.0);
    m.setObjective(sum, ObjSense::Minimize);

    auto result = solveLP(m);
    REQUIRE(result.status == LPStatus::Optimal);
    REQUIRE(result.objectiveValue == Approx(1.0));
}
