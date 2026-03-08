#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <stdexcept>

#include "baguette/model/Model.hpp"

using namespace baguette;
using Catch::Approx;

// ── addVar ────────────────────────────────────────────────────────────────────

TEST_CASE("Model starts empty", "[Model]") {
    Model m;
    REQUIRE(m.numVars() == 0);
    REQUIRE(m.numConstraints() == 0);
}

TEST_CASE("Model addVar assigns sequential IDs", "[Model]") {
    Model m;
    Variable x = m.addVar(0.0, 10.0);
    Variable y = m.addVar(0.0, 5.0);
    REQUIRE(x.id == 0);
    REQUIRE(y.id == 1);
    REQUIRE(m.numVars() == 2);
}

TEST_CASE("Model addVar stores bounds", "[Model]") {
    Model m;
    m.addVar(-3.0, 7.5);
    REQUIRE(m.getHot().lb[0] == Approx(-3.0));
    REQUIRE(m.getHot().ub[0] == Approx(7.5));
}

TEST_CASE("Model addVar stores type and label", "[Model]") {
    Model m;
    m.addVar(0.0, 1.0, VarType::Binary, "flag");
    REQUIRE(m.getCold().types[0]  == VarType::Binary);
    REQUIRE(m.getCold().labels[0] == "flag");
}

TEST_CASE("Model addVar initialises objective coefficient to zero", "[Model]") {
    Model m;
    m.addVar(0.0, 10.0);
    REQUIRE(m.getHot().obj[0] == Approx(0.0));
}

// ── addConstraint ─────────────────────────────────────────────────────────────

TEST_CASE("Model addConstraint stores sense and rhs", "[Model]") {
    Model m;
    Variable x = m.addVar(0.0, 10.0);
    Variable y = m.addVar(0.0, 10.0);

    m.addConstraint(2.0 * x + 3.0 * y, Sense::LessEq, 20.0);

    REQUIRE(m.numConstraints() == 1);
    const auto& c = m.getConstraints()[0];
    REQUIRE(c.sense == Sense::LessEq);
    REQUIRE(c.rhs   == Approx(20.0));
}

TEST_CASE("Model addConstraint stores lhs terms", "[Model]") {
    Model m;
    Variable x = m.addVar(0.0, 10.0);
    Variable y = m.addVar(0.0, 10.0);

    m.addConstraint(2.0 * x + 3.0 * y, Sense::Equal, 15.0);

    const auto& lhs = m.getConstraints()[0].lhs;
    REQUIRE(lhs.size() == 2);
    REQUIRE(lhs.coeffs[0] == Approx(2.0));  // x has id=0, sorted first
    REQUIRE(lhs.coeffs[1] == Approx(3.0));
}

TEST_CASE("Model addConstraint multiple constraints", "[Model]") {
    Model m;
    Variable x = m.addVar(0.0, 10.0);

    m.addConstraint(1.0 * x, Sense::GreaterEq, 1.0);
    m.addConstraint(1.0 * x, Sense::LessEq,    9.0);

    REQUIRE(m.numConstraints() == 2);
}

// ── setObjective ──────────────────────────────────────────────────────────────

TEST_CASE("Model setObjective sets dense coefficients", "[Model]") {
    Model m;
    Variable x = m.addVar(0.0, 10.0);
    Variable y = m.addVar(0.0, 10.0);

    m.setObjective(1.0 * x + 4.0 * y, ObjSense::Minimize);

    REQUIRE(m.getHot().obj[0] == Approx(1.0));
    REQUIRE(m.getHot().obj[1] == Approx(4.0));
    REQUIRE(m.getObjSense()   == ObjSense::Minimize);
}

TEST_CASE("Model setObjective resets previous objective", "[Model]") {
    Model m;
    Variable x = m.addVar(0.0, 10.0);
    Variable y = m.addVar(0.0, 10.0);

    m.setObjective(1.0 * x + 4.0 * y);
    m.setObjective(2.0 * y);              // x should now have coefficient 0

    REQUIRE(m.getHot().obj[0] == Approx(0.0));
    REQUIRE(m.getHot().obj[1] == Approx(2.0));
}

TEST_CASE("Model setObjective stores sense", "[Model]") {
    Model m;
    Variable x = m.addVar(0.0, 10.0);

    m.setObjective(1.0 * x, ObjSense::Maximize);
    REQUIRE(m.getObjSense() == ObjSense::Maximize);
}

TEST_CASE("Model setObjective throws on variable from empty model", "[Model]") {
    Model m1, m2;
    Variable x = m1.addVar(0.0, 10.0);   // x.id = 0, m1 has 1 var
    // m2 is empty: hot.obj.size() == 0, so x.id=0 is out of range
    REQUIRE_THROWS_AS(m2.setObjective(1.0 * x), std::out_of_range);
}

TEST_CASE("Model setObjective throws on out-of-range variable ID", "[Model]") {
    Model m;
    m.addVar(0.0, 1.0);   // only var id=0

    LinearExpr expr;
    expr.varIds = {5};    // id=5 does not exist
    expr.coeffs = {1.0};
    REQUIRE_THROWS_AS(m.setObjective(expr), std::out_of_range);
}
