#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "baguette/core/Variable.hpp"
#include "baguette/core/Domain.hpp"
#include "baguette/core/Sense.hpp"
#include "baguette/core/LinearExpr.hpp"

using namespace baguette;
using Catch::Approx;

// ── Variable ──────────────────────────────────────────────────────────────────

TEST_CASE("Variable equality", "[Variable]") {
    Variable a{0}, b{1}, c{0};
    REQUIRE(a == c);
    REQUIRE(a != b);
}

TEST_CASE("Variable ordering", "[Variable]") {
    Variable a{0}, b{1}, c{2};
    REQUIRE(a < b);
    REQUIRE(b < c);
    REQUIRE_FALSE(b < a);
}

// ── Domain ────────────────────────────────────────────────────────────────────

TEST_CASE("Domain isEmpty", "[Domain]") {
    REQUIRE_FALSE(Domain{0.0, 1.0}.isEmpty());
    REQUIRE_FALSE(Domain{3.0, 3.0}.isEmpty());
    REQUIRE(Domain{5.0, 2.0}.isEmpty());
}

TEST_CASE("Domain isFixed", "[Domain]") {
    REQUIRE(Domain{3.0, 3.0}.isFixed());
    REQUIRE_FALSE(Domain{0.0, 1.0}.isFixed());
    REQUIRE_FALSE(Domain{2.0, 1.0}.isFixed());
}

TEST_CASE("Domain contains", "[Domain]") {
    Domain d{1.0, 5.0};
    REQUIRE(d.contains(1.0));
    REQUIRE(d.contains(3.0));
    REQUIRE(d.contains(5.0));
    REQUIRE_FALSE(d.contains(0.9));
    REQUIRE_FALSE(d.contains(5.1));
}

// ── LinearExpr ────────────────────────────────────────────────────────────────

TEST_CASE("LinearExpr starts empty", "[LinearExpr]") {
    LinearExpr e;
    REQUIRE(e.empty());
    REQUIRE(e.size() == 0);
    REQUIRE(e.constant == Approx(0.0));
}

TEST_CASE("LinearExpr addTerm single", "[LinearExpr]") {
    LinearExpr e;
    e.addTerm(Variable{2}, 3.0);
    REQUIRE(e.size() == 1);
    REQUIRE(e.varIds[0] == 2);
    REQUIRE(e.coeffs[0] == Approx(3.0));
}

TEST_CASE("LinearExpr addTerm preserves sorted order", "[LinearExpr]") {
    LinearExpr e;
    e.addTerm(Variable{3}, 1.0);
    e.addTerm(Variable{1}, 2.0);
    e.addTerm(Variable{5}, 0.5);
    REQUIRE(e.size() == 3);
    REQUIRE(e.varIds[0] == 1);
    REQUIRE(e.varIds[1] == 3);
    REQUIRE(e.varIds[2] == 5);
}

TEST_CASE("LinearExpr addTerm merges same variable", "[LinearExpr]") {
    LinearExpr e;
    e.addTerm(Variable{1}, 2.0);
    e.addTerm(Variable{1}, 3.0);
    REQUIRE(e.size() == 1);
    REQUIRE(e.coeffs[0] == Approx(5.0));
}

TEST_CASE("LinearExpr addTerm removes cancelled term", "[LinearExpr]") {
    LinearExpr e;
    e.addTerm(Variable{1}, 2.0);
    e.addTerm(Variable{1}, -2.0);
    REQUIRE(e.empty());
}

TEST_CASE("LinearExpr scale", "[LinearExpr]") {
    LinearExpr e;
    e.addTerm(Variable{0}, 4.0);
    e.addTerm(Variable{1}, 2.0);
    e.constant = 1.0;
    e.scale(3.0);
    REQUIRE(e.coeffs[0] == Approx(12.0));
    REQUIRE(e.coeffs[1] == Approx(6.0));
    REQUIRE(e.constant == Approx(3.0));
}

TEST_CASE("LinearExpr operator* creates single term", "[LinearExpr]") {
    auto e = 2.5 * Variable{7};
    REQUIRE(e.size() == 1);
    REQUIRE(e.varIds[0] == 7);
    REQUIRE(e.coeffs[0] == Approx(2.5));
}

TEST_CASE("LinearExpr operator+ merges disjoint expressions", "[LinearExpr]") {
    auto a = 1.0 * Variable{0};
    auto b = 1.0 * Variable{2};
    auto c = a + b;
    REQUIRE(c.size() == 2);
    REQUIRE(c.varIds[0] == 0);
    REQUIRE(c.varIds[1] == 2);
}

TEST_CASE("LinearExpr operator+ merges common variable", "[LinearExpr]") {
    auto a = 3.0 * Variable{1};
    auto b = 2.0 * Variable{1};
    auto c = a + b;
    REQUIRE(c.size() == 1);
    REQUIRE(c.coeffs[0] == Approx(5.0));
}

TEST_CASE("LinearExpr operator+ cancels opposite coefficients", "[LinearExpr]") {
    auto a = 4.0 * Variable{1};
    auto b = -4.0 * Variable{1};
    auto c = a + b;
    REQUIRE(c.empty());
}

TEST_CASE("LinearExpr operator+ sums constants", "[LinearExpr]") {
    LinearExpr a, b;
    a.constant = 3.0;
    b.constant = 5.0;
    auto c = a + b;
    REQUIRE(c.constant == Approx(8.0));
}

TEST_CASE("LinearExpr operator+= merges disjoint expressions", "[LinearExpr]") {
    auto a = 1.0 * Variable{0};
    auto b = 1.0 * Variable{2};
    a += b;
    REQUIRE(a.size() == 2);
    REQUIRE(a.varIds[0] == 0);
    REQUIRE(a.varIds[1] == 2);
}

TEST_CASE("LinearExpr operator+= accumulates common variable", "[LinearExpr]") {
    auto a = 3.0 * Variable{1};
    a += 2.0 * Variable{1};
    REQUIRE(a.size() == 1);
    REQUIRE(a.coeffs[0] == Approx(5.0));
}

TEST_CASE("LinearExpr operator+= cancels opposite coefficients", "[LinearExpr]") {
    auto a = 4.0 * Variable{1};
    a += -4.0 * Variable{1};
    REQUIRE(a.empty());
}

TEST_CASE("LinearExpr operator+= sums constants", "[LinearExpr]") {
    LinearExpr a, b;
    a.constant = 2.0;
    b.constant = 7.0;
    a += b;
    REQUIRE(a.constant == Approx(9.0));
}

TEST_CASE("LinearExpr addTerm ignores zero coefficient", "[LinearExpr]") {
    LinearExpr e;
    e.addTerm(Variable{1}, 0.0);
    REQUIRE(e.empty());
}

TEST_CASE("LinearExpr operator* var * coeff", "[LinearExpr]") {
    auto e = Variable{4} * 3.0;
    REQUIRE(e.size() == 1);
    REQUIRE(e.varIds[0] == 4);
    REQUIRE(e.coeffs[0] == Approx(3.0));
}

TEST_CASE("Domain with floating-point bounds", "[Domain]") {
    Domain d{1.5, 2.5};
    REQUIRE(d.contains(1.5));
    REQUIRE(d.contains(2.0));
    REQUIRE(d.contains(2.5));
    REQUIRE_FALSE(d.contains(1.4));
    REQUIRE_FALSE(d.contains(2.6));
    REQUIRE_FALSE(d.isEmpty());
    REQUIRE_FALSE(d.isFixed());
}

// ── Sense ─────────────────────────────────────────────────────────────────────

TEST_CASE("Sense enum values are distinct", "[Sense]") {
    REQUIRE(Sense::LessEq    != Sense::Equal);
    REQUIRE(Sense::Equal     != Sense::GreaterEq);
    REQUIRE(Sense::LessEq    != Sense::GreaterEq);
}