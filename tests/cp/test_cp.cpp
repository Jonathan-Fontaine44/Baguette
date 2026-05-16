#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "baguette/cp/CPConstraints.hpp"
#include "baguette/cp/constraints/AllDiff.hpp"
#include "baguette/model/Model.hpp"
#include "baguette/model/ModelEnums.hpp"

using namespace baguette;
using Catch::Matchers::WithinAbs;

static constexpr double kTol = 1e-9;

// ── Test 1: empty CPConstraints ───────────────────────────────────────────────
//
// An empty CPConstraints propagates to Feasible with no bound changes.

TEST_CASE("CP: empty CPConstraints is no-op", "[cp]") {
    Model m;
    Variable x = m.addVar(0.0, 5.0, VarType::Integer, "x");
    (void)x;

    CPConstraints cp;
    REQUIRE(cp.empty());

    PropagationResult r = propagateCP(cp, m);
    REQUIRE(r.status == CPStatus::Feasible);
    REQUIRE(r.changedVarIds.empty());
}

// ── D6: removeBuiltin disables propagation ────────────────────────────────────
//
// Add AllDiff(x=[1,1], y=[1,3]). With the constraint: x=1 forces y.lb→2.
// After removeBuiltin(0): no propagation, y stays [1,3].

TEST_CASE("CPConstraints: removeBuiltin disables its propagation", "[cp]") {
    Model m;
    Variable x = m.addVar(1.0, 1.0, VarType::Integer, "x");
    Variable y = m.addVar(1.0, 3.0, VarType::Integer, "y");

    CPConstraints cp;
    REQUIRE(cp.numBuiltins() == 0);
    cp.add(AllDiffConstraint{x, y});
    REQUIRE(cp.numBuiltins() == 1);

    // With constraint: x=1 forces y.lb to 2
    {
        Model mCopy = m;
        PropagationResult r = propagateCP(cp, mCopy);
        REQUIRE(r.status == CPStatus::Feasible);
        REQUIRE_THAT(mCopy.getHot().lb[y.id], WithinAbs(2.0, kTol));
    }

    // After removal: no propagation
    cp.removeBuiltin(0);
    REQUIRE(cp.numBuiltins() == 0);
    REQUIRE(cp.empty());

    PropagationResult r = propagateCP(cp, m);
    REQUIRE(r.status == CPStatus::Feasible);
    REQUIRE_THAT(m.getHot().lb[y.id], WithinAbs(1.0, kTol)); // y unchanged
}

// ── D6: updateBuiltin replaces constraint in place ────────────────────────────
//
// Add AllDiff(x=[2,2], y=[2,4]): x=2 forces y.lb→3.
// Replace with AllDiff(x) (single-var, trivially satisfied): y stays [2,4].

TEST_CASE("CPConstraints: updateBuiltin replaces constraint in place", "[cp]") {
    Model m;
    Variable x = m.addVar(2.0, 2.0, VarType::Integer, "x");
    Variable y = m.addVar(2.0, 4.0, VarType::Integer, "y");

    CPConstraints cp;
    cp.add(AllDiffConstraint{x, y});
    REQUIRE(cp.numBuiltins() == 1);

    // With AllDiff(x, y): x=2 forces y.lb→3
    {
        Model mCopy = m;
        PropagationResult r = propagateCP(cp, mCopy);
        REQUIRE_THAT(mCopy.getHot().lb[y.id], WithinAbs(3.0, kTol));
    }

    // Replace with trivial AllDiff(x): no propagation on y
    cp.updateBuiltin(0, AllDiffConstraint{x});
    REQUIRE(cp.numBuiltins() == 1); // still one constraint

    PropagationResult r = propagateCP(cp, m);
    REQUIRE(r.status == CPStatus::Feasible);
    REQUIRE_THAT(m.getHot().lb[y.id], WithinAbs(2.0, kTol)); // y unchanged
}
