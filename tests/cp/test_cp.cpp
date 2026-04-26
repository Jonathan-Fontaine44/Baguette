#include <catch2/catch_test_macros.hpp>

#include "baguette/cp/CPConstraints.hpp"
#include "baguette/model/Model.hpp"
#include "baguette/model/ModelEnums.hpp"

using namespace baguette;

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
