#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "baguette/cp/CPConstraints.hpp"
#include "baguette/model/Model.hpp"
#include "baguette/model/ModelEnums.hpp"

using namespace baguette;
using Catch::Matchers::WithinAbs;

static constexpr double kTol = 1e-9;

// ── Test 1: sequential tasks, no conflict ────────────────────────────────────
//
// Task A: start ∈ [0,0] (fixed), dur=2, cns=1.  Compulsory: [0,2).
// Task B: start ∈ [2,4], dur=2, cns=1.  No compulsory region (lst=4 ≥ ect=4).
// Capacity = 1.  Tasks can execute sequentially — no overload, no tightening.

TEST_CASE("Cumulative: feasible sequential tasks, no bound tightening", "[cp][cumulative]") {
    Model m;
    Variable sA = m.addVar(0.0, 0.0, VarType::Integer, "sA");
    Variable sB = m.addVar(2.0, 4.0, VarType::Integer, "sB");

    CPConstraints cp;
    CumulativeConstraint cum;
    cum.capacity = 1;
    cum.tasks    = {{sA, 2, 1}, {sB, 2, 1}};
    cp.add(cum);

    PropagationResult r = propagateCP(cp, m);
    REQUIRE(r.status == CPStatus::Feasible);
    REQUIRE(r.changedVarIds.empty());
}

// ── Test 2: compulsory overload → infeasible ──────────────────────────────────
//
// Task A: start ∈ [0,1], dur=2, cns=2.  Compulsory: [1,2) — time 1.
// Task B: start ∈ [0,1], dur=2, cns=2.  Compulsory: [1,2) — time 1.
// Capacity = 3.  At time 1: A(2) + B(2) = 4 > 3.
// Both tasks always cover time 1, so no valid assignment exists.

TEST_CASE("Cumulative: infeasible when compulsory load exceeds capacity", "[cp][cumulative]") {
    Model m;
    Variable sA = m.addVar(0.0, 1.0, VarType::Integer, "sA");
    Variable sB = m.addVar(0.0, 1.0, VarType::Integer, "sB");

    CPConstraints cp;
    CumulativeConstraint cum;
    cum.capacity = 3;
    cum.tasks    = {{sA, 2, 2}, {sB, 2, 2}};
    cp.add(cum);

    PropagationResult r = propagateCP(cp, m);
    REQUIRE(r.status == CPStatus::Infeasible);
}

// ── Test 3: earliest-start tightening ────────────────────────────────────────
//
// Task A: start ∈ [0,5], dur=3, cns=1.  No compulsory region (lst=5 ≥ ect=3).
// Task B: start ∈ [2,2] (fixed), dur=3, cns=1.  Compulsory: [2,5).
// Capacity = 1.
//
// For A starting at t ∈ {0,1,2,3,4}: window [t, t+3) intersects [2,5) at
// some time point, causing load 2 > 1.  Only t=5 gives window [5,8) ∩ [2,5) = {}
// → est_A must advance to 5.

TEST_CASE("Cumulative: est advanced to avoid compulsory overload", "[cp][cumulative]") {
    Model m;
    Variable sA = m.addVar(0.0, 5.0, VarType::Integer, "sA");
    Variable sB = m.addVar(2.0, 2.0, VarType::Integer, "sB");

    CPConstraints cp;
    CumulativeConstraint cum;
    cum.capacity = 1;
    cum.tasks    = {{sA, 3, 1}, {sB, 3, 1}};
    cp.add(cum);

    PropagationResult r = propagateCP(cp, m);

    REQUIRE(r.status == CPStatus::Feasible);
    REQUIRE_THAT(m.getHot().lb[sA.id], WithinAbs(5.0, kTol));
    REQUIRE(std::find(r.changedVarIds.begin(), r.changedVarIds.end(), sA.id)
            != r.changedVarIds.end());
}
