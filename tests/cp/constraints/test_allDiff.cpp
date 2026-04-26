#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "baguette/cp/CPConstraints.hpp"
#include "baguette/milp/BranchAndBound.hpp"
#include "baguette/model/Model.hpp"
#include "baguette/model/ModelEnums.hpp"

using namespace baguette;
using Catch::Matchers::WithinAbs;

static constexpr double kTol = 1e-9;

// ── Test 1: single variable ───────────────────────────────────────────────────
//
// AllDiff on one variable is trivially satisfied — no propagation.

TEST_CASE("AllDiff: trivially satisfied with one variable", "[cp][alldiff]") {
    Model m;
    Variable x = m.addVar(1.0, 3.0, VarType::Integer, "x");

    CPConstraints cp;
    cp.add(AllDiffConstraint{x});

    PropagationResult r = propagateCP(cp, m);
    REQUIRE(r.status == CPStatus::Feasible);
    REQUIRE(r.changedVarIds.empty());
}

// ── Test 2: fixed-value elimination ──────────────────────────────────────────
//
// AllDiff(x, y) with x = [1,1], y = [1,3].
// x is fixed to 1 → y must avoid 1 → lb_y raised to 2 → y = [2,3].

TEST_CASE("AllDiff: raises lb when fixed value equals sibling lb", "[cp][alldiff]") {
    Model m;
    Variable x = m.addVar(1.0, 1.0, VarType::Integer, "x");
    Variable y = m.addVar(1.0, 3.0, VarType::Integer, "y");

    CPConstraints cp;
    cp.add(AllDiffConstraint{x, y});

    PropagationResult r = propagateCP(cp, m);

    REQUIRE(r.status == CPStatus::Feasible);
    REQUIRE_THAT(m.getHot().lb[y.id], WithinAbs(2.0, kTol));
    REQUIRE_THAT(m.getHot().ub[y.id], WithinAbs(3.0, kTol));
    REQUIRE(std::find(r.changedVarIds.begin(), r.changedVarIds.end(), y.id)
            != r.changedVarIds.end());
    REQUIRE(std::find(r.changedVarIds.begin(), r.changedVarIds.end(), x.id)
            == r.changedVarIds.end());
}

// ── Test 3: same fixed value → infeasible ────────────────────────────────────
//
// AllDiff(x, y) with x = y = 2: both fixed to the same value → infeasible.

TEST_CASE("AllDiff: infeasible when two variables fixed to same value", "[cp][alldiff]") {
    Model m;
    Variable x = m.addVar(2.0, 2.0, VarType::Integer, "x");
    Variable y = m.addVar(2.0, 2.0, VarType::Integer, "y");

    CPConstraints cp;
    cp.add(AllDiffConstraint{x, y});

    PropagationResult r = propagateCP(cp, m);
    REQUIRE(r.status == CPStatus::Infeasible);
}

// ── Test 4: range check → infeasible ─────────────────────────────────────────
//
// AllDiff(x, y, z) with all in [1,2]: only 2 distinct integers for 3 vars.

TEST_CASE("AllDiff: infeasible when range too small for constraint arity", "[cp][alldiff]") {
    Model m;
    Variable x = m.addVar(1.0, 2.0, VarType::Integer, "x");
    Variable y = m.addVar(1.0, 2.0, VarType::Integer, "y");
    Variable z = m.addVar(1.0, 2.0, VarType::Integer, "z");

    CPConstraints cp;
    cp.add(AllDiffConstraint{x, y, z});

    PropagationResult r = propagateCP(cp, m);
    REQUIRE(r.status == CPStatus::Infeasible);
}

// ── Test 5: cascade fixpoint → infeasible ────────────────────────────────────
//
// AllDiff(x, y, z) with x=[1,1], y=[1,2], z=[1,2].
// Pass 1: x=1 → y=[2,2], z=[2,2].
// Pass 2: y=2 → z: lb raised to 3, ub lowered to 1 → empty domain.

TEST_CASE("AllDiff: cascade elimination propagates to infeasibility", "[cp][alldiff]") {
    Model m;
    Variable x = m.addVar(1.0, 1.0, VarType::Integer, "x");
    Variable y = m.addVar(1.0, 2.0, VarType::Integer, "y");
    Variable z = m.addVar(1.0, 2.0, VarType::Integer, "z");

    CPConstraints cp;
    cp.add(AllDiffConstraint{x, y, z});

    PropagationResult r = propagateCP(cp, m);
    REQUIRE(r.status == CPStatus::Infeasible);
}

// ── Test 6: multi-step feasible, selective changedVarIds ─────────────────────
//
// AllDiff(x, y, z) with x=[1,1], y=[1,4], z=[2,4].
// Pass 1: x=1 → y: lb 1→2 (y=[2,4]).  z unchanged (lb=2 ≠ 1).
// changedVarIds must contain y.id; z.id must not appear.

TEST_CASE("AllDiff: only affected variable reported in changedVarIds", "[cp][alldiff]") {
    Model m;
    Variable x = m.addVar(1.0, 1.0, VarType::Integer, "x");
    Variable y = m.addVar(1.0, 4.0, VarType::Integer, "y");
    Variable z = m.addVar(2.0, 4.0, VarType::Integer, "z");

    CPConstraints cp;
    cp.add(AllDiffConstraint{x, y, z});

    PropagationResult r = propagateCP(cp, m);

    REQUIRE(r.status == CPStatus::Feasible);
    REQUIRE_THAT(m.getHot().lb[y.id], WithinAbs(2.0, kTol));
    REQUIRE_THAT(m.getHot().ub[z.id], WithinAbs(4.0, kTol)); // z unchanged
    REQUIRE(std::find(r.changedVarIds.begin(), r.changedVarIds.end(), y.id)
            != r.changedVarIds.end());
    REQUIRE(std::find(r.changedVarIds.begin(), r.changedVarIds.end(), z.id)
            == r.changedVarIds.end());
}

// ── Test 7: MILP + AllDiff, optimal permutation ──────────────────────────────
//
// min x + y + z,  x,y,z ∈ {1,2,3},  AllDiff(x,y,z).
// Feasible solutions are permutations of {1,2,3}.  Optimal value = 6.

TEST_CASE("AllDiff+MILP: 3-permutation minimize sum", "[cp][alldiff][milp]") {
    Model m;
    Variable x = m.addVar(1.0, 3.0, VarType::Integer, "x");
    Variable y = m.addVar(1.0, 3.0, VarType::Integer, "y");
    Variable z = m.addVar(1.0, 3.0, VarType::Integer, "z");
    m.setObjective(1.0 * x + 1.0 * y + 1.0 * z, ObjSense::Minimize);

    m.addCPConstraint(AllDiffConstraint{x, y, z});

    MILPResult r = solveMILP(m, BBOptions{});

    REQUIRE(r.status == MILPStatus::Optimal);
    REQUIRE_THAT(r.objectiveValue, WithinAbs(6.0, kTol));

    double vx = r.primalValues[x.id];
    double vy = r.primalValues[y.id];
    double vz = r.primalValues[z.id];
    REQUIRE_THAT(vx + vy + vz, WithinAbs(6.0, kTol));
    REQUIRE_THAT(std::abs(vx - vy), WithinAbs(std::abs(vx - vy), kTol)); // distinct
    REQUIRE(std::abs(vx - vy) > kTol);
    REQUIRE(std::abs(vy - vz) > kTol);
    REQUIRE(std::abs(vx - vz) > kTol);
}

// ── Test 8: MILP + AllDiff, CP prunes infeasibility at root ──────────────────
//
// x,y,z ∈ {1,2}, AllDiff(x,y,z): 2 distinct values for 3 variables.
// CP range check at the root immediately returns Infeasible.

TEST_CASE("AllDiff+MILP: range infeasibility detected at root node", "[cp][alldiff][milp]") {
    Model m;
    Variable x = m.addVar(1.0, 2.0, VarType::Integer, "x");
    Variable y = m.addVar(1.0, 2.0, VarType::Integer, "y");
    Variable z = m.addVar(1.0, 2.0, VarType::Integer, "z");
    m.setObjective(1.0 * x + 1.0 * y + 1.0 * z, ObjSense::Minimize);

    m.addCPConstraint(AllDiffConstraint{x, y, z});

    MILPResult r = solveMILP(m, BBOptions{});
    REQUIRE(r.status == MILPStatus::Infeasible);
}