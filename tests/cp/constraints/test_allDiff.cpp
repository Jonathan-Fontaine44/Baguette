#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>

#include "baguette/cp/CPConstraints.hpp"
#include "baguette/cp/constraints/AllDiff.hpp"
#include "baguette/milp/BranchAndBound.hpp"
#include "baguette/model/Model.hpp"
#include "baguette/model/ModelEnums.hpp"

using namespace baguette;
using Catch::Matchers::WithinAbs;

static constexpr double kTol = 1e-9;

// ── AllDiff: pas de doublons dans changedVarIds pour un seul appel ──────────────
//
// AllDiff(x1=[3,3], x2=[5,5], z=[3,5]).
// Propagation pass 1 :
//   x1 fixé à 3 → z.lb 3→4 : push z.id.
//   x2 fixé à 5 → z.ub 5→4 : push z.id à nouveau (doublon !).
// Propagation pass 2 : z=[4,4] fixé → aucun changement sur x1 ou x2.
// changedVarIds doit contenir z.id exactement une fois.

TEST_CASE("AllDiff propagate: no duplicate ids in changedVarIds", "[cp][alldiff]") {
    Model m;
    Variable x1 = m.addVar(3.0, 3.0, VarType::Integer, "x1");
    Variable x2 = m.addVar(5.0, 5.0, VarType::Integer, "x2");
    Variable z  = m.addVar(3.0, 5.0, VarType::Integer, "z");

    AllDiffConstraint con{{x1, x2, z}};
    PropagationResult r = propagate(con, m);

    REQUIRE(r.status == CPStatus::Feasible);
    REQUIRE_THAT(m.getHot().lb[z.id], WithinAbs(4.0, kTol));
    REQUIRE_THAT(m.getHot().ub[z.id], WithinAbs(4.0, kTol));

    const auto& changed = r.changedVarIds;
    const long  zCount  = std::count(changed.begin(), changed.end(), z.id);
    REQUIRE(zCount == 1); // échoue avant le fix (vaut 2)
}

// ── Test 1: single variable ───────────────────────────────────────────────────
//
// AllDiff on one variable is trivially satisfied - no propagation.

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

// ── Test 7: Hall interval propagation ────────────────────────────────────────
//
// AllDiff(x1=[1,2], x2=[1,2], x3=[1,3]):
// Hall interval [1,2]: x1 and x2 have domain ⊆ [1,2], count=2=capacity=2.
// → x3 must avoid [1,2]: lb 1→3. x3=[3,3].
// Fixed-value elimination alone would not propagate (x1,x2 not fixed).

TEST_CASE("AllDiff: Hall interval tightens bounds of non-Hall variables", "[cp][alldiff]") {
    Model m;
    Variable x1 = m.addVar(1.0, 2.0, VarType::Integer, "x1");
    Variable x2 = m.addVar(1.0, 2.0, VarType::Integer, "x2");
    Variable x3 = m.addVar(1.0, 3.0, VarType::Integer, "x3");

    AllDiffConstraint con{{x1, x2, x3}};
    PropagationResult r = propagate(con, m);

    REQUIRE(r.status == CPStatus::Feasible);
    // x3 forced to 3 (only value outside Hall set {1,2})
    REQUIRE_THAT(m.getHot().lb[x3.id], WithinAbs(3.0, kTol));
    REQUIRE_THAT(m.getHot().ub[x3.id], WithinAbs(3.0, kTol));
    // x1, x2 unchanged
    REQUIRE_THAT(m.getHot().lb[x1.id], WithinAbs(1.0, kTol));
    REQUIRE_THAT(m.getHot().ub[x1.id], WithinAbs(2.0, kTol));
    // x3 in changedVarIds; x1, x2 not
    const auto& ch = r.changedVarIds;
    REQUIRE(std::find(ch.begin(), ch.end(), x3.id) != ch.end());
    REQUIRE(std::find(ch.begin(), ch.end(), x1.id) == ch.end());
    REQUIRE(std::find(ch.begin(), ch.end(), x2.id) == ch.end());
}

// ── Test 8: Hall interval cascade ────────────────────────────────────────────
//
// AllDiff(x1=[1,2], x2=[1,2], x3=[1,2], x4=[3,4], x5=[3,4]):
// Hall [1,2]: count=3 > capacity=2 → infeasible.

TEST_CASE("AllDiff: Hall interval detects overcrowded sub-interval infeasibility", "[cp][alldiff]") {
    Model m;
    Variable x1 = m.addVar(1.0, 2.0, VarType::Integer, "x1");
    Variable x2 = m.addVar(1.0, 2.0, VarType::Integer, "x2");
    Variable x3 = m.addVar(1.0, 2.0, VarType::Integer, "x3");
    Variable x4 = m.addVar(3.0, 4.0, VarType::Integer, "x4");
    Variable x5 = m.addVar(3.0, 4.0, VarType::Integer, "x5");

    AllDiffConstraint con{{x1, x2, x3, x4, x5}};
    PropagationResult r = propagate(con, m);

    REQUIRE(r.status == CPStatus::Infeasible);
}

// ── MILP + AllDiff, optimal permutation ──────────────────────────────────────
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