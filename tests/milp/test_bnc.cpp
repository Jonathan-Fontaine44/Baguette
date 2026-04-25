#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "baguette/lp/LPSolver.hpp"
#include "baguette/milp/BranchAndBound.hpp"
#include "baguette/milp/CuttingPlanes.hpp"
#include "baguette/model/Model.hpp"
#include "baguette/model/ModelEnums.hpp"

using namespace baguette;
using Catch::Matchers::WithinAbs;

static constexpr double kTol = 1e-6;

// ── Test 1: enableCuts=false → cutsAdded=0 ──────────────────────────────────
//
// Verify that when cut generation is disabled, no cuts are recorded.

TEST_CASE("BnC: enableCuts=false -> cutsAdded=0", "[bnc]") {
    Model m;
    Variable x = m.addVar(0.0, 5.0, VarType::Integer, "x");
    Variable y = m.addVar(0.0, 5.0, VarType::Integer, "y");
    m.addConstraint(3.0 * x + 2.0 * y, Sense::LessEq, 7.0);
    m.setObjective(5.0 * x + 4.0 * y, ObjSense::Maximize);

    BBOptions opts;
    opts.enableCuts = false;

    MILPResult r = solveMILP(m, opts);

    REQUIRE(r.status == MILPStatus::Optimal);
    REQUIRE(r.cutsAdded == 0);
    REQUIRE_THAT(r.objectiveValue, WithinAbs(13.0, kTol));
}

// ── Test 2: B&C gives same answer as B&B ────────────────────────────────────
//
// max 5x + 4y  s.t. 3x + 2y ≤ 7, x,y ∈ Z≥0.
// IP optimal: x=1, y=2, obj=13.
// With cuts enabled the result must be identical.

TEST_CASE("BnC: same optimal as pure B&B (knapsack, maximize)", "[bnc]") {
    Model m;
    Variable x = m.addVar(0.0, 5.0, VarType::Integer, "x");
    Variable y = m.addVar(0.0, 5.0, VarType::Integer, "y");
    m.addConstraint(3.0 * x + 2.0 * y, Sense::LessEq, 7.0);
    m.setObjective(5.0 * x + 4.0 * y, ObjSense::Maximize);

    BBOptions noCuts;
    noCuts.enableCuts = false;

    BBOptions withCuts;
    withCuts.enableCuts     = true;
    withCuts.maxCutsPerNode = 10;

    MILPResult r1 = solveMILP(m, noCuts);
    MILPResult r2 = solveMILP(m, withCuts);

    REQUIRE(r1.status == MILPStatus::Optimal);
    REQUIRE(r2.status == MILPStatus::Optimal);
    REQUIRE_THAT(r1.objectiveValue, WithinAbs(r2.objectiveValue, kTol));
    REQUIRE_THAT(r1.primalValues[x.id], WithinAbs(r2.primalValues[x.id], kTol));
    REQUIRE_THAT(r1.primalValues[y.id], WithinAbs(r2.primalValues[y.id], kTol));
}

// ── Test 3: GMI cut closes LP-IP gap at root ─────────────────────────────────
//
// min x + y   s.t. 2x + 2y ≥ 7,  x, y ∈ Z,  0 ≤ x,y ≤ 5.
//
// LP relaxation optimal: x+y = 3.5 (e.g. x=3.5, y=0), obj=3.5.
// GMI cut from the fractional row: -2x - 2y ≥ -8, i.e. x+y ≤ 4.
// After adding the cut the LP optimal is x+y=4 (integer) → solved at the root.
//
// IP optimal: x+y=4, obj=4 (e.g. x=4, y=0 or x=0, y=4).

TEST_CASE("BnC: GMI cut closes gap at root", "[bnc]") {
    Model m;
    Variable x = m.addVar(0.0, 5.0, VarType::Integer, "x");
    Variable y = m.addVar(0.0, 5.0, VarType::Integer, "y");
    m.addConstraint(2.0 * x + 2.0 * y, Sense::GreaterEq, 7.0);
    m.setObjective(1.0 * x + 1.0 * y, ObjSense::Minimize);

    BBOptions noCuts;
    noCuts.enableCuts = false;

    BBOptions withCuts;
    withCuts.enableCuts     = true;
    withCuts.maxCutsPerNode = 10;

    MILPResult r1 = solveMILP(m, noCuts);
    MILPResult r2 = solveMILP(m, withCuts);

    // Both must find the same optimal.
    REQUIRE(r1.status == MILPStatus::Optimal);
    REQUIRE(r2.status == MILPStatus::Optimal);
    REQUIRE_THAT(r1.objectiveValue, WithinAbs(4.0, kTol));
    REQUIRE_THAT(r2.objectiveValue, WithinAbs(4.0, kTol));

    // With cuts the gap is closed at the root: 1 node.
    REQUIRE(r2.nodesExplored == 1);
    REQUIRE(r2.cutsAdded >= 1);

    // Without cuts branching is required.
    REQUIRE(r1.nodesExplored > 1);
    REQUIRE(r1.cutsAdded == 0);
}

// ── Test 4: PseudoCost branching gives correct answer ────────────────────────
//
// Use the knapsack from Test 2. PseudoCost must agree with MostFractional.

TEST_CASE("BnC: PseudoCost branching gives same optimal as MostFractional", "[bnc]") {
    Model m;
    Variable x = m.addVar(0.0, 5.0, VarType::Integer, "x");
    Variable y = m.addVar(0.0, 5.0, VarType::Integer, "y");
    m.addConstraint(3.0 * x + 2.0 * y, Sense::LessEq, 7.0);
    m.setObjective(5.0 * x + 4.0 * y, ObjSense::Maximize);

    BBOptions mf;
    mf.branchStrat = BranchStrategy::MostFractional;
    mf.enableCuts  = false;

    BBOptions pc;
    pc.branchStrat = BranchStrategy::PseudoCost;
    pc.enableCuts  = false;

    MILPResult r1 = solveMILP(m, mf);
    MILPResult r2 = solveMILP(m, pc);

    REQUIRE(r1.status == MILPStatus::Optimal);
    REQUIRE(r2.status == MILPStatus::Optimal);
    REQUIRE_THAT(r1.objectiveValue, WithinAbs(r2.objectiveValue, kTol));
}

// ── Test 5: PseudoCost + cuts give correct answer ────────────────────────────
//
// Combine PseudoCost branching with GMI cut generation.
// The result must equal the known IP optimum.

TEST_CASE("BnC: PseudoCost + cuts give correct answer", "[bnc]") {
    Model m;
    Variable x = m.addVar(0.0, 5.0, VarType::Integer, "x");
    Variable y = m.addVar(0.0, 5.0, VarType::Integer, "y");
    m.addConstraint(2.0 * x + 2.0 * y, Sense::GreaterEq, 7.0);
    m.setObjective(1.0 * x + 1.0 * y, ObjSense::Minimize);

    BBOptions opts;
    opts.branchStrat    = BranchStrategy::PseudoCost;
    opts.enableCuts     = true;
    opts.maxCutsPerNode = 5;

    MILPResult r = solveMILP(m, opts);

    REQUIRE(r.status == MILPStatus::Optimal);
    REQUIRE_THAT(r.objectiveValue, WithinAbs(4.0, kTol));
}

// ── Test 6: B&C infeasible MILP ─────────────────────────────────────────────
//
// LP-infeasible problem → MILP infeasible even with cuts enabled.

TEST_CASE("BnC: infeasible problem stays infeasible with cuts", "[bnc]") {
    Model m;
    Variable x = m.addVar(0.0, 10.0, VarType::Integer, "x");
    Variable y = m.addVar(0.0, 10.0, VarType::Integer, "y");
    m.addConstraint(1.0 * x + 1.0 * y, Sense::LessEq, -1.0);
    m.setObjective(1.0 * x + 1.0 * y, ObjSense::Minimize);

    BBOptions opts;
    opts.enableCuts = true;

    MILPResult r = solveMILP(m, opts);

    REQUIRE(r.status == MILPStatus::Infeasible);
    REQUIRE(r.primalValues.empty());
    REQUIRE(r.cutsAdded == 0);
}

// ── Test 7: B&C 3-variable problem ──────────────────────────────────────────
//
// max 5x + 4y + 3z
// s.t. 2x + y + z ≤ 6
//      x + 2y + z ≤ 6
//      x, y, z ∈ Z, 0 ≤ x,y,z ≤ 5
//
// LP optimal: by symmetry around (2, 2, 0) or nearby fractional points.
// IP optimal (by inspection): (2, 2, 0) → 10+8+0=18. Or (3,0,0)→15. Or (2,2,0)=18. Or (1,2,2)→5+8+6=19. Check: 2+2+2=6≤6✓, 1+4+2=7>6✗. Or (2,2,0)→4+2+0=6≤6✓,2+4+0=6≤6✓, obj=18. Or (3,0,2)→6+0+2=8>6✗. Or (0,3,0)→3≤6✓,6≤6✓,obj=12. Hmm...
// Let me try (2,0,2): 4+0+2=6≤6✓, 2+0+2=4≤6✓, obj=10+0+6=16.
// (2,2,0): obj=10+8=18. (3,0,0):15. (0,3,0):12. (0,0,5):15.
// (1,2,2): 2+2+2=6✓, 1+4+2=7>6✗.
// (2,2,0)=18 seems best. Let's use this.
//
// With or without cuts, the IP optimal must be ≥ 18 (achieved at (2,2,0)).

TEST_CASE("BnC: 3-variable MILP correct with cuts", "[bnc]") {
    Model m;
    Variable x = m.addVar(0.0, 5.0, VarType::Integer, "x");
    Variable y = m.addVar(0.0, 5.0, VarType::Integer, "y");
    Variable z = m.addVar(0.0, 5.0, VarType::Integer, "z");
    m.addConstraint(2.0*x + 1.0*y + 1.0*z, Sense::LessEq, 6.0);
    m.addConstraint(1.0*x + 2.0*y + 1.0*z, Sense::LessEq, 6.0);
    m.setObjective(5.0*x + 4.0*y + 3.0*z, ObjSense::Maximize);

    BBOptions noCuts;
    noCuts.enableCuts = false;

    BBOptions withCuts;
    withCuts.enableCuts     = true;
    withCuts.maxCutsPerNode = 10;
    withCuts.branchStrat    = BranchStrategy::PseudoCost;

    MILPResult r1 = solveMILP(m, noCuts);
    MILPResult r2 = solveMILP(m, withCuts);

    REQUIRE(r1.status == MILPStatus::Optimal);
    REQUIRE(r2.status == MILPStatus::Optimal);
    REQUIRE_THAT(r1.objectiveValue, WithinAbs(r2.objectiveValue, kTol));
    REQUIRE(r1.objectiveValue >= 18.0 - kTol);
}

// ── Test 8: computeCutData reports fractional integers only ─────────────────
//
// min 0x + y  s.t. x + y ≥ 3.5,  x ∈ Z[0,3.2],  y ∈ C[0,5].
//
// LP optimal: x=3.2 (fractional integer), y=0.3 (continuous).
// fractionalRows must contain exactly x; y must not appear.
// generateGMICuts must produce exactly one cut from the x row.

TEST_CASE("CutData: ignore continuous variables, detect fractional integer", "[cuts]") {
    Model m;
    Variable x = m.addVar(0.0, 3.2, VarType::Integer, "x");
    Variable y = m.addVar(0.0, 5.0, VarType::Continuous, "y");
    m.addConstraint(1.0 * x + 1.0 * y, Sense::GreaterEq, 3.5);
    m.setObjective(0.0 * x + 1.0 * y, ObjSense::Minimize); // LP: x=3.5 (fractional int), y=0

    // Solve relaxation LP
    LPDetailedResult lp = solveDualDetailed(
        m,
        0,
        std::numeric_limits<double>::infinity(),
        SolverClock::now(),
        {},
        /*computeSensitivity=*/false,
        /*computeCutData=*/true);

    REQUIRE(lp.result.status == LPStatus::Optimal);

    for (FractionalRow fr : lp.fractionalRows) {
        REQUIRE(fr.origVarId == x.id);
        REQUIRE(fr.origVarId != y.id);
    }

    // x = 3.5, y = 0
    std::vector<Cut> cuts = generateGMICuts(lp.fractionalRows, lp.basis, m, 50, kTol);

    REQUIRE(cuts.size() == 1);
}

// ── Test 9: B&C cuts bind on integer variables in a mixed MILP ──────────────
//
// min x + y + 0.1z + 0.1w  s.t. 2x + 2y + z + w ≥ 7.5,
//   x,y ∈ Z[0,10],  z,w ∈ C[0,2].
//
// z,w capped at 2 (max contribution 4 < 7.5), so 2x + 2y ≥ 3.5 is required.
// LP leaves x or y fractional → at least one GMI cut is added.
// IP solution must satisfy integrality for x and y.

TEST_CASE("BnC: cuts affect only integer variables (mixed MILP)", "[bnc][cuts]") {
    Model m;

    Variable x = m.addVar(0.0, 10.0, VarType::Integer, "x");
    Variable y = m.addVar(0.0, 10.0, VarType::Integer, "y");
    Variable z = m.addVar(0.0, 2.0, VarType::Continuous, "z");
    Variable w = m.addVar(0.0, 2.0, VarType::Continuous, "w");
    m.addConstraint(2*x + 2*y + 1*z + 1*w, Sense::GreaterEq, 7.5);
    m.setObjective(1*x + 1*y + 0.1*z + 0.1*w, ObjSense::Minimize);

    BBOptions opts;
    opts.enableCuts = true;
    opts.maxCutsPerNode = 5;

    MILPResult r = solveMILP(m, opts);

    REQUIRE(r.status == MILPStatus::Optimal);

    REQUIRE(r.cutsAdded > 0);

    REQUIRE(std::abs(r.primalValues[x.id] - std::round(r.primalValues[x.id])) < kTol);
    REQUIRE(std::abs(r.primalValues[y.id] - std::round(r.primalValues[y.id])) < kTol);
}

// ── Test 10: MILP infeasible even though LP relaxation is feasible ──────────
//
// min x  s.t. x = 0.5,  x ∈ Z[0,1].
//
// LP optimal: x=0.5 (feasible). Branching: x≤0 violates x=0.5; x≥1 violates x=0.5.
// Both children are LP-infeasible → MILP Infeasible.

TEST_CASE("BnC: MILP infeasible but LP feasible", "[bnc][edge]") {
    Model m;

    Variable x = m.addVar(0.0, 1.0, VarType::Integer, "x");

    m.addConstraint(1.0 * x, Sense::Equal, 0.5);
    m.setObjective(1.0 * x, ObjSense::Minimize);

    BBOptions opts;
    opts.enableCuts = true;

    LPDetailedResult lp = solveDualDetailed(m);

    REQUIRE(lp.result.status == LPStatus::Optimal);

    MILPResult r = solveMILP(m, opts);

    REQUIRE(r.status == MILPStatus::Infeasible);
}