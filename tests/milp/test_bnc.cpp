#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cstdio>

#include "baguette/lp/LPSolver.hpp"
#include "baguette/milp/BranchAndBound.hpp"
#include "baguette/milp/CuttingPlanes.hpp"
#include "baguette/model/Model.hpp"
#include "baguette/model/ModelEnums.hpp"
#include "lp/MILP_problems.hpp"

using namespace baguette;
using Catch::Matchers::WithinAbs;

static constexpr double kTol = 1e-6;

// ── Test 1: enableCuts=false → cutsAdded=0 ──────────────────────────────────
//
// Verify that when cut generation is disabled, no cuts are recorded.

TEST_CASE("BnC: enableCuts=false -> cutsAdded=0", "[bnc]") {
    auto method = GENERATE(LPMethod::Auto,
                           LPMethod::PrimalSimplex,   LPMethod::DualSimplex,
                           LPMethod::RevisedSimplex,
                           LPMethod::PrimalSimplexBV, LPMethod::DualSimplexBV,
                           LPMethod::RevisedSimplexBV);

    Model m;
    Variable x = m.addVar(0.0, 5.0, VarType::Integer, "x");
    Variable y = m.addVar(0.0, 5.0, VarType::Integer, "y");
    m.addLPConstraint(3.0 * x + 2.0 * y, Sense::LessEq, 7.0);
    m.setObjective(5.0 * x + 4.0 * y, ObjSense::Maximize);

    BBOptions opts;
    opts.enableCuts       = false;
    opts.collectStats     = true;
    opts.lpOpts.method    = method;
    opts.enablePresolve   = false;

    DYNAMIC_SECTION("method=" << to_string(method)) {
        MILPResult r = solveMILP(m, opts);

        REQUIRE(r.status == MILPStatus::Optimal);
        REQUIRE(r.stats->cutsAdded == 0);
        REQUIRE_THAT(r.objectiveValue, WithinAbs(13.0, kTol));
    }
}

// ── Test 2: B&C gives same answer as B&B ────────────────────────────────────
//
// max 5x + 4y  s.t. 3x + 2y ≤ 7, x,y ∈ Z≥0.
// IP optimal: x=1, y=2, obj=13.
// With cuts enabled the result must be identical.

TEST_CASE("BnC: same optimal as pure B&B (knapsack, maximize)", "[bnc][cuts]") {
    auto method = GENERATE(LPMethod::Auto,
                           LPMethod::PrimalSimplex,   LPMethod::DualSimplex,
                           LPMethod::RevisedSimplex,
                           LPMethod::PrimalSimplexBV, LPMethod::DualSimplexBV,
                           LPMethod::RevisedSimplexBV);

    Model m;
    Variable x = m.addVar(0.0, 5.0, VarType::Integer, "x");
    Variable y = m.addVar(0.0, 5.0, VarType::Integer, "y");
    m.addLPConstraint(3.0 * x + 2.0 * y, Sense::LessEq, 7.0);
    m.setObjective(5.0 * x + 4.0 * y, ObjSense::Maximize);

    BBOptions noCuts;
    noCuts.enableCuts        = false;
    noCuts.lpOpts.method     = method;
    noCuts.enablePresolve    = false;

    BBOptions withCuts;
    withCuts.enableCuts      = true;
    withCuts.maxCutsPerNode  = 10;
    withCuts.lpOpts.method   = method;
    withCuts.enablePresolve  = false;

    DYNAMIC_SECTION("method=" << to_string(method)) {
        MILPResult r1 = solveMILP(m, noCuts);
        MILPResult r2 = solveMILP(m, withCuts);

        REQUIRE(r1.status == MILPStatus::Optimal);
        REQUIRE(r2.status == MILPStatus::Optimal);
        REQUIRE_THAT(r1.objectiveValue, WithinAbs(r2.objectiveValue, kTol));
        REQUIRE_THAT(r1.primalValues[x.id], WithinAbs(r2.primalValues[x.id], kTol));
        REQUIRE_THAT(r1.primalValues[y.id], WithinAbs(r2.primalValues[y.id], kTol));
    }
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

TEST_CASE("BnC: GMI cut closes gap at root", "[bnc][cuts]") {
    auto method = GENERATE(LPMethod::Auto,
                           LPMethod::PrimalSimplex,   LPMethod::DualSimplex,
                           LPMethod::RevisedSimplex,
                           LPMethod::PrimalSimplexBV, LPMethod::DualSimplexBV,
                           LPMethod::RevisedSimplexBV);

    Model m;
    Variable x = m.addVar(0.0, 5.0, VarType::Integer, "x");
    Variable y = m.addVar(0.0, 5.0, VarType::Integer, "y");
    m.addLPConstraint(2.0 * x + 2.0 * y, Sense::GreaterEq, 7.0);
    m.setObjective(1.0 * x + 1.0 * y, ObjSense::Minimize);

    BBOptions noCuts;
    noCuts.enableCuts        = false;
    noCuts.collectStats      = true;
    noCuts.lpOpts.method     = method;
    noCuts.enablePresolve    = false;

    BBOptions withCuts;
    withCuts.enableCuts      = true;
    withCuts.maxCutsPerNode  = 10;
    withCuts.collectStats    = true;
    withCuts.lpOpts.method   = method;
    withCuts.enablePresolve  = false;

    DYNAMIC_SECTION("method=" << to_string(method)) {
        MILPResult r1 = solveMILP(m, noCuts);
        MILPResult r2 = solveMILP(m, withCuts);

        // Both must find the same optimal.
        REQUIRE(r1.status == MILPStatus::Optimal);
        REQUIRE(r2.status == MILPStatus::Optimal);
        REQUIRE_THAT(r1.objectiveValue, WithinAbs(4.0, kTol));
        REQUIRE_THAT(r2.objectiveValue, WithinAbs(4.0, kTol));

        REQUIRE(r1.stats->cutsAdded == 0);

        // With cuts the gap is closed at the root: 1 node.
        REQUIRE(r2.stats->nodesExplored == 1);
        REQUIRE(r2.stats->cutsAdded >= 1);

        // Without cuts branching is required.
        REQUIRE(r1.stats->nodesExplored > 1);
    }
}

// ── Test 4: PseudoCost branching gives correct answer ────────────────────────
//
// Use the knapsack from Test 2. PseudoCost must agree with MostFractional.

TEST_CASE("BnC: PseudoCost branching gives same optimal as MostFractional", "[bnc]") {
    auto method = GENERATE(LPMethod::Auto,
                           LPMethod::PrimalSimplex,   LPMethod::DualSimplex,
                           LPMethod::RevisedSimplex,
                           LPMethod::PrimalSimplexBV, LPMethod::DualSimplexBV,
                           LPMethod::RevisedSimplexBV);

    Model m;
    Variable x = m.addVar(0.0, 5.0, VarType::Integer, "x");
    Variable y = m.addVar(0.0, 5.0, VarType::Integer, "y");
    m.addLPConstraint(3.0 * x + 2.0 * y, Sense::LessEq, 7.0);
    m.setObjective(5.0 * x + 4.0 * y, ObjSense::Maximize);

    BBOptions mf;
    mf.branchStrat      = BranchStrategy::MostFractional;
    mf.enableCuts       = false;
    mf.lpOpts.method    = method;
    mf.enablePresolve   = false;

    BBOptions pc;
    pc.branchStrat      = BranchStrategy::PseudoCost;
    pc.enableCuts       = false;
    pc.lpOpts.method    = method;
    pc.enablePresolve   = false;

    DYNAMIC_SECTION("method=" << to_string(method)) {
        MILPResult r1 = solveMILP(m, mf);
        MILPResult r2 = solveMILP(m, pc);

        REQUIRE(r1.status == MILPStatus::Optimal);
        REQUIRE(r2.status == MILPStatus::Optimal);
        REQUIRE_THAT(r1.objectiveValue, WithinAbs(r2.objectiveValue, kTol));
    }
}

// ── Test 5: PseudoCost + cuts give correct answer ────────────────────────────
//
// Combine PseudoCost branching with GMI cut generation.
// The result must equal the known IP optimum.

TEST_CASE("BnC: PseudoCost + cuts give correct answer", "[bnc][cuts]") {
    auto method = GENERATE(LPMethod::Auto,
                           LPMethod::PrimalSimplex,   LPMethod::DualSimplex,
                           LPMethod::RevisedSimplex,
                           LPMethod::PrimalSimplexBV, LPMethod::DualSimplexBV,
                           LPMethod::RevisedSimplexBV);

    Model m;
    Variable x = m.addVar(0.0, 5.0, VarType::Integer, "x");
    Variable y = m.addVar(0.0, 5.0, VarType::Integer, "y");
    m.addLPConstraint(2.0 * x + 2.0 * y, Sense::GreaterEq, 7.0);
    m.setObjective(1.0 * x + 1.0 * y, ObjSense::Minimize);

    BBOptions opts;
    opts.branchStrat      = BranchStrategy::PseudoCost;
    opts.enableCuts       = true;
    opts.maxCutsPerNode   = 5;
    opts.lpOpts.method    = method;
    opts.enablePresolve   = false;

    DYNAMIC_SECTION("method=" << to_string(method)) {
        MILPResult r = solveMILP(m, opts);

        REQUIRE(r.status == MILPStatus::Optimal);
        REQUIRE_THAT(r.objectiveValue, WithinAbs(4.0, kTol));
    }
}

// ── Test 6: B&C infeasible MILP ─────────────────────────────────────────────
//
// LP-infeasible problem → MILP infeasible even with cuts enabled.

TEST_CASE("BnC: infeasible problem stays infeasible with cuts", "[bnc][cuts]") {
    auto method = GENERATE(LPMethod::Auto,
                           LPMethod::PrimalSimplex,   LPMethod::DualSimplex,
                           LPMethod::RevisedSimplex,
                           LPMethod::PrimalSimplexBV, LPMethod::DualSimplexBV,
                           LPMethod::RevisedSimplexBV);

    Model m;
    Variable x = m.addVar(0.0, 10.0, VarType::Integer, "x");
    Variable y = m.addVar(0.0, 10.0, VarType::Integer, "y");
    m.addLPConstraint(1.0 * x + 1.0 * y, Sense::LessEq, -1.0);
    m.setObjective(1.0 * x + 1.0 * y, ObjSense::Minimize);

    BBOptions opts;
    opts.enableCuts      = true;
    opts.collectStats    = true;
    opts.lpOpts.method   = method;
    opts.enablePresolve  = false;

    DYNAMIC_SECTION("method=" << to_string(method)) {
        MILPResult r = solveMILP(m, opts);

        REQUIRE(r.status == MILPStatus::Infeasible);
        REQUIRE(r.primalValues.empty());
        REQUIRE(r.stats->cutsAdded == 0);
    }
}

// ── Test 7: B&C 3-variable problem ──────────────────────────────────────────
//
// max 5x + 4y + 3z
// s.t. 2x + y + z ≤ 6
//      x + 2y + z ≤ 6
//      x, y, z ∈ Z, 0 ≤ x,y,z ≤ 5
//
// IP optimal: (2,2,0) → obj=18.

TEST_CASE("BnC: 3-variable MILP correct with cuts", "[bnc][cuts]") {
    auto method = GENERATE(LPMethod::Auto,
                           LPMethod::PrimalSimplex,   LPMethod::DualSimplex,
                           LPMethod::RevisedSimplex,
                           LPMethod::PrimalSimplexBV, LPMethod::DualSimplexBV,
                           LPMethod::RevisedSimplexBV);

    Model m;
    Variable x = m.addVar(0.0, 5.0, VarType::Integer, "x");
    Variable y = m.addVar(0.0, 5.0, VarType::Integer, "y");
    Variable z = m.addVar(0.0, 5.0, VarType::Integer, "z");
    m.addLPConstraint(2.0*x + 1.0*y + 1.0*z, Sense::LessEq, 6.0);
    m.addLPConstraint(1.0*x + 2.0*y + 1.0*z, Sense::LessEq, 6.0);
    m.setObjective(5.0*x + 4.0*y + 3.0*z, ObjSense::Maximize);

    BBOptions noCuts;
    noCuts.enableCuts      = false;
    noCuts.lpOpts.method   = method;
    noCuts.enablePresolve  = false;

    BBOptions withCuts;
    withCuts.enableCuts      = true;
    withCuts.maxCutsPerNode  = 10;
    withCuts.branchStrat     = BranchStrategy::PseudoCost;
    withCuts.lpOpts.method   = method;
    withCuts.enablePresolve  = false;

    DYNAMIC_SECTION("method=" << to_string(method)) {
        MILPResult r1 = solveMILP(m, noCuts);
        MILPResult r2 = solveMILP(m, withCuts);

        REQUIRE(r1.status == MILPStatus::Optimal);
        REQUIRE(r2.status == MILPStatus::Optimal);
        REQUIRE_THAT(r1.objectiveValue, WithinAbs(r2.objectiveValue, kTol));
        REQUIRE(r1.objectiveValue >= 18.0 - kTol);
    }
}

// ── Test 8: computeCutData reports fractional integers only ─────────────────
//
// min 0x + y  s.t. x + y ≥ 3.5,  x ∈ Z[0,3.2],  y ∈ C[0,5].
//
// LP optimal: x=3.2 (fractional integer at its UB), y=0.3 (continuous).
// fractionalRows must contain exactly x; y must not appear.
// generateGMICuts must produce exactly one cut from the x row.

TEST_CASE("CutData: ignore continuous variables, detect fractional integer", "[cuts]") {
    auto method = GENERATE(LPMethod::Auto,
                           LPMethod::PrimalSimplex,   LPMethod::DualSimplex,
                           LPMethod::RevisedSimplex,
                           LPMethod::PrimalSimplexBV, LPMethod::DualSimplexBV,
                           LPMethod::RevisedSimplexBV);

    Model m;
    Variable x = m.addVar(0.0, 3.2, VarType::Integer, "x");
    Variable y = m.addVar(0.0, 5.0, VarType::Continuous, "y");
    m.addLPConstraint(1.0 * x + 1.0 * y, Sense::GreaterEq, 3.5);
    m.setObjective(0.0 * x + 1.0 * y, ObjSense::Minimize);

    // Solve relaxation LP
    LPOptions lpOpts;
    lpOpts.computeCutData = true;
    lpOpts.method         = method;
    lpOpts.enablePresolve = false;
    LPDetailedResult lp = solveLPDetailed(m, lpOpts);

    DYNAMIC_SECTION("method=" << to_string(method)) {
        REQUIRE(lp.result.status == LPStatus::Optimal);

        for (FractionalRow fr : lp.fractionalRows) {
            REQUIRE(fr.origVarId == x.id);
            REQUIRE(fr.origVarId != y.id);
        }

        REQUIRE(lp.fractionalRows.size() >= 1);

        std::vector<Cut> cuts = generateGMICuts(lp.fractionalRows, lp.basis, m, 50, kTol);

        REQUIRE(cuts.size() == 1);
    }
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
    auto method = GENERATE(LPMethod::Auto,
                           LPMethod::PrimalSimplex,   LPMethod::DualSimplex,
                           LPMethod::RevisedSimplex,
                           LPMethod::PrimalSimplexBV, LPMethod::DualSimplexBV,
                           LPMethod::RevisedSimplexBV);

    Model m;

    Variable x = m.addVar(0.0, 10.0, VarType::Integer, "x");
    Variable y = m.addVar(0.0, 10.0, VarType::Integer, "y");
    Variable z = m.addVar(0.0, 2.0, VarType::Continuous, "z");
    Variable w = m.addVar(0.0, 2.0, VarType::Continuous, "w");
    m.addLPConstraint(2*x + 2*y + 1*z + 1*w, Sense::GreaterEq, 7.5);
    m.setObjective(1*x + 1*y + 0.1*z + 0.1*w, ObjSense::Minimize);

    BBOptions opts;
    opts.enableCuts      = true;
    opts.maxCutsPerNode  = 5;
    opts.collectStats    = true;
    opts.lpOpts.method   = method;

    DYNAMIC_SECTION("method=" << to_string(method)) {
        MILPResult r = solveMILP(m, opts);

        REQUIRE(r.status == MILPStatus::Optimal);

        REQUIRE(r.stats->cutsAdded > 0);

        REQUIRE(std::abs(r.primalValues[x.id] - std::round(r.primalValues[x.id])) < kTol);
        REQUIRE(std::abs(r.primalValues[y.id] - std::round(r.primalValues[y.id])) < kTol);
    }
}

// ── Diagnostic: knapsack-10 GMI cut coefficients ────────────────────────────
//
// Solve the LP relaxation of the 10-item knapsack with PrimalSimplexBV,
// generate GMI cuts, then check cut coefficients and RHS. Then add the cut
// to the model, re-solve, and verify the LP objective after the cut.

TEST_CASE("Diag: knapsack-10 GMI cut with BV methods", "[bnc][diag]") {
    auto method = GENERATE(LPMethod::PrimalSimplex, LPMethod::DualSimplex,
                           LPMethod::PrimalSimplexBV, LPMethod::DualSimplexBV,
                           LPMethod::RevisedSimplexBV);
    DYNAMIC_SECTION("method=" << to_string(method)) {
        Model m = baguette_test::makeKnapsack10();

        LPOptions lpOpts;
        lpOpts.method         = method;
        lpOpts.computeCutData = true;
        LPDetailedResult lp   = solveLPDetailed(m, lpOpts);

        REQUIRE(lp.result.status == LPStatus::Optimal);
        REQUIRE_THAT(lp.result.objectiveValue, WithinAbs(110.0, kTol));

        REQUIRE(!lp.fractionalRows.empty());

        std::vector<Cut> cuts = generateGMICuts(lp.fractionalRows, lp.basis, m, 10, kTol);
        REQUIRE(cuts.size() == 1);

        const Cut& cut = cuts[0];

        // Verify IP optimal satisfies the cut (x[0..7]=1, x[8]=0, x[9]=1).
        double lhsIP = 0.0;
        double ipVals[10] = {1,1,1,1,1,1,1,1,0,1};
        for (std::size_t k = 0; k < cut.expr.size(); ++k)
            lhsIP += cut.expr.coeffs[k] * ipVals[cut.expr.varIds[k]];
        
        REQUIRE(lhsIP >= cut.rhs - kTol);

        // Add the cut and re-solve.
        m.addLPConstraint(cut.expr, Sense::GreaterEq, cut.rhs);
        LPOptions lpOpts2;
        lpOpts2.method = method;
        LPDetailedResult lp2 = solveLPDetailed(m, lpOpts2);

        REQUIRE(lp2.result.status == LPStatus::Optimal);
        REQUIRE(lp2.result.objectiveValue >= 106.0 - kTol);
    }
}

// ── Test 10: MILP infeasible even though LP relaxation is feasible ──────────
//
// min x  s.t. x = 0.5,  x ∈ Z[0,1].
//
// LP optimal: x=0.5 (feasible). Branching: x≤0 violates x=0.5; x≥1 violates x=0.5.
// Both children are LP-infeasible → MILP Infeasible.

TEST_CASE("BnC: MILP infeasible but LP feasible", "[bnc][edge][cuts]") {
    auto method = GENERATE(LPMethod::Auto,
                           LPMethod::PrimalSimplex,   LPMethod::DualSimplex,
                           LPMethod::RevisedSimplex,
                           LPMethod::PrimalSimplexBV, LPMethod::DualSimplexBV,
                           LPMethod::RevisedSimplexBV);

    Model m;

    Variable x = m.addVar(0.0, 1.0, VarType::Integer, "x");

    m.addLPConstraint(1.0 * x, Sense::Equal, 0.5);
    m.setObjective(1.0 * x, ObjSense::Minimize);

    BBOptions opts;
    opts.enableCuts      = true;
    opts.lpOpts.method   = method;

    LPOptions lpOpts;
    lpOpts.method = method;
    LPDetailedResult lp = solveLPDetailed(m, lpOpts);

    DYNAMIC_SECTION("method=" << to_string(method)) {
        REQUIRE(lp.result.status == LPStatus::Optimal);

        MILPResult r = solveMILP(m, opts);

        REQUIRE(r.status == MILPStatus::Infeasible);
    }
}

// ── D4: maxTotalCuts caps total GMI cuts across all nodes ────────────────────
//
// min x + y,  2x + 2y >= 7,  x,y ∈ Z[0,5].
// LP relaxation: x+y = 3.5 (fractional) → the solver generates at least 1 cut
// without a cap.  With maxTotalCuts=1: cutsAdded ≤ 1, result still optimal.
// With maxTotalCuts=0 (unlimited): cutsAdded matches the uncapped run.

TEST_CASE("BnC: maxTotalCuts caps total cuts across all nodes", "[bnc][cuts]") {
    Model m;
    Variable x = m.addVar(0.0, 5.0, VarType::Integer, "x");
    Variable y = m.addVar(0.0, 5.0, VarType::Integer, "y");
    m.addLPConstraint(2.0*x + 2.0*y, Sense::GreaterEq, 7.0);
    m.setObjective(1.0*x + 1.0*y, ObjSense::Minimize);

    BBOptions base;
    base.enableCuts      = true;
    base.maxCutsPerNode  = 10;
    base.collectStats    = true;
    base.enablePresolve  = false;
    base.lpOpts.method   = LPMethod::DualSimplexBV;

    // Unlimited: at least 1 cut expected (LP relaxation is fractional at root).
    BBOptions unlimited = base;
    unlimited.maxTotalCuts = 0;
    MILPResult rUnlimited = solveMILP(m, unlimited);
    REQUIRE(rUnlimited.status == MILPStatus::Optimal);
    REQUIRE_THAT(rUnlimited.objectiveValue, WithinAbs(4.0, kTol));
    REQUIRE(rUnlimited.stats->cutsAdded >= 1);

    // Capped at 1: still optimal, total cuts ≤ 1.
    BBOptions capped = base;
    capped.maxTotalCuts = 1;
    MILPResult rCapped = solveMILP(m, capped);
    REQUIRE(rCapped.status == MILPStatus::Optimal);
    REQUIRE_THAT(rCapped.objectiveValue, WithinAbs(4.0, kTol));
    REQUIRE(rCapped.stats->cutsAdded <= 1);
}
