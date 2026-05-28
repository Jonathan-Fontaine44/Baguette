#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cstdio>

#include "baguette/lp/LPSolver.hpp"
#include "baguette/milp/BranchAndBound.hpp"
#include "baguette/model/Model.hpp"
#include "baguette/model/ModelEnums.hpp"
#include "lp/MILP_problems.hpp"

using namespace baguette;
using Catch::Matchers::WithinAbs;

static constexpr double kTol = 1e-6;

// â”€â”€ Test 1: enableCuts=false â†’ cutsAdded=0 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
    opts.presolveLevel   =  0;
    opts.timeLimitS       = 1.0;

    DYNAMIC_SECTION("method=" << to_string(method)) {
        MILPResult r = solveMILP(m, opts);

        REQUIRE(r.status == MILPStatus::Optimal);
        REQUIRE(r.stats->cutsAdded == 0);
        REQUIRE_THAT(r.objectiveValue, WithinAbs(13.0, kTol));
    }
}

// â”€â”€ Test 2: B&C gives same answer as B&B â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//
// max 5x + 4y  s.t. 3x + 2y â‰¤ 7, x,y âˆˆ Zâ‰¥0.
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
    noCuts.presolveLevel    =  0;
    noCuts.timeLimitS       = 1.0;

    BBOptions withCuts;
    withCuts.enableCuts      = true;
    withCuts.maxCutsPerNode  = 10;
    withCuts.lpOpts.method   = method;
    withCuts.presolveLevel  =  0;

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

// â”€â”€ Test 3: GMI cut closes LP-IP gap at root â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//
// min x + y   s.t. 2x + 2y â‰¥ 7,  x, y âˆˆ Z,  0 â‰¤ x,y â‰¤ 5.
//
// LP relaxation optimal: x+y = 3.5 (e.g. x=3.5, y=0), obj=3.5.
// GMI cut from the fractional row: -2x - 2y â‰¥ -8, i.e. x+y â‰¤ 4.
// After adding the cut the LP optimal is x+y=4 (integer) â†’ solved at the root.
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
    noCuts.presolveLevel    =  0;
    noCuts.timeLimitS       = 1.0;

    BBOptions withCuts;
    withCuts.enableCuts      = true;
    withCuts.maxCutsPerNode  = 10;
    withCuts.collectStats    = true;
    withCuts.lpOpts.method   = method;
    withCuts.presolveLevel  =  0;
    withCuts.timeLimitS       = 1.0;

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

// â”€â”€ Test 4: PseudoCost branching gives correct answer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
    mf.presolveLevel   =  0;
    mf.timeLimitS       = 1.0;

    BBOptions pc;
    pc.branchStrat      = BranchStrategy::PseudoCost;
    pc.enableCuts       = false;
    pc.lpOpts.method    = method;
    pc.presolveLevel   =  0;
    pc.timeLimitS       = 1.0;

    DYNAMIC_SECTION("method=" << to_string(method)) {
        MILPResult r1 = solveMILP(m, mf);
        MILPResult r2 = solveMILP(m, pc);

        REQUIRE(r1.status == MILPStatus::Optimal);
        REQUIRE(r2.status == MILPStatus::Optimal);
        REQUIRE_THAT(r1.objectiveValue, WithinAbs(r2.objectiveValue, kTol));
    }
}

// â”€â”€ Test 5: PseudoCost + cuts give correct answer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
    opts.presolveLevel   =  0;
    opts.timeLimitS       = 1.0;

    DYNAMIC_SECTION("method=" << to_string(method)) {
        MILPResult r = solveMILP(m, opts);

        REQUIRE(r.status == MILPStatus::Optimal);
        REQUIRE_THAT(r.objectiveValue, WithinAbs(4.0, kTol));
    }
}

// â”€â”€ Test 6: B&C infeasible MILP â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//
// LP-infeasible problem â†’ MILP infeasible even with cuts enabled.

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
    opts.presolveLevel  =  0;
    opts.timeLimitS       = 1.0;

    DYNAMIC_SECTION("method=" << to_string(method)) {
        MILPResult r = solveMILP(m, opts);

        REQUIRE(r.status == MILPStatus::Infeasible);
        REQUIRE(r.primalValues.empty());
        REQUIRE(r.stats->cutsAdded == 0);
    }
}

// â”€â”€ Test 7: B&C 3-variable problem â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//
// max 5x + 4y + 3z
// s.t. 2x + y + z â‰¤ 6
//      x + 2y + z â‰¤ 6
//      x, y, z âˆˆ Z, 0 â‰¤ x,y,z â‰¤ 5
//
// IP optimal: (2,2,0) â†’ obj=18.

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
    noCuts.presolveLevel  =  0;
    noCuts.timeLimitS      = 1.0;

    BBOptions withCuts;
    withCuts.enableCuts      = true;
    withCuts.maxCutsPerNode  = 10;
    withCuts.branchStrat     = BranchStrategy::PseudoCost;
    withCuts.lpOpts.method   = method;
    withCuts.presolveLevel  =  0;
    withCuts.timeLimitS      = 1.0;

    DYNAMIC_SECTION("method=" << to_string(method)) {
        MILPResult r1 = solveMILP(m, noCuts);
        MILPResult r2 = solveMILP(m, withCuts);

        REQUIRE(r1.status == MILPStatus::Optimal);
        REQUIRE(r2.status == MILPStatus::Optimal);
        REQUIRE_THAT(r1.objectiveValue, WithinAbs(r2.objectiveValue, kTol));
        REQUIRE(r1.objectiveValue >= 18.0 - kTol);
    }
}

// â”€â”€ Test 8: computeCutData reports fractional integers only â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//
// min 0x + y  s.t. x + y â‰¥ 3.5,  x âˆˆ Z[0,3.2],  y âˆˆ C[0,5].
//
// LP optimal: x=3.2 (fractional integer at its UB), y=0.3 (continuous).
// fractionalRows must contain exactly x (integer); y (continuous) must not appear.

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

    LPOptions lpOpts;
    lpOpts.computeCutData = true;
    lpOpts.method         = method;
    lpOpts.enablePresolve = false;
    lpOpts.timeLimitS     = 1.0;
    LPDetailedResult lp = solveLPDetailed(m, lpOpts);

    DYNAMIC_SECTION("method=" << to_string(method)) {
        REQUIRE(lp.result.status == LPStatus::Optimal);
        REQUIRE(lp.fractionalRows.size() >= 1);
        for (const FractionalRow& fr : lp.fractionalRows) {
            REQUIRE(fr.origVarId == x.id);   // only x (integer) appears
            REQUIRE(fr.origVarId != y.id);   // y (continuous) must not appear
        }
    }
}

// â”€â”€ Test 9: B&C cuts bind on integer variables in a mixed MILP â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//
// min x + y + 0.1z + 0.1w  s.t. 2x + 2y + z + w â‰¥ 7.5,
//   x,y âˆˆ Z[0,10],  z,w âˆˆ C[0,2].
//
// z,w capped at 2 (max contribution 4 < 7.5), so 2x + 2y â‰¥ 3.5 is required.
// LP leaves x or y fractional â†’ at least one GMI cut is added.
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
    opts.lpOpts.timeLimitS = 1.0;
    DYNAMIC_SECTION("method=" << to_string(method)) {
        MILPResult r = solveMILP(m, opts);

        REQUIRE(r.status == MILPStatus::Optimal);

        REQUIRE(r.stats->cutsAdded > 0);

        REQUIRE(std::abs(r.primalValues[x.id] - std::round(r.primalValues[x.id])) < kTol);
        REQUIRE(std::abs(r.primalValues[y.id] - std::round(r.primalValues[y.id])) < kTol);
    }
}


// â”€â”€ Test 10: MILP infeasible even though LP relaxation is feasible â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//
// min x  s.t. x = 0.5,  x âˆˆ Z[0,1].
//
// LP optimal: x=0.5 (feasible). Branching: xâ‰¤0 violates x=0.5; xâ‰¥1 violates x=0.5.
// Both children are LP-infeasible â†’ MILP Infeasible.

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
    lpOpts.timeLimitS = 1.0;
    LPDetailedResult lp = solveLPDetailed(m, lpOpts);

    DYNAMIC_SECTION("method=" << to_string(method)) {
        REQUIRE(lp.result.status == LPStatus::Optimal);

        MILPResult r = solveMILP(m, opts);

        REQUIRE(r.status == MILPStatus::Infeasible);
    }
}

// â”€â”€ User CutGenerator: gap closed at root without GMI â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//
// min x + y   s.t. 2x + 2y â‰¥ 7,  x, y âˆˆ Z[0,5].
// LP: x+y = 3.5.  IP: x+y = 4.
//
// The user generator injects x+y â‰¥ 4 the first time it sees a fractional LP.
// This closes the LP-IP gap at the root â†’ Optimal in 1 node, cutsAdded = 1.
// GMI is disabled so the generator is solely responsible for the cut.

TEST_CASE("BnC: user CutGenerator closes gap at root (no GMI)", "[bnc][cuts][callback]") {
    Model m;
    Variable x = m.addVar(0.0, 5.0, VarType::Integer, "x");
    Variable y = m.addVar(0.0, 5.0, VarType::Integer, "y");
    m.addLPConstraint(2.0 * x + 2.0 * y, Sense::GreaterEq, 7.0);
    m.setObjective(1.0 * x + 1.0 * y, ObjSense::Minimize);

    bool fired = false;
    CutGenerator gen = [&](const LPDetailedResult& lp, const Model&) -> std::vector<Cut> {
        const double sum = lp.result.primalValues[x.id] + lp.result.primalValues[y.id];
        if (fired || sum > 4.0 - kTol) return {};
        fired = true;
        Cut c;
        c.expr  = 1.0 * x + 1.0 * y;
        c.sense = Sense::GreaterEq;
        c.rhs   = 4.0;
        return {c};
    };

    BBOptions opts;
    opts.enableCuts     = false;   // GMI disabled â€” only user generator
    opts.collectStats   = true;
    opts.cutGenerators  = {gen};
    opts.presolveLevel = 0;
    opts.timeLimitS     = 5.0;

    MILPResult r = solveMILP(m, opts);

    REQUIRE(r.status == MILPStatus::Optimal);
    REQUIRE_THAT(r.objectiveValue, WithinAbs(4.0, kTol));
    REQUIRE(r.stats->nodesExplored == 1);  // gap closed at root, no branching
    REQUIRE(r.stats->cutsAdded == 1);
}

// â”€â”€ User CutGenerator: empty generator has no effect â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//
// A generator that always returns {} must leave the B&B result unchanged.

TEST_CASE("BnC: empty user CutGenerator has no effect", "[bnc][cuts][callback]") {
    Model m;
    Variable x = m.addVar(0.0, 5.0, VarType::Integer, "x");
    Variable y = m.addVar(0.0, 5.0, VarType::Integer, "y");
    m.addLPConstraint(3.0 * x + 2.0 * y, Sense::LessEq, 7.0);
    m.setObjective(5.0 * x + 4.0 * y, ObjSense::Maximize);

    BBOptions base;
    base.enableCuts     = false;
    base.presolveLevel = 0;
    base.timeLimitS     = 5.0;

    BBOptions withEmptyGen = base;
    withEmptyGen.cutGenerators = {
        [](const LPDetailedResult&, const Model&) { return std::vector<Cut>{}; }
    };

    MILPResult r1 = solveMILP(m, base);
    MILPResult r2 = solveMILP(m, withEmptyGen);

    REQUIRE(r1.status == MILPStatus::Optimal);
    REQUIRE(r2.status == MILPStatus::Optimal);
    REQUIRE_THAT(r1.objectiveValue, WithinAbs(r2.objectiveValue, kTol));
}

// â”€â”€ D4: maxTotalCuts caps total GMI cuts across all nodes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//
// min x + y,  2x + 2y >= 7,  x,y âˆˆ Z[0,5].
// LP relaxation: x+y = 3.5 (fractional) â†’ the solver generates at least 1 cut
// without a cap.  With maxTotalCuts=1: cutsAdded â‰¤ 1, result still optimal.
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
    base.presolveLevel  =  0;
    base.lpOpts.method   = LPMethod::DualSimplexBV;
    base.lpOpts.timeLimitS = 1.0;

    // Unlimited: at least 1 cut expected (LP relaxation is fractional at root).
    BBOptions unlimited = base;
    unlimited.maxTotalCuts = 0;
    MILPResult rUnlimited = solveMILP(m, unlimited);
    REQUIRE(rUnlimited.status == MILPStatus::Optimal);
    REQUIRE_THAT(rUnlimited.objectiveValue, WithinAbs(4.0, kTol));
    REQUIRE(rUnlimited.stats->cutsAdded >= 1);

    // Capped at 1: still optimal, total cuts â‰¤ 1.
    BBOptions capped = base;
    capped.maxTotalCuts = 1;
    MILPResult rCapped = solveMILP(m, capped);
    REQUIRE(rCapped.status == MILPStatus::Optimal);
    REQUIRE_THAT(rCapped.objectiveValue, WithinAbs(4.0, kTol));
    REQUIRE(rCapped.stats->cutsAdded <= 1);
}
