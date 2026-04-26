#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "baguette/milp/BranchAndBound.hpp"
#include "baguette/model/Model.hpp"
#include "baguette/model/ModelEnums.hpp"

using namespace baguette;
using Catch::Matchers::WithinAbs;

static constexpr double kTol = 1e-6;

// ── Test 1: collectStats=false → stats is nullopt ─────────────────────────────

TEST_CASE("Stats: collectStats=false yields nullopt", "[stats]") {
    Model m;
    Variable x = m.addVar(0.0, 1.0, VarType::Binary, "x");
    m.setObjective(1.0 * x, ObjSense::Maximize);

    BBOptions opts;
    opts.collectStats = false;
    MILPResult r = solveMILP(m, opts);

    REQUIRE(r.status == MILPStatus::Optimal);
    REQUIRE(!r.stats.has_value());
}

// ── Test 2: single-node optimal — nodesExplored=1, lpSolvesTotal=1 ───────────
//
// max x, x ∈ {0,1}.  LP relaxation already integer (x=1) → 1 node, 1 LP solve.

TEST_CASE("Stats: single-node optimal - nodesExplored=1 lpSolvesTotal=1", "[stats]") {
    Model m;
    Variable x = m.addVar(0.0, 1.0, VarType::Binary, "x");
    m.setObjective(1.0 * x, ObjSense::Maximize);

    BBOptions opts;
    opts.collectStats = true;
    MILPResult r = solveMILP(m, opts);

    REQUIRE(r.status == MILPStatus::Optimal);
    REQUIRE(r.stats->nodesExplored  == 1);
    REQUIRE(r.stats->lpSolvesTotal  == 1);
    REQUIRE(r.stats->cutsAdded      == 0);
    REQUIRE(r.stats->nodesWithCuts  == 0);
    REQUIRE(r.stats->cutsPerDepth.empty());
}

// ── Test 3: LP-infeasible root → nodesPrunedByInfeasibility = 1 ──────────────
//
// x + y ≤ -1, x,y ≥ 0  → root LP infeasible → 1 node pruned by infeasibility.

TEST_CASE("Stats: LP-infeasible root counts nodesPrunedByInfeasibility", "[stats]") {
    Model m;
    Variable x = m.addVar(0.0, 10.0, VarType::Integer, "x");
    Variable y = m.addVar(0.0, 10.0, VarType::Integer, "y");
    m.addLPConstraint(1.0 * x + 1.0 * y, Sense::LessEq, -1.0);
    m.setObjective(1.0 * x + 1.0 * y, ObjSense::Minimize);

    BBOptions opts;
    opts.collectStats = true;
    MILPResult r = solveMILP(m, opts);

    REQUIRE(r.status == MILPStatus::Infeasible);
    REQUIRE(r.stats->nodesPrunedByInfeasibility >= 1);
}

// ── Test 4: GMI cut at root ───────────────────────────────────────────────────
//
// min x + y  s.t. 2x + 2y ≥ 7,  x,y ∈ Z[0,5].
// LP relaxation optimal: x + y = 3.5 (fractional).
// GMI cut closes the gap to 4 (integer) at the root.
// Expected: 1 node, cutsAdded ≥ 1, nodesWithCuts = 1, lpSolvesTotal = 2,
//           cutsPerDepth[0] ≥ 1.

TEST_CASE("Stats: GMI cut at root populates cut stats", "[stats]") {
    Model m;
    Variable x = m.addVar(0.0, 5.0, VarType::Integer, "x");
    Variable y = m.addVar(0.0, 5.0, VarType::Integer, "y");
    m.addLPConstraint(2.0 * x + 2.0 * y, Sense::GreaterEq, 7.0);
    m.setObjective(1.0 * x + 1.0 * y, ObjSense::Minimize);

    BBOptions opts;
    opts.enableCuts     = true;
    opts.maxCutsPerNode = 10;
    opts.collectStats   = true;
    MILPResult r = solveMILP(m, opts);

    REQUIRE(r.status == MILPStatus::Optimal);
    REQUIRE_THAT(r.objectiveValue, WithinAbs(4.0, kTol));
    REQUIRE(r.stats->nodesExplored  == 1);
    REQUIRE(r.stats->cutsAdded      >= 1);
    REQUIRE(r.stats->nodesWithCuts  == 1);
    REQUIRE(r.stats->lpSolvesTotal  == 2); // base solve + re-solve after cut
    REQUIRE(!r.stats->cutsPerDepth.empty());
    REQUIRE(r.stats->cutsPerDepth[0] >= 1);
}

// ── Test 5: branching problem — nodesExplored and lpSolvesTotal ──────────────
//
// max 5x + 4y  s.t. 3x + 2y ≤ 7,  x,y ∈ Z[0,5].  Requires branching.
// nodesExplored ≥ 3, lpSolvesTotal ≥ nodesExplored (1 LP per node).

TEST_CASE("Stats: branching problem - lpSolvesTotal >= nodesExplored", "[stats]") {
    Model m;
    Variable x = m.addVar(0.0, 5.0, VarType::Integer, "x");
    Variable y = m.addVar(0.0, 5.0, VarType::Integer, "y");
    m.addLPConstraint(3.0 * x + 2.0 * y, Sense::LessEq, 7.0);
    m.setObjective(5.0 * x + 4.0 * y, ObjSense::Maximize);

    BBOptions opts;
    opts.enableCuts   = false;
    opts.collectStats = true;
    MILPResult r = solveMILP(m, opts);

    REQUIRE(r.status == MILPStatus::Optimal);
    REQUIRE(r.stats->nodesExplored >= 3);
    REQUIRE(r.stats->lpSolvesTotal >= r.stats->nodesExplored);
    REQUIRE(r.stats->cutsAdded     == 0);
    REQUIRE(r.stats->nodesWithCuts == 0);
}
