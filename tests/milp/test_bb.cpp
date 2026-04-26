#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "baguette/lp/LPSolver.hpp"
#include "baguette/milp/BranchAndBound.hpp"
#include "baguette/model/Model.hpp"
#include "baguette/model/ModelEnums.hpp"

using namespace baguette;
using Catch::Matchers::WithinAbs;

static constexpr double kTol  = 1e-9;

// ── Test 1: single binary variable, Maximize ──────────────────────────────────
//
// max x,  x ∈ {0,1}
// LP relaxation optimal: x = 1 (already integer) → 1 node.

TEST_CASE("BB: single binary variable, maximize", "[bb]") {
    Model m;
    Variable x = m.addVar(0.0, 1.0, VarType::Binary, "x");
    m.setObjective(1.0 * x, ObjSense::Maximize);

    MILPResult r = solveMILP(m);

    REQUIRE(r.status == MILPStatus::Optimal);
    REQUIRE_THAT(r.objectiveValue, WithinAbs(1.0, kTol));
    REQUIRE_THAT(r.primalValues[x.id], WithinAbs(1.0, kTol));
}

// ── Test 2: single binary variable, Minimize ──────────────────────────────────
//
// min x,  x ∈ {0,1}
// Optimal: x = 0, obj = 0.

TEST_CASE("BB: single binary variable, minimize", "[bb]") {
    Model m;
    Variable x = m.addVar(0.0, 1.0, VarType::Binary, "x");
    m.setObjective(1.0 * x, ObjSense::Minimize);

    MILPResult r = solveMILP(m);

    REQUIRE(r.status == MILPStatus::Optimal);
    REQUIRE_THAT(r.objectiveValue, WithinAbs(0.0, kTol));
    REQUIRE_THAT(r.primalValues[x.id], WithinAbs(0.0, kTol));
}

// ── Test 3: knapsack requiring branching ──────────────────────────────────────
//
// max 5x + 4y
// s.t. 3x + 2y ≤ 7
//      x, y ∈ Z, 0 ≤ x,y ≤ 5
//
// LP relaxation optimum: (x,y) = (0, 3.5), obj = 14 (fractional).
// B&B tree:
//   Root LP: (0, 3.5) → branch on y.
//   Left  (y≤3): LP (1/3, 3), obj ≈ 13.67 → branch on x.
//     Left-Left  (x≤0, y≤3): LP (0, 3), obj = 12. Integer! Incumbent = 12.
//     Left-Right (x≥1, y≤3): LP (1, 2), obj = 13. Integer! Incumbent = 13.
//   Right (y≥4): 3x + 8 ≤ 7 → 3x ≤ -1 → infeasible. Pruned.
// Optimal: x=1, y=2, obj=13.

TEST_CASE("BB: knapsack with branching, maximize", "[bb]") {
    Model m;
    Variable x = m.addVar(0.0, 5.0, VarType::Integer, "x");
    Variable y = m.addVar(0.0, 5.0, VarType::Integer, "y");
    m.addLPConstraint(3.0 * x + 2.0 * y, Sense::LessEq, 7.0);
    m.setObjective(5.0 * x + 4.0 * y, ObjSense::Maximize);

    BBOptions opts;
    opts.collectStats = true;
    MILPResult r = solveMILP(m, opts);

    REQUIRE(r.status == MILPStatus::Optimal);
    REQUIRE_THAT(r.objectiveValue, WithinAbs(13.0, kTol));
    REQUIRE_THAT(r.primalValues[x.id], WithinAbs(1.0, kTol));
    REQUIRE_THAT(r.primalValues[y.id], WithinAbs(2.0, kTol));
    // At least root + left + right children + their children.
    REQUIRE(r.stats->nodesExplored >= 3);
}

// ── Test 4: MILP infeasible by LP relaxation ──────────────────────────────────
//
// min x + y
// s.t. x + y ≤ -1   (infeasible for x,y ≥ 0)
//      x, y ∈ Z+

TEST_CASE("BB: LP relaxation infeasible -> MILP infeasible", "[bb]") {
    Model m;
    Variable x = m.addVar(0.0, 10.0, VarType::Integer, "x");
    Variable y = m.addVar(0.0, 10.0, VarType::Integer, "y");
    m.addLPConstraint(1.0 * x + 1.0 * y, Sense::LessEq, -1.0);
    m.setObjective(1.0 * x + 1.0 * y, ObjSense::Minimize);

    LPResult lpResult = solve(m);

    REQUIRE(lpResult.status == LPStatus::Infeasible);

    MILPResult r = solveMILP(m);

    REQUIRE(r.status == MILPStatus::Infeasible);
    REQUIRE(r.primalValues.empty());
}

// ── Test 5: MILP infeasible by integer constraints (LP relaxation feasible) ───
//
// min x
// s.t. x ≥ 0.3
//      x ≤ 0.7
//      x ∈ Z, 0 ≤ x ≤ 10
//
// LP relaxation: x = 0.3, feasible.
// Integer solutions in [0.3, 0.7]: none (floor=0 violates x≥0.3, ceil=1 violates x≤0.7).
// B&B tree:
//   Root LP: x = 0.3 → fractional → branch on x.
//   Left  (x ≤ 0): x ≥ 0.3 and x ≤ 0 → infeasible. Pruned.
//   Right (x ≥ 1): x ≤ 0.7 and x ≥ 1 → infeasible. Pruned.
// Result: Infeasible.

TEST_CASE("BB: integer infeasible (LP relaxation feasible)", "[bb]") {
    Model m;
    Variable x = m.addVar(0.0, 10.0, VarType::Integer, "x");
    m.addLPConstraint(1.0 * x, Sense::GreaterEq, 0.3);
    m.addLPConstraint(1.0 * x, Sense::LessEq,    0.7);
    m.setObjective(1.0 * x, ObjSense::Minimize);

    LPResult lpResult = solve(m);

    REQUIRE(lpResult.status == LPStatus::Optimal);

    MILPResult r = solveMILP(m);

    REQUIRE(r.status == MILPStatus::Infeasible);
    REQUIRE(r.primalValues.empty());
}

// ── Test 6: pure LP (no integer variables) ────────────────────────────────────
//
// min -x
// s.t. x ≤ 3  (all continuous)
// Optimal: x = 3, obj = -3.  Should be solved in exactly 1 node.

TEST_CASE("BB: pure LP (no integer variables)", "[bb]") {
    Model m;
    Variable x = m.addVar(0.0, 3.0, VarType::Continuous, "x");
    m.setObjective(-1.0 * x, ObjSense::Minimize);

    BBOptions opts;
    opts.collectStats = true;
    MILPResult r = solveMILP(m, opts);

    REQUIRE(r.status == MILPStatus::Optimal);
    REQUIRE_THAT(r.objectiveValue, WithinAbs(-3.0, kTol));
    REQUIRE_THAT(r.primalValues[x.id], WithinAbs(3.0, kTol));
    REQUIRE(r.stats->nodesExplored == 1);
}

// ── Test 7: depth-first gives same optimal as best-bound ─────────────────────

TEST_CASE("BB: depth-first gives same optimal as best-bound", "[bb]") {
    Model m;
    Variable x = m.addVar(0.0, 5.0, VarType::Integer, "x");
    Variable y = m.addVar(0.0, 5.0, VarType::Integer, "y");
    m.addLPConstraint(3.0 * x + 2.0 * y, Sense::LessEq, 7.0);
    m.setObjective(5.0 * x + 4.0 * y, ObjSense::Maximize);

    BBOptions bestBoundOpts;
    bestBoundOpts.nodeSelect = NodeSelection::BestBound;

    BBOptions depthFirstOpts;
    depthFirstOpts.nodeSelect = NodeSelection::DepthFirst;

    MILPResult r1 = solveMILP(m, bestBoundOpts);
    MILPResult r2 = solveMILP(m, depthFirstOpts);

    REQUIRE(r1.status == MILPStatus::Optimal);
    REQUIRE(r2.status == MILPStatus::Optimal);
    REQUIRE_THAT(r1.objectiveValue, WithinAbs(r2.objectiveValue, kTol));
    REQUIRE_THAT(r1.primalValues[x.id], WithinAbs(r2.primalValues[x.id], kTol));
    REQUIRE_THAT(r1.primalValues[y.id], WithinAbs(r2.primalValues[y.id], kTol));
}

// ── Test 8: node limit stops search ───────────────────────────────────────────

TEST_CASE("BB: maxNodes=1 stops after root", "[bb]") {
    Model m;
    Variable x = m.addVar(0.0, 5.0, VarType::Integer, "x");
    Variable y = m.addVar(0.0, 5.0, VarType::Integer, "y");
    m.addLPConstraint(3.0 * x + 2.0 * y, Sense::LessEq, 7.0);
    m.setObjective(5.0 * x + 4.0 * y, ObjSense::Maximize);

    BBOptions opts;
    opts.maxNodes     = 1;
    opts.collectStats = true;

    MILPResult r = solveMILP(m, opts);

    // Root LP is fractional → no integer solution found in 1 node.
    REQUIRE(r.status == MILPStatus::MaxNodes);
    REQUIRE(r.stats->nodesExplored == 1);
    REQUIRE(r.primalValues.empty());
}

// ── Test 9: delta-trail restore — deep tree, BestBound vs DepthFirst ────────
//
// max 3x + 5y + 2z + 4w
//   s.t. 2x + 3y + z + 2w ≤ 10
//        x + 2y + 3z + w  ≤ 10
//        x,y,z,w ∈ Z[0,4]
//
// Both BestBound (non-LIFO jumps between nodes) and DepthFirst (deep path then
// backtrack) exercise the delta-trail restore at multiple depths. They must
// agree on the optimal objective, validating that bound restoration is correct
// regardless of node processing order.

TEST_CASE("BB: delta-trail restore correct for deep tree (BestBound vs DepthFirst)", "[bb]") {
    Model m;
    Variable x = m.addVar(0.0, 4.0, VarType::Integer, "x");
    Variable y = m.addVar(0.0, 4.0, VarType::Integer, "y");
    Variable z = m.addVar(0.0, 4.0, VarType::Integer, "z");
    Variable w = m.addVar(0.0, 4.0, VarType::Integer, "w");
    m.addLPConstraint(2.0*x + 3.0*y + 1.0*z + 2.0*w, Sense::LessEq, 10.0);
    m.addLPConstraint(1.0*x + 2.0*y + 3.0*z + 1.0*w, Sense::LessEq, 10.0);
    m.setObjective(3.0*x + 5.0*y + 2.0*z + 4.0*w, ObjSense::Maximize);

    BBOptions bb, df;
    bb.nodeSelect = NodeSelection::BestBound;
    df.nodeSelect = NodeSelection::DepthFirst;

    MILPResult r1 = solveMILP(m, bb);
    MILPResult r2 = solveMILP(m, df);

    REQUIRE(r1.status == MILPStatus::Optimal);
    REQUIRE(r2.status == MILPStatus::Optimal);
    REQUIRE_THAT(r1.objectiveValue, WithinAbs(r2.objectiveValue, kTol));
}

// ── Test 10: HybridPlunge finds same optimal as BestBound ────────────────────
//
// max 5x + 4y  s.t. 3x + 2y ≤ 7,  x,y ∈ Z[0,5].  IP optimal: obj=13.
//
// HybridPlunge plunges DFS to the first incumbent, then heapifies the queue
// and finishes in BestBound mode. The optimal must match pure BestBound and
// pure DepthFirst.

TEST_CASE("BB: HybridPlunge finds same optimal as BestBound and DepthFirst", "[bb]") {
    Model m;
    Variable x = m.addVar(0.0, 5.0, VarType::Integer, "x");
    Variable y = m.addVar(0.0, 5.0, VarType::Integer, "y");
    m.addLPConstraint(3.0 * x + 2.0 * y, Sense::LessEq, 7.0);
    m.setObjective(5.0 * x + 4.0 * y, ObjSense::Maximize);

    BBOptions bb, df, hp;
    bb.nodeSelect = NodeSelection::BestBound;
    df.nodeSelect = NodeSelection::DepthFirst;
    hp.nodeSelect = NodeSelection::HybridPlunge;

    MILPResult r1 = solveMILP(m, bb);
    MILPResult r2 = solveMILP(m, df);
    MILPResult r3 = solveMILP(m, hp);

    REQUIRE(r1.status == MILPStatus::Optimal);
    REQUIRE(r2.status == MILPStatus::Optimal);
    REQUIRE(r3.status == MILPStatus::Optimal);
    REQUIRE_THAT(r3.objectiveValue, WithinAbs(r1.objectiveValue, kTol));
    REQUIRE_THAT(r3.objectiveValue, WithinAbs(r2.objectiveValue, kTol));
    REQUIRE_THAT(r3.primalValues[x.id], WithinAbs(r1.primalValues[x.id], kTol));
    REQUIRE_THAT(r3.primalValues[y.id], WithinAbs(r1.primalValues[y.id], kTol));
}
