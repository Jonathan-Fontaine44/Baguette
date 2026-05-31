#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <chrono>
#include <limits>

#include "baguette/core/Sense.hpp"
#include "baguette/lp/LPSolver.hpp"
#include "baguette/lp/Presolve.hpp"
#include "baguette/milp/BranchAndBound.hpp"
#include "baguette/model/Model.hpp"
#include "baguette/model/ModelEnums.hpp"

using namespace baguette;
using Catch::Matchers::WithinAbs;

static constexpr double kTol = 1e-9;

// ── presolveTBInPlace: LEQ ──────────────────────────────────────────────────────
//
// 2x + 3y <= 12, x∈[0,10], y∈[0,10]
// min activity = 0 → x <= 6, y <= 4

TEST_CASE("Presolve: LEQ bound tightening", "[presolve]") {
    Model m;
    auto x = m.addVar(0.0, 10.0, "x");
    auto y = m.addVar(0.0, 10.0, "y");
    m.addLPConstraint(2.0*x + 3.0*y, Sense::LessEq, 12.0);
    m.setObjective(1.0*x, ObjSense::Minimize);

    PresolveResult pr = presolveTBInPlace(m);

    REQUIRE_FALSE(pr.infeasible);
    REQUIRE(pr.boundsTightened >= 2);
    REQUIRE(pr.passesRun >= 1);
    REQUIRE_THAT(m.getHot().ub[x.id], WithinAbs(6.0, kTol));
    REQUIRE_THAT(m.getHot().ub[y.id], WithinAbs(4.0, kTol));
    REQUIRE_THAT(m.getHot().lb[x.id], WithinAbs(0.0, kTol));
    REQUIRE_THAT(m.getHot().lb[y.id], WithinAbs(0.0, kTol));
}

// ── presolveTBInPlace: GEQ ──────────────────────────────────────────────────────
//
// x + y >= 5, x∈[0,3], y∈[0,4]
// max activity = 7 → x >= 1, y >= 2

TEST_CASE("Presolve: GEQ bound tightening", "[presolve]") {
    Model m;
    auto x = m.addVar(0.0, 3.0, "x");
    auto y = m.addVar(0.0, 4.0, "y");
    m.addLPConstraint(1.0*x + 1.0*y, Sense::GreaterEq, 5.0);
    m.setObjective(1.0*x, ObjSense::Minimize);

    PresolveResult pr = presolveTBInPlace(m);

    REQUIRE_FALSE(pr.infeasible);
    REQUIRE(pr.boundsTightened >= 2);
    REQUIRE_THAT(m.getHot().lb[x.id], WithinAbs(1.0, kTol));
    REQUIRE_THAT(m.getHot().lb[y.id], WithinAbs(2.0, kTol));
    REQUIRE_THAT(m.getHot().ub[x.id], WithinAbs(3.0, kTol));
    REQUIRE_THAT(m.getHot().ub[y.id], WithinAbs(4.0, kTol));
}

// ── presolveTBInPlace: EQ ───────────────────────────────────────────────────────
//
// x + y = 5, x∈[0,10], y∈[0,10] → x<=5, y<=5

TEST_CASE("Presolve: EQ bound tightening", "[presolve]") {
    Model m;
    auto x = m.addVar(0.0, 10.0, "x");
    auto y = m.addVar(0.0, 10.0, "y");
    m.addLPConstraint(1.0*x + 1.0*y, Sense::Equal, 5.0);
    m.setObjective(1.0*x, ObjSense::Minimize);

    PresolveResult pr = presolveTBInPlace(m);

    REQUIRE_FALSE(pr.infeasible);
    REQUIRE_THAT(m.getHot().ub[x.id], WithinAbs(5.0, kTol));
    REQUIRE_THAT(m.getHot().ub[y.id], WithinAbs(5.0, kTol));
}

// ── presolveTBInPlace: multi-pass ───────────────────────────────────────────────
//
// Pass 1: x+y=5 → x∈[0,5], y∈[0,5]; y+z=3 → y∈[0,3], z∈[0,3]
// Pass 2: x+y=5 with y<=3 → x >= 2

TEST_CASE("Presolve: multi-pass tightening", "[presolve]") {
    Model m;
    auto x = m.addVar(0.0, 10.0, "x");
    auto y = m.addVar(0.0, 10.0, "y");
    auto z = m.addVar(0.0, 10.0, "z");
    m.addLPConstraint(1.0*x + 1.0*y, Sense::Equal, 5.0);
    m.addLPConstraint(1.0*y + 1.0*z, Sense::Equal, 3.0);
    m.setObjective(1.0*x, ObjSense::Minimize);

    PresolveResult pr = presolveTBInPlace(m);

    REQUIRE_FALSE(pr.infeasible);
    REQUIRE(pr.passesRun >= 2);
    REQUIRE_THAT(m.getHot().lb[x.id], WithinAbs(2.0, kTol));
    REQUIRE_THAT(m.getHot().ub[x.id], WithinAbs(5.0, kTol));
    REQUIRE_THAT(m.getHot().ub[y.id], WithinAbs(3.0, kTol));
    REQUIRE_THAT(m.getHot().ub[z.id], WithinAbs(3.0, kTol));
}

// ── presolveTBInPlace: infeasibility via LEQ ────────────────────────────────────
//
// x + y <= 3, x∈[2,10], y∈[2,10] → minActivity = 4 > 3

TEST_CASE("Presolve: infeasible via LEQ", "[presolve]") {
    Model m;
    auto x = m.addVar(2.0, 10.0, "x");
    auto y = m.addVar(2.0, 10.0, "y");
    m.addLPConstraint(1.0*x + 1.0*y, Sense::LessEq, 3.0);
    m.setObjective(1.0*x, ObjSense::Minimize);

    REQUIRE(presolveTBInPlace(m).infeasible);
}

// ── presolveTBInPlace: infeasibility via GEQ ────────────────────────────────────
//
// x + y >= 8, x∈[0,3], y∈[0,3] → maxActivity = 6 < 8

TEST_CASE("Presolve: infeasible via GEQ", "[presolve]") {
    Model m;
    auto x = m.addVar(0.0, 3.0, "x");
    auto y = m.addVar(0.0, 3.0, "y");
    m.addLPConstraint(1.0*x + 1.0*y, Sense::GreaterEq, 8.0);
    m.setObjective(1.0*x, ObjSense::Minimize);

    REQUIRE(presolveTBInPlace(m).infeasible);
}

// ── presolveTBInPlace: infinite lb ──────────────────────────────────────────────
//
// x + y <= 5, x∈(-inf,100), y∈[0,3]
// x has lb=-inf but is the only inf contributor → ub_x tightened to 5

TEST_CASE("Presolve: tighten UB with infinite LB", "[presolve]") {
    const double inf = std::numeric_limits<double>::infinity();
    Model m;
    auto x = m.addVar(-inf, 100.0, "x");
    auto y = m.addVar(0.0,  3.0,   "y");
    m.addLPConstraint(1.0*x + 1.0*y, Sense::LessEq, 5.0);
    m.setObjective(1.0*x, ObjSense::Minimize);

    PresolveResult pr = presolveTBInPlace(m);

    REQUIRE_FALSE(pr.infeasible);
    REQUIRE_THAT(m.getHot().ub[x.id], WithinAbs(5.0, kTol));
}

// ── presolveTBInPlace: fixed variable detection ─────────────────────────────────
//
// x+y=4, x∈[2,2] (fixed) → y fixed to 2

TEST_CASE("Presolve: fixed variable counting", "[presolve]") {
    Model m;
    auto x = m.addVar(2.0, 2.0, "x");
    auto y = m.addVar(0.0, 10.0, "y");
    m.addLPConstraint(1.0*x + 1.0*y, Sense::Equal, 4.0);
    m.setObjective(1.0*y, ObjSense::Minimize);

    PresolveResult pr = presolveTBInPlace(m);

    REQUIRE_FALSE(pr.infeasible);
    REQUIRE_THAT(m.getHot().lb[y.id], WithinAbs(2.0, kTol));
    REQUIRE_THAT(m.getHot().ub[y.id], WithinAbs(2.0, kTol));
    REQUIRE(pr.fixedVars >= 2);
}

// ── presolve (copy): original model unchanged ────────────────────────────────
//
// presolveTB(model) returns a presolved copy; the original bounds must not change.

TEST_CASE("Presolve: copy variant leaves original unchanged", "[presolve]") {
    Model original;
    auto x = original.addVar(0.0, 10.0, "x");
    auto y = original.addVar(0.0, 10.0, "y");
    original.addLPConstraint(2.0*x + 3.0*y, Sense::LessEq, 12.0);
    original.setObjective(1.0*x, ObjSense::Minimize);

    const double origLbX = original.getHot().lb[x.id]; // 0
    const double origUbX = original.getHot().ub[x.id]; // 10
    const double origLbY = original.getHot().lb[y.id]; // 0
    const double origUbY = original.getHot().ub[y.id]; // 10

    auto [presolved, pr] = presolveTB(original);

    // Original is untouched.
    REQUIRE_THAT(original.getHot().lb[x.id], WithinAbs(origLbX, kTol));
    REQUIRE_THAT(original.getHot().ub[x.id], WithinAbs(origUbX, kTol));
    REQUIRE_THAT(original.getHot().lb[y.id], WithinAbs(origLbY, kTol));
    REQUIRE_THAT(original.getHot().ub[y.id], WithinAbs(origUbY, kTol));

    // Copy has tighter bounds.
    REQUIRE_FALSE(pr.infeasible);
    REQUIRE(pr.boundsTightened >= 2);
    REQUIRE_THAT(presolved.getHot().ub[x.id], WithinAbs(6.0, kTol));
    REQUIRE_THAT(presolved.getHot().ub[y.id], WithinAbs(4.0, kTol));
}

// ── presolve (copy): presolved model can be solved independently ──────────────
//
// Solve the presolved copy and the original separately; both reach the same
// optimal. The original bounds (10/10) are still available for inspection.

TEST_CASE("Presolve: solve presolved copy, inspect original bounds", "[presolve]") {
    Model original;
    auto x = original.addVar(0.0, 10.0, "x");
    auto y = original.addVar(0.0, 10.0, "y");
    original.addLPConstraint(1.0*x + 1.0*y, Sense::GreaterEq, 3.0);
    original.setObjective(1.0*x + 1.0*y, ObjSense::Minimize);

    auto [presolved, pr] = presolveTB(original);

    LPResult rPresolved = solveLP(presolved);
    LPResult rOriginal  = solveLP(original);

    REQUIRE(rPresolved.status == LPStatus::Optimal);
    REQUIRE(rOriginal.status  == LPStatus::Optimal);
    REQUIRE_THAT(rPresolved.objectiveValue, WithinAbs(rOriginal.objectiveValue, kTol));

    // Original bounds unchanged - still usable for comparison/debugging.
    REQUIRE_THAT(original.getHot().ub[x.id], WithinAbs(10.0, kTol));
    REQUIRE_THAT(original.getHot().ub[y.id], WithinAbs(10.0, kTol));
}

// ── LP integration: enablePresolve gives same optimal ────────────────────────

TEST_CASE("Presolve: LP integration - same optimal", "[presolve]") {
    Model m;
    auto x = m.addVar(0.0, 5.0, "x");
    auto y = m.addVar(0.0, 5.0, "y");
    m.addLPConstraint(1.0*x + 1.0*y, Sense::GreaterEq, 3.0);
    m.setObjective(1.0*x + 1.0*y, ObjSense::Minimize);

    LPOptions optsBase, optsPS;
    optsPS.enablePresolve = true;
    optsPS.enableElimination = false;

    LPResult rBase = solveLP(m, optsBase);
    LPResult rPS   = solveLP(m, optsPS);

    REQUIRE(rBase.status == LPStatus::Optimal);
    REQUIRE(rPS.status   == LPStatus::Optimal);
    REQUIRE_THAT(rPS.objectiveValue, WithinAbs(rBase.objectiveValue, kTol));

    // enablePresolve must not mutate the caller's model.
    REQUIRE_THAT(m.getHot().ub[x.id], WithinAbs(5.0, kTol));
    REQUIRE_THAT(m.getHot().ub[y.id], WithinAbs(5.0, kTol));
}

// ── LP integration: presolveStat populated ───────────────────────────────────

TEST_CASE("Presolve: LP presolveStat populated", "[presolve]") {
    Model m;
    auto x = m.addVar(0.0, 10.0, "x");
    auto y = m.addVar(0.0, 10.0, "y");
    m.addLPConstraint(2.0*x + 3.0*y, Sense::LessEq, 12.0);
    m.setObjective(1.0*x + 1.0*y, ObjSense::Minimize);

    LPOptions opts;
    opts.enablePresolve = true;
    opts.enableElimination = false;

    LPDetailedResult r = solveLPDetailed(m, opts);

    REQUIRE(r.result.status == LPStatus::Optimal);
    REQUIRE(r.presolveStat.has_value());
    REQUIRE_FALSE(r.presolveStat->infeasible);
    REQUIRE(r.presolveStat->boundsTightened > 0);
}

// ── LP integration: presolveStat absent when disabled ────────────────────────

TEST_CASE("Presolve: LP presolveStat absent when disabled", "[presolve]") {
    Model m;
    auto x = m.addVar(0.0, 10.0, "x");
    m.setObjective(1.0*x, ObjSense::Minimize);

    REQUIRE_FALSE(solveLPDetailed(m, LPOptions{.enablePresolve = false}).presolveStat.has_value());
}

// ── LP integration: presolve detects infeasibility ───────────────────────────

TEST_CASE("Presolve: LP infeasibility detected by presolve", "[presolve]") {
    Model m;
    auto x = m.addVar(3.0, 10.0, "x");
    auto y = m.addVar(3.0, 10.0, "y");
    m.addLPConstraint(1.0*x + 1.0*y, Sense::LessEq, 5.0);
    m.setObjective(1.0*x, ObjSense::Minimize);

    LPOptions opts;
    opts.enablePresolve = true;
    opts.enableElimination = false;

    REQUIRE(solveLP(m, opts).status == LPStatus::Infeasible);
}

// ── MILP integration: enablePresolve gives same optimal ──────────────────────

TEST_CASE("Presolve: MILP integration - same optimal", "[presolve]") {
    Model m;
    auto x = m.addVar(0.0, 5.0, VarType::Integer, "x");
    auto y = m.addVar(0.0, 5.0, VarType::Integer, "y");
    m.addLPConstraint(3.0*x + 2.0*y, Sense::LessEq, 7.0);
    m.setObjective(5.0*x + 4.0*y, ObjSense::Maximize);

    MILPResult rBase = solveMILP(m);

    BBOptions optsPS;
    optsPS.presolveLevel = 1;
    optsPS.enableElimination = false;
    MILPResult rPS = solveMILP(m, optsPS);

    REQUIRE(rBase.status == MILPStatus::Optimal);
    REQUIRE(rPS.status   == MILPStatus::Optimal);
    REQUIRE_THAT(rPS.objectiveValue, WithinAbs(rBase.objectiveValue, kTol));
}

// ── MILP integration: presolveStat populated ─────────────────────────────────

TEST_CASE("Presolve: MILP presolveStat populated", "[presolve]") {
    Model m;
    auto x = m.addVar(0.0, 5.0, VarType::Integer, "x");
    auto y = m.addVar(0.0, 5.0, VarType::Integer, "y");
    m.addLPConstraint(3.0*x + 2.0*y, Sense::LessEq, 7.0);
    m.setObjective(5.0*x + 4.0*y, ObjSense::Maximize);

    BBOptions opts;
    opts.presolveLevel = 1;
    opts.enableElimination = false;

    MILPResult r = solveMILP(m, opts);

    REQUIRE(r.status == MILPStatus::Optimal);
    REQUIRE(r.presolveStat.has_value());
    REQUIRE_FALSE(r.presolveStat->infeasible);
}

// ── MILP integration: presolve detects infeasibility ─────────────────────────

TEST_CASE("Presolve: MILP infeasibility detected by presolve", "[presolve]") {
    Model m;
    auto x = m.addVar(4.0, 10.0, VarType::Integer, "x");
    auto y = m.addVar(4.0, 10.0, VarType::Integer, "y");
    m.addLPConstraint(1.0*x + 1.0*y, Sense::LessEq, 5.0);
    m.setObjective(1.0*x, ObjSense::Minimize);

    BBOptions opts;
    opts.presolveLevel = 1;
    opts.enableElimination = false;

    MILPResult r = solveMILP(m, opts);

    REQUIRE(r.status == MILPStatus::Infeasible);
    REQUIRE(r.presolveStat.has_value());
    REQUIRE(r.presolveStat->infeasible);
}

// ── presolveElim: trivial column ─────────────────────────────────────────────────
//
// x ∈ [3,3] (fixed), y ∈ [0,5], no LP constraints, min y.
// x is fixed → eliminated; y is kept; no rows to check.

TEST_CASE("PresolveElim: trivial column elimination", "[presolve][elim]") {
    Model m;
    auto x = m.addVar(3.0, 3.0, "x");
    auto y = m.addVar(0.0, 5.0, "y");
    m.setObjective(1.0*y, ObjSense::Minimize);

    EliminationRecord rec;
    Model reduced = presolveElim(m, rec);

    REQUIRE(rec.varsEliminated == 1);
    REQUIRE(rec.rowsEliminated == 0);
    REQUIRE(reduced.numVars() == 1);
    REQUIRE(reduced.numConstraints() == 0);
    REQUIRE_THAT(rec.objAdjustment, WithinAbs(0.0, kTol)); // x not in objective
}

// ── presolveElim: trivial row ─────────────────────────────────────────────────────
//
// x ∈ [0,2], y ∈ [0,3], x+y<=10. maxActivity=5 << 10 → redundant.

TEST_CASE("PresolveElim: trivial row elimination", "[presolve][elim]") {
    Model m;
    auto x = m.addVar(0.0, 2.0, "x");
    auto y = m.addVar(0.0, 3.0, "y");
    m.addLPConstraint(1.0*x + 1.0*y, Sense::LessEq, 10.0);
    m.setObjective(1.0*x + 1.0*y, ObjSense::Minimize);

    EliminationRecord rec;
    Model reduced = presolveElim(m, rec);

    REQUIRE(rec.varsEliminated == 0);
    REQUIRE(rec.rowsEliminated == 1);
    REQUIRE(reduced.numVars() == 2);
    REQUIRE(reduced.numConstraints() == 0);
}

// ── presolveElim: trivial combined ────────────────────────────────────────────────
//
// x ∈ [1,1] (fixed), y ∈ [0,3], y<=100.
// x eliminated; y<=100 trivially redundant (maxActivity=3); objAdjustment = 1×1 = 1.

TEST_CASE("PresolveElim: trivial combined elimination", "[presolve][elim]") {
    Model m;
    auto x = m.addVar(1.0, 1.0, "x");
    auto y = m.addVar(0.0, 3.0, "y");
    m.addLPConstraint(1.0*y, Sense::LessEq, 100.0);
    m.setObjective(1.0*x + 1.0*y, ObjSense::Minimize);

    EliminationRecord rec;
    Model reduced = presolveElim(m, rec);

    REQUIRE(rec.varsEliminated == 1);
    REQUIRE(rec.rowsEliminated == 1);
    REQUIRE(reduced.numVars() == 1);
    REQUIRE(reduced.numConstraints() == 0);
    REQUIRE_THAT(rec.objAdjustment, WithinAbs(1.0, kTol));
}

// ── presolveElim: non-trivial column ─────────────────────────────────────────────
//
// x ∈ [2,2] (fixed), y ∈ [0,10], z ∈ [0,10].
// Constraint: x+y+z<=15. After x=2: y+z<=13 (maxFin=20 > 13 → NOT redundant).
// Objective: 3x+y+z → objAdjustment=6. Reduced opt = 0 → postsolve opt = 6.

TEST_CASE("PresolveElim: non-trivial column - RHS adjusted, postsolve correct", "[presolve][elim]") {
    Model m;
    auto x = m.addVar(2.0, 2.0, "x");
    auto y = m.addVar(0.0, 10.0, "y");
    auto z = m.addVar(0.0, 10.0, "z");
    m.addLPConstraint(1.0*x + 1.0*y + 1.0*z, Sense::LessEq, 15.0);
    m.setObjective(3.0*x + 1.0*y + 1.0*z, ObjSense::Minimize);

    EliminationRecord rec;
    Model reduced = presolveElim(m, rec);

    REQUIRE(rec.varsEliminated == 1);
    REQUIRE(rec.rowsEliminated == 0);
    REQUIRE(reduced.numVars() == 2);
    REQUIRE(reduced.numConstraints() == 1);
    REQUIRE_THAT(rec.objAdjustment, WithinAbs(6.0, kTol));
    REQUIRE_THAT(reduced.getLPConstraints()[0].rhsConst, WithinAbs(13.0, kTol));

    LPOptions inner;
    inner.enablePresolve    = false;
    inner.enableElimination = false;
    LPDetailedResult r = solveLPDetailed(reduced, inner);
    REQUIRE(r.result.status == LPStatus::Optimal);
    postsolveElim(r, rec);
    REQUIRE_THAT(r.result.objectiveValue, WithinAbs(6.0, kTol));
    REQUIRE_THAT(r.result.primalValues[x.id], WithinAbs(2.0, kTol));
}

// ── presolveElim: non-trivial row ─────────────────────────────────────────────────
//
// x ∈ [0,3], y ∈ [0,2]. Three constraints:
//   x+y <= 2   (maxFin=5 > 2 → NOT redundant → kept)
//   x+y <= 100 (maxFin=5 <= 100 → redundant → eliminated)
//   2x+y >= 0  (minFin=0 >= 0 → redundant → eliminated)

TEST_CASE("PresolveElim: non-trivial row - selective elimination", "[presolve][elim]") {
    Model m;
    auto x = m.addVar(0.0, 3.0, "x");
    auto y = m.addVar(0.0, 2.0, "y");
    m.addLPConstraint(1.0*x + 1.0*y, Sense::LessEq,    2.0);
    m.addLPConstraint(1.0*x + 1.0*y, Sense::LessEq,  100.0);
    m.addLPConstraint(2.0*x + 1.0*y, Sense::GreaterEq,  0.0);
    m.setObjective(1.0*x + 1.0*y, ObjSense::Minimize);

    EliminationRecord rec;
    Model reduced = presolveElim(m, rec);

    REQUIRE(rec.varsEliminated == 0);
    REQUIRE(rec.rowsEliminated == 2);
    REQUIRE(reduced.numVars() == 2);
    REQUIRE(reduced.numConstraints() == 1);
    REQUIRE_THAT(reduced.getLPConstraints()[0].rhsConst, WithinAbs(2.0, kTol));
}

// ── presolveElim: non-trivial combined ───────────────────────────────────────────
//
// x ∈ [2,2] (fixed), y ∈ [0,5], z ∈ [0,3].
// Constraints:
//   x+y+z <= 15  (maxFin=2+5+3=10 <= 15 → redundant → eliminated)
//   y+z   >= 2   (minFin=0 < 2 → NOT redundant → kept)
//   x+z   <= 10  (maxFin=2+3=5 <= 10 → redundant → eliminated)
// Objective: 4x+y+z → objAdjustment=8. Reduced opt=2 → postsolve opt=10.

TEST_CASE("PresolveElim: non-trivial combined - fixed var + selective rows, postsolve correct", "[presolve][elim]") {
    Model m;
    auto x = m.addVar(2.0, 2.0, "x");
    auto y = m.addVar(0.0, 5.0, "y");
    auto z = m.addVar(0.0, 3.0, "z");
    m.addLPConstraint(1.0*x + 1.0*y + 1.0*z, Sense::LessEq,   15.0);
    m.addLPConstraint(            1.0*y + 1.0*z, Sense::GreaterEq, 2.0);
    m.addLPConstraint(1.0*x             + 1.0*z, Sense::LessEq,   10.0);
    m.setObjective(4.0*x + 1.0*y + 1.0*z, ObjSense::Minimize);

    EliminationRecord rec;
    Model reduced = presolveElim(m, rec);

    REQUIRE(rec.varsEliminated == 1);
    REQUIRE(rec.rowsEliminated == 2);
    REQUIRE(reduced.numVars() == 2);
    REQUIRE(reduced.numConstraints() == 1);
    REQUIRE_THAT(rec.objAdjustment, WithinAbs(8.0, kTol));

    LPOptions inner;
    inner.enablePresolve    = false;
    inner.enableElimination = false;
    LPDetailedResult r = solveLPDetailed(reduced, inner);
    REQUIRE(r.result.status == LPStatus::Optimal);
    postsolveElim(r, rec);
    REQUIRE_THAT(r.result.objectiveValue, WithinAbs(10.0, kTol));
    REQUIRE_THAT(r.result.primalValues[x.id], WithinAbs(2.0, kTol));
    REQUIRE_THAT(r.result.primalValues[y.id] + r.result.primalValues[z.id],
                 WithinAbs(2.0, kTol));
}

// ── presolveElim: contrainte all-fixed infaisable détectée à la source ────────────
//
// x∈[3,3], z∈[2,2] tous deux fixés.  Contrainte x+z <= 4 : 3+2=5 > 4 → infaisable.
// presolveTBInPlace détecterait ça, mais si presolveElim est appelé directement
// (ex. après un TB sans fixpoint), la contrainte vide avec RHS=-1 doit être signalée
// via EliminationRecord::infeasible plutôt que laissée silencieusement au solveur LP.

TEST_CASE("presolveElim: all-fixed infeasible constraint sets rec.infeasible",
          "[presolve][elim]") {
    Model m;
    Variable x = m.addVar(3.0, 3.0, "x");
    Variable z = m.addVar(2.0, 2.0, "z");
    m.addLPConstraint(1.0*x + 1.0*z, Sense::LessEq, 4.0); // 3+2=5 > 4 → violated
    m.setObjective(1.0*x, ObjSense::Minimize);

    EliminationRecord rec;
    presolveElim(m, rec);

    // Après élimination : LHS vide, adjustedRHS = 4-3-2 = -1 < 0 → infaisable.
    REQUIRE(rec.infeasible); // échoue avant le fix (champ non peuplé)
}

// ── postsolveElim: sensitivityResult remappé sur le modèle original ──────────────
//
// Modèle : x∈[5,5] (fixé), y∈[0,5], z∈[0,5].
// Contrainte 0 : y+z<=8  (survit : maxFin=10>8, TB ne peut pas la rendre redondante).
// Contrainte 1 : x<=6    (éliminée : LHS vide après fixage, 0.0<=1.0 → redondante).
// Après postsolveElim, sensitivity doit être re-dimensionnée au modèle original :
//   rhsRange.size() == numConstraints() (2, non 1)
//   objRange.size() == numVars()        (3, non 2)

TEST_CASE("postsolveElim: sensitivity remapped to original model size", "[presolve][elim][sensitivity]") {
    Model m;
    Variable x = m.addVar(5.0, 5.0, "x");
    Variable y = m.addVar(0.0, 5.0, "y");
    Variable z = m.addVar(0.0, 5.0, "z");
    m.addLPConstraint(1.0*y + 1.0*z, Sense::LessEq, 8.0); // con 0 - survit
    m.addLPConstraint(1.0*x,         Sense::LessEq, 6.0); // con 1 - éliminée
    m.setObjective(1.0*y + 1.0*z, ObjSense::Minimize);

    LPOptions opts;
    opts.computeSensitivity = true;
    opts.enablePresolve     = true;
    opts.enableElimination  = true;
    opts.method             = LPMethod::PrimalSimplex;

    LPDetailedResult r = solveLPDetailed(m, opts);
    REQUIRE(r.result.status == LPStatus::Optimal);

    // Les plages de sensibilité doivent être indexées sur le modèle original.
    REQUIRE(r.sensitivity.rhsRange.size() == m.numConstraints()); // 2, pas 1
    REQUIRE(r.sensitivity.objRange.size() == m.numVars());        // 3, pas 2
}

// ── presolveTBInPlace: ghost vars exclus du comptage fixedVars ───────────────────
//
// Modèle : x∈[3,3] (fixé), z∈[0,10], x+z<=12.
// Après presolveElim : z reste LP var (ub resserré à 9), x devient ghost.
// TB sur le modèle réduit : z non fixé → fixedVars doit être 0.
// hot.lb.size() = 2 (z + ghost x) ; model.numVars() = 1 (z seulement).

TEST_CASE("presolveTBInPlace: ghost vars excluded from fixedVars count", "[presolve][tb]") {
    Model orig;
    Variable x = orig.addVar(3.0,  3.0,  "x");
    Variable z = orig.addVar(0.0, 10.0,  "z");
    orig.addLPConstraint(1.0*x + 1.0*z, Sense::LessEq, 12.0);
    orig.setObjective(1.0*z, ObjSense::Minimize);

    EliminationRecord rec;
    Model reduced = presolveElim(orig, rec);

    REQUIRE(reduced.numVars()      == 1);
    REQUIRE(reduced.numTotalVars() == 2); // z + ghost x

    PresolveResult pr = presolveTBInPlace(reduced);
    REQUIRE_FALSE(pr.infeasible);

    // z ∈ [0,9] après tightening - pas fixé.
    // ghost x ∈ [3,3] doit être ignoré.
    REQUIRE(pr.fixedVars == 0);
}

// ── presolveTBInPlace: time limit stops passes ──────────────────────────────────
//
// startTime set 100 s in the past with timeLimitS=1 → already expired.
// Presolve must exit before running any pass.

TEST_CASE("Presolve: time limit stops passes early", "[presolve]") {
    using Clock = std::chrono::steady_clock;
    Model m;
    auto x = m.addVar(0.0, 10.0, "x");
    auto y = m.addVar(0.0, 10.0, "y");
    m.addLPConstraint(2.0*x + 3.0*y, Sense::LessEq, 12.0);
    m.setObjective(1.0*x, ObjSense::Minimize);

    const auto pastStart = Clock::now() - std::chrono::seconds(100);
    PresolveResult pr = presolveTBInPlace(m, 10, 1.0, pastStart);

    REQUIRE(pr.timeLimitReached);
    REQUIRE(pr.passesRun == 0);
    // Bounds must be untouched since no pass ran.
    REQUIRE_THAT(m.getHot().ub[x.id], WithinAbs(10.0, kTol));
    REQUIRE_THAT(m.getHot().ub[y.id], WithinAbs(10.0, kTol));
}
