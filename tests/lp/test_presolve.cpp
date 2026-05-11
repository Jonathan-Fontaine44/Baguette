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

// ── presolveInPlace: LEQ ──────────────────────────────────────────────────────
//
// 2x + 3y <= 12, x∈[0,10], y∈[0,10]
// min activity = 0 → x <= 6, y <= 4

TEST_CASE("Presolve: LEQ bound tightening", "[presolve]") {
    Model m;
    auto x = m.addVar(0.0, 10.0, "x");
    auto y = m.addVar(0.0, 10.0, "y");
    m.addLPConstraint(2.0*x + 3.0*y, Sense::LessEq, 12.0);
    m.setObjective(1.0*x, ObjSense::Minimize);

    PresolveResult pr = presolveInPlace(m);

    REQUIRE_FALSE(pr.infeasible);
    REQUIRE(pr.boundsTightened >= 2);
    REQUIRE(pr.passesRun >= 1);
    REQUIRE_THAT(m.getHot().ub[x.id], WithinAbs(6.0, kTol));
    REQUIRE_THAT(m.getHot().ub[y.id], WithinAbs(4.0, kTol));
    REQUIRE_THAT(m.getHot().lb[x.id], WithinAbs(0.0, kTol));
    REQUIRE_THAT(m.getHot().lb[y.id], WithinAbs(0.0, kTol));
}

// ── presolveInPlace: GEQ ──────────────────────────────────────────────────────
//
// x + y >= 5, x∈[0,3], y∈[0,4]
// max activity = 7 → x >= 1, y >= 2

TEST_CASE("Presolve: GEQ bound tightening", "[presolve]") {
    Model m;
    auto x = m.addVar(0.0, 3.0, "x");
    auto y = m.addVar(0.0, 4.0, "y");
    m.addLPConstraint(1.0*x + 1.0*y, Sense::GreaterEq, 5.0);
    m.setObjective(1.0*x, ObjSense::Minimize);

    PresolveResult pr = presolveInPlace(m);

    REQUIRE_FALSE(pr.infeasible);
    REQUIRE(pr.boundsTightened >= 2);
    REQUIRE_THAT(m.getHot().lb[x.id], WithinAbs(1.0, kTol));
    REQUIRE_THAT(m.getHot().lb[y.id], WithinAbs(2.0, kTol));
    REQUIRE_THAT(m.getHot().ub[x.id], WithinAbs(3.0, kTol));
    REQUIRE_THAT(m.getHot().ub[y.id], WithinAbs(4.0, kTol));
}

// ── presolveInPlace: EQ ───────────────────────────────────────────────────────
//
// x + y = 5, x∈[0,10], y∈[0,10] → x<=5, y<=5

TEST_CASE("Presolve: EQ bound tightening", "[presolve]") {
    Model m;
    auto x = m.addVar(0.0, 10.0, "x");
    auto y = m.addVar(0.0, 10.0, "y");
    m.addLPConstraint(1.0*x + 1.0*y, Sense::Equal, 5.0);
    m.setObjective(1.0*x, ObjSense::Minimize);

    PresolveResult pr = presolveInPlace(m);

    REQUIRE_FALSE(pr.infeasible);
    REQUIRE_THAT(m.getHot().ub[x.id], WithinAbs(5.0, kTol));
    REQUIRE_THAT(m.getHot().ub[y.id], WithinAbs(5.0, kTol));
}

// ── presolveInPlace: multi-pass ───────────────────────────────────────────────
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

    PresolveResult pr = presolveInPlace(m);

    REQUIRE_FALSE(pr.infeasible);
    REQUIRE(pr.passesRun >= 2);
    REQUIRE_THAT(m.getHot().lb[x.id], WithinAbs(2.0, kTol));
    REQUIRE_THAT(m.getHot().ub[x.id], WithinAbs(5.0, kTol));
    REQUIRE_THAT(m.getHot().ub[y.id], WithinAbs(3.0, kTol));
    REQUIRE_THAT(m.getHot().ub[z.id], WithinAbs(3.0, kTol));
}

// ── presolveInPlace: infeasibility via LEQ ────────────────────────────────────
//
// x + y <= 3, x∈[2,10], y∈[2,10] → minActivity = 4 > 3

TEST_CASE("Presolve: infeasible via LEQ", "[presolve]") {
    Model m;
    auto x = m.addVar(2.0, 10.0, "x");
    auto y = m.addVar(2.0, 10.0, "y");
    m.addLPConstraint(1.0*x + 1.0*y, Sense::LessEq, 3.0);
    m.setObjective(1.0*x, ObjSense::Minimize);

    REQUIRE(presolveInPlace(m).infeasible);
}

// ── presolveInPlace: infeasibility via GEQ ────────────────────────────────────
//
// x + y >= 8, x∈[0,3], y∈[0,3] → maxActivity = 6 < 8

TEST_CASE("Presolve: infeasible via GEQ", "[presolve]") {
    Model m;
    auto x = m.addVar(0.0, 3.0, "x");
    auto y = m.addVar(0.0, 3.0, "y");
    m.addLPConstraint(1.0*x + 1.0*y, Sense::GreaterEq, 8.0);
    m.setObjective(1.0*x, ObjSense::Minimize);

    REQUIRE(presolveInPlace(m).infeasible);
}

// ── presolveInPlace: infinite lb ──────────────────────────────────────────────
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

    PresolveResult pr = presolveInPlace(m);

    REQUIRE_FALSE(pr.infeasible);
    REQUIRE_THAT(m.getHot().ub[x.id], WithinAbs(5.0, kTol));
}

// ── presolveInPlace: fixed variable detection ─────────────────────────────────
//
// x+y=4, x∈[2,2] (fixed) → y fixed to 2

TEST_CASE("Presolve: fixed variable counting", "[presolve]") {
    Model m;
    auto x = m.addVar(2.0, 2.0, "x");
    auto y = m.addVar(0.0, 10.0, "y");
    m.addLPConstraint(1.0*x + 1.0*y, Sense::Equal, 4.0);
    m.setObjective(1.0*y, ObjSense::Minimize);

    PresolveResult pr = presolveInPlace(m);

    REQUIRE_FALSE(pr.infeasible);
    REQUIRE_THAT(m.getHot().lb[y.id], WithinAbs(2.0, kTol));
    REQUIRE_THAT(m.getHot().ub[y.id], WithinAbs(2.0, kTol));
    REQUIRE(pr.fixedVars >= 2);
}

// ── presolve (copy): original model unchanged ────────────────────────────────
//
// presolve(model) returns a presolved copy; the original bounds must not change.

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

    auto [presolved, pr] = presolve(original);

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

    auto [presolved, pr] = presolve(original);

    LPResult rPresolved = solveLP(presolved);
    LPResult rOriginal  = solveLP(original);

    REQUIRE(rPresolved.status == LPStatus::Optimal);
    REQUIRE(rOriginal.status  == LPStatus::Optimal);
    REQUIRE_THAT(rPresolved.objectiveValue, WithinAbs(rOriginal.objectiveValue, kTol));

    // Original bounds unchanged — still usable for comparison/debugging.
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
    optsPS.enablePresolve = true;
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
    opts.enablePresolve = true;

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
    opts.enablePresolve = true;

    MILPResult r = solveMILP(m, opts);

    REQUIRE(r.status == MILPStatus::Infeasible);
    REQUIRE(r.presolveStat.has_value());
    REQUIRE(r.presolveStat->infeasible);
}

// ── presolveInPlace: time limit stops passes ──────────────────────────────────
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
    PresolveResult pr = presolveInPlace(m, 10, 1.0, pastStart);

    REQUIRE(pr.timeLimitReached);
    REQUIRE(pr.passesRun == 0);
    // Bounds must be untouched since no pass ran.
    REQUIRE_THAT(m.getHot().ub[x.id], WithinAbs(10.0, kTol));
    REQUIRE_THAT(m.getHot().ub[y.id], WithinAbs(10.0, kTol));
}
