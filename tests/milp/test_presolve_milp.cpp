#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

#include "baguette/core/Sense.hpp"
#include "baguette/milp/BranchAndBound.hpp"
#include "baguette/milp/Presolve.hpp"
#include "baguette/model/Model.hpp"
#include "baguette/model/ModelEnums.hpp"

using namespace baguette;
using Catch::Matchers::WithinAbs;

static constexpr double kTol = 1e-9;

// ── Trivial 1: plain integer bound rounding ───────────────────────────────────
//
// x ∈ [0.5, 4.7] Integer — no constraints.
// LP does nothing; initial round snaps lb→1, ub→4.

TEST_CASE("presolveMILP: integer bound rounding", "[presolve][milp]") {
    Model m;
    Variable x = m.addVar(0.5, 4.7, VarType::Integer, "x");

    MILPPresolveResult res = presolveMILPInPlace(m);

    REQUIRE_FALSE(res.infeasible);
    REQUIRE(res.boundsRounded   == 1); // x rounded (lb 0.5→1, ub 4.7→4)
    REQUIRE(res.boundsTightened == 0); // no LP constraints
    REQUIRE_THAT(m.getHot().lb[x.id], WithinAbs(1.0, kTol));
    REQUIRE_THAT(m.getHot().ub[x.id], WithinAbs(4.0, kTol));
}

// ── Trivial 2: empty integer domain detected as infeasible ───────────────────
//
// x ∈ [1.1, 1.9] Integer → ceil(1.1)=2 > floor(1.9)=1 → infeasible.

TEST_CASE("presolveMILP: empty integer domain", "[presolve][milp]") {
    Model m;
    m.addVar(1.1, 1.9, VarType::Integer, "x");

    MILPPresolveResult res = presolveMILPInPlace(m);

    REQUIRE(res.infeasible);
}

// ── Trivial 3: LP tightening then integer rounding ────────────────────────────
//
// x ∈ [0, 10] Integer,  x <= 2.9
// LP: ub 10 → 2.9; round: ub 2.9 → 2.

TEST_CASE("presolveMILP: LP tightening cascades into rounding", "[presolve][milp]") {
    Model m;
    Variable x = m.addVar(0.0, 10.0, VarType::Integer, "x");
    m.addLPConstraint(1.0*x, Sense::LessEq, 2.9);

    MILPPresolveResult res = presolveMILPInPlace(m);

    REQUIRE_FALSE(res.infeasible);
    REQUIRE(res.boundsTightened == 1); // LP: ub 10→2.9
    REQUIRE(res.boundsRounded   == 1); // round: ub 2.9→2
    REQUIRE_THAT(m.getHot().ub[x.id], WithinAbs(2.0, kTol));
}

// ── Trivial 4: binary variable fixed by LP+round cascade ─────────────────────
//
// x ∈ [0, 1] Binary,  x >= 0.5
// LP: lb 0→0.5; round: lb 0.5→1. x fixed to 1.

TEST_CASE("presolveMILP: binary fixed by LP+round", "[presolve][milp]") {
    Model m;
    Variable x = m.addVar(0.0, 1.0, VarType::Binary, "x");
    m.addLPConstraint(1.0*x, Sense::GreaterEq, 0.5);

    MILPPresolveResult res = presolveMILPInPlace(m);

    REQUIRE_FALSE(res.infeasible);
    REQUIRE(res.boundsTightened == 1); // LP: lb 0→0.5
    REQUIRE(res.boundsRounded   == 1); // round: lb 0.5→1
    REQUIRE(res.fixedVars       == 1);
    REQUIRE_THAT(m.getHot().lb[x.id], WithinAbs(1.0, kTol));
    REQUIRE_THAT(m.getHot().ub[x.id], WithinAbs(1.0, kTol));
}

// ── presolveMILPInPlace: intFeasTol transmis depuis BBOptions ─────────────────────
//
// x ∈ [2.0005, 10] Integer, min x.
// Avec intFeasTol = 1e-3 :
//   ceil(2.0005 - 1e-3) = ceil(1.9995) = 2  → lb arrondi à 2, optimal = 2.
// Avec kIntTol = 1e-6 hardcodé (bug) :
//   ceil(2.0005 - 1e-6) = ceil(2.000499) = 3 → lb arrondi à 3, optimal = 3.

TEST_CASE("presolveMILP: intFeasTol honoured from BBOptions", "[presolve][milp]") {
    Model m;
    Variable x = m.addVar(2.0005, 10.0, VarType::Integer, "x");
    m.setObjective(1.0 * x, ObjSense::Minimize);

    BBOptions opts;
    opts.intFeasTol        = 1e-3;
    opts.lpOpts.method     = LPMethod::DualSimplexBV;

    MILPResult r = solveMILP(m, opts);
    REQUIRE(r.status == MILPStatus::Optimal);
    // Avec intFeasTol=1e-3 le presolve arrondit lb à 2, pas 3.
    REQUIRE_THAT(r.objectiveValue, WithinAbs(2.0, kTol));
}

// ── Non-trivial: 10 integer vars, two-round outer cascade ─────────────────────
//
// x[0..4] ∈ [0,10] Integer — individual upper bounds: x[i] <= 3.9
// x[5..9] ∈ [0,10] Integer — no individual constraints
// Sum: x[0]+x[1]+x[2]+x[3]+x[4] >= 13.5
// Objective: min sum(x[0..9])
//
// Outer iteration 1:
//   LP:   x[i].ub ← 3.9 (5 tightenings)   [excl_max=4×3.9=15.6 > 13.5, no lb change]
//   Round: x[i].ub ← 3  (5 roundings)
//
// Outer iteration 2:
//   LP:   x[i].lb ← 1.5 (5 tightenings)   [excl_max=4×3=12, newLb=(13.5−12)/1=1.5]
//   Round: x[i].lb ← 2  (5 roundings)
//
// Outer iteration 3: no change → fixed point.
//
// After presolve: x[0..4] ∈ [2, 3]; x[5..9] ∈ [0, 10] (untouched).
//
// MILP optimal: x[0..4] ∈ {2,3} summing to 14 (⌈13.5⌉=14); x[5..9]=0 → obj=14.

TEST_CASE("presolveMILP: 10-var two-round cascade", "[presolve][milp]") {
    Model m;
    std::vector<Variable> x(10);
    for (int i = 0; i < 10; ++i)
        x[i] = m.addVar(0.0, 10.0, VarType::Integer, "x" + std::to_string(i));

    for (int i = 0; i < 5; ++i)
        m.addLPConstraint(1.0*x[i], Sense::LessEq, 3.9);

    m.addLPConstraint(1.0*x[0] + 1.0*x[1] + 1.0*x[2] + 1.0*x[3] + 1.0*x[4],
                      Sense::GreaterEq, 13.5);

    m.setObjective(1.0*x[0] + 1.0*x[1] + 1.0*x[2] + 1.0*x[3] + 1.0*x[4] +
                   1.0*x[5] + 1.0*x[6] + 1.0*x[7] + 1.0*x[8] + 1.0*x[9],
                   ObjSense::Minimize);

    // ── Structure checks ──────────────────────────────────────────────────────
    {
        Model mCopy = m;
        MILPPresolveResult res = presolveMILPInPlace(mCopy);

        REQUIRE_FALSE(res.infeasible);
        REQUIRE(res.boundsTightened == 10); // 5 ub (LP) + 5 lb (LP)
        REQUIRE(res.boundsRounded   == 10); // 5 ub (round) + 5 lb (round)
        REQUIRE(res.fixedVars       == 0);  // x[i] ∈ [2,3], not fixed

        const auto& hot = mCopy.getHot();
        for (int i = 0; i < 5; ++i) {
            REQUIRE_THAT(hot.lb[x[i].id], WithinAbs(2.0, kTol));
            REQUIRE_THAT(hot.ub[x[i].id], WithinAbs(3.0, kTol));
        }
        for (int i = 5; i < 10; ++i) {
            REQUIRE_THAT(hot.lb[x[i].id], WithinAbs(0.0, kTol));
            REQUIRE_THAT(hot.ub[x[i].id], WithinAbs(10.0, kTol));
        }
    }

    // ── MILP solve coherence ──────────────────────────────────────────────────
    BBOptions opts;
    opts.lpOpts.method = LPMethod::DualSimplexBV;
    MILPResult r = solveMILP(m, opts);

    REQUIRE(r.status == MILPStatus::Optimal);
    REQUIRE_THAT(r.objectiveValue, WithinAbs(14.0, kTol));

    // x[5..9] unconstrained: minimise to 0
    for (int i = 5; i < 10; ++i)
        REQUIRE_THAT(r.primalValues[x[i].id], WithinAbs(0.0, kTol));

    // x[0..4] ∈ {2, 3} with sum == 14
    double sumFront = 0.0;
    for (int i = 0; i < 5; ++i) {
        const double vi = r.primalValues[x[i].id];
        REQUIRE((std::abs(vi - 2.0) < 1e-6 || std::abs(vi - 3.0) < 1e-6));
        sumFront += vi;
    }
    REQUIRE_THAT(sumFront, WithinAbs(14.0, kTol));
}

// ── D2: milpPresolveMaxCycles distinct de lpOpts.presolveMaxPasses ────────────
//
// Same 5-var setup as above. With maxCycles = 1, only the first outer iteration
// runs (LP: ub 10→3.9, round: ub→3). The second outer iteration (LP: lb→1.5,
// round: lb→2) is skipped. B&B still finds the correct optimal.

TEST_CASE("presolveMILP: milpPresolveMaxCycles limits outer iterations", "[presolve][milp]") {
    Model m;
    std::vector<Variable> x(5);
    for (int i = 0; i < 5; ++i)
        x[i] = m.addVar(0.0, 10.0, VarType::Integer, "x" + std::to_string(i));

    for (int i = 0; i < 5; ++i)
        m.addLPConstraint(1.0*x[i], Sense::LessEq, 3.9);
    m.addLPConstraint(1.0*x[0] + 1.0*x[1] + 1.0*x[2] + 1.0*x[3] + 1.0*x[4],
                      Sense::GreaterEq, 13.5);
    m.setObjective(1.0*x[0] + 1.0*x[1] + 1.0*x[2] + 1.0*x[3] + 1.0*x[4],
                   ObjSense::Minimize);

    // 1 cycle: LP tightens ub (5 bounds), round snaps ub (5 bounds).
    // lb tightening (requires a second outer cycle) is NOT done.
    {
        Model mCopy = m;
        MILPPresolveResult res = presolveMILPInPlace(mCopy, /*maxCycles=*/1);
        REQUIRE_FALSE(res.infeasible);
        REQUIRE(res.boundsTightened == 5); // only ub tightened
        REQUIRE(res.boundsRounded   == 5); // only ub rounded
        const auto& hot = mCopy.getHot();
        for (int i = 0; i < 5; ++i) {
            REQUIRE_THAT(hot.lb[x[i].id], WithinAbs(0.0, kTol)); // lb unchanged
            REQUIRE_THAT(hot.ub[x[i].id], WithinAbs(3.0, kTol)); // ub rounded
        }
    }

    // BBOptions::milpPresolveMaxCycles controls the outer loop;
    // lpOpts.presolveMaxPasses (LP node passes) is independent.
    BBOptions opts;
    opts.milpPresolveMaxCycles  = 1;
    opts.lpOpts.presolveMaxPasses = 0; // unlimited LP passes per node
    opts.lpOpts.method          = LPMethod::DualSimplexBV;
    MILPResult r = solveMILP(m, opts);

    REQUIRE(r.status == MILPStatus::Optimal);
    REQUIRE_THAT(r.objectiveValue, WithinAbs(14.0, kTol));
}
