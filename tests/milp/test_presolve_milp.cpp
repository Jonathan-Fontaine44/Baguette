#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

#include "baguette/core/Sense.hpp"
#include "baguette/milp/BranchAndBound.hpp"
#include "baguette/milp/Presolve.hpp"
#include "baguette/model/Model.hpp"
#include "baguette/model/ModelEnums.hpp"
#include "lp/MILP_problems.hpp"

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
// PR1 rounds constraint RHS: x <= 2.9 → x <= 2 (floor).
// LP: ub 10 → 2 (1 tightening, already integer → 0 roundings).

TEST_CASE("presolveMILP: LP tightening cascades into rounding", "[presolve][milp]") {
    Model m;
    Variable x = m.addVar(0.0, 10.0, VarType::Integer, "x");
    m.addLPConstraint(1.0*x, Sense::LessEq, 2.9);

    MILPPresolveResult res = presolveMILPInPlace(m);

    REQUIRE_FALSE(res.infeasible);
    REQUIRE(res.rhsRounded      == 1); // PR1: 2.9 → floor(2.9) = 2
    REQUIRE(res.boundsTightened == 1); // LP: ub 10→2
    REQUIRE(res.boundsRounded   == 0); // LP gave integer bound directly
    REQUIRE_THAT(m.getHot().ub[x.id], WithinAbs(2.0, kTol));
}

// ── Trivial 4: binary variable fixed by LP+round cascade ─────────────────────
//
// x ∈ [0, 1] Binary,  x >= 0.5
// PR1 rounds constraint RHS: x >= 0.5 → x >= 1 (ceil).
// LP: lb 0→1 (1 tightening, already integer → 0 roundings). x fixed to 1.

TEST_CASE("presolveMILP: binary fixed by LP+round", "[presolve][milp]") {
    Model m;
    Variable x = m.addVar(0.0, 1.0, VarType::Binary, "x");
    m.addLPConstraint(1.0*x, Sense::GreaterEq, 0.5);

    MILPPresolveResult res = presolveMILPInPlace(m);

    REQUIRE_FALSE(res.infeasible);
    REQUIRE(res.rhsRounded      == 1); // PR1: 0.5 → ceil(0.5) = 1
    REQUIRE(res.boundsTightened == 1); // LP: lb 0→1
    REQUIRE(res.boundsRounded   == 0); // LP gave integer bound directly
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

// ── Non-trivial: 10 integer vars, PR1 + LP cascade ────────────────────────────
//
// x[0..4] ∈ [0,10] Integer — individual upper bounds: x[i] <= 3.9
// x[5..9] ∈ [0,10] Integer — no individual constraints
// Sum: x[0]+x[1]+x[2]+x[3]+x[4] >= 13.5
// Objective: min sum(x[0..9])
//
// PR1 (before outer loop):
//   x[i] <= 3.9  →  x[i] <= 3   (floor, 5 constraints)
//   sum   >= 13.5 →  sum  >= 14  (ceil,  1 constraint)
//   rhsRounded = 6
//
// Outer iteration 1 (LP pass — constraints processed in order):
//   x[i] <= 3:   x[i].ub 10→3 (5 LP tightenings)
//   sum  >= 14:  excl_max=4×3=12, x[i].lb 0→2 (5 LP tightenings) — in same pass!
//   LP result: x[i] ∈ [2, 3]   (already integer → 0 roundings)
//
// Outer iteration 2: LP finds no change → fixed point.
//
// After presolve: x[0..4] ∈ [2, 3]; x[5..9] ∈ [0, 10] (untouched).
// MILP optimal: sum(x[0..4]) = 14 (=⌈13.5⌉), x[5..9]=0 → obj=14.

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
        REQUIRE(res.rhsRounded      == 6);  // PR1: 5×(3.9→3) + 1×(13.5→14)
        REQUIRE(res.boundsTightened == 10); // 5 ub (LP) + 5 lb (LP) in one pass
        REQUIRE(res.boundsRounded   == 0);  // LP gave integer bounds directly
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
// a, b ∈ [0, 100] Integer.  Constraints added in this order:
//   C1: 2b - a <= 1   (added first — processed first in each LP pass)
//   C2: 2a    <= 9
//
// PR1: both RHS are already integers → rhsRounded = 0.
//
// With maxCycles=1 (1 LP pass + 1 round):
//   LP pass:  C1 with a.ub=100 → b.ub ≤ (1+100)/2 = 50.5 (tightened from 100)
//             C2            → a.ub ≤ 9/2 = 4.5   (tightened from 100)
//   Round:    a.ub 4.5→4, b.ub 50.5→50   (2 roundings)
//   Result:   a ∈ [0,4], b ∈ [0,50]      — b NOT fully tightened
//
// With maxCycles=2 (two LP+round cycles):
//   Cycle 2 LP:  C1 with a.ub=4 → b.ub ≤ (1+4)/2 = 2.5 (tightened from 50)
//   Cycle 2 Round: b.ub 2.5→2   (1 more rounding)
//   Result:   a ∈ [0,4], b ∈ [0,2]       — fully tightened
//
// The key: C1 is processed before a.ub is tightened in cycle 1, so it cannot
// propagate to b.ub yet.  Only after a.ub is rounded (cycle 1) and cycle 2's
// LP pass runs can C1 propagate tightly.
//
// Also verifies milpPresolveMaxCycles ⊥ lpOpts.presolveMaxPasses.

TEST_CASE("presolveMILP: milpPresolveMaxCycles limits outer iterations", "[presolve][milp]") {
    Model m;
    Variable a = m.addVar(0.0, 100.0, VarType::Integer, "a");
    Variable b = m.addVar(0.0, 100.0, VarType::Integer, "b");

    // C1 first so it processes before a.ub is tightened in cycle-1 LP.
    m.addLPConstraint(2.0*b - 1.0*a, Sense::LessEq, 1.0);
    m.addLPConstraint(2.0*a,         Sense::LessEq, 9.0);

    m.setObjective(1.0*a + 1.0*b, ObjSense::Minimize);

    // maxCycles=1: b.ub only tightened to 50 (a.ub not yet rounded when C1 runs).
    {
        Model mCopy = m;
        MILPPresolveResult res = presolveMILPInPlace(mCopy, MILPPresolveOpts{.maxCycles = 1});
        REQUIRE_FALSE(res.infeasible);
        REQUIRE(res.rhsRounded      == 0); // all RHS already integer
        REQUIRE(res.boundsTightened == 2); // b.ub 100→50.5, a.ub 100→4.5
        REQUIRE(res.boundsRounded   == 2); // a.ub 4.5→4, b.ub 50.5→50
        const auto& hot = mCopy.getHot();
        REQUIRE_THAT(hot.ub[a.id], WithinAbs(4.0,  kTol));
        REQUIRE_THAT(hot.ub[b.id], WithinAbs(50.0, kTol)); // NOT yet 2
    }

    // maxCycles=2: b.ub fully tightened to 2 using rounded a.ub=4.
    {
        Model mCopy = m;
        MILPPresolveResult res = presolveMILPInPlace(mCopy, MILPPresolveOpts{.maxCycles = 2});
        REQUIRE_FALSE(res.infeasible);
        REQUIRE(res.boundsTightened == 3); // +1: b.ub 50→2.5 in cycle 2
        REQUIRE(res.boundsRounded   == 3); // +1: b.ub 2.5→2  in cycle 2
        const auto& hot = mCopy.getHot();
        REQUIRE_THAT(hot.ub[a.id], WithinAbs(4.0, kTol));
        REQUIRE_THAT(hot.ub[b.id], WithinAbs(2.0, kTol)); // fully tightened
    }

    // BBOptions::milpPresolveMaxCycles controls the outer loop;
    // lpOpts.presolveMaxPasses (LP node passes) is independent.
    BBOptions opts;
    opts.milpPresolveMaxCycles    = 1;
    opts.lpOpts.presolveMaxPasses = 0; // unlimited LP passes per node
    opts.lpOpts.method            = LPMethod::DualSimplexBV;
    MILPResult r = solveMILP(m, opts);

    REQUIRE(r.status == MILPStatus::Optimal);
    REQUIRE_THAT(r.objectiveValue, WithinAbs(0.0, kTol)); // min = a=b=0
}

// ════════════════════════════════════════════════════════════════════════════
// PR1 — Integer RHS rounding
// ════════════════════════════════════════════════════════════════════════════

// ── PR1.1: LessEq with non-integer RHS ───────────────────────────────────────
//
// x ∈ [0, 10] Integer, x <= 3.7  →  PR1 rounds to x <= 3.
// LP then tightens x.ub from 10 to 3; result is already integer → 0 roundings.

TEST_CASE("presolveMILP: PR1 LessEq RHS rounded to floor", "[presolve][milp][PR1]") {
    Model m;
    Variable x = m.addVar(0.0, 10.0, VarType::Integer, "x");
    m.addLPConstraint(1.0*x, Sense::LessEq, 3.7);

    MILPPresolveResult res = presolveMILPInPlace(m);

    REQUIRE_FALSE(res.infeasible);
    REQUIRE(res.rhsRounded      == 1); // 3.7 → floor(3.7) = 3
    REQUIRE(res.boundsTightened == 1); // x.ub 10 → 3
    REQUIRE(res.boundsRounded   == 0); // result already integer
    REQUIRE_THAT(m.getHot().ub[x.id], WithinAbs(3.0, kTol));
}

// ── PR1.2: GreaterEq with non-integer RHS ────────────────────────────────────
//
// x ∈ [0, 10] Integer, x >= 2.3  →  PR1 rounds to x >= 3.
// LP then tightens x.lb from 0 to 3.

TEST_CASE("presolveMILP: PR1 GreaterEq RHS rounded to ceil", "[presolve][milp][PR1]") {
    Model m;
    Variable x = m.addVar(0.0, 10.0, VarType::Integer, "x");
    m.addLPConstraint(1.0*x, Sense::GreaterEq, 2.3);

    MILPPresolveResult res = presolveMILPInPlace(m);

    REQUIRE_FALSE(res.infeasible);
    REQUIRE(res.rhsRounded      == 1); // 2.3 → ceil(2.3) = 3
    REQUIRE(res.boundsTightened == 1); // x.lb 0 → 3
    REQUIRE(res.boundsRounded   == 0); // result already integer
    REQUIRE_THAT(m.getHot().lb[x.id], WithinAbs(3.0, kTol));
}

// ── PR1.3: Already-integer RHS → no rounding ─────────────────────────────────

TEST_CASE("presolveMILP: PR1 integer RHS unchanged", "[presolve][milp][PR1]") {
    Model m;
    Variable x = m.addVar(0.0, 10.0, VarType::Integer, "x");
    m.addLPConstraint(1.0*x, Sense::LessEq, 4.0);

    MILPPresolveResult res = presolveMILPInPlace(m);

    REQUIRE(res.rhsRounded == 0); // 4.0 is already integer
    REQUIRE_THAT(m.getHot().ub[x.id], WithinAbs(4.0, kTol));
}

// ── PR1.4: Equality constraint → no rounding (no safe direction) ─────────────

TEST_CASE("presolveMILP: PR1 equality constraint not rounded", "[presolve][milp][PR1]") {
    Model m;
    Variable x = m.addVar(0.0, 10.0, VarType::Integer, "x");
    // x == 3.7 → LP fixes x ∈ [3.7, 3.7] → roundIntBounds: infeasible
    m.addLPConstraint(1.0*x, Sense::Equal, 3.7);

    MILPPresolveResult res = presolveMILPInPlace(m);

    REQUIRE(res.rhsRounded == 0); // equality skipped by PR1
    REQUIRE(res.infeasible);      // LP pins x=3.7; rounding detects lb>ub
}

// ── PR1.5: Continuous variable in constraint → no rounding ───────────────────

TEST_CASE("presolveMILP: PR1 skips constraint with continuous variable", "[presolve][milp][PR1]") {
    Model m;
    Variable x = m.addVar(0.0, 10.0, VarType::Integer,    "x");
    Variable y = m.addVar(0.0, 10.0, VarType::Continuous, "y");
    m.addLPConstraint(1.0*x + 1.0*y, Sense::LessEq, 3.7);

    MILPPresolveResult res = presolveMILPInPlace(m);

    REQUIRE(res.rhsRounded == 0); // y is Continuous → ineligible
}

// ── PR1.6: Non-integer coefficient → no rounding ─────────────────────────────

TEST_CASE("presolveMILP: PR1 skips constraint with non-integer coefficient", "[presolve][milp][PR1]") {
    Model m;
    Variable x = m.addVar(0.0, 10.0, VarType::Integer, "x");
    m.addLPConstraint(1.5*x, Sense::LessEq, 3.7); // coeff 1.5 is not integer

    MILPPresolveResult res = presolveMILPInPlace(m);

    REQUIRE(res.rhsRounded == 0); // non-integer coefficient → ineligible
}

// ── PR1.7: Binary variable eligible ──────────────────────────────────────────
//
// x ∈ {0,1} Binary, y ∈ {0,1} Binary, x + y <= 1.5 → rounds to x + y <= 1.

TEST_CASE("presolveMILP: PR1 binary variable eligible for rounding", "[presolve][milp][PR1]") {
    Model m;
    Variable x = m.addVar(0.0, 1.0, VarType::Binary, "x");
    Variable y = m.addVar(0.0, 1.0, VarType::Binary, "y");
    m.addLPConstraint(1.0*x + 1.0*y, Sense::LessEq, 1.5);

    MILPPresolveResult res = presolveMILPInPlace(m);

    REQUIRE(res.rhsRounded == 1); // 1.5 → floor(1.5) = 1
}

// ════════════════════════════════════════════════════════════════════════════
// Integration: all presolve levels yield the correct optimal
// ════════════════════════════════════════════════════════════════════════════

// ── Knapsack-10: optimal = 106 across levels 0-6 ─────────────────────────────
//
// 10-item 0/1 knapsack (capacity 50, optimal profit 106).
// LP relaxation takes item 8 at fraction 1/2 → B&B required for levels 0-1.
// Higher levels may tighten bounds or fix variables before branching.
// All levels must reach the same optimal.

TEST_CASE("presolveMILP: knapsack-10 optimal across all presolve levels", "[presolve][milp][levels]") {
    const auto level = GENERATE(0u, 1u, 2u, 3u, 4u, 5u, 6u);

    DYNAMIC_SECTION("presolveLevel=" << level) {
        BBOptions opts;
        opts.presolveLevel = level;
        opts.timeLimitS    = 60.0;

        MILPResult r = solveMILP(baguette_test::makeKnapsack10(), opts);

        REQUIRE(r.status == MILPStatus::Optimal);
        REQUIRE_THAT(r.objectiveValue, WithinAbs(106.0, kTol));
    }
}

// ── TSP-10 (MTZ): optimal = 10 across levels 0-6 ─────────────────────────────
//
// 10-city cyclic TSP (MTZ formulation, cycle 0→1→…→9→0, cost 1 per arc).
// LP relaxation is already integer at the cyclic tour → B&B terminates at root.
// Presolve may tighten MTZ position variables; correctness must be preserved.

TEST_CASE("presolveMILP: TSP-10 optimal across all presolve levels", "[presolve][milp][levels]") {
    const auto level = GENERATE(0u, 1u, 2u, 3u, 4u, 5u, 6u);

    DYNAMIC_SECTION("presolveLevel=" << level) {
        BBOptions opts;
        opts.presolveLevel = level;
        opts.timeLimitS    = 60.0;

        MILPResult r = solveMILP(baguette_test::makeTSP10(), opts);

        REQUIRE(r.status == MILPStatus::Optimal);
        REQUIRE_THAT(r.objectiveValue, WithinAbs(10.0, kTol));
    }
}
