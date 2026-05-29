#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <variant>

#include "baguette/core/Sense.hpp"
#include "baguette/cp/constraints/AllDiff.hpp"
#include "baguette/lp/LPSolver.hpp"
#include "baguette/milp/BranchAndBound.hpp"
#include "baguette/model/Model.hpp"
#include "baguette/model/ModelEnums.hpp"
#include "baguette/model/Presolve.hpp"

using namespace baguette;
using Catch::Matchers::WithinAbs;

static constexpr double kTol = 1e-9;

// ── LP reduce ─────────────────────────────────────────────────────────────────
//
// x∈[3,3], y∈[2,2] (fixed), z∈[0,10], w∈[0,10]
// x + y + z + w <= 12  →  z + w <= 7  (RHS adjusted by 3+2=5)
// min z + w  →  optimal z=0, w=0, obj=0

TEST_CASE("presolveElim: 2 fixed vars - RHS adjustment and LP solve coherence",
          "[presolve][elim]") {
    Model m;
    Variable x = m.addVar(3.0,  3.0,  "x");
    Variable y = m.addVar(2.0,  2.0,  "y");
    Variable z = m.addVar(0.0, 10.0,  "z");
    Variable w = m.addVar(0.0, 10.0,  "w");
    m.addLPConstraint(1.0*x + 1.0*y + 1.0*z + 1.0*w, Sense::LessEq, 12.0);
    m.setObjective(1.0*z + 1.0*w, ObjSense::Minimize);

    EliminationRecord rec;
    Model reduced = presolveElim(m, rec);

    // Structure
    REQUIRE(rec.lpVarCount    == 2);
    REQUIRE(rec.varsEliminated == 2);
    REQUIRE(reduced.numVars()       == 2);
    REQUIRE(reduced.numTotalVars()  == 4);

    // RHS: 12 - 3 - 2 = 7
    REQUIRE(reduced.getLPConstraints().size() == 1);
    REQUIRE_THAT(reduced.getLPConstraints()[0].rhsConst, WithinAbs(7.0, kTol));

    // Solve reduced LP
    LPOptions lpOpts;
    lpOpts.method = LPMethod::DualSimplexBV;
    auto lp = solveLPDetailed(reduced, lpOpts);

    REQUIRE(lp.result.status == LPStatus::Optimal);
    REQUIRE_THAT(lp.result.objectiveValue, WithinAbs(0.0, kTol));
    REQUIRE_THAT(lp.result.primalValues[rec.varMap[z.id]], WithinAbs(0.0, kTol));
    REQUIRE_THAT(lp.result.primalValues[rec.varMap[w.id]], WithinAbs(0.0, kTol));
}

// ── CP reduce (AllDiff + ghost vars) ─────────────────────────────────────────
//
// a∈[1,1], b∈[2,2] (fixed), c∈{1..5} Integer
// AllDiff(a, b, c)
// min c
//
// After elimination: a, b become ghost vars in reduced AllDiff.
// Propagation sees a fixed to 1 and b fixed to 2 → c ∈ {3,4,5}.
// Optimal: c = 3.

TEST_CASE("presolveElimCP: AllDiff with 2 ghost vars - structure and MILP coherence",
          "[presolve][elim][cp]") {
    Model m;
    Variable a = m.addVar(1.0, 1.0, VarType::Integer, "a");
    Variable b = m.addVar(2.0, 2.0, VarType::Integer, "b");
    Variable c = m.addVar(1.0, 5.0, VarType::Integer, "c");
    m.addCPConstraint(AllDiffConstraint{a, b, c});
    m.setObjective(1.0*c, ObjSense::Minimize);

    // Manual elimination for structure checks
    const CPConstraints cpOrig = m.getCPConstraints();
    EliminationRecord rec;
    Model reduced = presolveElim(m, rec);
    presolveElimCP(cpOrig, rec, reduced);

    // 1 LP var (c), 2 ghost vars (a, b)
    REQUIRE(rec.lpVarCount     == 1);
    REQUIRE(rec.varsEliminated == 2);
    REQUIRE(reduced.numVars()      == 1);
    REQUIRE(reduced.numTotalVars() == 3);

    // AllDiff in reduced model keeps all 3 vars (ghosts included)
    const auto& builtins = reduced.getCPConstraints().builtins();
    REQUIRE(builtins.size() == 1);
    const auto& ad = std::get<AllDiffConstraint>(builtins[0]);
    REQUIRE(ad.vars.size() == 3);

    // MILP solve: c must differ from a=1 and b=2, so min c = 3
    BBOptions opts;
    opts.lpOpts.method = LPMethod::DualSimplexBV;
    MILPResult r = solveMILP(m, opts);

    REQUIRE(r.status == MILPStatus::Optimal);
    REQUIRE_THAT(r.objectiveValue,       WithinAbs(3.0, kTol));
    REQUIRE_THAT(r.primalValues[c.id],   WithinAbs(3.0, kTol));
    REQUIRE_THAT(r.primalValues[a.id],   WithinAbs(1.0, kTol));
    REQUIRE_THAT(r.primalValues[b.id],   WithinAbs(2.0, kTol));
}

// ── Large test: 10 vars, 4 fixed by equality constraints ─────────────────────
//
// x0..x9 ∈ [0,100] Integer (wide bounds simulate a pre-branching node)
// Equality constraints fix x0..x3 (simulates B&B constraint tightening):
//   x0==3, x1==5, x2==2, x3==7
// TB presolve detects lb==ub on x0..x3; presolveElim promotes them to ghost vars.
//
// AllDiff(x0, x1, x2, x3, x4, x5)      → 6 vars in reduced constraint (4 ghost + 2 LP)
// x4+x5+x6+x7+x8+x9 <= 50              → capacity (survives, RHS unchanged)
// min x4+x5+x6+x7+x8+x9
//
// After propagation: x4 ∉ {2,3,5,7}, x5 ∉ {0,2,3,5,7} → min x4=0, x5=1, rest=0 → obj=1.

TEST_CASE("presolveElim+CP: 10 vars, 4 fixed by equality constraints",
          "[presolve][elim][cp]") {
    Model m;
    std::vector<Variable> x(10);
    for (int i = 0; i < 10; ++i)
        x[i] = m.addVar(0.0, 100.0, VarType::Integer, "x" + std::to_string(i));

    // Equality constraints that simulate B&B node tightening
    m.addLPConstraint(1.0*x[0], Sense::Equal, 3.0);
    m.addLPConstraint(1.0*x[1], Sense::Equal, 5.0);
    m.addLPConstraint(1.0*x[2], Sense::Equal, 2.0);
    m.addLPConstraint(1.0*x[3], Sense::Equal, 7.0);

    // Capacity constraint on free variables only
    m.addLPConstraint(1.0*x[4] + 1.0*x[5] + 1.0*x[6] + 1.0*x[7] + 1.0*x[8] + 1.0*x[9],
                      Sense::LessEq, 50.0);

    m.addCPConstraint(AllDiffConstraint{x[0], x[1], x[2], x[3], x[4], x[5]});

    m.setObjective(1.0*x[4] + 1.0*x[5] + 1.0*x[6] + 1.0*x[7] + 1.0*x[8] + 1.0*x[9],
                   ObjSense::Minimize);

    // ── Structure checks (manual presolve on a copy) ───────────────────────────
    {
        Model mCopy = m;
        presolveTBInPlace(mCopy);

        const CPConstraints cpOrig = mCopy.getCPConstraints();
        EliminationRecord rec;
        Model reduced = presolveElim(mCopy, rec);
        presolveElimCP(cpOrig, rec, reduced);

        // 4 LP vars eliminated → 4 ghost vars; 6 LP vars remain
        REQUIRE(rec.varsEliminated  == 4);
        REQUIRE(rec.lpVarCount      == 6);
        REQUIRE(reduced.numVars()      == 6);
        REQUIRE(reduced.numTotalVars() == 10);

        // 4 equality rows become empty after fixing → eliminated; capacity row kept
        REQUIRE(rec.rowsEliminated     == 4);
        REQUIRE(reduced.numConstraints() == 1);
        REQUIRE_THAT(reduced.getLPConstraints()[0].rhsConst, WithinAbs(50.0, kTol));

        // AllDiff in reduced model: 4 ghost vars + 2 LP vars = 6 total
        const auto& builtins = reduced.getCPConstraints().builtins();
        REQUIRE(builtins.size() == 1);
        REQUIRE(std::get<AllDiffConstraint>(builtins[0]).vars.size() == 6);
    }

    // ── MILP solve coherence ───────────────────────────────────────────────────
    BBOptions opts;
    opts.lpOpts.method = LPMethod::DualSimplexBV;
    MILPResult r = solveMILP(m, opts);

    REQUIRE(r.status == MILPStatus::Optimal);
    REQUIRE_THAT(r.objectiveValue, WithinAbs(1.0, kTol));

    // Fixed vars restored by postsolve
    REQUIRE_THAT(r.primalValues[x[0].id], WithinAbs(3.0, kTol));
    REQUIRE_THAT(r.primalValues[x[1].id], WithinAbs(5.0, kTol));
    REQUIRE_THAT(r.primalValues[x[2].id], WithinAbs(2.0, kTol));
    REQUIRE_THAT(r.primalValues[x[3].id], WithinAbs(7.0, kTol));

    // x4 and x5 must be {0,1} in some order: min pair satisfying AllDiff with {2,3,5,7}
    const double v4 = r.primalValues[x[4].id];
    const double v5 = r.primalValues[x[5].id];
    REQUIRE_THAT(v4 + v5, WithinAbs(1.0, kTol)); // sum == 1
    REQUIRE_THAT(std::min(v4, v5), WithinAbs(0.0, kTol));
    REQUIRE_THAT(std::max(v4, v5), WithinAbs(1.0, kTol));

    // x6..x9 unconstrained by AllDiff → minimize to 0
    REQUIRE_THAT(r.primalValues[x[6].id], WithinAbs(0.0, kTol));
    REQUIRE_THAT(r.primalValues[x[7].id], WithinAbs(0.0, kTol));
    REQUIRE_THAT(r.primalValues[x[8].id], WithinAbs(0.0, kTol));
    REQUIRE_THAT(r.primalValues[x[9].id], WithinAbs(0.0, kTol));
}
