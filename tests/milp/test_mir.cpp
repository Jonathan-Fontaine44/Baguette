#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "baguette/lp/LPSolver.hpp"
#include "baguette/milp/BranchAndBound.hpp"
#include "baguette/milp/CuttingPlanes.hpp"
#include "baguette/model/Model.hpp"
#include "baguette/model/ModelEnums.hpp"
#include "milp/cuts/mir.hpp"

using namespace baguette;
using Catch::Matchers::WithinAbs;

static constexpr double kTol = 1e-6;

// ── MIR: cut from LessEq constraint with fractional RHS ──────────────────────
//
// max x + y   s.t. 2x + 3y ≤ 8.5,  x,y ∈ ℤ[0,5]
//
// LP optimal: (4.25, 0), obj = 4.25.
// MIR cut from the constraint: 2x + 3y ≤ 8   (f=0.5, int coeffs → ⌊RHS⌋)
//   - LP solution (4.25, 0) violates: 2*4.25 = 8.5 > 8.
//   - IP optimal (4, 0): 2*4 = 8 ≤ 8.  ✓
// After adding the MIR cut, the LP optimal becomes (4, 0) = IP optimal.

TEST_CASE("MIR: cut closes LP-IP gap on LessEq constraint", "[mir]") {
    auto method = GENERATE(LPMethod::PrimalSimplex,   LPMethod::DualSimplex,
                           LPMethod::PrimalSimplexBV, LPMethod::DualSimplexBV,
                           LPMethod::RevisedSimplexBV);
    DYNAMIC_SECTION("method=" << to_string(method)) {
        Model m;
        Variable x = m.addVar(0.0, 5.0, VarType::Integer, "x");
        Variable y = m.addVar(0.0, 5.0, VarType::Integer, "y");
        m.addLPConstraint(2.0 * x + 3.0 * y, Sense::LessEq, 8.5);
        m.setObjective(1.0 * x + 1.0 * y, ObjSense::Maximize);

        LPOptions lpOpts;
        lpOpts.method         = method;
        lpOpts.enablePresolve = false;
        lpOpts.timeLimitS     = 1.0;
        LPDetailedResult lp = solveLPDetailed(m, lpOpts);

        REQUIRE(lp.result.status == LPStatus::Optimal);
        REQUIRE_THAT(lp.result.objectiveValue, WithinAbs(4.25, kTol));

        std::vector<Cut> cuts = generateMIRCuts(lp, m, 10, kTol);
        REQUIRE(cuts.size() >= 1);

        const Cut& cut = cuts[0];
        REQUIRE(cut.sense == Sense::LessEq);

        // IP optimal (4, 0) satisfies the cut.
        double ipLhs = 0.0;
        for (std::size_t k = 0; k < cut.expr.size(); ++k) {
            double ipVal = (cut.expr.varIds[k] == x.id) ? 4.0 : 0.0;
            ipLhs += cut.expr.coeffs[k] * ipVal;
        }
        REQUIRE(ipLhs <= cut.rhs + kTol);

        // LP with cut: optimal must be ≥ IP optimal (valid cut only tightens).
        m.addLPConstraint(cut.expr, cut.sense, cut.rhs);
        LPDetailedResult lp2 = solveLPDetailed(m, lpOpts);
        REQUIRE(lp2.result.status == LPStatus::Optimal);
        REQUIRE(lp2.result.objectiveValue >= 4.0 - kTol);
        REQUIRE(lp2.result.objectiveValue <= lp.result.objectiveValue + kTol);
    }
}

// ── CMIR: cut from GreaterEq constraint via complementation ──────────────────
//
// min x + y   s.t. 3x + 4y ≥ 9.5,  x,y ∈ ℤ[0,3]
//
// LP optimal: (0, 2.375), obj = 2.375.
// CMIR: complement with ub=3 → 3x'+4y' ≤ 11.5 (x'=3-x, y'=3-y).
//   f = 0.5; int coeffs (frac=0 ≤ 0.5) → α̃=3,4; MIR: 3x'+4y' ≤ 11.
//   Back to original: 3x+4y ≥ 10.
//   - LP solution (0, 2.375): 4*2.375 = 9.5 < 10.  Violated ✓
//   - IP optimal (0,3): 12 ≥ 10  ✓
// After adding the CMIR cut, LP bound ≥ 2.5 (tightened from 2.375).

TEST_CASE("CMIR: cut from GreaterEq constraint via complementation", "[mir][cmir]") {
    auto method = GENERATE(LPMethod::PrimalSimplex,   LPMethod::DualSimplex,
                           LPMethod::PrimalSimplexBV, LPMethod::DualSimplexBV,
                           LPMethod::RevisedSimplexBV);
    DYNAMIC_SECTION("method=" << to_string(method)) {
        Model m;
        Variable x = m.addVar(0.0, 3.0, VarType::Integer, "x");
        Variable y = m.addVar(0.0, 3.0, VarType::Integer, "y");
        m.addLPConstraint(3.0 * x + 4.0 * y, Sense::GreaterEq, 9.5);
        m.setObjective(1.0 * x + 1.0 * y, ObjSense::Minimize);

        LPOptions lpOpts;
        lpOpts.method         = method;
        lpOpts.enablePresolve = false;
        lpOpts.timeLimitS     = 1.0;
        LPDetailedResult lp = solveLPDetailed(m, lpOpts);

        REQUIRE(lp.result.status == LPStatus::Optimal);
        REQUIRE_THAT(lp.result.objectiveValue, WithinAbs(2.375, kTol));

        std::vector<Cut> cuts = generateCMIRCuts(lp, m, 10, kTol);
        REQUIRE(cuts.size() >= 1);

        const Cut& cut = cuts[0];
        REQUIRE(cut.sense == Sense::GreaterEq);

        // IP optimals (0,3), (1,2), (2,1) all satisfy the cut.
        auto check = [&](double xv, double yv) {
            double lhsVal = 0.0;
            for (std::size_t k = 0; k < cut.expr.size(); ++k) {
                double v = (cut.expr.varIds[k] == x.id) ? xv : yv;
                lhsVal += cut.expr.coeffs[k] * v;
            }
            return lhsVal >= cut.rhs - kTol;
        };
        REQUIRE(check(0.0, 3.0));
        REQUIRE(check(1.0, 2.0));
        REQUIRE(check(2.0, 1.0));

        // LP with cut: bound tightened from 2.375.
        m.addLPConstraint(cut.expr, cut.sense, cut.rhs);
        LPDetailedResult lp2 = solveLPDetailed(m, lpOpts);
        REQUIRE(lp2.result.status == LPStatus::Optimal);
        REQUIRE(lp2.result.objectiveValue >= lp.result.objectiveValue - kTol);
        REQUIRE(lp2.result.objectiveValue >= 2.5 - kTol);
    }
}

// ── MIR via BBOptions::enableMIR ─────────────────────────────────────────────
//
// Same model as the MIR test. With enableMIR=true, the B&B must find the IP
// optimal of 4, and cutsAdded ≥ 1.

TEST_CASE("MIR: enableMIR flag activates cuts in B&B", "[mir][bb]") {
    Model m;
    Variable x = m.addVar(0.0, 5.0, VarType::Integer, "x");
    Variable y = m.addVar(0.0, 5.0, VarType::Integer, "y");
    m.addLPConstraint(2.0 * x + 3.0 * y, Sense::LessEq, 8.5);
    m.setObjective(1.0 * x + 1.0 * y, ObjSense::Maximize);

    BBOptions opts;
    opts.enableCuts     = false;
    opts.enableMIR      = true;
    opts.collectStats   = true;
    opts.presolveLevel = 0;
    opts.timeLimitS     = 5.0;

    MILPResult r = solveMILP(m, opts);

    REQUIRE(r.status == MILPStatus::Optimal);
    REQUIRE_THAT(r.objectiveValue, WithinAbs(4.0, kTol));
    REQUIRE(r.stats->cutsAdded >= 1);
}
