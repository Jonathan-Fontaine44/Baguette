#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "baguette/lp/LPSolver.hpp"
#include "baguette/milp/CuttingPlanes.hpp"
#include "baguette/model/Model.hpp"
#include "baguette/model/ModelEnums.hpp"
#include "milp/cuts/gmi.hpp"

#include "lp/MILP_problems.hpp"

using namespace baguette;
using Catch::Matchers::WithinAbs;

static constexpr double kTol = 1e-6;

// ── GMI cut coefficients on knapsack-10 ─────────────────────────────────────
//
// Solve the LP relaxation of the 10-item knapsack, generate one GMI cut,
// verify the cut is satisfied by the known IP optimal, then re-solve with
// the cut added and check that the LP bound tightens to ≥ 106.
//
// LP optimal = 110 (item 9 at fraction ½).
// IP optimal = 106 (items 0-7 + item 9).

TEST_CASE("GMI: knapsack-10 cut is valid and tightens the LP bound", "[gmi]") {
    auto method = GENERATE(LPMethod::PrimalSimplex, LPMethod::DualSimplex,
                           LPMethod::PrimalSimplexBV, LPMethod::DualSimplexBV,
                           LPMethod::RevisedSimplexBV);
    DYNAMIC_SECTION("method=" << to_string(method)) {
        Model m = baguette_test::makeKnapsack10();

        LPOptions lpOpts;
        lpOpts.method         = method;
        lpOpts.computeCutData = true;
        lpOpts.timeLimitS     = 1.0;
        LPDetailedResult lp   = solveLPDetailed(m, lpOpts);

        REQUIRE(lp.result.status == LPStatus::Optimal);
        REQUIRE_THAT(lp.result.objectiveValue, WithinAbs(110.0, kTol));
        REQUIRE(!lp.fractionalRows.empty());

        std::vector<Cut> cuts = generateGMICuts(lp.fractionalRows, lp.basis, m, 10, kTol);
        REQUIRE(cuts.size() == 1);

        const Cut& cut = cuts[0];

        // IP optimal: x[0..7]=1, x[8]=0, x[9]=1.
        double lhsIP = 0.0;
        double ipVals[10] = {1,1,1,1,1,1,1,1,0,1};
        for (std::size_t k = 0; k < cut.expr.size(); ++k)
            lhsIP += cut.expr.coeffs[k] * ipVals[cut.expr.varIds[k]];
        REQUIRE(lhsIP >= cut.rhs - kTol);

        // Add the cut and re-solve: LP bound must tighten to ≥ IP optimal.
        m.addLPConstraint(cut.expr, cut.sense, cut.rhs);
        LPOptions lpOpts2;
        lpOpts2.method     = method;
        lpOpts2.timeLimitS = 1.0;
        LPDetailedResult lp2 = solveLPDetailed(m, lpOpts2);

        REQUIRE(lp2.result.status == LPStatus::Optimal);
        REQUIRE(lp2.result.objectiveValue >= 106.0 - kTol);
    }
}
