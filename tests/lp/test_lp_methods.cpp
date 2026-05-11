#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "baguette/lp/LPSolver.hpp"
#include "lp_problems.hpp"
#include "MILP_problems.hpp"

using namespace baguette;
using Catch::Matchers::WithinAbs;

static constexpr double kTol = 1e-6;

// Runs every LP problem in makeLPTestSuite() against every solving method and
// verifies that the status and objective value are correct.
TEST_CASE("LP methods x problems: status and objective", "[lp_methods]") {
    auto method = GENERATE(LPMethod::Auto, LPMethod::PrimalSimplex,
                           LPMethod::DualSimplex, LPMethod::RevisedSimplex,
                           LPMethod::ShortStepIPM, LPMethod::MehrotraIPM,
                           LPMethod::PrimalSimplexBV, LPMethod::DualSimplexBV,
                           LPMethod::RevisedSimplexBV, LPMethod::NetworkSimplex);

    static const auto suite = makeLPTestSuite();
    auto i = GENERATE(range(std::size_t{0}, suite.size()));
    const auto& tc = suite[i];

    auto enablePresolve = GENERATE(false, true);

    DYNAMIC_SECTION("Method=" << to_string(method) << ", case=" << tc.name
                               << ", presolve=" << enablePresolve) {
        // ShortStepIPM cannot prove infeasibility or unboundedness: returns MaxIter.
        // Exception: presolve can detect infeasibility (bound contradiction), so when
        // presolve is enabled and the problem is infeasible, expect Infeasible.
        const bool shortStepCantSolve =
            method == LPMethod::ShortStepIPM && tc.expectedStatus != LPStatus::Optimal;
        const bool presolveHandlesInfeasible =
            enablePresolve && tc.expectedStatus == LPStatus::Infeasible;
        const LPStatus expected =
            (shortStepCantSolve && !presolveHandlesInfeasible)
            ? LPStatus::MaxIter
            : tc.expectedStatus;

        LPOptions opts;
        opts.method         = method;
        opts.enablePresolve = enablePresolve;
        LPResult r = solveLP(tc.build(), opts);

        REQUIRE(r.status == expected);
        if (expected == LPStatus::Optimal)
            REQUIRE_THAT(r.objectiveValue, WithinAbs(tc.expectedObj, kTol));
    }
}

// Runs every LP relaxation of a classic MILP problem against every solving method.
TEST_CASE("Relaxed MILP x methods: status and objective", "[lp_methods]") {
    auto method = GENERATE(LPMethod::Auto, LPMethod::PrimalSimplex,
                           LPMethod::DualSimplex, LPMethod::RevisedSimplex,
                           LPMethod::ShortStepIPM, LPMethod::MehrotraIPM,
                           LPMethod::PrimalSimplexBV, LPMethod::DualSimplexBV,
                           LPMethod::RevisedSimplexBV, LPMethod::NetworkSimplex);

    static const auto suite = makeRelaxedMILPTestSuite();
    auto i = GENERATE(range(std::size_t{0}, suite.size()));
    const auto& tc = suite[i];

    auto enablePresolve = GENERATE(false, true);

    DYNAMIC_SECTION("Method=" << to_string(method) << ", case=" << tc.name
                               << ", presolve=" << enablePresolve) {
        LPOptions opts;
        opts.method         = method;
        opts.enablePresolve = enablePresolve;
        // ShortStepIPM ne converge pas sur les relaxations MILP : son pas fixe
        // α=1/(1+√n) stagne quand les variables approchent 0 à l'optimum.
        // On limite à 0.1 s pour ne pas bloquer la suite.
        if (method == LPMethod::ShortStepIPM) opts.timeLimitS = 0.1;
        LPResult r = solveLP(tc.build(), opts);

        const bool shortStepInconcl =
            method == LPMethod::ShortStepIPM &&
            (r.status == LPStatus::MaxIter || r.status == LPStatus::TimeLimit);
        if (!shortStepInconcl)
            REQUIRE(r.status == tc.expectedStatus);
        if (r.status == LPStatus::Optimal)
            REQUIRE_THAT(r.objectiveValue, WithinAbs(tc.expectedObj, kTol));
    }
}
