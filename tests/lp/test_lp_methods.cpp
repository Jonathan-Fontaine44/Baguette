#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "baguette/lp/LPSolver.hpp"
#include "lp_problems.hpp"

using namespace baguette;
using Catch::Matchers::WithinAbs;

static constexpr double kTol = 1e-6;

// Runs every LP problem in makeLPTestSuite() against every solving method and
// verifies that the status and objective value are correct.
TEST_CASE("LP methods x problems: status and objective", "[lp_methods]") {
    auto method = GENERATE(LPMethod::Auto, LPMethod::PrimalSimplex,
                           LPMethod::DualSimplex, LPMethod::RevisedSimplex);

    static const auto suite = makeLPTestSuite();
    auto i = GENERATE(range(std::size_t{0}, suite.size()));
    const auto& tc = suite[i];

    const char* mname =
        (method == LPMethod::Auto)          ? "Auto" :
        (method == LPMethod::PrimalSimplex) ? "PrimalSimplex" :
        (method == LPMethod::DualSimplex)   ? "DualSimplex" : "RevisedSimplex";

    DYNAMIC_SECTION("Method=" << mname << ", case=" << tc.name) {
        LPOptions opts;
        opts.method = method;
        LPResult r = solveLP(tc.build(), opts);

        REQUIRE(r.status == tc.expectedStatus);
        if (tc.expectedStatus == LPStatus::Optimal)
            REQUIRE_THAT(r.objectiveValue, WithinAbs(tc.expectedObj, kTol));
    }
}
