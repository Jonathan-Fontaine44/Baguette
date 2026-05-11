#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "baguette/lp/LPSolver.hpp"
#include "lp_problems.hpp"

using namespace baguette;
using Catch::Matchers::WithinAbs;

static constexpr double kTol = 1e-5;

static LPResult ipm(Model m) {
    LPOptions opts;
    opts.method = LPMethod::ShortStepIPM;
    opts.enablePresolve = false;
    return solveLP(m, opts);
}

TEST_CASE("IPM: simple max", "[ipm]") {
    auto r = ipm(makeSimpleMax());
    REQUIRE(r.status == LPStatus::Optimal);
    REQUIRE_THAT(r.objectiveValue, WithinAbs(4.0, kTol));
}

TEST_CASE("IPM: min with GEQ", "[ipm]") {
    auto r = ipm(makeMinWithGEQ());
    REQUIRE(r.status == LPStatus::Optimal);
    REQUIRE_THAT(r.objectiveValue, WithinAbs(8.0, kTol));
}

TEST_CASE("IPM: equality constraint", "[ipm]") {
    auto r = ipm(makeEqualityConstraint());
    REQUIRE(r.status == LPStatus::Optimal);
    REQUIRE_THAT(r.objectiveValue, WithinAbs(5.0, kTol));
}

TEST_CASE("IPM: upper bound", "[ipm]") {
    auto r = ipm(makeUpperBound());
    REQUIRE(r.status == LPStatus::Optimal);
    REQUIRE_THAT(r.objectiveValue, WithinAbs(5.0, kTol));
}

TEST_CASE("IPM: lower bound", "[ipm]") {
    auto r = ipm(makeLowerBound());
    REQUIRE(r.status == LPStatus::Optimal);
    REQUIRE_THAT(r.objectiveValue, WithinAbs(3.0, kTol));
}

TEST_CASE("IPM: simple min LEQ", "[ipm]") {
    auto r = ipm(makeSimpleMinLEQ());
    REQUIRE(r.status == LPStatus::Optimal);
    REQUIRE_THAT(r.objectiveValue, WithinAbs(4.0, kTol));
}

TEST_CASE("IPM: three variables", "[ipm]") {
    // max -2x1 - 3x2 - x3 s.t. ... expected obj = -26
    static const auto suite = makeLPTestSuite();
    const auto& tc = suite[8]; // three_var
    auto r = ipm(tc.build());
    REQUIRE(r.status == LPStatus::Optimal);
    REQUIRE_THAT(r.objectiveValue, WithinAbs(tc.expectedObj, kTol));
}

TEST_CASE("IPM: max 5x+4y", "[ipm]") {
    static const auto suite = makeLPTestSuite();
    const auto& tc = suite[9]; // max_5x4y
    auto r = ipm(tc.build());
    REQUIRE(r.status == LPStatus::Optimal);
    REQUIRE_THAT(r.objectiveValue, WithinAbs(tc.expectedObj, kTol));
}