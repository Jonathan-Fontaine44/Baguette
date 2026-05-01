#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "baguette/lp/LPSolver.hpp"
#include "lp_problems.hpp"

using namespace baguette;
using Catch::Matchers::WithinAbs;

static constexpr double kTol = 1e-5;

static LPResult mehrotra(Model m) {
    LPOptions opts;
    opts.method = LPMethod::MehrotraIPM;
    return solveLP(m, opts);
}

TEST_CASE("Mehrotra IPM: simple max", "[mehrotra_ipm]") {
    auto r = mehrotra(makeSimpleMax());
    REQUIRE(r.status == LPStatus::Optimal);
    REQUIRE_THAT(r.objectiveValue, WithinAbs(4.0, kTol));
}

TEST_CASE("Mehrotra IPM: min with GEQ", "[mehrotra_ipm]") {
    auto r = mehrotra(makeMinWithGEQ());
    REQUIRE(r.status == LPStatus::Optimal);
    REQUIRE_THAT(r.objectiveValue, WithinAbs(8.0, kTol));
}

TEST_CASE("Mehrotra IPM: equality constraint", "[mehrotra_ipm]") {
    auto r = mehrotra(makeEqualityConstraint());
    REQUIRE(r.status == LPStatus::Optimal);
    REQUIRE_THAT(r.objectiveValue, WithinAbs(5.0, kTol));
}

TEST_CASE("Mehrotra IPM: upper bound", "[mehrotra_ipm]") {
    auto r = mehrotra(makeUpperBound());
    REQUIRE(r.status == LPStatus::Optimal);
    REQUIRE_THAT(r.objectiveValue, WithinAbs(5.0, kTol));
}

TEST_CASE("Mehrotra IPM: lower bound", "[mehrotra_ipm]") {
    auto r = mehrotra(makeLowerBound());
    REQUIRE(r.status == LPStatus::Optimal);
    REQUIRE_THAT(r.objectiveValue, WithinAbs(3.0, kTol));
}

TEST_CASE("Mehrotra IPM: simple min LEQ", "[mehrotra_ipm]") {
    auto r = mehrotra(makeSimpleMinLEQ());
    REQUIRE(r.status == LPStatus::Optimal);
    REQUIRE_THAT(r.objectiveValue, WithinAbs(4.0, kTol));
}

TEST_CASE("Mehrotra IPM: three variables", "[mehrotra_ipm]") {
    static const auto suite = makeLPTestSuite();
    const auto& tc = suite[8]; // three_var, obj = -26
    auto r = mehrotra(tc.build());
    REQUIRE(r.status == LPStatus::Optimal);
    REQUIRE_THAT(r.objectiveValue, WithinAbs(tc.expectedObj, kTol));
}

TEST_CASE("Mehrotra IPM: max 5x+4y", "[mehrotra_ipm]") {
    static const auto suite = makeLPTestSuite();
    const auto& tc = suite[9]; // max_5x4y, obj = 21
    auto r = mehrotra(tc.build());
    REQUIRE(r.status == LPStatus::Optimal);
    REQUIRE_THAT(r.objectiveValue, WithinAbs(tc.expectedObj, kTol));
}

TEST_CASE("Mehrotra IPM: infeasible", "[mehrotra_ipm]") {
    auto r = mehrotra(makeInfeasible());
    REQUIRE(r.status == LPStatus::Infeasible);
}

TEST_CASE("Mehrotra IPM: unbounded", "[mehrotra_ipm]") {
    auto r = mehrotra(makeUnbounded());
    REQUIRE(r.status == LPStatus::Unbounded);
}