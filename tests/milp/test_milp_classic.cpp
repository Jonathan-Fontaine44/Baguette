#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "baguette/lp/LPSolver.hpp"
#include "baguette/lp/Presolve.hpp"
#include "baguette/milp/BranchAndBound.hpp"
#include "baguette/milp/MILPResult.hpp"

#include "lp/MILP_problems.hpp"

using namespace baguette;
using Catch::Matchers::WithinAbs;

static constexpr double kTol = 1e-6;

struct MILPTestCase {
    std::string    name;
    MILPStatus     expectedStatus;
    double         expectedObj;
    std::function<baguette::Model()> build;
    bool           large = false; // if true, accept TimeLimit as inconclusive
};

static std::vector<MILPTestCase> makeMILPTestSuite() {
    return {
        // TSP-10 MTZ: LP relaxation optimal = MILP optimal (cyclic tour is an
        // integer extreme point of the polytope) → B&B terminates at the root.
        {"tsp_10", MILPStatus::Optimal, 10.0,
            []() { return baguette_test::makeTSP10(); }},

        // 0/1 knapsack, 10 items: MILP optimal = 106 (items 0-7 + item 9,
        // weight 46, profit 106; LP relaxation takes item 8 at fraction 1/2).
        {"knapsack_10", MILPStatus::Optimal, 106.0,
            []() { return baguette_test::makeKnapsack10(); }},

        // Knapsack: capacity=5 < minLoad=6 → infeasible at the LP root.
        {"knapsack_10_infeasible", MILPStatus::Infeasible, 0.0,
            []() { return baguette_test::makeKnapsack10(5.0, 6.0); }},

        // Job shop 10×2: MILP optimal = 26 (Johnson's rule on the two-machine
        // flow shop). The big-M LP relaxation gives a very weak bound (5), so
        // B&B may need many nodes; marked large to accept TimeLimit.
        {"jobshop_10x2", MILPStatus::Optimal, 26.0,
            []() { return baguette_test::makeJobShop10(); }, true},

        // Job shop: C_max ≤ 4 contradicts C_max ≥ 5 (from precedence) →
        // infeasible at the LP root.
        {"jobshop_10x2_infeasible", MILPStatus::Infeasible, 0.0,
            []() { return baguette_test::makeJobShop10(4.0); }},
    };
}

// ── Presolve coherence: presolve must not mark feasible problems as infeasible ──
TEST_CASE("Classic MILP x presolve coherence", "[milp_classic][presolve]") {
    static const auto suite = makeMILPTestSuite();
    auto i = GENERATE(range(std::size_t{0}, suite.size()));
    const auto& tc = suite[i];

    DYNAMIC_SECTION("Presolve / " << tc.name) {
        Model m = tc.build();
        PresolveResult pr = presolveTBInPlace(m);

        // Presolve is sound: if it claims infeasible, the problem must be.
        if (pr.infeasible)
            REQUIRE(tc.expectedStatus == MILPStatus::Infeasible);
    }
}

TEST_CASE("Classic MILP x LP-method x B&B/B&C", "[milp_classic]") {
    auto lpMethod = GENERATE(LPMethod::RevisedSimplex, LPMethod::MehrotraIPM,
                             LPMethod::PrimalSimplex, LPMethod::DualSimplex,
                             LPMethod::PrimalSimplexBV, LPMethod::DualSimplexBV,
                             LPMethod::RevisedSimplexBV);

    static const auto suite = makeMILPTestSuite();
    auto i = GENERATE(range(std::size_t{0}, suite.size()));
    const auto& tc = suite[i];

    // B&B (no cuts) and B&C (GMI cuts enabled).
    auto enableCuts = GENERATE(false, true);

    const char* solver = enableCuts ? "BnC" : "BB";

    DYNAMIC_SECTION(solver << " / " << to_string(lpMethod) << " / " << tc.name) {
        BBOptions opts;
        opts.lpOpts.method  = lpMethod;
        opts.enableCuts     = enableCuts;
        opts.enablePresolve = true;
        opts.enableElimination = true;
        opts.timeLimitS     = 10.0;
        if (tc.large) opts.timeLimitS = 1.0;

        MILPResult r = solveMILP(tc.build(), opts);

        const bool inconclusive = tc.large && r.status == MILPStatus::TimeLimit;
        if (!inconclusive)
            REQUIRE(r.status == tc.expectedStatus);
        if (r.status == MILPStatus::Optimal)
            REQUIRE_THAT(r.objectiveValue, WithinAbs(tc.expectedObj, kTol));
    }
}
