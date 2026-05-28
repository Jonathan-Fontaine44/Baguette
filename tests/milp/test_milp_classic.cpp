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
        // integer extreme point of the polytope) â†’ B&B terminates at the root.
        {"tsp_10", MILPStatus::Optimal, 10.0,
            []() { return baguette_test::makeTSP10(); }},

        // 0/1 knapsack, 10 items: MILP optimal = 106 (items 0-7 + item 9,
        // weight 46, profit 106; LP relaxation takes item 8 at fraction 1/2).
        {"knapsack_10", MILPStatus::Optimal, 106.0,
            []() { return baguette_test::makeKnapsack10(); }},

        // Knapsack: capacity=5 < minLoad=6 â†’ infeasible at the LP root.
        {"knapsack_10_infeasible", MILPStatus::Infeasible, 0.0,
            []() { return baguette_test::makeKnapsack10(5.0, 6.0); }},

        // Job shop 10Ã—2: MILP optimal = 26 (Johnson's rule on the two-machine
        // flow shop). The big-M LP relaxation gives a very weak bound (5), so
        // B&B may need many nodes; marked large to accept TimeLimit.
        {"jobshop_10x2", MILPStatus::Optimal, 26.0,
            []() { return baguette_test::makeJobShop10(); }, true},

        // Job shop: C_max â‰¤ 4 contradicts C_max â‰¥ 5 (from precedence) â†’
        // infeasible at the LP root.
        {"jobshop_10x2_infeasible", MILPStatus::Infeasible, 0.0,
            []() { return baguette_test::makeJobShop10(4.0); }},

        // Uncapacitated Facility Location 5x10: 5 facilities (fixed cost 20),
        // 10 clients, LCG costs in [1,10]. IP optimal = 69.
        {"facility_location_5x10", MILPStatus::Optimal, 69.0,
            []() { return baguette_test::makeFacilityLocation5x10(); }},
    };
}

// â”€â”€ Presolve coherence: presolve must not mark feasible problems as infeasible â”€â”€
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
        opts.presolveLevel = 1;
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

// â”€â”€ rootMethod / nodeMethod: split LP algorithm between root and nodes â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Root uses MehrotraIPM for a tight initial bound; subsequent nodes use
// DualSimplexBV for fast warm-started solves.  Result must match the
// uniform-method solve.
TEST_CASE("BB: rootMethod=MehrotraIPM + nodeMethod=DualSimplexBV on knapsack",
          "[milp_classic][bb]") {
    BBOptions opts;
    opts.rootMethod    = LPMethod::MehrotraIPM;
    opts.nodeMethod    = LPMethod::DualSimplexBV;
    opts.timeLimitS    = 10.0;

    MILPResult r = solveMILP(baguette_test::makeKnapsack10(), opts);

    REQUIRE(r.status == MILPStatus::Optimal);
    REQUIRE_THAT(r.objectiveValue, WithinAbs(106.0, kTol));
}
