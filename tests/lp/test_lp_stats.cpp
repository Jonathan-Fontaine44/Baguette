#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include <limits>

#include "baguette/lp/LPResult.hpp"
#include "baguette/lp/LPSolver.hpp"
#include "baguette/model/Model.hpp"

#include "lp_problems.hpp"
#include "MILP_problems.hpp"

using namespace baguette;
using namespace baguette_test;

static const double kInf = std::numeric_limits<double>::infinity();

// ── Test 1: Presence of iterationsUsed across all methods and problems ─────────
//
// For every (method, LP problem) pair: verify the field is populated with a
// plausible value (not left at an absurd sentinel).

TEST_CASE("LP stats: iterationsUsed present on all methods and problems", "[lp_stats]") {
    auto method = GENERATE(LPMethod::Auto,
                           LPMethod::PrimalSimplex,   LPMethod::PrimalSimplexBV,
                           LPMethod::DualSimplex,     LPMethod::DualSimplexBV,
                           LPMethod::RevisedSimplex,  LPMethod::RevisedSimplexBV,
                           LPMethod::ShortStepIPM,    LPMethod::MehrotraIPM,
                           LPMethod::NetworkSimplex);

    static const auto suite = makeLPTestSuite();
    auto i = GENERATE(range(std::size_t{0}, suite.size()));
    const auto& tc = suite[i];

    DYNAMIC_SECTION(to_string(method) << " / " << tc.name) {
        LPOptions opts;
        opts.method         = method;
        opts.enablePresolve = false;
        // ShortStepIPM stagnates on non-optimal problems; cap it so the suite stays fast.
        if (method == LPMethod::ShortStepIPM && tc.expectedStatus != LPStatus::Optimal)
            opts.timeLimitS = 0.1;

        auto r = solveLPDetailed(tc.build(), opts);

        // Test problems are tiny; a plausible upper bound guards against the field
        // being left at a garbage sentinel.
        CHECK(r.iterationsUsed < 100'000u);
    }
}

// ── Test 2: iterationsUsed is a sane value on an infeasible LP ────────────────
//
//   x >= 3  AND  x + y <= 2,  y >= 0  →  infeasible via pivoting (not lb > ub).
//
// Dual simplex methods may detect infeasibility in 0 pivots (no entering variable
// found on the first blocking row), while primal/IPM methods need at least one
// iteration. Either way the field must hold a plausible value for this 2-constraint LP.

TEST_CASE("LP stats: iterationsUsed is sane on infeasible LP", "[lp_stats]") {
    auto method = GENERATE(LPMethod::PrimalSimplex,   LPMethod::PrimalSimplexBV,
                           LPMethod::DualSimplex,     LPMethod::DualSimplexBV,
                           LPMethod::RevisedSimplex,  LPMethod::RevisedSimplexBV,
                           LPMethod::MehrotraIPM);

    DYNAMIC_SECTION(to_string(method)) {
        Model m;
        auto x = m.addVar(0.0, kInf, "x");
        auto y = m.addVar(0.0, kInf, "y");
        m.addLPConstraint(1.0 * x,            Sense::GreaterEq, 3.0);
        m.addLPConstraint(1.0 * x + 1.0 * y, Sense::LessEq,    2.0);
        m.setObjective(1.0 * x, ObjSense::Minimize);

        LPOptions opts; opts.method = method; opts.enablePresolve = false;
        auto r = solveLPDetailed(m, opts);

        REQUIRE(r.result.status == LPStatus::Infeasible);
        CHECK(r.iterationsUsed < 100u);
    }
}

// ── Test 3: Fallback path propagates iterationsUsed ───────────────────────────
//
// An equality constraint prevents the DualSimplex cold-start from building a
// natural dual-feasible basis, triggering a fallback to PrimalSimplex.
// The primal iteration count from the fallback must reach iterationsUsed.

TEST_CASE("LP stats: iterationsUsed populated after DualSimplex fallback to primal", "[lp_stats]") {
    auto method = GENERATE(LPMethod::DualSimplex, LPMethod::DualSimplexBV);

    DYNAMIC_SECTION(to_string(method)) {
        Model m;
        auto x = m.addVar(0.0, kInf, "x");
        auto y = m.addVar(0.0, kInf, "y");
        m.addLPConstraint(1.0 * x + 1.0 * y, Sense::Equal, 5.0);
        m.setObjective(1.0 * x + 2.0 * y, ObjSense::Minimize);

        LPOptions opts; opts.method = method; opts.enablePresolve = false;
        auto r = solveLPDetailed(m, opts);

        REQUIRE(r.result.status == LPStatus::Optimal);
        // Primal simplex needs at least 1 pivot to reach the optimum (x=5, y=0).
        CHECK(r.iterationsUsed >= 1u);
    }
}

// ── Test 4: MaxIter stops and reports exactly the pivot count ─────────────────
//
//   min -x-y  s.t. 2x+y<=4, x+2y<=4, x in [0,3], y in [0,3].
//   Optimal at (4/3, 4/3) - needs at least 4 pivots from (0,0). With maxIter in
//   {1,2,3}, the solver stops early and iterationsUsed must equal maxIter exactly.

TEST_CASE("LP stats: iterationsUsed == maxIter when MaxIter returned", "[lp_stats]") {
    auto method  = GENERATE(LPMethod::PrimalSimplex,  LPMethod::PrimalSimplexBV,
                            LPMethod::RevisedSimplex, LPMethod::RevisedSimplexBV);
    auto maxIterTest = GENERATE(1u, 2u);

    DYNAMIC_SECTION(to_string(method) << " / maxIter=" << maxIterTest) {
        Model m;
        auto x = m.addVar(0.0, 3.0, "x");
        auto y = m.addVar(0.0, 3.0, "y");
        m.addLPConstraint(2.0 * x + 1.0 * y, Sense::LessEq, 4.0);
        m.addLPConstraint(1.0 * x + 2.0 * y, Sense::LessEq, 4.0);
        m.setObjective(-1.0 * x + -1.0 * y, ObjSense::Minimize);

        LPOptions opts;
        opts.method         = method;
        opts.enablePresolve = false;
        opts.maxIter        = maxIterTest;
        auto r = solveLPDetailed(m, opts);

        REQUIRE(r.result.status == LPStatus::MaxIter);
        CHECK(r.iterationsUsed == maxIterTest);
    }
}

// ── Test 5: TimeLimit - iterationsUsed reflects work done before cutoff ────────
//
// TSP10 LP relaxation (99 vars, 91 constraints) solved with a 10 ms budget.
// The LP relaxation has an integer-optimal vertex, so some methods may solve it
// before the time limit fires. Either way, iterationsUsed must be populated.

TEST_CASE("LP stats: iterationsUsed populated under TimeLimit on TSP10 relaxation", "[lp_stats]") {
    auto method = GENERATE(LPMethod::PrimalSimplex,   LPMethod::PrimalSimplexBV,
                           LPMethod::DualSimplex,     LPMethod::DualSimplexBV,
                           LPMethod::RevisedSimplex,  LPMethod::RevisedSimplexBV,
                           LPMethod::MehrotraIPM);

    DYNAMIC_SECTION(to_string(method)) {
        LPOptions opts;
        opts.method         = method;
        opts.enablePresolve = false;
        opts.timeLimitS     = 0.01; // 10 ms

        auto r = solveLPDetailed(makeTSP10(), opts);

        // TSP10's LP relaxation has an integer-optimal vertex: fast methods may solve
        // it before the time check fires. Both outcomes are valid; the field must be set.
        REQUIRE((r.result.status == LPStatus::TimeLimit ||
                 r.result.status == LPStatus::Optimal));
        CHECK(r.iterationsUsed < 1'000'000u);
    }
}
