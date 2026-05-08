#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "baguette/lp/LPSolver.hpp"
#include "baguette/model/Model.hpp"
#include "lp_problems.hpp"

using namespace baguette;
using Catch::Matchers::WithinAbs;

static constexpr double kTol = 1e-6;

// ── Pure network problem suite ────────────────────────────────────────────────

inline std::vector<LPTestCase> makeNetworkSuite() {
    using namespace baguette;
    return {
        // 3-node: supply→transit→demand
        // Arcs: (0→1) c=2, (0→2) c=7, (1→2) c=3. Supply 4 at 0, demand 4 at 2.
        // Optimal: all flow via 0→1→2, cost = 4*(2+3) = 20.
        {"3-node transport", LPStatus::Optimal, 20.0, []() {
            Model m;
            auto x01 = m.addVar(0.0, 10.0, "x01");
            auto x02 = m.addVar(0.0, 10.0, "x02");
            auto x12 = m.addVar(0.0, 10.0, "x12");
            m.addLPConstraint( 1.0*x01 + 1.0*x02,              Sense::Equal,  4.0);
            m.addLPConstraint(-1.0*x01              + 1.0*x12, Sense::Equal,  0.0);
            m.addLPConstraint(             -1.0*x02 - 1.0*x12, Sense::Equal, -4.0);
            m.setObjective(2.0*x01 + 7.0*x02 + 3.0*x12, ObjSense::Minimize);
            return m;
        }},

        // 4-node diamond
        // Arcs: (0→1) c=1, (0→2) c=2, (1→3) c=3, (2→3) c=1.
        // Supply 6 at 0, demand 6 at 3. Optimal: all via 0→2→3, cost = 6*(2+1) = 18.
        {"4-node diamond", LPStatus::Optimal, 18.0, []() {
            Model m;
            auto x01 = m.addVar(0.0, 10.0);
            auto x02 = m.addVar(0.0, 10.0);
            auto x13 = m.addVar(0.0, 10.0);
            auto x23 = m.addVar(0.0, 10.0);
            m.addLPConstraint( 1.0*x01 + 1.0*x02,              Sense::Equal,  6.0);
            m.addLPConstraint(-1.0*x01              + 1.0*x13, Sense::Equal,  0.0);
            m.addLPConstraint(             -1.0*x02 + 1.0*x23, Sense::Equal,  0.0);
            m.addLPConstraint(                       -1.0*x13 - 1.0*x23, Sense::Equal, -6.0);
            m.setObjective(1.0*x01 + 2.0*x02 + 3.0*x13 + 1.0*x23, ObjSense::Minimize);
            return m;
        }},

        // Capacity forces flow split
        // x01 capped at 2: cost(x01+x12) = 2+3=5, cost(x02) = 7.
        // Without cap: x01=4, x02=0, cost=20. With cap: obj = 28 - 2*x01 (decreasing
        // in x01), so maximize x01=2 → x01=2, x02=2, x12=2, cost=4+14+6=24.
        {"3-node capacity split", LPStatus::Optimal, 24.0, []() {
            Model m;
            auto x01 = m.addVar(0.0,  2.0);  // capped at 2
            auto x02 = m.addVar(0.0, 10.0);
            auto x12 = m.addVar(0.0, 10.0);
            m.addLPConstraint( 1.0*x01 + 1.0*x02,              Sense::Equal,  4.0);
            m.addLPConstraint(-1.0*x01              + 1.0*x12, Sense::Equal,  0.0);
            m.addLPConstraint(             -1.0*x02 - 1.0*x12, Sense::Equal, -4.0);
            m.setObjective(2.0*x01 + 7.0*x02 + 3.0*x12, ObjSense::Minimize);
            return m;
        }},

        // Lower-bound shift: x01 ≥ 1 (committed flow, not binding at optimum)
        {"3-node lb-shift", LPStatus::Optimal, 20.0, []() {
            Model m;
            auto x01 = m.addVar(1.0, 10.0);  // lb = 1
            auto x02 = m.addVar(0.0, 10.0);
            auto x12 = m.addVar(0.0, 10.0);
            m.addLPConstraint( 1.0*x01 + 1.0*x02,              Sense::Equal,  4.0);
            m.addLPConstraint(-1.0*x01              + 1.0*x12, Sense::Equal,  0.0);
            m.addLPConstraint(             -1.0*x02 - 1.0*x12, Sense::Equal, -4.0);
            m.setObjective(2.0*x01 + 7.0*x02 + 3.0*x12, ObjSense::Minimize);
            return m;
        }},

        // Unbalanced flow → infeasible (supply 5, demand 3 only)
        {"infeasible unbalanced", LPStatus::Infeasible, 0.0, []() {
            Model m;
            auto x = m.addVar(0.0, 10.0);
            m.addLPConstraint( 1.0*x, Sense::Equal,  5.0);
            m.addLPConstraint(-1.0*x, Sense::Equal, -3.0);
            m.setObjective(1.0*x, ObjSense::Minimize);
            return m;
        }},

        // Maximize: 3-node network, best route is direct (0→2), profit = 7/unit → 28.
        {"3-node maximize", LPStatus::Optimal, 28.0, []() {
            Model m;
            auto x01 = m.addVar(0.0, 10.0);
            auto x02 = m.addVar(0.0, 10.0);
            auto x12 = m.addVar(0.0, 10.0);
            m.addLPConstraint( 1.0*x01 + 1.0*x02,              Sense::Equal,  4.0);
            m.addLPConstraint(-1.0*x01              + 1.0*x12, Sense::Equal,  0.0);
            m.addLPConstraint(             -1.0*x02 - 1.0*x12, Sense::Equal, -4.0);
            m.setObjective(2.0*x01 + 7.0*x02 + 3.0*x12, ObjSense::Maximize);
            return m;
        }},
    };
}

// ── Cross-method: NetworkSimplex vs PrimalSimplex vs DualSimplexBV ─────────────

TEST_CASE("NetworkSimplex: pure network problems vs reference methods",
          "[network_simplex]") {
    auto method = GENERATE(LPMethod::NetworkSimplex,
                           LPMethod::PrimalSimplex,
                           LPMethod::DualSimplexBV);

    static const auto suite = makeNetworkSuite();
    auto i = GENERATE(range(std::size_t{0}, suite.size()));
    const auto& tc = suite[i];

    DYNAMIC_SECTION("Method=" << to_string(method) << ", case=" << tc.name) {
        LPOptions opts;
        opts.method = method;
        LPResult r = solveLP(tc.build(), opts);

        REQUIRE(r.status == tc.expectedStatus);
        if (tc.expectedStatus == LPStatus::Optimal)
            REQUIRE_THAT(r.objectiveValue, WithinAbs(tc.expectedObj, kTol));
    }
}

// ── Fallback: non-network LP (LessEq constraint) ──────────────────────────────

TEST_CASE("NetworkSimplex: fallback on non-network LP", "[network_simplex]") {
    Model mdl;
    auto x = mdl.addVar(0.0, 10.0);
    auto y = mdl.addVar(0.0, 10.0);
    mdl.addLPConstraint(1.0 * x + 1.0 * y, Sense::LessEq, 8.0);
    mdl.setObjective(1.0 * x + 2.0 * y, ObjSense::Minimize);

    LPOptions opts;
    opts.method = LPMethod::NetworkSimplex;
    LPResult r = solveLP(mdl, opts);

    REQUIRE(r.status == LPStatus::Optimal);
    REQUIRE_THAT(r.objectiveValue, WithinAbs(0.0, kTol));
}

// ── Strong duality: Σ dual[i]*rhs[i] = objective ──────────────────────────────

TEST_CASE("NetworkSimplex: strong duality 3-node", "[network_simplex]") {
    Model mdl;
    auto x01 = mdl.addVar(0.0, 10.0);
    auto x02 = mdl.addVar(0.0, 10.0);
    auto x12 = mdl.addVar(0.0, 10.0);
    mdl.addLPConstraint( 1.0*x01 + 1.0*x02,              Sense::Equal,  4.0);
    mdl.addLPConstraint(-1.0*x01              + 1.0*x12, Sense::Equal,  0.0);
    mdl.addLPConstraint(             -1.0*x02 - 1.0*x12, Sense::Equal, -4.0);
    mdl.setObjective(2.0*x01 + 7.0*x02 + 3.0*x12, ObjSense::Minimize);

    LPOptions opts;
    opts.method = LPMethod::NetworkSimplex;
    LPDetailedResult det = solveLPDetailed(mdl, opts);

    REQUIRE(det.result.status == LPStatus::Optimal);
    REQUIRE(det.dualValues.size() == 3);

    // Strong duality: Σ y_i * b_i = obj
    double dualObj = det.dualValues[0] * 4.0
                   + det.dualValues[1] * 0.0
                   + det.dualValues[2] * (-4.0);
    REQUIRE_THAT(dualObj, WithinAbs(det.result.objectiveValue, kTol));
}
