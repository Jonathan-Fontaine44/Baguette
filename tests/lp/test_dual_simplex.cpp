#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <limits>

#include "baguette/lp/LPSolver.hpp"
#include "baguette/model/Model.hpp"
#include "lp_problems.hpp"

using namespace baguette;
using Catch::Matchers::WithinAbs;

static constexpr double kTol = 1e-6;
static const double     kInf = std::numeric_limits<double>::infinity();

// ── Tests: dual simplex matches primal ───────────────────────────────────────

TEST_CASE("dual simplex - GEQ min: matches primal solve", "[dual_simplex]") {
    auto method = GENERATE(LPMethod::DualSimplex, LPMethod::DualSimplexBV);
    LPOptions primalOpts; primalOpts.method = LPMethod::PrimalSimplex; primalOpts.enablePresolve = false;
    LPOptions dualOpts;   dualOpts.method   = method;                   dualOpts.enablePresolve   = false;

    auto primal = solveLP(makeMinWithGEQ(), primalOpts);
    auto dual   = solveLP(makeMinWithGEQ(), dualOpts);

    REQUIRE(primal.status == LPStatus::Optimal);
    REQUIRE(dual.status   == LPStatus::Optimal);
    CHECK_THAT(dual.objectiveValue, WithinAbs(primal.objectiveValue, kTol));
    REQUIRE(dual.primalValues.size() == primal.primalValues.size());
    for (std::size_t j = 0; j < primal.primalValues.size(); ++j)
        CHECK_THAT(dual.primalValues[j], WithinAbs(primal.primalValues[j], kTol));
}

TEST_CASE("dual simplex - mixed LEQ+GEQ: matches primal solve", "[dual_simplex]") {
    auto method = GENERATE(LPMethod::DualSimplex, LPMethod::DualSimplexBV);
    LPOptions primalOpts; primalOpts.method = LPMethod::PrimalSimplex; primalOpts.enablePresolve = false;
    LPOptions dualOpts;   dualOpts.method   = method;                   dualOpts.enablePresolve   = false;
    auto m1 = makeSimpleMinLEQ();

    auto primal = solveLP(m1, primalOpts);
    auto dual   = solveLP(m1, dualOpts);

    REQUIRE(primal.status == LPStatus::Optimal);
    REQUIRE(dual.status   == LPStatus::Optimal);
    CHECK_THAT(dual.objectiveValue, WithinAbs(primal.objectiveValue, kTol));
}

TEST_CASE("dual simplex - infeasible detected correctly", "[dual_simplex]") {
    auto method = GENERATE(LPMethod::DualSimplex, LPMethod::DualSimplexBV);
    // x >= 3 AND x <= 2: infeasible
    LPOptions dualOpts; dualOpts.method = method; dualOpts.enablePresolve = false;
    auto res = solveLP(makeInfeasible(), dualOpts);
    REQUIRE(res.status == LPStatus::Infeasible);
    CHECK(res.primalValues.empty());
}

TEST_CASE("dual simplex - single variable bounded: min x in [3, inf]", "[dual_simplex]") {
    auto method = GENERATE(LPMethod::DualSimplex, LPMethod::DualSimplexBV);
    // min x,  x >= 3  (lb-shift: x' = x-3, min x' with x'>=0 → x'=0, x=3)
    Model m;
    auto x = m.addVar(3.0, kInf, "x");
    m.setObjective(1.0 * x, ObjSense::Minimize);

    LPOptions dualOpts; dualOpts.method = method; dualOpts.enablePresolve = false;
    auto res = solveLP(m, dualOpts);
    REQUIRE(res.status == LPStatus::Optimal);
    CHECK_THAT(res.objectiveValue, WithinAbs(3.0, kTol));
    CHECK_THAT(res.primalValues[0], WithinAbs(3.0, kTol));
}

TEST_CASE("dual simplex - LessEq only, non-negative costs", "[dual_simplex]") {
    auto method = GENERATE(LPMethod::DualSimplex, LPMethod::DualSimplexBV);
    // min 5x + 4y  s.t. x+y<=10, x<=7, y<=8,  x,y>=0
    // Minimum at (0,0): obj=0
    Model m;
    auto x = m.addVar(0.0, kInf, "x");
    auto y = m.addVar(0.0, kInf, "y");
    m.addLPConstraint(1.0 * x + 1.0 * y, Sense::LessEq, 10.0);
    m.addLPConstraint(1.0 * x,            Sense::LessEq,  7.0);
    m.addLPConstraint(1.0 * y,            Sense::LessEq,  8.0);
    m.setObjective(5.0 * x + 4.0 * y, ObjSense::Minimize);

    LPOptions dualOpts; dualOpts.method = method; dualOpts.enablePresolve = false;
    auto res = solveLP(m, dualOpts);
    REQUIRE(res.status == LPStatus::Optimal);
    CHECK_THAT(res.objectiveValue, WithinAbs(0.0, kTol));
}

TEST_CASE("dual simplex - objective value matches primal (5-var problem)", "[dual_simplex]") {
    auto method = GENERATE(LPMethod::DualSimplex, LPMethod::DualSimplexBV);
    // min 3x1 + 2x2 + x3 + 4x4 + 2x5
    // s.t. x1 + x2 + x3 >= 6
    //      x2 + x3 + x4 >= 4
    //      x3 + x4 + x5 >= 3
    //      xi >= 0
    Model m;
    auto x1 = m.addVar(0.0, kInf, "x1");
    auto x2 = m.addVar(0.0, kInf, "x2");
    auto x3 = m.addVar(0.0, kInf, "x3");
    auto x4 = m.addVar(0.0, kInf, "x4");
    auto x5 = m.addVar(0.0, kInf, "x5");
    m.addLPConstraint(1.0*x1 + 1.0*x2 + 1.0*x3,                    Sense::GreaterEq, 6.0);
    m.addLPConstraint(           1.0*x2 + 1.0*x3 + 1.0*x4,          Sense::GreaterEq, 4.0);
    m.addLPConstraint(                    1.0*x3 + 1.0*x4 + 1.0*x5, Sense::GreaterEq, 3.0);
    m.setObjective(3.0*x1 + 2.0*x2 + 1.0*x3 + 4.0*x4 + 2.0*x5, ObjSense::Minimize);

    LPOptions primalOpts; primalOpts.method = LPMethod::PrimalSimplex; primalOpts.enablePresolve = false;
    LPOptions dualOpts;   dualOpts.method   = method;                   dualOpts.enablePresolve   = false;

    auto primal = solveLP(m, primalOpts);
    auto dual   = solveLP(m, dualOpts);

    REQUIRE(primal.status == LPStatus::Optimal);
    REQUIRE(dual.status   == LPStatus::Optimal);
    CHECK_THAT(dual.objectiveValue, WithinAbs(primal.objectiveValue, kTol));
}

TEST_CASE("dual simplex - Upper-bound variable", "[dual_simplex]") {
    auto method = GENERATE(LPMethod::DualSimplex, LPMethod::DualSimplexBV);
    // min x,  0 <= x <= 5  → optimal x=0, obj=0
    Model m;
    auto x = m.addVar(0.0, 5.0, "x");
    m.setObjective(1.0 * x, ObjSense::Minimize);

    LPOptions primalOpts; primalOpts.method = LPMethod::PrimalSimplex; primalOpts.enablePresolve = false;
    LPOptions dualOpts;   dualOpts.method   = method;                   dualOpts.enablePresolve   = false;

    auto primal = solveLP(m, primalOpts);
    auto dual   = solveLP(m, dualOpts);

    REQUIRE(primal.status == LPStatus::Optimal);
    REQUIRE(dual.status   == LPStatus::Optimal);
    CHECK_THAT(dual.objectiveValue, WithinAbs(primal.objectiveValue, kTol));
    CHECK_THAT(dual.primalValues[0], WithinAbs(primal.primalValues[0], kTol));
}

// ── Tests: solveLPDetailed (dual path) ───────────────────────────────────────

TEST_CASE("dual simplex - solveDualDetailed primal matches solve", "[dual_simplex]") {
    auto method = GENERATE(LPMethod::DualSimplex, LPMethod::DualSimplexBV);
    LPOptions primalOpts; primalOpts.method = LPMethod::PrimalSimplex; primalOpts.enablePresolve = false;
    LPOptions dualOpts;   dualOpts.method   = method;                   dualOpts.enablePresolve   = false;

    auto detailed = solveLPDetailed(makeMinWithGEQ(), dualOpts);
    auto simple   = solveLP(makeMinWithGEQ(), primalOpts);

    REQUIRE(detailed.result.status == LPStatus::Optimal);
    CHECK_THAT(detailed.result.objectiveValue, WithinAbs(simple.objectiveValue, kTol));
    REQUIRE(detailed.result.primalValues.size() == simple.primalValues.size());
    for (std::size_t j = 0; j < simple.primalValues.size(); ++j)
        CHECK_THAT(detailed.result.primalValues[j],
                   WithinAbs(simple.primalValues[j], kTol));
}

TEST_CASE("dual simplex - solveDualDetailed dual values non-negative (GEQ min)", "[dual_simplex]") {
    auto method = GENERATE(LPMethod::DualSimplex, LPMethod::DualSimplexBV);
    LPOptions dualOpts; dualOpts.method = method; dualOpts.enablePresolve = false;
    auto det = solveLPDetailed(makeMinWithGEQ(), dualOpts);
    REQUIRE(det.result.status == LPStatus::Optimal);
    REQUIRE(det.dualValues.size() == 2);
    // GEQ constraints, min problem → shadow prices >= 0
    for (double y : det.dualValues)
        CHECK(y >= -kTol);
}

TEST_CASE("dual simplex - solveDualDetailed reduced costs at optimum", "[dual_simplex]") {
    auto method = GENERATE(LPMethod::DualSimplex, LPMethod::DualSimplexBV);
    LPOptions dualOpts; dualOpts.method = method; dualOpts.enablePresolve = false;
    auto det = solveLPDetailed(makeMinWithGEQ(), dualOpts);
    REQUIRE(det.result.status == LPStatus::Optimal);
    // Basic (non-zero) variables must have zero reduced cost
    for (std::size_t j = 0; j < det.result.primalValues.size(); ++j)
        if (det.result.primalValues[j] > kTol)
            CHECK_THAT(det.reducedCosts[j], WithinAbs(0.0, kTol));
}

TEST_CASE("dual simplex - fallback for Equal constraint gives correct result", "[dual_simplex]") {
    auto method = GENERATE(LPMethod::DualSimplex, LPMethod::DualSimplexBV);
    // Model with Equal constraint: dual simplex falls back to primal.
    // min x + y  s.t. x + y = 5,  x,y >= 0
    Model m;
    auto x = m.addVar(0.0, kInf, "x");
    auto y = m.addVar(0.0, kInf, "y");
    m.addLPConstraint(1.0 * x + 1.0 * y, Sense::Equal, 5.0);
    m.setObjective(1.0 * x + 1.0 * y, ObjSense::Minimize);

    LPOptions dualOpts; dualOpts.method = method; dualOpts.enablePresolve = false;
    auto res = solveLP(m, dualOpts);
    REQUIRE(res.status == LPStatus::Optimal);
    CHECK_THAT(res.objectiveValue, WithinAbs(5.0, kTol));
}

TEST_CASE("dual simplex - maxIter limit respected", "[dual_simplex]") {
    auto method = GENERATE(LPMethod::DualSimplex, LPMethod::DualSimplexBV);
    LPOptions opts;
    opts.method         = method;
    opts.maxIter        = 1;
    opts.enablePresolve = false;
    auto res = solveLP(makeMinWithGEQ(), opts);
    CHECK((res.status == LPStatus::MaxIter || res.status == LPStatus::Optimal));
}

TEST_CASE("dual simplex - solveDualDetailed result sub-object accessible", "[dual_simplex]") {
    auto method = GENERATE(LPMethod::DualSimplex, LPMethod::DualSimplexBV);
    LPOptions dualOpts; dualOpts.method = method; dualOpts.enablePresolve = false;
    LPDetailedResult det = solveLPDetailed(makeMinWithGEQ(), dualOpts);
    const LPResult& res  = det.result;
    CHECK(res.status == LPStatus::Optimal);
    CHECK_THAT(res.objectiveValue, WithinAbs(det.result.objectiveValue, kTol));
}

TEST_CASE("dual simplex - strong duality: solveDual obj == solve obj", "[dual_simplex]") {
    auto method = GENERATE(LPMethod::DualSimplex, LPMethod::DualSimplexBV);
    // Verify strong duality through both solvers on a richer instance.
    // min 6x + 4y + 3z
    // s.t. 2x + y + z >= 4
    //      x + 2y + z >= 3
    //      x + y + 2z >= 2
    //      x,y,z >= 0
    Model m;
    auto x = m.addVar(0.0, kInf, "x");
    auto y = m.addVar(0.0, kInf, "y");
    auto z = m.addVar(0.0, kInf, "z");
    m.addLPConstraint(2.0*x + 1.0*y + 1.0*z, Sense::GreaterEq, 4.0);
    m.addLPConstraint(1.0*x + 2.0*y + 1.0*z, Sense::GreaterEq, 3.0);
    m.addLPConstraint(1.0*x + 1.0*y + 2.0*z, Sense::GreaterEq, 2.0);
    m.setObjective(6.0*x + 4.0*y + 3.0*z, ObjSense::Minimize);

    LPOptions primalOpts; primalOpts.method = LPMethod::PrimalSimplex; primalOpts.enablePresolve = false;
    LPOptions dualOpts;   dualOpts.method   = method;                   dualOpts.enablePresolve   = false;

    auto primal = solveLP(m, primalOpts);
    auto dual   = solveLP(m, dualOpts);

    REQUIRE(primal.status == LPStatus::Optimal);
    REQUIRE(dual.status   == LPStatus::Optimal);
    CHECK_THAT(dual.objectiveValue, WithinAbs(primal.objectiveValue, kTol));
}

TEST_CASE("dual simplex - timeLimitS = 0 returns TimeLimit", "[dual_simplex][timelimit]") {
    auto method = GENERATE(LPMethod::DualSimplex, LPMethod::DualSimplexBV);
    LPOptions opts;
    opts.method         = method;
    opts.timeLimitS     = 0.0;
    opts.enablePresolve = false;
    auto res = solveLP(makeMinWithGEQ(), opts);
    REQUIRE(res.status == LPStatus::TimeLimit);
}

TEST_CASE("dual simplex - LP relaxation infeasible from B&B test", "[dual_simplex]") {
    auto method = GENERATE(LPMethod::DualSimplex, LPMethod::DualSimplexBV);
    Model m;
    Variable x = m.addVar(0.0, 10.0, VarType::Integer, "x");
    Variable y = m.addVar(0.0, 10.0, VarType::Integer, "y");
    m.addLPConstraint(1.0 * x + 1.0 * y, Sense::LessEq, -1.0);
    m.setObjective(1.0 * x + 1.0 * y, ObjSense::Minimize);

    LPOptions dualOpts; dualOpts.method = method; dualOpts.enablePresolve = false;
    LPResult  lpResult = solveLP(m, dualOpts);

    REQUIRE(lpResult.status == LPStatus::Infeasible);
}
