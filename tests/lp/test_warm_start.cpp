#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <limits>

#include "baguette/lp/LPResult.hpp"
#include "baguette/lp/LPSolver.hpp"
#include "baguette/model/Model.hpp"

using namespace baguette;
using Catch::Matchers::WithinAbs;

static constexpr double kTol = 1e-6;
static const double kInf = std::numeric_limits<double>::infinity();

// ── Reference LP ──────────────────────────────────────────────────────────────
//
//   min  -x1 - x2
//   s.t.  2*x1 +   x2  <=  4
//           x1 + 2*x2  <=  4
//         x1 in [0, 3],  x2 in [0, 3]
//
// Standard form: nRows = 4  (2 model rows + 2 upper-bound rows for x1, x2).
// LP optimum:  x1 = x2 = 4/3,  obj = -8/3.

static Model makeFractionalLP() {
    Model m;
    auto x1 = m.addVar(0.0, 3.0, "x1");
    auto x2 = m.addVar(0.0, 3.0, "x2");

    LinearExpr obj;
    obj.addTerm(x1, -1.0);
    obj.addTerm(x2, -1.0);
    m.setObjective(obj);

    LinearExpr c1;
    c1.addTerm(x1, 2.0);
    c1.addTerm(x2, 1.0);
    m.addLPConstraint(c1, Sense::LessEq, 4.0);

    LinearExpr c2;
    c2.addTerm(x1, 1.0);
    c2.addTerm(x2, 2.0);
    m.addLPConstraint(c2, Sense::LessEq, 4.0);

    return m;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

TEST_CASE("Warm-start: identity returns same result", "[warm_start]") {
    auto method = GENERATE(LPMethod::DualSimplex, LPMethod::DualSimplexBV);
    LPOptions coldOpts; coldOpts.method = method; coldOpts.enablePresolve = false;
    Model m      = makeFractionalLP();
    auto  parent = solveLPDetailed(m, coldOpts);
    REQUIRE(parent.result.status == LPStatus::Optimal);

    LPOptions opts; opts.method = method; opts.warmBasis = parent.basis; opts.enablePresolve = false;
    auto warm = solveLPDetailed(m, opts);
    REQUIRE(warm.result.status == LPStatus::Optimal);
    REQUIRE_THAT(warm.result.objectiveValue,
                 WithinAbs(parent.result.objectiveValue, kTol));
    REQUIRE_THAT(warm.result.primalValues[0],
                 WithinAbs(parent.result.primalValues[0], kTol));
    REQUIRE_THAT(warm.result.primalValues[1],
                 WithinAbs(parent.result.primalValues[1], kTol));
}

TEST_CASE("Warm-start: left branch x1 <= 1", "[warm_start]") {
    // Branch: x1 <= floor(4/3) = 1.
    // Child optimum: x1=1, x2=1.5, obj=-2.5.
    auto method = GENERATE(LPMethod::DualSimplex, LPMethod::DualSimplexBV);
    LPOptions coldOpts; coldOpts.method = method; coldOpts.enablePresolve = false;
    Model    parent_m = makeFractionalLP();
    Variable x1{0};
    auto     parent   = solveLPDetailed(parent_m, coldOpts);
    REQUIRE(parent.result.status == LPStatus::Optimal);

    Model child_m = parent_m.withVarBounds(x1, 0.0, 1.0);

    LPOptions opts; opts.method = method; opts.warmBasis = parent.basis; opts.enablePresolve = false;
    auto warm = solveLPDetailed(child_m, opts);
    auto cold = solveLPDetailed(child_m, LPOptions{.enablePresolve = false});

    REQUIRE(warm.result.status == LPStatus::Optimal);
    REQUIRE(warm.result.primalValues[0] <= 1.0 + kTol);
    REQUIRE_THAT(warm.result.objectiveValue,
                 WithinAbs(cold.result.objectiveValue, kTol));
    REQUIRE_THAT(warm.result.objectiveValue, WithinAbs(-2.5, kTol));
}

TEST_CASE("Warm-start: right branch x1 >= 2", "[warm_start]") {
    // Branch: x1 >= ceil(4/3) = 2.
    // Child optimum: x1=2, x2=0, obj=-2.0.
    auto method = GENERATE(LPMethod::DualSimplex, LPMethod::DualSimplexBV);
    LPOptions coldOpts; coldOpts.method = method; coldOpts.enablePresolve = false;
    Model    parent_m = makeFractionalLP();
    Variable x1{0};
    auto     parent   = solveLPDetailed(parent_m, coldOpts);
    REQUIRE(parent.result.status == LPStatus::Optimal);

    Model child_m = parent_m.withVarBounds(x1, 2.0, 3.0);

    LPOptions opts; opts.method = method; opts.warmBasis = parent.basis; opts.enablePresolve = false;
    auto warm = solveLPDetailed(child_m, opts);
    auto cold = solveLPDetailed(child_m, LPOptions{.enablePresolve = false});

    REQUIRE(warm.result.status == LPStatus::Optimal);
    REQUIRE(warm.result.primalValues[0] >= 2.0 - kTol);
    REQUIRE_THAT(warm.result.objectiveValue,
                 WithinAbs(cold.result.objectiveValue, kTol));
    REQUIRE_THAT(warm.result.objectiveValue, WithinAbs(-2.0, kTol));
}

TEST_CASE("Warm-start: infeasible branch (empty domain lb > ub)", "[warm_start]") {
    auto method = GENERATE(LPMethod::DualSimplex, LPMethod::DualSimplexBV);
    LPOptions coldOpts; coldOpts.method = method; coldOpts.enablePresolve = false;
    Model    parent_m = makeFractionalLP();
    Variable x1{0};
    auto     parent   = solveLPDetailed(parent_m, coldOpts);
    REQUIRE(parent.result.status == LPStatus::Optimal);

    // Explicitly contradictory bounds: lb > ub.
    Model child_m = parent_m.withVarBounds(x1, 2.0, 1.0);

    LPOptions opts; opts.method = method; opts.warmBasis = parent.basis; opts.enablePresolve = false;
    auto warm = solveLPDetailed(child_m, opts);
    REQUIRE(warm.result.status == LPStatus::Infeasible);
}

TEST_CASE("Warm-start: infeasible by constraints after tight bounds", "[warm_start]") {
    // x1 >= 3 and x2 >= 3 violates  2*x1 + x2 <= 4  (gives 9 <= 4).
    auto method = GENERATE(LPMethod::DualSimplex, LPMethod::DualSimplexBV);
    LPOptions coldOpts; coldOpts.method = method; coldOpts.enablePresolve = false;
    Model    parent_m = makeFractionalLP();
    Variable x1{0};
    Variable x2{1};
    auto     parent   = solveLPDetailed(parent_m, coldOpts);
    REQUIRE(parent.result.status == LPStatus::Optimal);

    Model child_m = parent_m.withVarBounds(x1, 3.0, 3.0)
                             .withVarBounds(x2, 3.0, 3.0);

    LPOptions opts; opts.method = method; opts.warmBasis = parent.basis; opts.enablePresolve = false;
    auto warm = solveLPDetailed(child_m, opts);
    REQUIRE(warm.result.status == LPStatus::Infeasible);
}

TEST_CASE("Warm-start: incompatible basis falls back to cold solve", "[warm_start]") {
    // An incomplete BasisRecord (no atUBCache for DualSimplexBV, wrong size for
    // DualSimplex) is rejected and falls back transparently to a cold solve.
    auto method = GENERATE(LPMethod::DualSimplex, LPMethod::DualSimplexBV);
    Model m = makeFractionalLP();

    BasisRecord bad_basis;
    bad_basis.basicCols = {0, 1, 2}; // 3 entries — wrong for both methods
    bad_basis.colKind   = {ColumnKind::Original,
                           ColumnKind::Original,
                           ColumnKind::Original};

    LPOptions opts; opts.method = method; opts.warmBasis = bad_basis; opts.enablePresolve = false;
    auto result = solveLPDetailed(m, opts);
    REQUIRE(result.result.status == LPStatus::Optimal);
    REQUIRE_THAT(result.result.objectiveValue, WithinAbs(-8.0 / 3.0, kTol));
}

TEST_CASE("Warm-start: backtrack left then right via setVarBounds", "[warm_start]") {
    // Simulate a B&B node with in-place mutation and restore (no copies).
    //
    //   root (x1* = x2* = 4/3,  obj = -8/3)
    //   ├── left  x1 in [0, 1]  →  x1=1,  x2=1.5,  obj=-2.5
    //   └── right x1 in [2, 3]  →  x1=2,  x2=0,    obj=-2.0
    //
    // After each branch the bounds are restored with setVarBounds() and the
    // root solution is re-verified to confirm the backtrack was complete.
    auto method = GENERATE(LPMethod::DualSimplex, LPMethod::DualSimplexBV);
    LPOptions baseOpts; baseOpts.method = method; baseOpts.enablePresolve = false;
    Model    m  = makeFractionalLP();
    Variable x1{0};

    auto root = solveLPDetailed(m, baseOpts);
    REQUIRE(root.result.status == LPStatus::Optimal);

    const double savedLb = m.getHot().lb[x1.id]; // 0.0
    const double savedUb = m.getHot().ub[x1.id]; // 3.0

    // ── Left branch ──────────────────────────────────────────────────────────
    m.setVarBounds(x1, 0.0, 1.0);
    LPOptions leftOpts; leftOpts.method = method; leftOpts.warmBasis = root.basis; leftOpts.enablePresolve = false;
    auto left = solveLPDetailed(m, leftOpts);
    REQUIRE(left.result.status == LPStatus::Optimal);
    REQUIRE_THAT(left.result.objectiveValue, WithinAbs(-2.5, kTol));
    REQUIRE(left.result.primalValues[0] <= 1.0 + kTol);

    // Backtrack
    m.setVarBounds(x1, savedLb, savedUb);
    REQUIRE(m.getHot().lb[x1.id] == savedLb);
    REQUIRE(m.getHot().ub[x1.id] == savedUb);

    // ── Right branch (same model instance, same root basis) ──────────────────
    m.setVarBounds(x1, 2.0, 3.0);
    LPOptions rightOpts; rightOpts.method = method; rightOpts.warmBasis = root.basis; rightOpts.enablePresolve = false;
    auto right = solveLPDetailed(m, rightOpts);
    REQUIRE(right.result.status == LPStatus::Optimal);
    REQUIRE_THAT(right.result.objectiveValue, WithinAbs(-2.0, kTol));
    REQUIRE(right.result.primalValues[0] >= 2.0 - kTol);

    // Backtrack
    m.setVarBounds(x1, savedLb, savedUb);

    // Root model is fully restored: re-solving gives the original optimum.
    auto restored = solveLPDetailed(m, baseOpts);
    REQUIRE(restored.result.status == LPStatus::Optimal);
    REQUIRE_THAT(restored.result.objectiveValue, WithinAbs(-8.0 / 3.0, kTol));
}

TEST_CASE("Warm-start: two-level B&B tree gives consistent results", "[warm_start]") {
    // Root → left (x1<=1) → left again (x2<=1).
    // Final optimum: x1=1, x2=1, obj=-2.
    auto method = GENERATE(LPMethod::DualSimplex, LPMethod::DualSimplexBV);
    LPOptions baseOpts; baseOpts.method = method; baseOpts.enablePresolve = false;
    Model    root_m = makeFractionalLP();
    Variable x1{0};
    Variable x2{1};

    auto root = solveLPDetailed(root_m, baseOpts);
    REQUIRE(root.result.status == LPStatus::Optimal);

    Model child_m = root_m.withVarBounds(x1, 0.0, 1.0);
    LPOptions childOpts; childOpts.method = method; childOpts.warmBasis = root.basis; childOpts.enablePresolve = false;
    auto  child   = solveLPDetailed(child_m, childOpts);
    REQUIRE(child.result.status == LPStatus::Optimal);

    Model grand_m = child_m.withVarBounds(x2, 0.0, 1.0);
    LPOptions grandOpts; grandOpts.method = method; grandOpts.warmBasis = child.basis; grandOpts.enablePresolve = false;
    auto  grand   = solveLPDetailed(grand_m, grandOpts);
    REQUIRE(grand.result.status == LPStatus::Optimal);
    REQUIRE_THAT(grand.result.objectiveValue, WithinAbs(-2.0, kTol));
    REQUIRE_THAT(grand.result.primalValues[0], WithinAbs(1.0, kTol));
    REQUIRE_THAT(grand.result.primalValues[1], WithinAbs(1.0, kTol));
}

// ── sfCache ───────────────────────────────────────────────────────────────────

TEST_CASE("sfCache: populated on Optimal, null on non-Optimal", "[warm_start]") {
    Model    m  = makeFractionalLP();
    Variable x1{0};

    // Cold DualSimplex solve → Optimal → sfCache set (DualSimplexBV uses atUBCache instead)
    LPOptions coldOpts; coldOpts.method = LPMethod::DualSimplex; coldOpts.enablePresolve = false;
    auto res = solveLPDetailed(m, coldOpts);
    REQUIRE(res.result.status == LPStatus::Optimal);
    REQUIRE(res.basis.sfCache != nullptr);

    // Infeasible node: lb > ub → sfCache not set
    Model inf_m = m.withVarBounds(x1, 2.0, 1.0);
    LPOptions infOpts; infOpts.method = LPMethod::DualSimplex; infOpts.warmBasis = res.basis; infOpts.enablePresolve = false;
    auto  inf   = solveLPDetailed(inf_m, infOpts);
    REQUIRE(inf.result.status == LPStatus::Infeasible);
    REQUIRE(inf.basis.sfCache == nullptr);
}

TEST_CASE("sfCache: three-level chain produces correct results and propagates cache", "[warm_start]") {
    // Simulate a B&B path: root → child (x1<=1) → grandchild (x2<=1).
    // Each level uses the sfCache from the previous level's BasisRecord so
    // that toStandardFormBoundsOnly is exercised at every step.
    // Uses DualSimplex explicitly: sfCache is a DualSimplex warm-start mechanism
    // (DualSimplexBV uses atUBCache instead and does not populate sfCache).
    Model    root_m = makeFractionalLP();
    Variable x1{0}, x2{1};

    // Level 0: cold DualSimplex solve populates sfCache
    LPOptions coldOpts; coldOpts.method = LPMethod::DualSimplex; coldOpts.enablePresolve = false;
    auto root = solveLPDetailed(root_m, coldOpts);
    REQUIRE(root.result.status == LPStatus::Optimal);
    REQUIRE(root.basis.sfCache != nullptr);

    // Level 1: warm start using root.basis (which carries sfCache)
    Model child_m = root_m.withVarBounds(x1, 0.0, 1.0);
    LPOptions childOpts; childOpts.method = LPMethod::DualSimplex; childOpts.warmBasis = root.basis; childOpts.enablePresolve = false;
    auto  child   = solveLPDetailed(child_m, childOpts);
    REQUIRE(child.result.status == LPStatus::Optimal);
    REQUIRE_THAT(child.result.objectiveValue, WithinAbs(-2.5, kTol));
    REQUIRE(child.basis.sfCache != nullptr);

    // Level 2: warm start using child.basis (which also carries sfCache)
    Model grand_m = child_m.withVarBounds(x2, 0.0, 1.0);
    LPOptions grandOpts; grandOpts.method = LPMethod::DualSimplex; grandOpts.warmBasis = child.basis; grandOpts.enablePresolve = false;
    auto  grand   = solveLPDetailed(grand_m, grandOpts);
    REQUIRE(grand.result.status == LPStatus::Optimal);
    REQUIRE_THAT(grand.result.objectiveValue, WithinAbs(-2.0, kTol));
    REQUIRE_THAT(grand.result.primalValues[0], WithinAbs(1.0, kTol));
    REQUIRE_THAT(grand.result.primalValues[1], WithinAbs(1.0, kTol));
}

TEST_CASE("Warm-start: usedWarmStart confirms effective warm on branch, false on cold", "[warm_start]") {
    // Verifies that usedWarmStart == true when the warm path is taken (no sfCache
    // mismatch, no dual-feasibility fallback), and false on a plain cold solve.
    // Covers DualSimplex (sfCache), DualSimplexBV (atUBCache), RevisedSimplex and
    // RevisedSimplexBV (sfbvCache + atUBCache + BV dual repair).
    auto method = GENERATE(LPMethod::DualSimplex,    LPMethod::DualSimplexBV,
                           LPMethod::RevisedSimplex, LPMethod::RevisedSimplexBV);

    Model    parent_m = makeFractionalLP();
    Variable x1{0};

    LPOptions coldOpts; coldOpts.method = method; coldOpts.enablePresolve = false;
    auto parent = solveLPDetailed(parent_m, coldOpts);
    REQUIRE(parent.result.status == LPStatus::Optimal);
    REQUIRE_FALSE(parent.usedWarmStart); // no warm basis supplied → cold path

    // Branch: x1 <= floor(4/3) = 1.  Tightening the bound makes the parent basis
    // primal-infeasible; the dual simplex repairs it via a single-pivot ratio test.
    Model child_m = parent_m.withVarBounds(x1, 0.0, 1.0);

    LPOptions warmOpts;
    warmOpts.method        = method;
    warmOpts.warmBasis     = parent.basis;
    warmOpts.enablePresolve = false;
    auto warm = solveLPDetailed(child_m, warmOpts);
    REQUIRE(warm.result.status == LPStatus::Optimal);
    REQUIRE(warm.usedWarmStart);                              // warm-start accepted — no fallback
    REQUIRE_THAT(warm.result.objectiveValue, WithinAbs(-2.5, kTol));

    LPOptions coldChildOpts; coldChildOpts.method = method; coldChildOpts.enablePresolve = false;
    auto cold = solveLPDetailed(child_m, coldChildOpts);
    REQUIRE(cold.result.status == LPStatus::Optimal);
    REQUIRE_FALSE(cold.usedWarmStart);                        // cold path: no basis supplied
    REQUIRE_THAT(cold.result.objectiveValue, WithinAbs(warm.result.objectiveValue, kTol));
    // Warm start repairs the parent basis with 1-2 dual pivots; cold rebuilds from scratch.
    CHECK(warm.iterationsUsed < cold.iterationsUsed);
}

TEST_CASE("sfCache: result matches cold solve at every level", "[warm_start]") {
    // Verify that the sfCache optimisation does not affect correctness by
    // comparing the warm (cached) result to an independent cold solve.
    Model    root_m = makeFractionalLP();
    Variable x1{0};

    auto root = solveLPDetailed(root_m, LPOptions{.enablePresolve = false});
    REQUIRE(root.result.status == LPStatus::Optimal);

    Model child_m = root_m.withVarBounds(x1, 0.0, 1.0);
    LPOptions warmOpts; warmOpts.warmBasis = root.basis; warmOpts.enablePresolve = false;
    auto  warm    = solveLPDetailed(child_m, warmOpts);
    auto  cold    = solveLPDetailed(child_m, LPOptions{.enablePresolve = false});

    REQUIRE(warm.result.status == LPStatus::Optimal);
    REQUIRE(cold.result.status == LPStatus::Optimal);
    REQUIRE_THAT(warm.result.objectiveValue,
                 WithinAbs(cold.result.objectiveValue, kTol));
    REQUIRE_THAT(warm.result.primalValues[0],
                 WithinAbs(cold.result.primalValues[0], kTol));
    REQUIRE_THAT(warm.result.primalValues[1],
                 WithinAbs(cold.result.primalValues[1], kTol));
    // The sfCache reuses the parent basis: warm repairs primal infeasibility with fewer
    // pivots than a cold solve reconstructing the optimal vertex from the slack basis.
    CHECK(warm.iterationsUsed < cold.iterationsUsed);
}

// ── Non-implementing methods: basis ignored, identical work ───────────────────

TEST_CASE("Warm-start: non-implementing methods ignore basis and produce identical results", "[warm_start]") {
    // Methods that do not implement warm-start ignore warmBasis entirely.
    // Passing a basis record must be a no-op: usedWarmStart stays false and the
    // solver does exactly the same work (same iterationsUsed) as a cold call.
    auto method = GENERATE(LPMethod::PrimalSimplex, LPMethod::PrimalSimplexBV,
                           LPMethod::ShortStepIPM,  LPMethod::MehrotraIPM,
                           LPMethod::NetworkSimplex);

    DYNAMIC_SECTION(to_string(method)) {
        Model    parent_m = makeFractionalLP();
        Variable x1{0};

        // Solve the parent to obtain a basis record (even if the method doesn't
        // produce one, passing it to the warm call below must be harmless).
        LPOptions parentOpts; parentOpts.method = method; parentOpts.enablePresolve = false;
        auto parent = solveLPDetailed(parent_m, parentOpts);

        // Child LP: x1 <= 1 (tightened upper bound → different feasible region).
        Model child_m = parent_m.withVarBounds(x1, 0.0, 1.0);

        LPOptions coldOpts; coldOpts.method = method; coldOpts.enablePresolve = false;
        LPOptions warmOpts; warmOpts.method = method; warmOpts.warmBasis = parent.basis;
        warmOpts.enablePresolve = false;

        auto cold = solveLPDetailed(child_m, coldOpts);
        auto warm = solveLPDetailed(child_m, warmOpts);

        // Basis ignored: the warm path must not be entered.
        CHECK_FALSE(warm.usedWarmStart);
        // Same algorithm, same initial conditions → identical amount of work.
        REQUIRE(warm.result.status == cold.result.status);
        CHECK(warm.iterationsUsed == cold.iterationsUsed);
    }
}
