#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <limits>
#include <vector>

#include "lp/StandardForm.hpp"
#include "lp/Tableau.hpp"
#include "baguette/core/Config.hpp"
#include "baguette/model/Model.hpp"

using namespace baguette;
using namespace baguette::internal;
using Catch::Matchers::WithinAbs;

static constexpr double kTol  = 1e-8;
static const double     kInf  = std::numeric_limits<double>::infinity();

// ── Helpers ───────────────────────────────────────────────────────────────────

static double Ap(const LPStandardForm& sf, std::size_t row, std::size_t col) {
    return (*sf.A)[row * sf.nCols + col];
}

/// Solve an LPStandardForm with the primal two-phase simplex.
/// Returns {status, objectiveValue} using only the Tableau machinery.
static std::pair<LPStatus, double> solveRawSF(const LPStandardForm& sf) {
    // Phase I: add one artificial per row (all rows may need one since we
    // build from scratch without knowing which rows already have naturals).
    const std::size_t m    = sf.nRows;
    const std::size_t nOld = sf.nCols;
    const std::size_t nNew = nOld + m;

    LPStandardForm aug = sf;
    aug.nCols = nNew;
    aug.A = std::make_shared<std::vector<double>>(m * nNew, 0.0);
    aug.c.assign(nNew, 0.0);
    aug.colKind.resize(nNew);
    aug.colOrigin.resize(nNew);

    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < nOld; ++j)
            (*aug.A)[i * nNew + j] = (*sf.A)[i * nOld + j];
        (*aug.A)[i * nNew + nOld + i] = 1.0;
        aug.c[nOld + i]            = 1.0; // phase-I objective
        aug.colKind[nOld + i]      = ColumnKind::Slack;
        aug.colOrigin[nOld + i]    = 0;
    }

    std::vector<uint32_t> basis(m);
    for (std::size_t i = 0; i < m; ++i)
        basis[i] = static_cast<uint32_t>(nOld + i);

    Tableau tab;
    tab.init(aug, basis);

    // Phase-I loop
    while (true) {
        std::size_t entering = tab.selectEntering();
        if (entering == tab.n) break;
        std::size_t leaving = tab.selectLeaving(entering);
        if (leaving == tab.m) break; // shouldn't happen in phase I
        tab.pivot(leaving, entering);
    }

    if (tab.objectiveValue() > baguette::lp_feasibility_tol)
        return {LPStatus::Infeasible, 0.0};

    // Shrink to original columns and re-price phase-II objective
    std::vector<double> newTab(m * (nOld + 1));
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < nOld; ++j)
            newTab[i * (nOld + 1) + j] = tab.tab[i * (nNew + 1) + j];
        newTab[i * (nOld + 1) + nOld] = tab.tab[i * (nNew + 1) + nNew];
    }
    tab.tab = std::move(newTab);
    tab.n   = nOld;

    // Repair basis entries that still point to artificials
    for (std::size_t i = 0; i < m; ++i)
        if (tab.basicCols[i] >= nOld)
            tab.basicCols[i] = 0; // degenerate row: assign col 0

    // Re-price phase-II objective
    tab.rc.assign(nOld + 1, 0.0);
    for (std::size_t j = 0; j < nOld; ++j)
        tab.rc[j] = sf.c[j];
    for (std::size_t i = 0; i < m; ++i) {
        double cb = sf.c[tab.basicCols[i]];
        if (cb == 0.0) continue;
        for (std::size_t j = 0; j <= nOld; ++j)
            tab.rc[j] -= cb * tab.tab[i * (nOld + 1) + j];
    }

    // Phase-II loop
    while (true) {
        std::size_t entering = tab.selectEntering();
        if (entering == tab.n) return {LPStatus::Optimal, tab.objectiveValue()};
        std::size_t leaving = tab.selectLeaving(entering);
        if (leaving == tab.m) return {LPStatus::Unbounded, 0.0};
        tab.pivot(leaving, entering);
    }
}

// ── Dimensions ────────────────────────────────────────────────────────────────

TEST_CASE("dual SF - dimensions match transposition", "[dual_sf]") {
    // Primal: 2 rows, 3 cols  →  Dual: 3 rows, 2*2+3 = 7 cols
    Model m;
    auto x = m.addVar(0.0, kInf, "x");
    auto y = m.addVar(0.0, kInf, "y");
    m.addLPConstraint(1.0 * x + 2.0 * y, Sense::LessEq, 4.0);
    m.addLPConstraint(3.0 * x + 1.0 * y, Sense::LessEq, 6.0);
    m.setObjective(1.0 * x + 1.0 * y, ObjSense::Minimize);

    auto primal = toStandardForm(m);  // 2 rows, 4 cols (x, y, s0, s1)
    auto dual   = dualStandardForm(primal);

    CHECK(dual.nRows    == primal.nCols);          // n = 4 rows
    CHECK(dual.nCols    == 2 * primal.nRows + primal.nCols); // 2m+n = 8
    CHECK(dual.nOrig    == 2 * primal.nRows);      // y+ and y- blocks
    CHECK(dual.nSlack   == primal.nCols);
}

TEST_CASE("dual SF - single LessEq constraint dimensions", "[dual_sf]") {
    // Primal: min x  s.t. x <= 5,  x >= 0
    // SF: 1 row, 2 cols (x, slack)
    // Dual SF: 2 rows, 2*1+2 = 4 cols (y+, y-, s0, s1)
    Model m;
    auto x = m.addVar(0.0, kInf, "x");
    m.addLPConstraint(1.0 * x, Sense::LessEq, 5.0);
    m.setObjective(1.0 * x, ObjSense::Minimize);

    auto primal = toStandardForm(m);
    auto dual   = dualStandardForm(primal);

    CHECK(dual.nRows  == 2);
    CHECK(dual.nCols  == 4);
    CHECK(dual.nOrig  == 2);
    CHECK(dual.nSlack == 2);
}

// ── Matrix structure: A_dual[j,i] = A_primal[i,j] ───────────────────────────

TEST_CASE("dual SF - A matrix is transposed primal A (y+ block)", "[dual_sf]") {
    // Primal SF: min c^T x, Ax=b, x>=0
    // Dual SF row j: sum_i A_primal[i,j] * y+_i + ... + s_j = c_primal[j]
    // Check A_dual[j, i] == A_primal[i, j] for j=0..n-1, i=0..m-1
    Model m;
    auto x = m.addVar(0.0, kInf, "x");
    auto y = m.addVar(0.0, kInf, "y");
    m.addLPConstraint(2.0 * x + 3.0 * y, Sense::LessEq, 10.0);
    m.addLPConstraint(1.0 * x + 4.0 * y, Sense::LessEq,  8.0);
    m.setObjective(5.0 * x + 2.0 * y, ObjSense::Minimize);

    auto primal = toStandardForm(m);
    auto dual   = dualStandardForm(primal);

    const std::size_t mP = primal.nRows;
    const std::size_t nP = primal.nCols;

    // For each dual row j and primal row i:
    //   if rowNegated[j] == false: A_dual[j, i] == A_primal[i, j]
    //   if rowNegated[j] == true:  A_dual[j, i] == -A_primal[i, j]
    for (std::size_t j = 0; j < nP; ++j) {
        for (std::size_t i = 0; i < mP; ++i) {
            double expected = (*primal.A)[i * nP + j];
            if (dual.rowNegated[j]) expected = -expected;
            CHECK_THAT(Ap(dual, j, i), WithinAbs(expected, kTol));
        }
    }
}

TEST_CASE("dual SF - y- block is negation of y+ block", "[dual_sf]") {
    Model m;
    auto x = m.addVar(0.0, kInf, "x");
    auto y = m.addVar(0.0, kInf, "y");
    m.addLPConstraint(2.0 * x + 3.0 * y, Sense::LessEq, 10.0);
    m.setObjective(1.0 * x + 1.0 * y, ObjSense::Minimize);

    auto primal = toStandardForm(m);
    auto dual   = dualStandardForm(primal);

    const std::size_t mP = primal.nRows;
    const std::size_t nP = primal.nCols;

    for (std::size_t j = 0; j < nP; ++j)
        for (std::size_t i = 0; i < mP; ++i)
            CHECK_THAT(Ap(dual, j, mP + i), WithinAbs(-Ap(dual, j, i), kTol));
}

TEST_CASE("dual SF - slack block is identity", "[dual_sf]") {
    Model m;
    auto x = m.addVar(0.0, kInf, "x");
    m.addLPConstraint(1.0 * x, Sense::LessEq, 5.0);
    m.setObjective(1.0 * x, ObjSense::Minimize);

    auto primal = toStandardForm(m);
    auto dual   = dualStandardForm(primal);

    const std::size_t nP = primal.nCols;
    const std::size_t mP = primal.nRows;

    // Slack block: columns [2m .. 2m+n-1], should form identity (before any negation)
    for (std::size_t j = 0; j < nP; ++j) {
        for (std::size_t jj = 0; jj < nP; ++jj) {
            double expected = (j == jj) ? 1.0 : 0.0;
            if (dual.rowNegated[j]) expected = -expected;
            CHECK_THAT(Ap(dual, j, 2 * mP + jj), WithinAbs(expected, kTol));
        }
    }
}

// ── Objective and rhs ─────────────────────────────────────────────────────────

TEST_CASE("dual SF - objective is negated primal rhs", "[dual_sf]") {
    // Dual obj: min -b^T y+ + b^T y-  →  c_dual[i] = -b[i], c_dual[m+i] = +b[i]
    Model m;
    auto x = m.addVar(0.0, kInf, "x");
    auto y = m.addVar(0.0, kInf, "y");
    m.addLPConstraint(1.0 * x + 2.0 * y, Sense::LessEq, 7.0);
    m.addLPConstraint(3.0 * x + 1.0 * y, Sense::LessEq, 9.0);
    m.setObjective(1.0 * x + 1.0 * y, ObjSense::Minimize);

    auto primal = toStandardForm(m);
    auto dual   = dualStandardForm(primal);

    const std::size_t mP = primal.nRows;

    for (std::size_t i = 0; i < mP; ++i) {
        CHECK_THAT(dual.c[i],      WithinAbs(-primal.b[i], kTol)); // y+_i
        CHECK_THAT(dual.c[mP + i], WithinAbs(+primal.b[i], kTol)); // y-_i
    }
    // Slack objective coefficients are 0
    for (std::size_t j = 0; j < primal.nCols; ++j)
        CHECK_THAT(dual.c[2 * mP + j], WithinAbs(0.0, kTol));
}

TEST_CASE("dual SF - rhs is primal objective (normalised)", "[dual_sf]") {
    // b_dual[j] = |c_primal[j]| after row normalisation
    Model m;
    auto x = m.addVar(0.0, kInf, "x");
    m.addLPConstraint(1.0 * x, Sense::LessEq, 5.0);
    m.setObjective(3.0 * x, ObjSense::Minimize);

    auto primal = toStandardForm(m);
    auto dual   = dualStandardForm(primal);

    for (std::size_t j = 0; j < dual.nRows; ++j)
        CHECK(dual.b[j] >= 0.0); // always normalised
}

TEST_CASE("dual SF - rowSlackCol points to slack block", "[dual_sf]") {
    Model m;
    auto x = m.addVar(0.0, kInf, "x");
    auto y = m.addVar(0.0, kInf, "y");
    m.addLPConstraint(1.0 * x + 1.0 * y, Sense::LessEq, 4.0);
    m.setObjective(1.0 * x, ObjSense::Minimize);

    auto primal = toStandardForm(m);
    auto dual   = dualStandardForm(primal);

    const std::size_t mP = primal.nRows;

    for (std::size_t j = 0; j < dual.nRows; ++j)
        CHECK(dual.rowSlackCol[j] == 2 * mP + j);
}

// ── Strong duality ────────────────────────────────────────────────────────────

TEST_CASE("dual SF - strong duality: simple min LessEq", "[dual_sf]") {
    // Primal: min 2x + 3y  s.t. x+y<=4, x<=3, y>=0, x>=0
    // Optimal primal: obj = 8  (x=4, but x<=3 is binding... let me recalculate)
    // Actually: min 2x+3y  s.t. x+y<=4  → optimal at x=4, y=0 → obj=8? No, x+y<=4 → x can be 4.
    // Wait: only constraint is x+y<=4, x,y>=0. Minimum of 2x+3y with x+y<=4:
    // Corner (0,0) → obj=0. But wait x,y>=0 so (0,0) is feasible with obj=0.
    // That's the minimum. But 2x+3y is minimised at (0,0) not at the constraint boundary.
    // Use a GreaterEq constraint to force a non-trivial solution.
    //
    // Primal: min 2x + 3y  s.t. x+y>=4, x,y>=0
    // Optimal: x=4, y=0, obj=8
    // Dual:    max 4*y  s.t. y<=2, y<=3  →  max at y=2, obj=8 ✓
    Model m;
    auto x = m.addVar(0.0, kInf, "x");
    auto y = m.addVar(0.0, kInf, "y");
    m.addLPConstraint(1.0 * x + 1.0 * y, Sense::GreaterEq, 4.0);
    m.setObjective(2.0 * x + 3.0 * y, ObjSense::Minimize);

    auto primal_sf = toStandardForm(m);
    auto dual_sf   = dualStandardForm(primal_sf);

    auto [ps, pObj] = solveRawSF(primal_sf);
    auto [ds, dObj] = solveRawSF(dual_sf);

    REQUIRE(ps == LPStatus::Optimal);
    REQUIRE(ds == LPStatus::Optimal);

    // Primal obj: min in standard form (includes objOffset=0 for this construction)
    // Dual obj: min -b^T y → dObj = -primalObj at optimality (strong duality)
    CHECK_THAT(pObj + dObj, WithinAbs(0.0, kTol));
}

TEST_CASE("dual SF - strong duality: two LessEq constraints", "[dual_sf]") {
    // Primal: min 5x + 4y  s.t.  6x + 4y >= 24,  x + 2y >= 6,  x,y >= 0
    // Optimal: intersection of 6x+4y=24 and x+2y=6 → x=3, y=1.5 → obj=5*3+4*1.5=21
    Model m;
    auto x = m.addVar(0.0, kInf, "x");
    auto y = m.addVar(0.0, kInf, "y");
    m.addLPConstraint(6.0 * x + 4.0 * y, Sense::GreaterEq, 24.0);
    m.addLPConstraint(1.0 * x + 2.0 * y, Sense::GreaterEq,  6.0);
    m.setObjective(5.0 * x + 4.0 * y, ObjSense::Minimize);

    auto primal_sf = toStandardForm(m);
    auto dual_sf   = dualStandardForm(primal_sf);

    auto [ps, pObj] = solveRawSF(primal_sf);
    auto [ds, dObj] = solveRawSF(dual_sf);

    REQUIRE(ps == LPStatus::Optimal);
    REQUIRE(ds == LPStatus::Optimal);
    CHECK_THAT(pObj + dObj, WithinAbs(0.0, kTol));
}

TEST_CASE("dual SF - double dual has same optimal value as primal", "[dual_sf]") {
    // Primal: min 2x + 3y  s.t. x + y >= 1.6,  x,y >= 0
    // Optimal: x = 1.6, y = 0, obj = 3.2  (non-trivial, driven by GEQ constraint)
    //
    // By strong duality applied twice:
    //   dual_sf   opt = −pObj = −3.2   (min −b^T y)
    //   DS2       opt = −(−pObj) = pObj = 3.2
    //
    // Each solve may accumulate up to kTol of floating-point error, so the
    // comparison pObj ≈ ppObj uses 2*kTol to avoid spurious failures when one
    // result is just above and the other just below the true value.
    Model m;
    auto x = m.addVar(0.0, kInf, "x");
    auto y = m.addVar(0.0, kInf, "y");
    m.addLPConstraint(1.0 * x + 1.0 * y, Sense::GreaterEq, 1.6);
    m.setObjective(2.0 * x + 3.0 * y, ObjSense::Minimize);

    auto primal_sf   = toStandardForm(m);
    auto dual_sf     = dualStandardForm(primal_sf);
    auto dualdual_sf = dualStandardForm(dual_sf);

    auto [ps,  pObj]  = solveRawSF(primal_sf);
    auto [pps, ppObj] = solveRawSF(dualdual_sf);

    REQUIRE(ps  == LPStatus::Optimal);
    REQUIRE(pps == LPStatus::Optimal);
    CHECK_THAT(pObj,  WithinAbs(3.2,  kTol));        // primal optimal
    CHECK_THAT(ppObj, WithinAbs(3.2,  kTol));        // double-dual optimal
    CHECK_THAT(ppObj, WithinAbs(pObj, 2.0 * kTol)); // equal up to two rounding budgets
}

TEST_CASE("dual SF - primal infeasible implies dual unbounded", "[dual_sf]") {
    // Infeasible primal: x >= 3 AND x <= 2
    Model m;
    auto x = m.addVar(0.0, kInf, "x");
    m.addLPConstraint(1.0 * x, Sense::GreaterEq, 3.0);
    m.addLPConstraint(1.0 * x, Sense::LessEq,    2.0);
    m.setObjective(1.0 * x, ObjSense::Minimize);

    auto primal_sf = toStandardForm(m);
    auto dual_sf   = dualStandardForm(primal_sf);

    auto [ps, _p] = solveRawSF(primal_sf);
    auto [ds, _d] = solveRawSF(dual_sf);

    CHECK(ps == LPStatus::Infeasible);
    // By LP duality: primal infeasible → dual is infeasible or unbounded
    CHECK((ds == LPStatus::Unbounded || ds == LPStatus::Infeasible));
}
