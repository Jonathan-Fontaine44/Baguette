#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <limits>

#include "lp/StandardForm.hpp"
#include "lp/Tableau.hpp"
#include "baguette/model/Model.hpp"

// Regression tests for Bug #1: Tableau::init() must handle any valid basis
// regardless of column/row ordering. The fix adds partial pivoting so that
// reinvert() correctly rebuilds the tableau even when basicCols is not
// triangularly compatible with the original row order.

using namespace baguette;
using namespace baguette::internal;
using Catch::Matchers::WithinAbs;

static constexpr double kTol = 1e-10;
static const double kInf = std::numeric_limits<double>::infinity();

// ── LP fixture ────────────────────────────────────────────────────────────────
//
//   min  x + 2y
//   s.t. x + y <= 4   (row 0, slack s0 = col 2)
//        x     <= 3   (row 1, slack s1 = col 3)
//            y <= 3   (row 2, slack s2 = col 4)
//        x, y >= 0
//
// Standard-form column layout: [x(0), y(1), s0(2), s1(3), s2(4)]
// Standard-form matrix A (row-major):
//   row 0: [1, 1, 1, 0, 0]  b = 4
//   row 1: [1, 0, 0, 1, 0]  b = 3
//   row 2: [0, 1, 0, 0, 1]  b = 3
//
// Optimal corner: x=3, y=1, s0=0, s1=0, s2=2   →  obj = 5.
// Optimal basic variables: {x(0), y(1), s2(4)}.

static Model makeTestLP() {
    Model m;
    auto x = m.addVar(0.0, kInf, "x");
    auto y = m.addVar(0.0, kInf, "y");
    m.addConstraint(1.0*x + 1.0*y, Sense::LessEq, 4.0);
    m.addConstraint(1.0*x,          Sense::LessEq, 3.0);
    m.addConstraint(        1.0*y,  Sense::LessEq, 3.0);
    m.setObjective(1.0*x + 2.0*y, ObjSense::Minimize);
    return m;
}

// ── Bug trigger at step i=0 ───────────────────────────────────────────────────

TEST_CASE("Tableau::reinvert - permuted basis, zero at step 0", "[tableau]") {
    // basicCols = [4, 0, 1]:  row0→s2(4),  row1→x(0),  row2→y(1)
    //
    // Basis matrix B = A[:, [4,0,1]] = [[0,1,1],[0,1,0],[1,0,1]]
    // det(B) = -1  →  valid, non-singular.
    //
    // Sequential GJ at step i=0:
    //   col = basicCols[0] = 4,  A[row=0, col=4] = 0  →  false pivot (bug).
    //
    // Solution: x=3, y=1, s2=2 (the optimal corner), obj=5.
    auto sf = toStandardForm(makeTestLP());

    REQUIRE(sf.nOrig  == 2);
    REQUIRE(sf.nSlack == 3);
    REQUIRE(sf.nCols  == 5);
    REQUIRE(sf.nRows  == 3);
    REQUIRE((*sf.A)[0*5 + 4] == 0.0);  // confirms A[row=0, col=4] = 0

    // Init with the natural slack basis (triangular, always safe).
    Tableau tab;
    assert(tab.init(sf, {2, 3, 4}));

    // Simulate the warm-start path: overwrite basicCols with the permuted
    // optimal basis, then reinvert (same two lines as LPSolver.cpp:508-509).
    tab.basicCols = {4, 0, 1};
    REQUIRE(tab.reinvert(sf));

    auto x_sol = tab.primalSolution();
    REQUIRE(x_sol.size() == 5);
    CHECK_THAT(x_sol[0], WithinAbs(3.0, kTol));  // x  = 3
    CHECK_THAT(x_sol[1], WithinAbs(1.0, kTol));  // y  = 1
    CHECK_THAT(x_sol[2], WithinAbs(0.0, kTol));  // s0 = 0
    CHECK_THAT(x_sol[3], WithinAbs(0.0, kTol));  // s1 = 0
    CHECK_THAT(x_sol[4], WithinAbs(2.0, kTol));  // s2 = 2
    CHECK_THAT(tab.objectiveValue(), WithinAbs(5.0, kTol));
}

// ── Bug trigger at step i=1 (first step passes, second fails) ─────────────────

TEST_CASE("Tableau::reinvert - permuted basis, zero appears at step 1", "[tableau]") {
    // basicCols = [0, 4, 1]:  row0→x(0),  row1→s2(4),  row2→y(1)
    //
    // Basis matrix B = A[:, [0,4,1]] = [[1,0,1],[1,0,0],[0,1,1]]
    // det(B) = 1  →  valid, non-singular.
    //
    // Sequential GJ at step i=0:
    //   col = 0,  A[row=0, col=0] = 1  →  OK, eliminates col 0.
    //   After elimination: A_updated[row=1, col=0] = 0, A_updated[row=1, col=4] = ?
    //   row1 := row1 - 1*row0 = [1,0,0,1,0] - [1,1,1,0,0] = [0,-1,-1,1,0]
    // Sequential GJ at step i=1:
    //   col = basicCols[1] = 4,  A_updated[row=1, col=4] = 0  →  false pivot (bug).
    //
    // Same optimal corner solution.
    auto sf = toStandardForm(makeTestLP());

    // Verify: after eliminating col 0 from row 1 using row 0 as pivot,
    // the entry at (row=1, col=4) becomes 0.
    // A[row=1] = [1,0,0,1,0], A[row=0] = [1,1,1,0,0] → row1 - row0 = [0,-1,-1,1,0]
    // col=4 entry: 0 - 0 = 0. Confirmed.
    REQUIRE((*sf.A)[1*5 + 4] == 0.0);  // A[row=1, col=4] = 0 in the original matrix

    Tableau tab;
    assert(tab.init(sf, {2, 3, 4}));

    tab.basicCols = {0, 4, 1};
    REQUIRE(tab.reinvert(sf));

    auto x_sol = tab.primalSolution();
    REQUIRE(x_sol.size() == 5);
    CHECK_THAT(x_sol[0], WithinAbs(3.0, kTol));
    CHECK_THAT(x_sol[1], WithinAbs(1.0, kTol));
    CHECK_THAT(x_sol[2], WithinAbs(0.0, kTol));
    CHECK_THAT(x_sol[3], WithinAbs(0.0, kTol));
    CHECK_THAT(x_sol[4], WithinAbs(2.0, kTol));
    CHECK_THAT(tab.objectiveValue(), WithinAbs(5.0, kTol));
}
