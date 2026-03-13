#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <limits>

#include "lp/StandardForm.hpp"
#include "baguette/model/Model.hpp"

using namespace baguette;
using namespace baguette::internal;
using Catch::Matchers::WithinAbs;

static constexpr double kTol = 1e-12;
static const double kInf = std::numeric_limits<double>::infinity();

// ── Helpers ───────────────────────────────────────────────────────────────────

/// A(row, col) accessor for the dense row-major matrix.
static double A(const LPStandardForm& sf, std::size_t row, std::size_t col) {
    return sf.A[row * sf.nCols + col];
}

// ── Dimensions ────────────────────────────────────────────────────────────────

TEST_CASE("SF dimensions - single LessEq, no bounds", "[standard_form]") {
    // min x  s.t. x <= 5,  x >= 0
    Model m;
    auto x = m.addVar(0.0, kInf);
    m.addConstraint(1.0 * x, Sense::LessEq, 5.0);
    m.setObjective(1.0 * x, ObjSense::Minimize);

    auto sf = toStandardForm(m);

    CHECK(sf.nOrig     == 1); // x'
    CHECK(sf.nSlack    == 1); // slack for LessEq
    CHECK(sf.nOrigRows == 1);
    CHECK(sf.nRows     == 1); // no UB row
    CHECK(sf.nCols     == 2); // x', s0
}

TEST_CASE("SF dimensions - finite upper bound adds UB row", "[standard_form]") {
    // min x  s.t. x <= 3,  0 <= x <= 5
    Model m;
    auto x = m.addVar(0.0, 5.0);
    m.addConstraint(1.0 * x, Sense::LessEq, 3.0);
    m.setObjective(1.0 * x, ObjSense::Minimize);

    auto sf = toStandardForm(m);

    CHECK(sf.nOrigRows == 1);
    CHECK(sf.nRows     == 2); // 1 constraint + 1 UB row
    CHECK(sf.nCols     == 3); // x', s_leq, s_ub
}

TEST_CASE("SF dimensions - two vars, mixed senses", "[standard_form]") {
    // min x + y  s.t. x + y <= 4, x >= 1, x = y
    Model m;
    auto x = m.addVar(0.0, kInf);
    auto y = m.addVar(0.0, kInf);
    m.addConstraint(1.0 * x + 1.0 * y, Sense::LessEq,    4.0);
    m.addConstraint(1.0 * x,            Sense::GreaterEq, 1.0);
    m.addConstraint(1.0 * x - 1.0 * y, Sense::Equal,     0.0);
    m.setObjective(1.0 * x + 1.0 * y, ObjSense::Minimize);

    auto sf = toStandardForm(m);

    CHECK(sf.nOrig     == 2); // x', y'
    CHECK(sf.nSlack    == 2); // LessEq + GreaterEq only (Equal has no slack)
    CHECK(sf.nOrigRows == 3);
    CHECK(sf.nRows     == 3); // no finite UBs
    CHECK(sf.nCols     == 4); // x', y', s_leq, s_surplus
}

// ── Objective vector ──────────────────────────────────────────────────────────

TEST_CASE("SF objective - minimize, no lb shift", "[standard_form]") {
    Model m;
    auto x = m.addVar(0.0, kInf);
    auto y = m.addVar(0.0, kInf);
    m.addConstraint(1.0 * x, Sense::LessEq, 10.0);
    m.setObjective(3.0 * x + 5.0 * y, ObjSense::Minimize);

    auto sf = toStandardForm(m);

    CHECK_THAT(sf.c[0], WithinAbs( 3.0, kTol)); // x'
    CHECK_THAT(sf.c[1], WithinAbs( 5.0, kTol)); // y'
    CHECK_THAT(sf.c[2], WithinAbs( 0.0, kTol)); // slack
    CHECK_THAT(sf.objOffset, WithinAbs(0.0, kTol)); // lb = 0
}

TEST_CASE("SF objective - minimize, non-zero lb shifts offset", "[standard_form]") {
    // min 2x  s.t. x <= 10,  3 <= x <= 10
    // After shift: min 2(x'+3) = 2x' + 6  → c[0]=2, objOffset=6
    Model m;
    auto x = m.addVar(3.0, 10.0);
    m.addConstraint(1.0 * x, Sense::LessEq, 10.0);
    m.setObjective(2.0 * x, ObjSense::Minimize);

    auto sf = toStandardForm(m);

    CHECK_THAT(sf.c[0], WithinAbs(2.0, kTol));
    CHECK_THAT(sf.objOffset, WithinAbs(6.0, kTol)); // 2 * 3
}

TEST_CASE("SF objective - maximize negates costs", "[standard_form]") {
    // max 3x  →  standard form: min -3x',  objOffset = 0
    Model m;
    auto x = m.addVar(0.0, kInf);
    m.addConstraint(1.0 * x, Sense::LessEq, 10.0);
    m.setObjective(3.0 * x, ObjSense::Maximize);

    auto sf = toStandardForm(m);

    CHECK_THAT(sf.c[0], WithinAbs(-3.0, kTol)); // negated for min
    CHECK_THAT(sf.objOffset, WithinAbs(0.0, kTol));
}

TEST_CASE("SF objective - maximize with lb: offset negated", "[standard_form]") {
    // max 2x,  2 <= x  →  standard form: min -2x', objOffset = -2*2 = -4
    Model m;
    auto x = m.addVar(2.0, kInf);
    m.addConstraint(1.0 * x, Sense::LessEq, 10.0);
    m.setObjective(2.0 * x, ObjSense::Maximize);

    auto sf = toStandardForm(m);

    CHECK_THAT(sf.c[0], WithinAbs(-2.0, kTol));
    CHECK_THAT(sf.objOffset, WithinAbs(-4.0, kTol)); // (-2) * 2
}

// ── Constraint matrix ─────────────────────────────────────────────────────────

TEST_CASE("SF constraint - LessEq: slack has +1", "[standard_form]") {
    // x + 2y <= 6,  x,y >= 0
    // Row 0: [1, 2, +1 | 6]
    Model m;
    auto x = m.addVar(0.0, kInf);
    auto y = m.addVar(0.0, kInf);
    m.addConstraint(1.0 * x + 2.0 * y, Sense::LessEq, 6.0);
    m.setObjective(1.0 * x, ObjSense::Minimize);

    auto sf = toStandardForm(m);

    CHECK_THAT(A(sf, 0, 0), WithinAbs( 1.0, kTol)); // x'
    CHECK_THAT(A(sf, 0, 1), WithinAbs( 2.0, kTol)); // y'
    CHECK_THAT(A(sf, 0, 2), WithinAbs(+1.0, kTol)); // slack
    CHECK_THAT(sf.b[0],     WithinAbs( 6.0, kTol));
    CHECK(sf.rowNegated[0] == false);
}

TEST_CASE("SF constraint - GreaterEq: surplus has -1", "[standard_form]") {
    // x + y >= 3,  x,y >= 0
    // Row 0: [1, 1, -1 | 3]
    Model m;
    auto x = m.addVar(0.0, kInf);
    auto y = m.addVar(0.0, kInf);
    m.addConstraint(1.0 * x + 1.0 * y, Sense::GreaterEq, 3.0);
    m.setObjective(1.0 * x, ObjSense::Minimize);

    auto sf = toStandardForm(m);

    CHECK_THAT(A(sf, 0, 0), WithinAbs( 1.0, kTol)); // x'
    CHECK_THAT(A(sf, 0, 1), WithinAbs( 1.0, kTol)); // y'
    CHECK_THAT(A(sf, 0, 2), WithinAbs(-1.0, kTol)); // surplus
    CHECK_THAT(sf.b[0],     WithinAbs( 3.0, kTol));
    CHECK(sf.rowNegated[0] == false);
}

TEST_CASE("SF constraint - Equal: no slack column entry", "[standard_form]") {
    // x + y = 5 — Equal rows have no slack column (Option A).
    // nCols == nOrig == 2; column index 2 is out of bounds.
    Model m;
    auto x = m.addVar(0.0, kInf);
    auto y = m.addVar(0.0, kInf);
    m.addConstraint(1.0 * x + 1.0 * y, Sense::Equal, 5.0);
    m.setObjective(1.0 * x, ObjSense::Minimize);

    auto sf = toStandardForm(m);

    CHECK_THAT(A(sf, 0, 0), WithinAbs(1.0, kTol)); // x'
    CHECK_THAT(A(sf, 0, 1), WithinAbs(1.0, kTol)); // y'
    CHECK(sf.nCols == 2); // no slack column allocated for Equal row
    CHECK_THROWS_AS(sf.A.at(0 * sf.nCols + 2), std::out_of_range); // column 2 doesn't exist
    CHECK_THAT(sf.b[0],     WithinAbs(5.0, kTol));
}

TEST_CASE("SF constraint - negative rhs is normalised", "[standard_form]") {
    // x <= -2  → rhs < 0 after shift (x,lb=0) → row is negated
    // Original: x + s = -2  →  normalised: -x - s = 2
    Model m;
    auto x = m.addVar(0.0, kInf);
    m.addConstraint(1.0 * x, Sense::LessEq, -2.0);
    m.setObjective(1.0 * x, ObjSense::Minimize);

    auto sf = toStandardForm(m);

    CHECK(sf.rowNegated[0] == true);
    CHECK_THAT(sf.b[0], WithinAbs(2.0, kTol)); // b >= 0 after normalisation
    CHECK_THAT(A(sf, 0, 0), WithinAbs(-1.0, kTol)); // row flipped
    CHECK_THAT(A(sf, 0, 1), WithinAbs(-1.0, kTol)); // slack sign flipped
}

// ── Lower-bound shift ─────────────────────────────────────────────────────────

TEST_CASE("SF lb-shift - rhs absorbs lb contribution", "[standard_form]") {
    // x + y <= 10,  lb_x = 2, lb_y = 3
    // After shift: x' = x-2, y' = y-3
    // Row: x' + y' <= 10 - 2 - 3 = 5
    Model m;
    auto x = m.addVar(2.0, kInf);
    auto y = m.addVar(3.0, kInf);
    m.addConstraint(1.0 * x + 1.0 * y, Sense::LessEq, 10.0);
    m.setObjective(1.0 * x, ObjSense::Minimize);

    auto sf = toStandardForm(m);

    CHECK_THAT(sf.b[0], WithinAbs(5.0, kTol)); // 10 - 2 - 3
    CHECK_THAT(A(sf, 0, 0), WithinAbs(1.0, kTol)); // coeff unchanged
    CHECK_THAT(A(sf, 0, 1), WithinAbs(1.0, kTol));
}

// ── Upper-bound row ───────────────────────────────────────────────────────────

TEST_CASE("SF UB row - structure", "[standard_form]") {
    // min x,  0 <= x <= 7  (no constraints)
    // UB row: x' + s_ub = 7 - 0 = 7
    Model m;
    auto x = m.addVar(0.0, 7.0);
    m.setObjective(1.0 * x, ObjSense::Minimize);

    auto sf = toStandardForm(m);

    // 0 original constraints + 1 UB row
    CHECK(sf.nOrigRows == 0);
    CHECK(sf.nRows     == 1);
    // Columns: x'(0), s_slack(none since 0 constraints), s_ub(1)
    // nSlack = nOrigRows = 0 → nCols = 1 + 0 + 1 = 2
    CHECK(sf.nCols == 2);

    // UB row: row 0, x' col=0, s_ub col=1
    CHECK_THAT(A(sf, 0, 0), WithinAbs(1.0, kTol)); // x'
    CHECK_THAT(A(sf, 0, 1), WithinAbs(1.0, kTol)); // s_ub
    CHECK_THAT(sf.b[0],     WithinAbs(7.0, kTol)); // ub - lb = 7 - 0

    CHECK(sf.colKind[1]   == ColumnKind::UpperSlack);
    CHECK(sf.colOrigin[1] == 0); // Variable::id of x
}

TEST_CASE("SF UB row - lb-shifted range", "[standard_form]") {
    // 2 <= x <= 8  → UB row: x' + s_ub = 8 - 2 = 6
    Model m;
    auto x = m.addVar(2.0, 8.0);
    m.setObjective(1.0 * x, ObjSense::Minimize);

    auto sf = toStandardForm(m);

    CHECK_THAT(sf.b[0], WithinAbs(6.0, kTol)); // ub - lb
}

// ── Column metadata ───────────────────────────────────────────────────────────

TEST_CASE("SF column metadata - kinds and origins", "[standard_form]") {
    // Two vars, one LessEq constraint, both with finite ub
    // Columns: x'(0), y'(1), s_leq(2), s_ub_x(3), s_ub_y(4)
    Model m;
    auto x = m.addVar(0.0, 5.0, "x"); // id=0
    auto y = m.addVar(0.0, 3.0, "y"); // id=1
    m.addConstraint(1.0 * x + 1.0 * y, Sense::LessEq, 4.0);
    m.setObjective(1.0 * x, ObjSense::Minimize);

    auto sf = toStandardForm(m);

    REQUIRE(sf.nCols == 5);

    CHECK(sf.colKind[0]   == ColumnKind::Original);
    CHECK(sf.colOrigin[0] == x.id);

    CHECK(sf.colKind[1]   == ColumnKind::Original);
    CHECK(sf.colOrigin[1] == y.id);

    CHECK(sf.colKind[2]   == ColumnKind::Slack);
    CHECK(sf.colOrigin[2] == 0); // constraint index 0

    CHECK(sf.colKind[3]   == ColumnKind::UpperSlack);
    CHECK(sf.colOrigin[3] == x.id);

    CHECK(sf.colKind[4]   == ColumnKind::UpperSlack);
    CHECK(sf.colOrigin[4] == y.id);
}

TEST_CASE("SF rowSlackCol - points to correct column", "[standard_form]") {
    // 3 constraints → slack cols at nOrig + 0, 1, 2
    Model m;
    auto x = m.addVar(0.0, kInf);
    auto y = m.addVar(0.0, kInf);
    m.addConstraint(1.0 * x,            Sense::LessEq,    5.0);
    m.addConstraint(1.0 * y,            Sense::GreaterEq, 1.0);
    m.addConstraint(1.0 * x + 1.0 * y, Sense::Equal,     4.0);
    m.setObjective(1.0 * x, ObjSense::Minimize);

    auto sf = toStandardForm(m);

    CHECK(sf.rowSlackCol[0] == 2); // nOrig(2) + 0
    CHECK(sf.rowSlackCol[1] == 3); // nOrig(2) + 1
    CHECK(sf.rowSlackCol[2] == 4); // nOrig(2) + 2
}

// ── Infinite upper bound ──────────────────────────────────────────────────────

TEST_CASE("SF infinite ub - no UB row added", "[standard_form]") {
    Model m;
    auto x = m.addVar(0.0, kInf);
    m.addConstraint(1.0 * x, Sense::LessEq, 10.0);
    m.setObjective(1.0 * x, ObjSense::Minimize);

    auto sf = toStandardForm(m);

    CHECK(sf.nRows == 1); // only the LessEq row, no UB row
}

// ── Free-split (fully free variables) ─────────────────────────────────────────

TEST_CASE("SF free-split - dimensions and column count", "[standard_form]") {
    // min x,  x ∈ (−∞, +∞),  x <= 4
    // nOrig=1, nSlack=1 (LessEq), nUBSlack=0, nFree=1
    // → nCols = 1 + 1 + 0 + 1 = 3  (x⁺, s_leq, x⁻)
    // No UB row (ub = +inf).
    Model m;
    auto x = m.addVar(-kInf, kInf, "x");
    m.addConstraint(1.0 * x, Sense::LessEq, 4.0);
    m.setObjective(1.0 * x, ObjSense::Minimize);

    auto sf = toStandardForm(m);

    CHECK(sf.nOrig     == 1);
    CHECK(sf.nSlack    == 1);
    CHECK(sf.nOrigRows == 1);
    CHECK(sf.nRows     == 1); // no UB row
    CHECK(sf.nCols     == 3); // x⁺(0), s_leq(1), x⁻(2)
}

TEST_CASE("SF free-split - varFreeNegCol and ColumnKind::FreeNeg", "[standard_form]") {
    // Same model as above.
    Model m;
    auto x = m.addVar(-kInf, kInf, "x"); // id = 0
    m.addConstraint(1.0 * x, Sense::LessEq, 4.0);
    m.setObjective(1.0 * x, ObjSense::Minimize);

    auto sf = toStandardForm(m);

    // x⁻ is at column 2 (nOrig + nSlack + nUBSlack + 0)
    REQUIRE(sf.varFreeNegCol.size() == 1);
    CHECK(sf.varFreeNegCol[0] == 2);

    CHECK(sf.colKind[0]   == ColumnKind::Original); // x⁺
    CHECK(sf.colKind[1]   == ColumnKind::Slack);    // s_leq
    CHECK(sf.colKind[2]   == ColumnKind::FreeNeg);  // x⁻
    CHECK(sf.colOrigin[2] == x.id);
}

TEST_CASE("SF free-split - A matrix: xneg column is negation of xpos column", "[standard_form]") {
    // min x,  x ∈ (−∞, +∞),  x <= 4
    // Row 0: [A(x⁺)=1, A(s)=+1, A(x⁻)=-1 | b=4]  (LessEq, not negated)
    Model m;
    auto x = m.addVar(-kInf, kInf, "x");
    m.addConstraint(1.0 * x, Sense::LessEq, 4.0);
    m.setObjective(1.0 * x, ObjSense::Minimize);

    auto sf = toStandardForm(m);

    CHECK_THAT(A(sf, 0, 0), WithinAbs( 1.0, kTol)); // x⁺ coefficient
    CHECK_THAT(A(sf, 0, 1), WithinAbs(+1.0, kTol)); // slack (+1 for LessEq)
    CHECK_THAT(A(sf, 0, 2), WithinAbs(-1.0, kTol)); // x⁻ = −(x⁺ coeff)
    CHECK_THAT(sf.b[0],     WithinAbs( 4.0, kTol));
    CHECK(sf.rowNegated[0] == false);
}
