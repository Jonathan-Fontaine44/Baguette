#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "baguette/lp/LPResult.hpp"
#include "baguette/model/Model.hpp"

namespace baguette::internal {

/// Internal standard-form LP: minimise c^T x  subject to  A x = b,  x ≥ 0.
///
/// Built by toStandardForm() from a user-facing Model.  Never exposed in a
/// public header — callers only see LPResult / LPDetailedResult.
///
/// Each original variable is shifted to a non-negative auxiliary x'_j ≥ 0:
///   - lb-shift (lb finite):  x'_j = x_j − lb_j  → x_j = lb_j + x'_j
///   - ub-shift (lb = −∞, ub finite): x'_j = ub_j − x_j → x_j = ub_j − x'_j
///   - free-split (lb = −∞, ub = +∞): x_j = x⁺_j − x⁻_j, both x⁺_j, x⁻_j ≥ 0.
///
/// Column ordering (contiguous blocks):
///   [0 .. nOrig-1]                  shifted original variables (x⁺ for free-split vars)
///   [nOrig .. nOrig+nSlack-1]       slack / surplus variables (one per LessEq / GreaterEq row; Equal rows have none)
///   [nOrig+nSlack .. nOrig+nSlack+nUBSlack-1]  upper-bound slacks  x'_j + s = ub_j − lb_j
///   [nOrig+nSlack+nUBSlack .. nCols-1]          free-split negative parts x⁻_j
///                                               (one per fully free variable)
///
/// Row ordering (contiguous blocks):
///   [0 .. nOrigRows-1]              converted model constraints
///   [nOrigRows .. nRows-1]          upper-bound rows (one per lb-shifted finite-ub variable)
struct LPStandardForm {
    std::size_t nRows;      ///< Total rows = nOrigRows + nUBRows.
    std::size_t nOrigRows;  ///< Rows from model constraints.
    std::size_t nCols;      ///< Total columns.
    std::size_t nOrig;      ///< Number of shifted original variables.
    std::size_t nSlack;     ///< Number of slack / surplus variables (Equal rows excluded).

    /// Dense constraint matrix, row-major: (*A)[i * nCols + j].
    std::shared_ptr<std::vector<double>> A;

    /// RHS vector b, length nRows.  All entries ≥ 0 after normalisation
    /// (rows where the original rhs < 0 are multiplied by −1).
    std::vector<double> b;

    /// Objective vector c, length nCols.
    /// Always a minimisation objective (Maximize is handled by negating c).
    std::vector<double> c;

    /// Constant added back to the objective value when reporting results.
    /// Equals sum(obj_j * lb_j) due to the lb-shift, adjusted for Maximize.
    double objOffset = 0.0;

    /// Kind of each column (length nCols).  Used to populate BasisRecord.
    std::vector<ColumnKind> colKind;

    /// Origin index of each column (length nCols).
    ///   ColumnKind::Original   → Variable::id of the model variable.
    ///   ColumnKind::Slack      → index of the model constraint.
    ///   ColumnKind::UpperSlack → Variable::id of the bounded variable.
    std::vector<uint32_t> colOrigin;

    /// For each model constraint row i, the column index of its slack/surplus.
    /// Equal rows have no slack; for those, rowSlackCol[i] == nCols (sentinel).
    /// Used when extracting dual variables from the tableau's reduced-cost row.
    std::vector<uint32_t> rowSlackCol;

    /// For each model constraint row i, true if the row was negated during
    /// normalisation (original rhs < 0).  Needed to sign-correct dual values.
    std::vector<bool> rowNegated;

    /// Per-variable shift value (length nOrig).
    ///   lb-shift: varShiftVal[j] = lb_j   →  x_j = varShiftVal[j] + x'_j
    ///   ub-shift: varShiftVal[j] = ub_j   →  x_j = varShiftVal[j] − x'_j
    ///   free-split: varShiftVal[j] = 0.0  →  x_j = x⁺_j − x⁻_j
    std::vector<double> varShiftVal;

    /// Per-variable column sign (length nOrig): +1 for lb-shift or free-split, −1 for ub-shift.
    /// Applied to all A entries and the objective coefficient for that variable.
    std::vector<int8_t> varColSign;

    /// For each original variable j: column index of its x⁻ part if it is a fully
    /// free variable (lb = −∞, ub = +∞), or nCols if not free.
    /// Length nOrig.
    std::vector<uint32_t> varFreeNegCol;
};

/// Convert a Model into standard form.
///
/// Variable bounds are handled as follows:
///   - lb shift:  x'_j = x_j − lb_j  so that x'_j ≥ 0.
///   - finite ub: an extra row  x'_j + s = ub_j − lb_j  is appended.
///   - infinite ub (std::numeric_limits<double>::infinity()): no extra row.
///
/// Constraint senses:
///   - LessEq   → slack s ≥ 0 added;    row:  lhs_terms + s = rhs
///   - GreaterEq → surplus s ≥ 0 added; row:  lhs_terms − s = rhs
///   - Equal    → no slack/surplus
///
/// Integer and Binary variables are treated as continuous (LP relaxation).
///
/// @throws std::invalid_argument if any variable ID in a constraint or in the
///         objective exceeds Model::numVars().
LPStandardForm toStandardForm(const Model& model);

/// Lightweight bounds-only update of an existing LPStandardForm.
///
/// Recomputes b, varShiftVal, and objOffset from the updated variable bounds
/// in @p model.  A, c, colKind, colOrigin, rowSlackCol, rowNegated, varColSign,
/// and varFreeNegCol are assumed unchanged and are NOT touched.
///
/// Returns true if the update succeeded.  Returns false if:
///   - The number of variables or constraints changed.
///   - Any variable's shift type changed (e.g. a bound crossed finite/infinite).
///   - Any constraint row's shifted RHS would change sign (requiring a row
///     negation in A that this function does not perform).
/// The caller must fall back to toStandardForm() when false is returned.
bool toStandardFormBoundsOnly(LPStandardForm& sf, const Model& model);

/// Build the LP dual of a standard-form LP.
///
/// Given the primal   min  c^T x   s.t.  Ax = b,  x ≥ 0   (m rows, n cols),
/// the dual is        max  b^T y   s.t.  A^T y ≤ c,  y free.
///
/// Each free dual variable y_i is split as y_i = y⁺_i − y⁻_i, and each
/// dual inequality is converted to an equality by adding a slack s_j ≥ 0:
///
///     min  −b^T y⁺ + b^T y⁻
///     s.t.  A^T (y⁺ − y⁻) + s = c,   y⁺, y⁻, s ≥ 0.
///
/// Column layout of the returned SF:
///   [0 .. m-1]          y⁺ variables  (one per primal row)
///   [m .. 2m-1]         y⁻ variables  (one per primal row)
///   [2m .. 2m+n-1]      slack s        (one per primal column / dual constraint)
///
/// The slack columns form a natural initial basis; their rhs is primal.c[j],
/// which is normalised to ≥ 0 by negating the row when primal.c[j] < 0
/// (recorded in rowNegated).
///
/// @note objOffset is always 0: the dual is derived from the pure
///       standard-form coefficients, not from a user Model with variable shifts.
LPStandardForm dualStandardForm(const LPStandardForm& primal);

} // namespace baguette::internal
