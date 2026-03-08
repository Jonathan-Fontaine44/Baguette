#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "baguette/lp/LPResult.hpp"
#include "baguette/model/Model.hpp"

namespace baguette::internal {

/// Internal standard-form LP: minimise c^T x  subject to  A x = b,  x ≥ 0.
///
/// Built by toStandardForm() from a user-facing Model.  Never exposed in a
/// public header — callers only see LPResult / LPDetailedResult.
///
/// Column ordering (contiguous blocks):
///   [0 .. nOrig-1]                  shifted original variables  x'_j = x_j − lb_j
///   [nOrig .. nOrig+nSlack-1]       slack / surplus variables (one per model row)
///   [nOrig+nSlack .. nCols-1]       upper-bound slacks  x'_j + s = ub_j − lb_j
///                                   (one per original variable with finite ub)
///
/// Row ordering (contiguous blocks):
///   [0 .. nOrigRows-1]              converted model constraints
///   [nOrigRows .. nRows-1]          upper-bound rows (one per finite-ub variable)
struct LPStandardForm {
    std::size_t nRows;      ///< Total rows = nOrigRows + nUBRows.
    std::size_t nOrigRows;  ///< Rows from model constraints.
    std::size_t nCols;      ///< Total columns.
    std::size_t nOrig;      ///< Number of shifted original variables.
    std::size_t nSlack;     ///< Number of slack / surplus variables.

    /// Dense constraint matrix, row-major: A[i * nCols + j].
    std::vector<double> A;

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
    /// Used when extracting dual variables from the tableau's reduced-cost row.
    std::vector<uint32_t> rowSlackCol;

    /// For each model constraint row i, true if the row was negated during
    /// normalisation (original rhs < 0).  Needed to sign-correct dual values.
    std::vector<bool> rowNegated;
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

} // namespace baguette::internal
