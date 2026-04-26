#include "StandardForm.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace baguette::internal {

LPStandardForm toStandardForm(const Model& model) {
    const auto& hot         = model.getHot();
    const auto& constraints = model.getLPConstraints();
    const std::size_t nOrig     = model.numVars();
    const std::size_t nOrigRows = model.numConstraints();

    // ── Determine shift type ─────────────────────────────────────────────────
    // lb-shift (lb finite):          x' = x − lb,    colSign = +1
    // ub-shift (lb = −∞, ub finite): x' = ub − x,    colSign = −1
    // free-split (both non-finite):  x = x⁺ − x⁻,   colSign = +1, shiftVal = 0
    std::vector<double> varShiftVal(nOrig);
    std::vector<int8_t> varColSign(nOrig);
    std::size_t nFree = 0; // fully free variables needing a split column
    for (std::size_t j = 0; j < nOrig; ++j) {
        if (std::isfinite(hot.lb[j])) {
            varShiftVal[j] = hot.lb[j];
            varColSign[j]  = +1;
        } else if (std::isfinite(hot.ub[j])) {
            varShiftVal[j] = hot.ub[j];
            varColSign[j]  = -1;
        } else {
            // Fully free: treat x⁺ like a lb-shifted var with lb = 0.
            varShiftVal[j] = 0.0;
            varColSign[j]  = +1;
            ++nFree;
        }
    }

    // ── Count upper-bound rows (only for lb-shifted vars with finite ub) ────
    // ub-shifted vars are naturally bounded: x' = ub − x ≥ 0, no UB row needed.
    // free-split vars have ub = +inf, so no UB row needed either.
    std::size_t nUBRows = 0;
    for (std::size_t j = 0; j < nOrig; ++j) {
        if (varColSign[j] == +1 && std::isfinite(hot.ub[j]))
            ++nUBRows;
    }

    const std::size_t nRows    = nOrigRows + nUBRows;
    // Equal rows have no slack/surplus; only LessEq and GEQ get one.
    std::size_t nSlack = 0;
    for (std::size_t i = 0; i < nOrigRows; ++i)
        if (constraints[i].sense != Sense::Equal)
            ++nSlack;
    const std::size_t nUBSlack = nUBRows;
    const std::size_t nCols    = nOrig + nSlack + nUBSlack + nFree;

    LPStandardForm sf;
    sf.nRows     = nRows;
    sf.nOrigRows = nOrigRows;
    sf.nCols     = nCols;
    sf.nOrig     = nOrig;
    sf.nSlack    = nSlack;
    sf.varShiftVal   = std::move(varShiftVal);
    sf.varColSign    = std::move(varColSign);
    sf.varFreeNegCol.assign(nOrig, static_cast<uint32_t>(nCols)); // default: not free

    sf.A = std::make_shared<std::vector<double>>(nRows * nCols, 0.0);
    sf.b.resize(nRows, 0.0);
    sf.c.resize(nCols, 0.0);
    sf.colKind.resize(nCols);
    sf.colOrigin.resize(nCols);
    sf.rowSlackCol.resize(nOrigRows);
    sf.rowNegated.resize(nOrigRows, false);

    // ── Column metadata ─────────────────────────────────────────────────────
    for (std::size_t j = 0; j < nOrig; ++j) {
        sf.colKind[j]   = ColumnKind::Original;
        sf.colOrigin[j] = static_cast<uint32_t>(j);
    }
    {
        std::size_t slackIdx = 0;
        for (std::size_t i = 0; i < nOrigRows; ++i) {
            if (constraints[i].sense == Sense::Equal) continue;
            std::size_t col = nOrig + slackIdx;
            sf.colKind[col]   = ColumnKind::Slack;
            sf.colOrigin[col] = static_cast<uint32_t>(i);
            ++slackIdx;
        }
    }

    // ── Objective vector (lb-shifted, always minimise) ──────────────────────
    const bool maximize = (model.getObjSense() == ObjSense::Maximize);
    const double objSign = maximize ? -1.0 : 1.0;

    sf.objOffset = objSign * model.getObjConstant();
    for (std::size_t j = 0; j < nOrig; ++j) {
        double cj = objSign * hot.obj[j];
        sf.c[j]       = sf.varColSign[j] * cj;
        sf.objOffset += cj * sf.varShiftVal[j];
    }

    // ── Model constraint rows ───────────────────────────────────────────────
    // rowSlackCol[i] = nCols (sentinel) for Equal rows (no slack column).
    // All consumers (extractDetailed, buildDualBasis, buildPhaseOne) skip
    // Equal rows before dereferencing rowSlackCol, so the sentinel is safe.
    std::size_t slackColIdx = 0;
    for (std::size_t i = 0; i < nOrigRows; ++i) {
        const auto& con = constraints[i];
        const std::size_t slackCol = (con.sense != Sense::Equal)
            ? nOrig + slackColIdx
            : sf.nCols; // sentinel: Equal rows have no slack
        sf.rowSlackCol[i] = static_cast<uint32_t>(slackCol);

        double rhs = con.rhs;
        for (std::size_t k = 0; k < con.lhs.size(); ++k) {
            uint32_t varId = con.lhs.varIds[k];
            if (varId >= nOrig)
                throw std::invalid_argument(
                    "toStandardForm: variable ID out of range in constraint");
            double aij = con.lhs.coeffs[k];
            (*sf.A)[i * nCols + varId] += sf.varColSign[varId] * aij;
            rhs -= aij * sf.varShiftVal[varId];
        }

        switch (con.sense) {
            case Sense::LessEq:
                (*sf.A)[i * nCols + slackCol] = +1.0;
                ++slackColIdx;
                break;
            case Sense::GreaterEq:
                (*sf.A)[i * nCols + slackCol] = -1.0;
                ++slackColIdx;
                break;
            case Sense::Equal:
                break;
        }

        if (rhs < 0.0) {
            for (std::size_t j = 0; j < nCols; ++j)
                (*sf.A)[i * nCols + j] = -(*sf.A)[i * nCols + j];
            rhs = -rhs;
            sf.rowNegated[i] = true;
        }
        sf.b[i] = rhs;
    }

    // ── Upper-bound rows (lb-shifted vars with finite ub only) ──────────────
    std::size_t ubRow     = nOrigRows;
    std::size_t ubSlackCol = nOrig + nSlack;
    for (std::size_t j = 0; j < nOrig; ++j) {
        if (sf.varColSign[j] != +1 || !std::isfinite(hot.ub[j]))
            continue;

        (*sf.A)[ubRow * nCols + j]           = 1.0;
        (*sf.A)[ubRow * nCols + ubSlackCol]  = 1.0;
        sf.b[ubRow]                       = hot.ub[j] - sf.varShiftVal[j];

        sf.colKind[ubSlackCol]   = ColumnKind::UpperSlack;
        sf.colOrigin[ubSlackCol] = static_cast<uint32_t>(j);

        ++ubRow;
        ++ubSlackCol;
    }

    // ── Free-split negative columns (fully free variables) ──────────────────
    // Column layout: [nOrig+nSlack+nUBSlack .. nOrig+nSlack+nUBSlack+nFree-1]
    // For variable j with x_j = x⁺_j − x⁻_j:
    //   Objective:   c[negCol] = −c[j]   (x⁻ enters with opposite sign)
    //   Constraints: A[i][negCol] = −A[i][j]  (before any row negation)
    //   The row-negation loop above has already been applied to model rows, so
    //   we must replicate the negation here.
    std::size_t negCol = nOrig + nSlack + nUBSlack;
    for (std::size_t j = 0; j < nOrig; ++j) {
        if (std::isfinite(hot.lb[j]) || std::isfinite(hot.ub[j]))
            continue; // not fully free

        sf.varFreeNegCol[j] = static_cast<uint32_t>(negCol);
        sf.colKind[negCol]   = ColumnKind::FreeNeg;
        sf.colOrigin[negCol] = static_cast<uint32_t>(j);

        // Objective: negate x⁺ coefficient
        sf.c[negCol] = -sf.c[j];

        // Model constraint rows: mirror x⁺ coefficient with opposite sign,
        // respecting the row negation already applied.
        for (std::size_t i = 0; i < nOrigRows; ++i)
            (*sf.A)[i * nCols + negCol] = -(*sf.A)[i * nCols + j];

        ++negCol;
    }

    return sf;
}

LPStandardForm dualStandardForm(const LPStandardForm& primal) {
    // Primal: min c^T x,  Ax = b,  x ≥ 0   (m rows, n cols)
    // Dual:   max b^T y,  A^T y ≤ c,  y free  (m vars, n constraints)
    //
    // Standard form of the dual (min, equality, non-negative):
    //   Split y_i = y⁺_i − y⁻_i  (y⁺, y⁻ ≥ 0)
    //   Add slack s_j ≥ 0 per dual constraint j.
    //
    //   min  −b^T y⁺ + b^T y⁻
    //   s.t. A^T (y⁺ − y⁻) + s = c   (n rows, one per primal column)
    //        y⁺, y⁻, s ≥ 0

    const std::size_t m    = primal.nRows;
    const std::size_t n    = primal.nCols;
    const std::size_t dCols = 2 * m + n;

    LPStandardForm dual;
    dual.nRows     = n;
    dual.nOrigRows = n;
    dual.nCols     = dCols;
    dual.nOrig     = 2 * m;  // y⁺ and y⁻ treated as "original" columns
    dual.nSlack    = n;      // one slack per dual constraint row

    dual.A = std::make_shared<std::vector<double>>(n * dCols, 0.0);
    dual.b.resize(n);
    dual.c.resize(dCols, 0.0);
    dual.colKind.resize(dCols);
    dual.colOrigin.resize(dCols);
    dual.rowSlackCol.resize(n);
    dual.rowNegated.resize(n, false);

    // varShiftVal / varColSign / varFreeNegCol: the y⁺/y⁻ variables are
    // already non-negative (no shift, sign = +1, not free-split).
    dual.varShiftVal.assign(2 * m, 0.0);
    dual.varColSign.assign(2 * m, static_cast<int8_t>(+1));
    dual.varFreeNegCol.assign(2 * m, static_cast<uint32_t>(dCols)); // sentinel: not free

    // Fill constraint matrix rows j = 0..n-1
    //   A_dual[j, i]       = A_primal[i, j]   (y⁺_i coefficient)
    //   A_dual[j, m+i]     = −A_primal[i, j]  (y⁻_i coefficient)
    //   A_dual[j, 2m+j]    = 1.0              (slack s_j)
    //   b_dual[j]          = c_primal[j]       (rhs of dual constraint j)
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < m; ++i) {
            double aij = (*primal.A)[i * n + j];
            (*dual.A)[j * dCols + i]       =  aij;
            (*dual.A)[j * dCols + m + i]   = -aij;
        }
        (*dual.A)[j * dCols + 2 * m + j] = 1.0;
        dual.b[j] = primal.c[j];
        dual.rowSlackCol[j] = static_cast<uint32_t>(2 * m + j);
    }

    // Objective: min −b_primal^T y⁺ + b_primal^T y⁻
    for (std::size_t i = 0; i < m; ++i) {
        dual.c[i]       = -primal.b[i];  // y⁺_i
        dual.c[m + i]   = +primal.b[i];  // y⁻_i
    }
    // c[2m..2m+n-1] = 0.0  (slacks have zero cost)

    // Normalise: b_dual[j] must be ≥ 0 for the Tableau initialisation.
    // Negate any row where b_dual[j] = primal.c[j] < 0.
    for (std::size_t j = 0; j < n; ++j) {
        if (dual.b[j] < 0.0) {
            for (std::size_t k = 0; k < dCols; ++k)
                (*dual.A)[j * dCols + k] = -(*dual.A)[j * dCols + k];
            dual.b[j]          = -dual.b[j];
            dual.rowNegated[j] = true;
        }
    }

    // Column metadata
    for (std::size_t i = 0; i < m; ++i) {
        // y⁺_i: maps to primal row i
        dual.colKind[i]       = ColumnKind::Original;
        dual.colOrigin[i]     = static_cast<uint32_t>(i);
        // y⁻_i: also maps to primal row i (negative counterpart)
        dual.colKind[m + i]   = ColumnKind::Original;
        dual.colOrigin[m + i] = static_cast<uint32_t>(i);
    }
    for (std::size_t j = 0; j < n; ++j) {
        dual.colKind[2 * m + j]   = ColumnKind::Slack;
        dual.colOrigin[2 * m + j] = static_cast<uint32_t>(j);
    }

    return dual;
}

bool toStandardFormBoundsOnly(LPStandardForm& sf, const Model& model) {
    const auto& hot         = model.getHot();
    const auto& constraints = model.getLPConstraints();
    const std::size_t nOrig     = model.numVars();
    const std::size_t nOrigRows = model.numConstraints();

    if (nOrig != sf.nOrig || nOrigRows != sf.nOrigRows)
        return false;

    // Verify that each variable's shift type is unchanged.
    for (std::size_t j = 0; j < nOrig; ++j) {
        const bool lbFin   = std::isfinite(hot.lb[j]);
        const bool ubFin   = std::isfinite(hot.ub[j]);
        const bool wasFree   = (sf.varFreeNegCol[j] < sf.nCols);
        const bool wasUbShift = (sf.varColSign[j] == -1);
        const bool wasLbShift = !wasFree && !wasUbShift;

        if (lbFin  && !wasLbShift)           return false;
        if (!lbFin && ubFin  && !wasUbShift) return false;
        if (!lbFin && !ubFin && !wasFree)    return false;
    }

    // Update varShiftVal.
    for (std::size_t j = 0; j < nOrig; ++j) {
        if (sf.varColSign[j] == +1 && sf.varFreeNegCol[j] == sf.nCols)
            sf.varShiftVal[j] = hot.lb[j];   // lb-shift
        else if (sf.varColSign[j] == -1)
            sf.varShiftVal[j] = hot.ub[j];   // ub-shift
        // free-split: varShiftVal[j] remains 0.0
    }

    // Update objOffset.
    const bool   maximize = (model.getObjSense() == ObjSense::Maximize);
    const double objSign  = maximize ? -1.0 : 1.0;
    sf.objOffset = objSign * model.getObjConstant();
    for (std::size_t j = 0; j < nOrig; ++j)
        sf.objOffset += objSign * hot.obj[j] * sf.varShiftVal[j];

    // Recompute b for model constraint rows.
    // Reject if any row's sign would need to flip (A row negation not performed here).
    for (std::size_t i = 0; i < nOrigRows; ++i) {
        const auto& con = constraints[i];
        double rhs = con.rhs;
        for (std::size_t k = 0; k < con.lhs.size(); ++k)
            rhs -= con.lhs.coeffs[k] * sf.varShiftVal[con.lhs.varIds[k]];

        const bool needsNeg = (rhs < 0.0);
        if (needsNeg != sf.rowNegated[i])
            return false; // sign flip would require updating A — fall back
        sf.b[i] = needsNeg ? -rhs : rhs;
    }

    // Recompute b for upper-bound rows.
    std::size_t ubRow = nOrigRows;
    for (std::size_t j = 0; j < nOrig; ++j) {
        if (sf.varColSign[j] != +1 || !std::isfinite(hot.ub[j]))
            continue;
        sf.b[ubRow] = hot.ub[j] - sf.varShiftVal[j];
        ++ubRow;
    }

    return true;
}

} // namespace baguette::internal
