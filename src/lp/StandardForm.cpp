#include "StandardForm.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace baguette::internal {

LPStandardForm toStandardForm(const Model& model) {
    const auto& hot         = model.getHot();
    const auto& constraints = model.getConstraints();
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
    const std::size_t nSlack   = nOrigRows;   // one slack/surplus per model row
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

    sf.A.assign(nRows * nCols, 0.0);
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
    for (std::size_t i = 0; i < nSlack; ++i) {
        std::size_t col = nOrig + i;
        sf.colKind[col]   = ColumnKind::Slack;
        sf.colOrigin[col] = static_cast<uint32_t>(i);
    }

    // ── Objective vector (lb-shifted, always minimise) ──────────────────────
    const bool maximize = (model.getObjSense() == ObjSense::Maximize);
    const double objSign = maximize ? -1.0 : 1.0;

    for (std::size_t j = 0; j < nOrig; ++j) {
        double cj = objSign * hot.obj[j];
        sf.c[j]       = sf.varColSign[j] * cj;
        sf.objOffset += cj * sf.varShiftVal[j];
    }

    // ── Model constraint rows ───────────────────────────────────────────────
    for (std::size_t i = 0; i < nOrigRows; ++i) {
        const auto& con = constraints[i];
        const std::size_t slackCol = nOrig + i;
        sf.rowSlackCol[i] = static_cast<uint32_t>(slackCol);

        double rhs = con.rhs;
        for (std::size_t k = 0; k < con.lhs.size(); ++k) {
            uint32_t varId = con.lhs.varIds[k];
            if (varId >= nOrig)
                throw std::invalid_argument(
                    "toStandardForm: variable ID out of range in constraint");
            double aij = con.lhs.coeffs[k];
            sf.A[i * nCols + varId] += sf.varColSign[varId] * aij;
            rhs -= aij * sf.varShiftVal[varId];
        }

        switch (con.sense) {
            case Sense::LessEq:
                sf.A[i * nCols + slackCol] = +1.0;
                break;
            case Sense::GreaterEq:
                sf.A[i * nCols + slackCol] = -1.0;
                break;
            case Sense::Equal:
                break;
        }

        if (rhs < 0.0) {
            for (std::size_t j = 0; j < nCols; ++j)
                sf.A[i * nCols + j] = -sf.A[i * nCols + j];
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

        sf.A[ubRow * nCols + j]           = 1.0;
        sf.A[ubRow * nCols + ubSlackCol]  = 1.0;
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
            sf.A[i * nCols + negCol] = -sf.A[i * nCols + j];

        ++negCol;
    }

    return sf;
}

} // namespace baguette::internal
