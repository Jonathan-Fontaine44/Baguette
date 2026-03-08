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

    // ── Determine shift type and validate bounds ─────────────────────────────
    // lb-shift (lb finite):         x' = x − lb, colSign = +1
    // ub-shift (lb = −∞, ub finite): x' = ub − x, colSign = −1
    // fully free (both non-finite):  not supported
    std::vector<double> varShiftVal(nOrig);
    std::vector<int8_t> varColSign(nOrig);
    for (std::size_t j = 0; j < nOrig; ++j) {
        if (std::isfinite(hot.lb[j])) {
            varShiftVal[j] = hot.lb[j];
            varColSign[j]  = +1;
        } else if (std::isfinite(hot.ub[j])) {
            varShiftVal[j] = hot.ub[j];
            varColSign[j]  = -1;
        } else {
            throw std::invalid_argument(
                "toStandardForm: fully free variables (lb = -inf, ub = +inf) are not yet supported");
        }
    }

    // ── Count upper-bound rows (only for lb-shifted vars with finite ub) ────
    // ub-shifted vars are naturally bounded: x' = ub − x ≥ 0, no UB row needed.
    std::size_t nUBRows = 0;
    for (std::size_t j = 0; j < nOrig; ++j) {
        if (varColSign[j] == +1 && std::isfinite(hot.ub[j]))
            ++nUBRows;
    }

    const std::size_t nRows  = nOrigRows + nUBRows;
    const std::size_t nSlack = nOrigRows;          // one slack/surplus per model row
    const std::size_t nUBSlack = nUBRows;
    const std::size_t nCols  = nOrig + nSlack + nUBSlack;

    LPStandardForm sf;
    sf.nRows     = nRows;
    sf.nOrigRows = nOrigRows;
    sf.nCols     = nCols;
    sf.nOrig     = nOrig;
    sf.nSlack    = nSlack;
    sf.varShiftVal = std::move(varShiftVal);
    sf.varColSign  = std::move(varColSign);

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
        sf.colOrigin[j] = static_cast<uint32_t>(j); // Variable::id == index
    }
    for (std::size_t i = 0; i < nSlack; ++i) {
        std::size_t col = nOrig + i;
        sf.colKind[col]   = ColumnKind::Slack;
        sf.colOrigin[col] = static_cast<uint32_t>(i); // constraint index
    }

    // ── Objective vector (lb-shifted, always minimise) ──────────────────────
    const bool maximize = (model.getObjSense() == ObjSense::Maximize);
    const double objSign = maximize ? -1.0 : 1.0;

    // After the variable shift, the objective becomes:
    //   lb-shift (x = shiftVal + x'):  c_j * x = c_j * x' + c_j * shiftVal
    //   ub-shift (x = shiftVal − x'):  c_j * x = −c_j * x' + c_j * shiftVal
    // In both cases: tableau coefficient = colSign * c_j, offset += c_j * shiftVal.
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

        // Substitute the shifted variable into each constraint term:
        //   lb-shift: a * x = a * x' + a * shiftVal  → coeff = +a, rhs -= a * shiftVal
        //   ub-shift: a * x = −a * x' + a * shiftVal → coeff = −a, rhs -= a * shiftVal
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

        // Slack / surplus column
        switch (con.sense) {
            case Sense::LessEq:
                sf.A[i * nCols + slackCol] = +1.0; // x'_j terms + s = rhs
                break;
            case Sense::GreaterEq:
                sf.A[i * nCols + slackCol] = -1.0; // x'_j terms − s = rhs
                break;
            case Sense::Equal:
                // No slack; column stays 0 (artificial added in phase I)
                break;
        }

        // Normalise: ensure b[i] >= 0
        if (rhs < 0.0) {
            for (std::size_t j = 0; j < nCols; ++j)
                sf.A[i * nCols + j] = -sf.A[i * nCols + j];
            rhs = -rhs;
            sf.rowNegated[i] = true;
        }
        sf.b[i] = rhs;
    }

    // ── Upper-bound rows (lb-shifted vars with finite ub only) ──────────────
    std::size_t ubRow = nOrigRows;
    std::size_t ubSlackCol = nOrig + nSlack;
    for (std::size_t j = 0; j < nOrig; ++j) {
        if (sf.varColSign[j] != +1 || !std::isfinite(hot.ub[j]))
            continue;

        // Row: x'_j + s_ub = ub_j − lb_j  (lb_j = varShiftVal[j])
        sf.A[ubRow * nCols + j]          = 1.0;
        sf.A[ubRow * nCols + ubSlackCol] = 1.0;
        sf.b[ubRow]                      = hot.ub[j] - sf.varShiftVal[j];

        sf.colKind[ubSlackCol]   = ColumnKind::UpperSlack;
        sf.colOrigin[ubSlackCol] = static_cast<uint32_t>(j);

        ++ubRow;
        ++ubSlackCol;
    }

    return sf;
}

} // namespace baguette::internal
