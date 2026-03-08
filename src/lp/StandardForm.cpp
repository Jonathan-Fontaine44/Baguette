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

    // ── Count upper-bound rows ──────────────────────────────────────────────
    std::size_t nUBRows = 0;
    for (std::size_t j = 0; j < nOrig; ++j) {
        if (std::isfinite(hot.ub[j]))
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

    // After lb-shift x_j = x'_j + lb_j, the original objective becomes:
    //   sum(c_j * x_j) = sum(c_j * x'_j) + sum(c_j * lb_j)
    // The tableau minimises sum(c_j * x'_j); objOffset = sum(c_j * lb_j)
    // is added back in extractDetailed() to recover the original value.
    // For Maximize, c_j = -hot.obj[j], so objOffset = -sum(hot.obj[j]*lb_j),
    // and extractDetailed() negates the total (tableau_obj + objOffset).
    for (std::size_t j = 0; j < nOrig; ++j) {
        double cj = objSign * hot.obj[j];
        sf.c[j]       = cj;
        sf.objOffset += cj * hot.lb[j];
    }

    // ── Model constraint rows ───────────────────────────────────────────────
    for (std::size_t i = 0; i < nOrigRows; ++i) {
        const auto& con = constraints[i];
        const std::size_t slackCol = nOrig + i;
        sf.rowSlackCol[i] = static_cast<uint32_t>(slackCol);

        // Fill original-variable columns (lb-shifted: replace x_j with x'_j + lb_j)
        double rhs = con.rhs;
        for (std::size_t k = 0; k < con.lhs.size(); ++k) {
            uint32_t varId = con.lhs.varIds[k];
            if (varId >= nOrig)
                throw std::invalid_argument(
                    "toStandardForm: variable ID out of range in constraint");
            double aij = con.lhs.coeffs[k];
            sf.A[i * nCols + varId] += aij;
            rhs -= aij * hot.lb[varId]; // absorb the lb shift into rhs
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

    // ── Upper-bound rows ────────────────────────────────────────────────────
    std::size_t ubRow = nOrigRows;
    std::size_t ubSlackCol = nOrig + nSlack;
    for (std::size_t j = 0; j < nOrig; ++j) {
        if (!std::isfinite(hot.ub[j]))
            continue;

        // Row: x'_j + s_ub = ub_j − lb_j
        sf.A[ubRow * nCols + j]          = 1.0;
        sf.A[ubRow * nCols + ubSlackCol] = 1.0;
        sf.b[ubRow]                      = hot.ub[j] - hot.lb[j];

        sf.colKind[ubSlackCol]   = ColumnKind::UpperSlack;
        sf.colOrigin[ubSlackCol] = static_cast<uint32_t>(j);

        ++ubRow;
        ++ubSlackCol;
    }

    return sf;
}

} // namespace baguette::internal
