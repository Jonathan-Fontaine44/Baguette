#include "Extractor.hpp"

#include <cmath>
#include <limits>

#include "baguette/core/Config.hpp"
#include "baguette/core/Sense.hpp"

namespace baguette {

namespace {

/// Compute RHS and objective sensitivity ranges from an optimal tableau.
///
/// RHS ranging: for each model constraint i, the range [lo, hi] of b[i] for
/// which the current basis remains primal feasible.  Found by requiring that
/// x_B + Δ·(B⁻¹ e_i) ≥ 0 for all basic rows, where B⁻¹ e_i is read from
/// the slack/surplus column of row i (or the artificial column for Equal rows).
///
/// Objective ranging: for each original variable j, the range [lo, hi] of
/// c[j] for which the current basis remains dual feasible.
///   - Non-basic j: only rc[j] changes → rc[j] + Δc'_sf ≥ 0 → Δc'_sf ≥ −rc[j].
///   - Basic j in row r: rc[k] -= Δc'_sf · tab[r,k] for each non-basic k →
///     dual ratio test over all non-basic columns.
///
/// All ranges are expressed as actual parameter values (not deltas).
/// ±infinity entries indicate the bound is unlimited in that direction.
///
/// @note Complexity: O(m · n_eff) where m = tab.m (standard-form rows) and
///   n_eff is the number of active columns (sf.nCols on the primal path,
///   tab.n on the dual path).  RHS ranging: O(m · nOrigRows).
///   Objective ranging: O(nOrig · (m + n_eff)).  Dominant term: O(m · n_eff).
SensitivityResult extractSensitivity(const internal::SimplexTableau& tab,
                                      const internal::LPStandardForm& sf,
                                      const Model&                    model,
                                      const std::vector<uint32_t>&    equalArtCol) {
    using Lim = std::numeric_limits<double>;
    const bool   maximize    = (model.getObjSense() == ObjSense::Maximize);
    const auto&  constraints = model.getLPConstraints();
    const auto&  objCoeffs   = model.getHot().obj;
    const double inf         = Lim::infinity();

    const std::size_t m  = tab.m;
    const std::size_t n  = tab.n;
    const std::size_t np = n + 1; // row stride in tab.tab

    // Effective active columns: original SF cols only (excludes artificials on the
    // primal path, where tab.nActive == sf.nCols < tab.n).
    // On the dual path there are no artificials, so tab.nActive == 0 and we use tab.n.
    const std::size_t nEff = (tab.nActive > 0) ? tab.nActive : n;

    // Build a fast basic-column lookup over [0, nEff).
    std::vector<bool> isBasic(nEff, false);
    for (std::size_t r = 0; r < m; ++r)
        if (tab.basicCols[r] < nEff)
            isBasic[tab.basicCols[r]] = true;

    SensitivityResult sens;

    // ── RHS ranging ──────────────────────────────────────────────────────────────
    // For each model constraint i, find the standard-form delta interval [Δlo, Δhi]
    // such that all basic solution values remain non-negative, then translate to the
    // model b[i] value range.
    sens.rhsRange.resize(sf.nOrigRows);

    for (std::size_t i = 0; i < sf.nOrigRows; ++i) {
        uint32_t colI    = sf.rowSlackCol[i];
        double   dirSign = 1.0;

        // sf.rowSlackCol[i] == sf.nCols is the sentinel for Equal rows (no slack/surplus).
        if (colI < static_cast<uint32_t>(sf.nCols)) {
            // Slack (LessEq) or surplus (GreaterEq) column.
            const bool negated = sf.rowNegated[i];
            if (constraints[i].sense == Sense::LessEq)
                dirSign = negated ? -1.0 : +1.0;
            else // GreaterEq
                dirSign = negated ? +1.0 : -1.0;
        } else {
            // Equal row: use the artificial column kept for dual extraction.
            if (equalArtCol.empty() || equalArtCol[i] >= static_cast<uint32_t>(n)) {
                sens.rhsRange[i] = {-inf, +inf};
                continue;
            }
            colI    = equalArtCol[i];
            dirSign = 1.0;
        }

        // Ratio test: largest Δ_sf interval such that x_B + Δ·d ≥ 0 for all rows.
        double deltaLo = -inf, deltaHi = +inf;
        for (std::size_t r = 0; r < m; ++r) {
            const double xBr = tab.tab[r * np + n];
            const double dr  = dirSign * tab.tab[r * np + colI];
            if (dr > baguette::pivot_tol)
                deltaLo = std::max(deltaLo, -xBr / dr);
            else if (dr < -baguette::pivot_tol)
                deltaHi = std::min(deltaHi, -xBr / dr);
        }

        // Translate from standard-form Δ to model-space b[i] range.
        const double modelRHS = constraints[i].rhs;
        if (!sf.rowNegated[i])
            sens.rhsRange[i] = {modelRHS + deltaLo, modelRHS + deltaHi};
        else
            sens.rhsRange[i] = {modelRHS - deltaHi, modelRHS - deltaLo};
    }

    // ── Objective ranging ─────────────────────────────────────────────────────────
    // For each original variable j, find the standard-form delta interval [Δlo, Δhi]
    // such that all reduced costs remain non-negative, then translate to c[j] range.
    sens.objRange.resize(sf.nOrig);

    for (std::size_t j = 0; j < sf.nOrig; ++j) {
        // Free variables are split as x⁺ − x⁻; coupled analysis not implemented.
        if (sf.varFreeNegCol[j] < static_cast<uint32_t>(sf.nCols)) {
            sens.objRange[j] = {-inf, +inf};
            continue;
        }

        // Conversion factor between the standard-form coefficient c'_j and the
        // model coefficient c_model[j]:  c'_j = factor · c_model[j].
        const double factor = static_cast<double>(sf.varColSign[j]) * (maximize ? -1.0 : 1.0);
        const double cModel = objCoeffs[j];

        double deltaLoSF = -inf, deltaHiSF = +inf;

        if (!isBasic[j]) {
            // Non-basic: only rc[j] is affected by changing c'_j.
            deltaLoSF = -tab.rc[j];
        } else {
            // Basic in row r: changing c'_j shifts all non-basic reduced costs.
            std::size_t r = 0;
            while (r < m && tab.basicCols[r] != static_cast<uint32_t>(j)) ++r;

            for (std::size_t k = 0; k < nEff; ++k) {
                if (isBasic[k]) continue;
                const double t   = tab.tab[r * np + k];
                const double rck = tab.rc[k];
                if (t > baguette::pivot_tol)
                    deltaHiSF = std::min(deltaHiSF, rck / t);
                else if (t < -baguette::pivot_tol)
                    deltaLoSF = std::max(deltaLoSF, rck / t);
            }
        }

        // Convert from standard-form Δ range to model coefficient range.
        double lo, hi;
        if (factor > 0.0) {
            lo = cModel + deltaLoSF;
            hi = cModel + deltaHiSF;
        } else {
            lo = cModel - deltaHiSF;
            hi = cModel - deltaLoSF;
        }
        sens.objRange[j] = {lo, hi};
    }

    return sens;
}

} // anonymous namespace

namespace internal {

/// @note Complexity: O(nOrig + nOrigRows + nCols) when computeSensitivity is
///   false. When true, dominated by extractSensitivity at O(m·n_eff).
LPDetailedResult extractDetailed(const SimplexTableau&        tab,
                                  const LPStandardForm&        sf,
                                  const Model&                 model,
                                  LPStatus                     status,
                                  const std::vector<uint32_t>& equalArtCol,
                                  bool                         computeSensitivity) {
    LPDetailedResult det;
    det.result.status = status;

    const std::size_t nOrig = sf.nOrig;
    const bool maximize     = (model.getObjSense() == ObjSense::Maximize);

    // Primal: un-shift using varShiftVal and varColSign.
    //   lb-shift:  x_j = varShiftVal[j] + x'_j
    //   ub-shift:  x_j = varShiftVal[j] − x'_j
    //   free-split: x_j = x⁺_j − x⁻_j  (varShiftVal=0, varColSign=+1,
    //               x⁻ column at sf.varFreeNegCol[j])
    std::vector<double> xPrime = tab.primalSolution();
    det.result.primalValues.resize(nOrig);
    for (std::size_t j = 0; j < nOrig; ++j) {
        double val = sf.varShiftVal[j] + sf.varColSign[j] * xPrime[j];
        uint32_t negCol = sf.varFreeNegCol[j];
        if (negCol < sf.nCols)          // fully free variable
            val -= xPrime[negCol];
        det.result.primalValues[j] = val;
    }

    // Objective: add back the shift offset and flip sign for Maximize
    double obj = tab.objectiveValue() + sf.objOffset;
    det.result.objectiveValue = maximize ? -obj : obj;

    if (status != LPStatus::Optimal)
        return det;

    // Dual variables
    const auto& constraints = model.getLPConstraints();
    det.dualValues.resize(sf.nOrigRows);
    for (std::size_t i = 0; i < sf.nOrigRows; ++i) {
        uint32_t slackCol = sf.rowSlackCol[i];
        double raw = tab.rc[slackCol];

        double sign = 0.0;
        if (constraints[i].sense == Sense::Equal) {
            if (!equalArtCol.empty() && equalArtCol[i] < tab.n) {
                raw  = tab.rc[equalArtCol[i]];
                sign = -1.0;
                if (sf.rowNegated[i]) sign = -sign;
                if (maximize)         sign = -sign;
                det.dualValues[i] = sign * raw;
            } else {
                det.dualValues[i] = 0.0; // dual simplex path: no artificial kept
            }
            continue;
        } else if (constraints[i].sense == Sense::LessEq) {
            sign = -1.0;
        } else { // GreaterEq
            sign = +1.0;
        }
        if (sf.rowNegated[i]) sign = -sign;
        if (maximize)         sign = -sign;
        det.dualValues[i] = sign * raw;
    }

    // Reduced costs for original variables.
    det.reducedCosts.resize(nOrig);
    for (std::size_t j = 0; j < nOrig; ++j)
        det.reducedCosts[j] = sf.varColSign[j] * (maximize ? -tab.rc[j] : tab.rc[j]);

    // Basis record (in terms of the original sf column space, no artificials)
    det.basis.basicCols.assign(tab.basicCols.begin(), tab.basicCols.end());
    det.basis.colKind   = sf.colKind;
    det.basis.colOrigin = sf.colOrigin;

    if (computeSensitivity)
        det.sensitivity = extractSensitivity(tab, sf, model, equalArtCol);

    return det;
}

} // namespace internal
} // namespace baguette
