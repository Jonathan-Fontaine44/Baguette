#include "SimplexTableauBV.hpp"

#include <cassert>
#include <cmath>
#include <limits>

#include "baguette/core/Config.hpp"
#include "baguette/core/Sense.hpp"
#include "baguette/model/ModelEnums.hpp"

namespace baguette::internal {

// ── Construction ──────────────────────────────────────────────────────────────

bool SimplexTableauBV::init(const LPStandardFormBV& sfbv,
                             std::vector<uint32_t>   initialBasis) {
    assert(initialBasis.size() == sfbv.nRows);

    m = sfbv.nRows;
    n = sfbv.nCols;

    tab.resize(m * (n + 1));
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j)
            tab[i * (n + 1) + j] = (*sfbv.A)[i * n + j];
        tab[i * (n + 1) + n] = sfbv.b[i];
    }

    basicCols = std::move(initialBasis);

    // Gauss-Jordan with partial pivoting (same as SimplexTableau::init)
    const std::size_t w = n + 1;
    for (std::size_t i = 0; i < m; ++i) {
        std::size_t col = basicCols[i];

        std::size_t pivotRow = i;
        double      maxAbs   = 0.0;
        for (std::size_t r = i; r < m; ++r) {
            double val = std::abs(tab[r * w + col]);
            if (val > maxAbs) { maxAbs = val; pivotRow = r; }
        }
        if (maxAbs < cfg.pivotTol) return false;

        if (pivotRow != i)
            for (std::size_t j = 0; j <= n; ++j)
                std::swap(tab[i * w + j], tab[pivotRow * w + j]);

        double inv = 1.0 / tab[i * w + col];
        for (std::size_t j = 0; j <= n; ++j)
            tab[i * w + j] *= inv;

        for (std::size_t r = 0; r < m; ++r) {
            if (r == i) continue;
            double factor = tab[r * w + col];
            if (factor == 0.0) continue;
            for (std::size_t j = 0; j <= n; ++j)
                tab[r * w + j] -= factor * tab[i * w + j];
        }
    }

    // Price objective row
    rc.assign(n + 1, 0.0);
    for (std::size_t j = 0; j < n; ++j)
        rc[j] = sfbv.c[j];
    for (std::size_t i = 0; i < m; ++i) {
        double cb = sfbv.c[basicCols[i]];
        if (cb == 0.0) continue;
        for (std::size_t j = 0; j <= n; ++j)
            rc[j] -= cb * tab[i * (n + 1) + j];
    }

    colUB = sfbv.colUB;
    atUB.assign(n, false);
    return true;
}

bool SimplexTableauBV::reinvert(const LPStandardFormBV& sfbv) {
    // Save complement state before init() resets it
    std::vector<bool> savedAtUB = atUB;
    if (!init(sfbv, std::move(basicCols))) return false;
    // Restore complement state for non-basic AT_UB variables
    for (std::size_t j = 0; j < n; ++j)
        if (savedAtUB[j]) complement(j);
    return true;
}

// ── Complement ────────────────────────────────────────────────────────────────

void SimplexTableauBV::complement(std::size_t j) {
    const std::size_t w  = n + 1;
    const double      uj = colUB[j];
    // Negate column j and update RHS using the new (negated) value
    for (std::size_t r = 0; r <= m; ++r) {
        double* row = (r < m) ? &tab[r * w] : rc.data();
        row[j]  = -row[j];
        row[n] += uj * row[j]; // += uj * new_col[j]
    }
    atUB[j] = !atUB[j];
}

// ── Simplex operations ────────────────────────────────────────────────────────

SimplexTableauBV::DualLeavingResult SimplexTableauBV::selectLeavingDualBV() const {
    const std::size_t w   = n + 1;
    std::size_t bestRow   = m;
    uint32_t    bestIdx   = std::numeric_limits<uint32_t>::max();
    bool        bestExitsToUB = false;

    // Bland's rule: select the infeasible basic variable with the smallest
    // column index. Guarantees finite termination on degenerate LPs.
    for (std::size_t i = 0; i < m; ++i) {
        const double bi  = tab[i * w + n];
        const double ubi = colUB[basicCols[i]];

        const bool infeasL = (bi < -cfg.feasibilityTol);
        const bool infeasU = (std::isfinite(ubi) && bi > ubi + cfg.feasibilityTol);

        if (!infeasL && !infeasU) continue;

        const uint32_t idx = basicCols[i];
        if (idx < bestIdx) {
            bestIdx       = idx;
            bestRow       = i;
            bestExitsToUB = (infeasU && (!infeasL || bi - ubi > -bi));
        }
    }
    return {bestRow, bestExitsToUB};
}

std::size_t SimplexTableauBV::selectEnteringDualBV(std::size_t leavingRow,
                                                    bool        exitsToUB) const {
    const std::size_t w     = n + 1;
    const std::size_t limit = (nActive > 0) ? nActive : n;
    double      minRatio = std::numeric_limits<double>::infinity();
    std::size_t entering = n; // sentinel: infeasible

    const uint32_t currentBasic = basicCols[leavingRow];

    for (std::size_t j = 0; j < limit; ++j) {
        if (j == currentBasic) continue; // never enter the variable that is already basic here

        const double eta = tab[leavingRow * w + j];
        double ratio;

        if (!exitsToUB) {
            // xBi < LB → wants to increase → need eta_ij < 0
            if (eta >= -cfg.pivotTol) continue;
            ratio = rc[j] / (-eta);
        } else {
            // xBi > UB → wants to decrease → need eta_ij > 0
            if (eta <= cfg.pivotTol) continue;
            ratio = rc[j] / eta;
        }

        if (ratio < minRatio - cfg.pivotTol ||
            (ratio < minRatio + cfg.pivotTol && j < entering)) {
            minRatio = ratio;
            entering = j;
        }
    }
    return entering;
}

std::size_t SimplexTableauBV::selectEntering() const {
    const std::size_t limit = (nActive > 0) ? nActive : n;
    if (cfg.useDantzig) {
        std::size_t best  = n;
        double      bestRc = -cfg.optimalityTol;
        for (std::size_t j = 0; j < limit; ++j)
            if (rc[j] < bestRc) { bestRc = rc[j]; best = j; }
        return best;
    }
    for (std::size_t j = 0; j < limit; ++j)
        if (rc[j] < -cfg.optimalityTol)
            return j;
    return n;
}

SimplexTableauBV::RatioResult
SimplexTableauBV::selectLeavingBV(std::size_t e) const {
    const std::size_t w = n + 1;

    // Initialise with the entering variable's own UB (bound flip candidate).
    const bool   entUBFin = std::isfinite(colUB[e]);
    double       best     = entUBFin ? colUB[e]
                                     : std::numeric_limits<double>::infinity();
    std::size_t  bestRow  = m;
    uint32_t     bestIdx  = entUBFin ? static_cast<uint32_t>(e)
                                     : std::numeric_limits<uint32_t>::max();
    bool         bflip    = entUBFin;
    bool         bestAtUB = false;

    for (std::size_t i = 0; i < m; ++i) {
        const double eta = tab[i * w + e];
        const double xBi = tab[i * w + n];

        double   ratio;
        bool     thisAtUB;

        if (eta > cfg.pivotTol) {
            // xBi decreases: check LB = 0
            ratio    = xBi / eta;
            thisAtUB = false;
        } else if (eta < -cfg.pivotTol) {
            // xBi increases: check its UB
            const double ubi = colUB[basicCols[i]];
            if (!std::isfinite(ubi)) continue;
            ratio    = (ubi - xBi) / (-eta);
            thisAtUB = true;
        } else {
            continue;
        }

        const uint32_t idx = basicCols[i];
        if (ratio < best - cfg.pivotTol ||
            (ratio < best + cfg.pivotTol && idx < bestIdx)) {
            best     = ratio;
            bestRow  = i;
            bestIdx  = idx;
            bestAtUB = thisAtUB;
            bflip    = false;
        }
    }

    if (bflip)        return {m, true,  false};
    if (bestRow == m) return {m, false, false}; // unbounded
    return {bestRow, false, bestAtUB};
}

// ── Pivot ──────────────────────────────────────────────────────────────────────

void SimplexTableauBV::pivotBV(std::size_t leavingRow, std::size_t enteringCol,
                                bool leavingAtUB) {
    const std::size_t w = n + 1;

    // If the entering column is AT_UB (complemented), restore it to AT_LB form
    // before the GJ update so the RHS stores the actual value, not the complement.
    if (atUB[enteringCol])
        complement(enteringCol);

    double inv = 1.0 / tab[leavingRow * w + enteringCol];
    for (std::size_t j = 0; j <= n; ++j)
        tab[leavingRow * w + j] *= inv;

    for (std::size_t r = 0; r <= m; ++r) {
        double* row = (r < m) ? &tab[r * w] : rc.data();
        if (r == leavingRow) continue;
        double factor = row[enteringCol];
        if (factor == 0.0) continue;
        for (std::size_t j = 0; j <= n; ++j)
            row[j] -= factor * tab[leavingRow * w + j];
    }

    if (hasRedundantRow) {
        for (std::size_t i = 0; i < m; ++i) {
            if (i != leavingRow &&
                basicCols[i] == static_cast<uint32_t>(enteringCol)) {
                basicCols[i] = basicCols[leavingRow];
                break;
            }
        }
    }

    const uint32_t leavingCol = basicCols[leavingRow];
    basicCols[leavingRow]     = static_cast<uint32_t>(enteringCol);
    atUB[enteringCol]         = false; // basic vars always satisfy complement invariant

    if (leavingAtUB)
        complement(leavingCol); // leaving exits to UB: negate col + adjust RHS
    else
        atUB[leavingCol] = false; // leaving exits to LB
}

// ── Solution extraction ───────────────────────────────────────────────────────

std::vector<double> SimplexTableauBV::primalSolution() const {
    std::vector<double> x(n, 0.0);
    for (std::size_t i = 0; i < m; ++i)
        x[basicCols[i]] = tab[i * (n + 1) + n];
    // Non-basic AT_UB: actual shifted value is colUB[j]
    // (complement invariant guarantees atUB[basicCols[i]] = false)
    for (std::size_t j = 0; j < n; ++j)
        if (atUB[j]) x[j] = colUB[j];
    return x;
}

// ── Sensitivity analysis ──────────────────────────────────────────────────────

SensitivityResult extractSensitivityBV(const SimplexTableauBV&      tab,
                                        const LPStandardFormBV&      sfbv,
                                        const Model&                 model,
                                        const std::vector<uint32_t>& equalArtCol) {
    using Lim = std::numeric_limits<double>;
    const bool   maximize    = (model.getObjSense() == ObjSense::Maximize);
    const auto&  constraints = model.getLPConstraints();
    const auto&  objCoeffs   = model.getHot().obj;
    const double inf         = Lim::infinity();

    const SimplexConfig& cfg = tab.cfg;
    const std::size_t m   = tab.m;
    const std::size_t n   = tab.n;
    const std::size_t np  = n + 1;
    const std::size_t nEff = (tab.nActive > 0) ? tab.nActive : n;

    std::vector<bool> isBasic(nEff, false);
    for (std::size_t r = 0; r < m; ++r)
        if (tab.basicCols[r] < nEff)
            isBasic[tab.basicCols[r]] = true;

    SensitivityResult sens;

    // ── RHS ranging ──────────────────────────────────────────────────────────
    // For each constraint i, find [Δlo, Δhi] such that all basic variables
    // remain within [0, colUB] after the perturbation Δ in b[i].
    sens.rhsRange.resize(sfbv.nOrigRows);
    for (std::size_t i = 0; i < sfbv.nOrigRows; ++i) {
        uint32_t colI    = sfbv.rowSlackCol[i];
        double   dirSign = 1.0;

        if (colI >= static_cast<uint32_t>(sfbv.nCols)) {
            // Equal row: use the artificial column kept after Phase I.
            if (equalArtCol.empty() || equalArtCol[i] >= static_cast<uint32_t>(n)) {
                sens.rhsRange[i] = {-inf, +inf};
                continue;
            }
            colI    = equalArtCol[i];
            dirSign = 1.0;
        } else {
            dirSign = (constraints[i].sense == Sense::LessEq)
                          ? (sfbv.rowNegated[i] ? -1.0 : +1.0)
                          : (sfbv.rowNegated[i] ? +1.0 : -1.0);
        }

        double deltaLo = -inf, deltaHi = +inf;
        for (std::size_t r = 0; r < m; ++r) {
            const double xBr = tab.tab[r * np + n];
            const double dr  = dirSign * tab.tab[r * np + colI];
            const double ubr = tab.colUB[tab.basicCols[r]];
            if (dr > cfg.pivotTol) {
                deltaLo = std::max(deltaLo, -xBr / dr);
                if (std::isfinite(ubr))
                    deltaHi = std::min(deltaHi, (ubr - xBr) / dr);
            } else if (dr < -cfg.pivotTol) {
                deltaHi = std::min(deltaHi, -xBr / dr);
                if (std::isfinite(ubr))
                    deltaLo = std::max(deltaLo, (ubr - xBr) / dr);
            }
        }

        const double modelRHS = constraints[i].rhsConst;
        if (!sfbv.rowNegated[i])
            sens.rhsRange[i] = {modelRHS + deltaLo, modelRHS + deltaHi};
        else
            sens.rhsRange[i] = {modelRHS - deltaHi, modelRHS - deltaLo};
    }

    // ── Objective ranging ─────────────────────────────────────────────────────
    // For each original variable j, find [Δlo, Δhi] in c[j] such that all
    // non-basic reduced costs remain non-negative.
    sens.objRange.resize(sfbv.nOrig);
    for (std::size_t j = 0; j < sfbv.nOrig; ++j) {
        if (sfbv.varFreeNegCol[j] < static_cast<uint32_t>(sfbv.nCols)) {
            sens.objRange[j] = {-inf, +inf}; // free variable: not implemented
            continue;
        }

        const double factor = static_cast<double>(sfbv.varColSign[j]) * (maximize ? -1.0 : 1.0);
        const double cModel = objCoeffs[j];
        double deltaLoSF = -inf, deltaHiSF = +inf;

        if (!isBasic[j]) {
            // Non-basic AT_LB: rc[j] + δ ≥ 0  →  δ ≥ -rc[j]
            // Non-basic AT_UB: rc[j] - δ ≥ 0  →  δ ≤  rc[j]
            if (!tab.atUB[j]) deltaLoSF = -tab.rc[j];
            else               deltaHiSF =  tab.rc[j];
        } else {
            std::size_t r = 0;
            while (r < m && tab.basicCols[r] != static_cast<uint32_t>(j)) ++r;
            // Changing c'_j by δ shifts rc[k] by -δ * tab[r][k] for all non-basics k.
            // (Holds for both AT_LB and AT_UB non-basics — see comment in header.)
            for (std::size_t k = 0; k < nEff; ++k) {
                if (isBasic[k]) continue;
                const double t   = tab.tab[r * np + k];
                const double rck = tab.rc[k];
                if (t > cfg.pivotTol)
                    deltaHiSF = std::min(deltaHiSF, rck / t);
                else if (t < -cfg.pivotTol)
                    deltaLoSF = std::max(deltaLoSF, rck / t);
            }
        }

        double lo, hi;
        if (factor > 0.0) { lo = cModel + deltaLoSF; hi = cModel + deltaHiSF; }
        else               { lo = cModel - deltaHiSF; hi = cModel - deltaLoSF; }
        sens.objRange[j] = {lo, hi};
    }

    return sens;
}

} // namespace baguette::internal
