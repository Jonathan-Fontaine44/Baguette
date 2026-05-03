#include "SimplexTableauBV.hpp"

#include <cassert>
#include <cmath>
#include <limits>

#include "baguette/core/Config.hpp"

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
        if (maxAbs < baguette::pivot_tol) return false;

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

std::size_t SimplexTableauBV::selectEntering() const {
    // Bland's rule — complement invariant makes this identical to the standard tableau.
    const std::size_t limit = (nActive > 0) ? nActive : n;
    for (std::size_t j = 0; j < limit; ++j)
        if (rc[j] < -baguette::lp_optimality_tol)
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

        if (eta > baguette::pivot_tol) {
            // xBi decreases: check LB = 0
            ratio    = xBi / eta;
            thisAtUB = false;
        } else if (eta < -baguette::pivot_tol) {
            // xBi increases: check its UB
            const double ubi = colUB[basicCols[i]];
            if (!std::isfinite(ubi)) continue;
            ratio    = (ubi - xBi) / (-eta);
            thisAtUB = true;
        } else {
            continue;
        }

        const uint32_t idx = basicCols[i];
        if (ratio < best - baguette::pivot_tol ||
            (ratio < best + baguette::pivot_tol && idx < bestIdx)) {
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

} // namespace baguette::internal
