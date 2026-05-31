#include "SimplexTableau.hpp"

#include <cassert>
#include <cmath>
#include <limits>

namespace baguette::internal {

// ── Construction ─────────────────────────────────────────────────────────────

bool SimplexTableau::init(const LPStandardForm& sf,
                           std::vector<uint32_t> initialBasis) {
    assert(initialBasis.size() == sf.nRows);

    m = sf.nRows;
    n = sf.nCols;

    // Copy the full augmented matrix [A | b] into the tableau
    tab.resize(m * (n + 1));
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j)
            tab[i * (n + 1) + j] = (*sf.A)[i * n + j];
        tab[i * (n + 1) + n] = sf.b[i];
    }

    basicCols = std::move(initialBasis);

    // Gauss-Jordan with partial pivoting: pivot on each basis column so that
    // B becomes the identity. Partial pivoting avoids false "degenerate basis"
    // failures when the row order is not triangularly compatible with basicCols
    // (e.g. after B&B warm-start or mid-solve reinversion).
    const std::size_t w = n + 1;
    for (std::size_t i = 0; i < m; ++i) {
        std::size_t col = basicCols[i];

        // Find the row r in [i, m-1] with the largest |tab[r, col]|.
        // col is fixed as basicCols[i], so basicCols is never swapped.
        std::size_t pivotRow = i;
        double      maxAbs   = 0.0;
        for (std::size_t r = i; r < m; ++r) {
            double val = std::abs(tab[r * w + col]);
            if (val > maxAbs) { maxAbs = val; pivotRow = r; }
        }

        if (maxAbs < cfg.pivotTol)
            return false; // truly singular basis

        // Swap physical rows (not basicCols: col stays basicCols[i]).
        if (pivotRow != i)
            for (std::size_t j = 0; j <= n; ++j)
                std::swap(tab[i * w + j], tab[pivotRow * w + j]);

        // Scale pivot row
        double inv = 1.0 / tab[i * w + col];
        for (std::size_t j = 0; j <= n; ++j)
            tab[i * w + j] *= inv;

        // Eliminate column in all other rows
        for (std::size_t r = 0; r < m; ++r) {
            if (r == i) continue;
            double factor = tab[r * w + col];
            if (factor == 0.0) continue;
            for (std::size_t j = 0; j <= n; ++j)
                tab[r * w + j] -= factor * tab[i * w + j];
        }
    }

    // Price the objective row: rc_j = c_j − c_B * B^{-1} a_j.
    // sf.c encodes the right objective for both phases:
    //   Phase I:  aug.sf.c has 1 for artificial columns, 0 for others.
    //   Phase II: original sf.c with the actual objective costs.
    rc.assign(n + 1, 0.0);
    for (std::size_t j = 0; j < n; ++j)
        rc[j] = sf.c[j];

    for (std::size_t i = 0; i < m; ++i) {
        double cb = sf.c[basicCols[i]];
        if (cb == 0.0) continue;
        for (std::size_t j = 0; j <= n; ++j)
            rc[j] -= cb * tab[i * (n + 1) + j];
    }
    return true;
}

// ── Reinversion ──────────────────────────────────────────────────────────────

bool SimplexTableau::reinvert(const LPStandardForm& sf) {
    // Rebuild the tableau from scratch using the current basis.
    // std::move avoids a copy: init() receives the buffer and moves it back
    // into basicCols via the by-value parameter, reusing the same allocation.
    //
    // After reinversion with sfOrig (nCols == nOld), the tableau shrinks to
    // nOld columns and artificial columns are gone - clear the tracking list.
    artColsForDual.clear();
    return init(sf, std::move(basicCols));
}

// ── Simplex operations ────────────────────────────────────────────────────────

std::size_t SimplexTableau::selectEntering() const {
    const std::size_t limit = (nActive > 0) ? nActive : n;
    if (cfg.useDantzig) {
        std::size_t best  = n;
        double      bestRc = -cfg.optimalityTol;
        for (std::size_t j = 0; j < limit; ++j)
            if (rc[j] < bestRc) { bestRc = rc[j]; best = j; }
        return best;
    }
    // Bland's rule: smallest index with rc[j] < -optimalityTol.
    for (std::size_t j = 0; j < limit; ++j)
        if (rc[j] < -cfg.optimalityTol)
            return j;
    return n;
}

std::size_t SimplexTableau::selectLeaving(std::size_t enteringCol) const {
    double minRatio = std::numeric_limits<double>::infinity();
    std::size_t leavingRow = m; // sentinel: unbounded

    for (std::size_t i = 0; i < m; ++i) {
        double aij = tab[i * (n + 1) + enteringCol];
        if (aij <= cfg.pivotTol) continue; // skip non-positive entries

        double ratio = tab[i * (n + 1) + n] / aij;
        // Full Bland's rule: on ties (within pivot_tol), prefer the row whose
        // basic variable has the smallest column index. This guarantees finite
        // termination even on degenerate pivots. The tolerance guards against
        // floating-point drift causing near-equal ratios to be missed.
        if (ratio < minRatio - cfg.pivotTol) {
            minRatio   = ratio;
            leavingRow = i;
        } else if (ratio < minRatio + cfg.pivotTol &&
                   leavingRow < m &&
                   basicCols[i] < basicCols[leavingRow]) {
            leavingRow = i;
        }
    }
    return leavingRow;
}

std::size_t SimplexTableau::selectLeavingDual() const {
    std::size_t leavingRow = m; // sentinel: primal feasible (all bi >= -tol)
    uint32_t    bestIdx    = std::numeric_limits<uint32_t>::max();

    // Bland's rule: select the infeasible row with the smallest basic column
    // index. Guarantees finite termination on degenerate LPs.
    for (std::size_t i = 0; i < m; ++i) {
        double bi = tab[i * (n + 1) + n];
        if (bi >= -cfg.feasibilityTol) continue;

        if (basicCols[i] < bestIdx) {
            bestIdx    = basicCols[i];
            leavingRow = i;
        }
    }
    return leavingRow;
}

std::size_t SimplexTableau::selectEnteringDual(std::size_t leavingRow) const {
    double      minRatio    = std::numeric_limits<double>::infinity();
    std::size_t enteringCol = n; // sentinel: primal infeasible

    for (std::size_t j = 0; j < n; ++j) {
        double aij = tab[leavingRow * (n + 1) + j];
        if (aij >= -cfg.pivotTol) continue; // only strictly negative entries

        // rc[j] ≥ 0 (dual feasibility) and −aij > 0, so ratio ≥ 0.
        double ratio = rc[j] / (-aij);
        if (ratio < minRatio - cfg.pivotTol) {
            minRatio    = ratio;
            enteringCol = j;
        } else if (ratio < minRatio + cfg.pivotTol && j < enteringCol) {
            enteringCol = j; // tie-break: smallest column index
        }
    }
    return enteringCol;
}

void SimplexTableau::pivot(std::size_t leavingRow, std::size_t enteringCol) {
    const std::size_t w = n + 1;
    // In Phase II nActive == nOld limits updates to original SF columns, skipping
    // non-tracked artificial columns.  artColsForDual tracks Equal-row artificials
    // still needed for dual extraction.  Phase I uses nActive == 0 (all n columns).
    const std::size_t pEnd = (nActive > 0) ? nActive : n;

    // Scale pivot row: original cols [0, pEnd) + Equal-art tracked cols + rhs
    double inv = 1.0 / tab[leavingRow * w + enteringCol];
    for (std::size_t j = 0; j < pEnd; ++j)
        tab[leavingRow * w + j] *= inv;
    for (uint32_t ac : artColsForDual)
        tab[leavingRow * w + ac] *= inv;
    tab[leavingRow * w + n] *= inv;

    // Eliminate entering column from all other rows (including rc row at r == m)
    for (std::size_t r = 0; r <= m; ++r) {
        double* row = (r < m) ? &tab[r * w] : rc.data();
        if (r == leavingRow) continue;
        double factor = row[enteringCol];
        if (factor == 0.0) continue;
        for (std::size_t j = 0; j < pEnd; ++j)
            row[j] -= factor * tab[leavingRow * w + j];
        for (uint32_t ac : artColsForDual)
            row[ac] -= factor * tab[leavingRow * w + ac];
        row[n] -= factor * tab[leavingRow * w + n];
    }

    // If enteringCol was already assigned to another row (a dummy redundant row
    // from repairRedundantRows), give that row the leaving column instead.
    // The flag avoids this O(m) scan in the common case (no redundant rows).
    if (hasRedundantRow) {
        for (std::size_t i = 0; i < m; ++i) {
            if (i != leavingRow && basicCols[i] == static_cast<uint32_t>(enteringCol)) {
                basicCols[i] = basicCols[leavingRow];
                break; // at most one dummy row per entering column
            }
        }
    }

    basicCols[leavingRow] = static_cast<uint32_t>(enteringCol);
}

// ── Solution extraction ───────────────────────────────────────────────────────

std::vector<double> SimplexTableau::primalSolution() const {
    std::vector<double> x(n, 0.0);
    for (std::size_t i = 0; i < m; ++i)
        x[basicCols[i]] = tab[i * (n + 1) + n];
    return x;
}

} // namespace baguette::internal
