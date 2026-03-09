#include "Tableau.hpp"

#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "baguette/core/Config.hpp"

namespace baguette::internal {

// ── Construction ─────────────────────────────────────────────────────────────

void Tableau::init(const LPStandardForm& sf,
                   const std::vector<uint32_t>& initialBasis) {
    assert(initialBasis.size() == sf.nRows);

    m = sf.nRows;
    n = sf.nCols;

    // Copy the full augmented matrix [A | b] into the tableau
    tab.resize(m * (n + 1));
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j)
            tab[i * (n + 1) + j] = sf.A[i * n + j];
        tab[i * (n + 1) + n] = sf.b[i];
    }

    basicCols.assign(initialBasis.begin(), initialBasis.end());

    // Gauss-Jordan: pivot on each basis column so that B becomes the identity
    for (std::size_t i = 0; i < m; ++i) {
        std::size_t col = basicCols[i];
        double pivot = tab[i * (n + 1) + col];
        if (std::abs(pivot) < baguette::pivot_tol)
            throw std::runtime_error("Tableau::init: degenerate initial basis");

        // Scale pivot row
        double inv = 1.0 / pivot;
        for (std::size_t j = 0; j <= n; ++j)
            tab[i * (n + 1) + j] *= inv;

        // Eliminate column in all other rows
        for (std::size_t r = 0; r < m; ++r) {
            if (r == i) continue;
            double factor = tab[r * (n + 1) + col];
            if (factor == 0.0) continue;
            for (std::size_t j = 0; j <= n; ++j)
                tab[r * (n + 1) + j] -= factor * tab[i * (n + 1) + j];
        }
    }

    // Price the objective row: rc_j = c_j − c_B * B^{-1} a_j.
    // sf.c encodes the right objective for both phases:
    //   Phase I:  aug.sf.c has 1 for artificial columns, 0 for others
    //             (set by buildPhaseOne in LPSolver.cpp).
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
}

// ── Reinversion ──────────────────────────────────────────────────────────────

void Tableau::reinvert(const LPStandardForm& sf) {
    // Rebuild the tableau from scratch using the current basis.
    // This resets accumulated floating-point errors.
    std::vector<uint32_t> basis = basicCols; // keep current basis
    init(sf, basis);
}

// ── Simplex operations ────────────────────────────────────────────────────────

std::size_t Tableau::selectEntering() const {
    // Bland's rule: smallest index with rc[j] < -lp_optimality_tol
    for (std::size_t j = 0; j < n; ++j)
        if (rc[j] < -baguette::lp_optimality_tol)
            return j;
    return n; // optimal
}

std::size_t Tableau::selectLeaving(std::size_t enteringCol) const {
    double minRatio = std::numeric_limits<double>::infinity();
    std::size_t leavingRow = m; // sentinel: unbounded

    for (std::size_t i = 0; i < m; ++i) {
        double aij = tab[i * (n + 1) + enteringCol];
        if (aij <= baguette::pivot_tol) continue; // skip non-positive entries

        double ratio = tab[i * (n + 1) + n] / aij;
        // Full Bland's rule: on ties (within pivot_tol), prefer the row whose
        // basic variable has the smallest column index. This guarantees finite
        // termination even on degenerate pivots. The tolerance guards against
        // floating-point drift causing near-equal ratios to be missed.
        if (ratio < minRatio - baguette::pivot_tol) {
            minRatio   = ratio;
            leavingRow = i;
        } else if (ratio < minRatio + baguette::pivot_tol &&
                   leavingRow < m &&
                   basicCols[i] < basicCols[leavingRow]) {
            leavingRow = i;
        }
    }
    return leavingRow;
}

void Tableau::pivot(std::size_t leavingRow, std::size_t enteringCol) {
    const std::size_t w = n + 1;

    // Scale pivot row
    double pivotVal = tab[leavingRow * w + enteringCol];
    double inv = 1.0 / pivotVal;
    for (std::size_t j = 0; j <= n; ++j)
        tab[leavingRow * w + j] *= inv;

    // Eliminate entering column from all other rows (including rc)
    for (std::size_t r = 0; r <= m; ++r) { // r == m means the rc row
        double* row = (r < m) ? &tab[r * w] : rc.data();
        if (r == leavingRow) continue;
        double factor = row[enteringCol];
        if (factor == 0.0) continue;
        for (std::size_t j = 0; j <= n; ++j)
            row[j] -= factor * tab[leavingRow * w + j];
    }

    basicCols[leavingRow] = static_cast<uint32_t>(enteringCol);
}

// ── Solution extraction ───────────────────────────────────────────────────────

std::vector<double> Tableau::primalSolution() const {
    std::vector<double> x(n, 0.0);
    for (std::size_t i = 0; i < m; ++i)
        x[basicCols[i]] = tab[i * (n + 1) + n];
    return x;
}

std::vector<double> Tableau::dualSolution(const LPStandardForm& sf) const {
    // Raw dual: rc[slackCol] for each model row.
    // Sign corrections for GEQ surplus (coeff −1), row negation, and
    // Maximize are applied by the caller (LPSolver.cpp) which has access
    // to the Model's constraint senses.
    std::vector<double> y(sf.nOrigRows, 0.0);
    for (std::size_t i = 0; i < sf.nOrigRows; ++i) {
        double val = rc[sf.rowSlackCol[i]];
        y[i] = sf.rowNegated[i] ? -val : val;
    }
    return y;
}

} // namespace baguette::internal
