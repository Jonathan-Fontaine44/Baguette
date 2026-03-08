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
        if (ratio < minRatio) {
            minRatio   = ratio;
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
    // For the full tableau, the dual variable for constraint i is read from
    // the reduced-cost entry of the slack column for that row:
    //   y_i = rc[slackCol_i]  (for LessEq rows)
    //   y_i = −rc[slackCol_i] (for GreaterEq rows, surplus has coefficient −1)
    //   y_i = 0               (for Equal rows; no slack)
    // Sign is also flipped if the row was negated during normalisation.
    std::vector<double> y(sf.nOrigRows, 0.0);
    for (std::size_t i = 0; i < sf.nOrigRows; ++i) {
        uint32_t slackCol = sf.rowSlackCol[i];
        double val = rc[slackCol];

        // GreaterEq surplus has coefficient −1, so dual is negated
        const auto& con = // we need the sense — use colKind or rowNegated
            // The surplus sign is already baked into A; if the row was a
            // GreaterEq with A[i, slackCol] == −1 before normalisation, the
            // rc entry directly gives y_i * (−1), so we negate.
            // Detect this via the original sense stored in sf (not available
            // here directly). Instead, inspect A[i * n + slackCol] after
            // reinversion: if it was −1 (surplus), then negate.
            // Actually the simplest approach: use the fact that for a surplus
            // column the original A entry is −1, and after Gauss-Jordan the
            // rc entry satisfies rc[slackCol] = −y_i. So:
            val; // placeholder — see note below

        // Note: the sign convention for the dual depends on whether the slack
        // coefficient is +1 (LessEq) or −1 (GreaterEq). We cannot recover the
        // original sense from the tableau alone without keeping extra metadata.
        // The rowNegated flag accounts for the rhs flip; the sense information
        // is handled in LPSolver.cpp where we have access to the Model constraints.
        // Here we return the raw rc value; LPSolver.cpp applies the correction.
        y[i] = sf.rowNegated[i] ? -val : val;
    }
    return y;
}

} // namespace baguette::internal
