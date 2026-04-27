#include "LUTableau.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>

#include "baguette/core/Config.hpp"

namespace baguette::internal {

namespace {

// ── LU factorisation with partial pivoting ────────────────────────────────────
//
// Factors mat (m×m, row-major) in-place into L (below diagonal, implicit 1s)
// and U (on + above diagonal). perm[i] = original row index at position i.
// Returns false if any pivot is below pivot_tol (singular or near-singular).

bool luFactorise(std::vector<double>& mat, std::size_t m,
                 std::vector<std::size_t>& perm) {
    perm.resize(m);
    std::iota(perm.begin(), perm.end(), 0);

    for (std::size_t k = 0; k < m; ++k) {
        // Partial pivot: largest |a[i,k]| for i >= k
        std::size_t pivot = k;
        double      maxv  = std::abs(mat[k * m + k]);
        for (std::size_t i = k + 1; i < m; ++i) {
            double v = std::abs(mat[i * m + k]);
            if (v > maxv) { maxv = v; pivot = i; }
        }
        if (maxv < baguette::pivot_tol) return false;

        if (pivot != k) {
            std::swap(perm[k], perm[pivot]);
            for (std::size_t j = 0; j < m; ++j)
                std::swap(mat[k * m + j], mat[pivot * m + j]);
        }

        double ukk = mat[k * m + k];
        for (std::size_t i = k + 1; i < m; ++i) {
            double factor  = mat[i * m + k] / ukk;
            mat[i * m + k] = factor; // store L multiplier in-place
            for (std::size_t j = k + 1; j < m; ++j)
                mat[i * m + j] -= factor * mat[k * m + j];
        }
    }
    return true;
}

// Solve LU x = P rhs using forward + backward substitution.
std::vector<double> luSolve(const std::vector<double>&      lu,
                              std::size_t                     m,
                              const std::vector<std::size_t>& perm,
                              const std::vector<double>&      rhs) {
    std::vector<double> x(m);
    for (std::size_t i = 0; i < m; ++i) x[i] = rhs[perm[i]];

    // Forward: L x = b (L has 1s on diagonal)
    for (std::size_t k = 0; k < m; ++k)
        for (std::size_t i = k + 1; i < m; ++i)
            x[i] -= lu[i * m + k] * x[k];

    // Backward: U x = b
    for (std::ptrdiff_t k = static_cast<std::ptrdiff_t>(m) - 1; k >= 0; --k) {
        x[k] /= lu[k * m + k];
        for (std::size_t i = 0; i < static_cast<std::size_t>(k); ++i)
            x[i] -= lu[i * m + k] * x[k];
    }
    return x;
}

} // anonymous namespace

// ── LUTableau implementation ──────────────────────────────────────────────────

bool LUTableau::init(const LPStandardForm& sf, std::vector<uint32_t> initialBasis) {
    assert(initialBasis.size() == sf.nRows);
    m         = sf.nRows;
    n         = sf.nCols;
    basicCols = std::move(initialBasis);
    Binv.assign(m * m, 0.0);
    xB.resize(m, 0.0);
    pi.resize(m, 0.0);
    rc.resize(n + 1, 0.0);
    return reinvert(sf);
}

bool LUTableau::reinvert(const LPStandardForm& sf) {
    // Refresh stored data (handles bounds-only sf updates between B&B nodes)
    A_ptr = sf.A;
    c     = sf.c;
    b     = sf.b;

    const auto& Avec = *A_ptr;

    // Build basis matrix B (m×m): column k of B = column basicCols[k] of A
    std::vector<double> B(m * m);
    for (std::size_t k = 0; k < m; ++k)
        for (std::size_t i = 0; i < m; ++i)
            B[i * m + k] = Avec[i * n + basicCols[k]];

    std::vector<std::size_t> perm;
    if (!luFactorise(B, m, perm)) return false;

    // Compute B⁻¹ column by column: solve B x = e_k for each k
    std::vector<double> ek(m, 0.0);
    for (std::size_t k = 0; k < m; ++k) {
        std::fill(ek.begin(), ek.end(), 0.0);
        ek[k]    = 1.0;
        auto col = luSolve(B, m, perm, ek);
        for (std::size_t i = 0; i < m; ++i)
            Binv[i * m + k] = col[i];
    }

    // xB = B⁻¹ b
    for (std::size_t i = 0; i < m; ++i) {
        xB[i] = 0.0;
        for (std::size_t k = 0; k < m; ++k)
            xB[i] += Binv[i * m + k] * b[k];
    }

    recomputePi();
    recomputeReducedCosts();
    return true;
}

void LUTableau::repriceObjective(const std::vector<double>& newC,
                                  std::size_t                newNActive) {
    c       = newC;
    nActive = newNActive;
    recomputePi();
    recomputeReducedCosts();
}

std::vector<double> LUTableau::enteringColumn(std::size_t j) const {
    const auto& Avec = *A_ptr;
    std::vector<double> eta(m, 0.0);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t k = 0; k < m; ++k)
            eta[i] += Binv[i * m + k] * Avec[k * n + j];
    return eta;
}

std::vector<double> LUTableau::tableauRow(std::size_t r) const {
    const auto& Avec = *A_ptr;
    const double* yr = &Binv[r * m];
    std::vector<double> t(n, 0.0);
    for (std::size_t k = 0; k < m; ++k) {
        if (yr[k] == 0.0) continue;
        for (std::size_t j = 0; j < n; ++j)
            t[j] += yr[k] * Avec[k * n + j];
    }
    return t;
}

std::size_t LUTableau::selectEntering() const {
    const std::size_t limit = (nActive > 0) ? nActive : n;
    for (std::size_t j = 0; j < limit; ++j)
        if (rc[j] < -baguette::lp_optimality_tol)
            return j;
    return n;
}

std::size_t LUTableau::selectLeaving(std::size_t j) const {
    auto eta = enteringColumn(j);

    std::size_t leaving  = m;
    double      minRatio = std::numeric_limits<double>::infinity();

    for (std::size_t i = 0; i < m; ++i) {
        if (eta[i] <= baguette::pivot_tol) continue;
        double ratio = xB[i] / eta[i];
        if (ratio < minRatio - baguette::pivot_tol) {
            minRatio = ratio;
            leaving  = i;
        } else if (ratio < minRatio + baguette::pivot_tol &&
                   leaving < m && basicCols[i] < basicCols[leaving]) {
            leaving = i;
        }
    }
    return leaving;
}

std::size_t LUTableau::selectLeavingDual() const {
    std::size_t leaving = m;
    double      minB    = std::numeric_limits<double>::infinity();

    for (std::size_t i = 0; i < m; ++i) {
        double bi = xB[i];
        if (bi >= -baguette::lp_feasibility_tol) continue;

        if (bi < minB - baguette::lp_feasibility_tol) {
            minB    = bi;
            leaving = i;
        } else if (bi < minB + baguette::lp_feasibility_tol &&
                   leaving < m && basicCols[i] < basicCols[leaving]) {
            leaving = i;
        }
    }
    return leaving;
}

std::size_t LUTableau::selectEnteringDual(std::size_t leavingRow) const {
    auto        t        = tableauRow(leavingRow);
    std::size_t entering = n;
    double      minRatio = std::numeric_limits<double>::infinity();

    for (std::size_t j = 0; j < n; ++j) {
        if (t[j] >= -baguette::pivot_tol) continue;
        double ratio = rc[j] / (-t[j]);
        if (ratio < minRatio - baguette::pivot_tol) {
            minRatio = ratio;
            entering = j;
        } else if (ratio < minRatio + baguette::pivot_tol && j < entering) {
            entering = j;
        }
    }
    return entering;
}

void LUTableau::pivot(std::size_t r, std::size_t j) {
    auto   eta     = enteringColumn(j);
    double rho     = eta[r];
    double inv_rho = 1.0 / rho;

    // Eta-file update of B⁻¹:
    //   New row r   = old row r / ρ
    //   New row i≠r = old row i − (η[i]/ρ) × old row r
    for (std::size_t k = 0; k < m; ++k) {
        double old_rk = Binv[r * m + k];
        for (std::size_t i = 0; i < m; ++i) {
            if (i == r) continue;
            Binv[i * m + k] -= eta[i] * inv_rho * old_rk;
        }
        Binv[r * m + k] = old_rk * inv_rho;
    }

    // Update xB
    double old_xBr = xB[r];
    for (std::size_t i = 0; i < m; ++i) {
        if (i != r) xB[i] -= eta[i] * inv_rho * old_xBr;
    }
    xB[r] = old_xBr * inv_rho;

    basicCols[r] = static_cast<uint32_t>(j);

    recomputePi();
    recomputeReducedCosts();
}

std::vector<double> LUTableau::primalSolution() const {
    std::vector<double> x(n, 0.0);
    for (std::size_t i = 0; i < m; ++i)
        x[basicCols[i]] = xB[i];
    return x;
}

void LUTableau::recomputePi() {
    // π[k] = Σ_i c[basicCols[i]] · Binv[i*m+k]
    std::fill(pi.begin(), pi.end(), 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        if (basicCols[i] >= c.size()) continue;
        double cb = c[basicCols[i]];
        if (cb == 0.0) continue;
        for (std::size_t k = 0; k < m; ++k)
            pi[k] += cb * Binv[i * m + k];
    }
}

void LUTableau::recomputeReducedCosts() {
    const auto& Avec = *A_ptr;
    for (std::size_t j = 0; j < n; ++j) {
        double piTaj = 0.0;
        for (std::size_t k = 0; k < m; ++k)
            piTaj += pi[k] * Avec[k * n + j];
        rc[j] = (j < c.size() ? c[j] : 0.0) - piTaj;
    }
    // rc[n] = −(π^T b) = −(cB^T xB)
    double piTb = 0.0;
    for (std::size_t k = 0; k < m; ++k)
        piTb += pi[k] * b[k];
    rc[n] = -piTb;
}

} // namespace baguette::internal
