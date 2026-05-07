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

bool LUTableau::doReinvert() {
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

bool LUTableau::init(const LPStandardForm& sf, std::vector<uint32_t> initialBasis) {
    assert(initialBasis.size() == sf.nRows);
    m         = sf.nRows;
    n         = sf.nCols;
    basicCols = std::move(initialBasis);
    Binv.assign(m * m, 0.0);
    xB.resize(m, 0.0);
    pi.resize(m, 0.0);
    rc.resize(n + 1, 0.0);
    A_ptr = sf.A; c = sf.c; b = sf.b;
    return doReinvert();
}

bool LUTableau::reinvert(const LPStandardForm& sf) {
    A_ptr = sf.A;
    c     = sf.c;
    b     = sf.b;
    return doReinvert();
}

bool LUTableau::initBV(const LPStandardFormBV& sfbv, std::vector<uint32_t> initialBasis) {
    assert(initialBasis.size() == sfbv.nRows);
    m         = sfbv.nRows;
    n         = sfbv.nCols;
    basicCols = std::move(initialBasis);
    Binv.assign(m * m, 0.0);
    xB.resize(m, 0.0);
    pi.resize(m, 0.0);
    rc.resize(n + 1, 0.0);
    nActive = 0;
    colUB = sfbv.colUB;
    atUB.assign(n, false);
    A_ptr = sfbv.A; c = sfbv.c; b = sfbv.b;
    return doReinvert();
}

bool LUTableau::reinvertBV(const LPStandardFormBV& sfbv) {
    A_ptr = sfbv.A; c = sfbv.c; b = sfbv.b;
    colUB = sfbv.colUB;
    std::vector<bool> savedAtUB = atUB;
    atUB.assign(n, false);
    if (!doReinvert()) return false;
    for (std::size_t j = 0; j < n; ++j)
        if (savedAtUB[j]) complement(j);
    return true;
}

void LUTableau::repriceObjective(const std::vector<double>& newC,
                                  std::size_t                newNActive) {
    c       = newC;
    nActive = newNActive;
    recomputePi();
    recomputeReducedCosts();
}

void LUTableau::applyAtUBToRc() {
    for (std::size_t j = 0; j < n; ++j) {
        if (!atUB[j]) continue;
        const double rcj = rc[j];
        rc[n] -= colUB[j] * rcj;
        rc[j] = -rcj;
    }
}

void LUTableau::repriceBV(const std::vector<double>& newC, std::size_t newNActive) {
    c       = newC;
    nActive = newNActive;
    recomputePi();
    recomputeReducedCosts();
    applyAtUBToRc();
}

void LUTableau::complement(std::size_t j) {
    const double uj  = colUB[j];
    const auto   eta = enteringColumn(j);
    // AT_LB → AT_UB: xB -= uj * eta; AT_UB → AT_LB: xB += uj * eta
    const double sign = atUB[j] ? +1.0 : -1.0;
    for (std::size_t i = 0; i < m; ++i)
        xB[i] += sign * uj * eta[i];
    const double old_rcj = rc[j];
    rc[j]  = -old_rcj;
    rc[n] -= uj * old_rcj; // equivalent to rc[n] += uj * rc[j]_new
    atUB[j] = !atUB[j];
}

std::pair<LUTableau::RatioResultBV, std::vector<double>>
LUTableau::selectLeavingBVWithEta(std::size_t e) const {
    auto eta_orig = enteringColumn(e);
    // Effective entering direction: negate for AT_UB (complemented) column
    const double sign = atUB[e] ? -1.0 : 1.0;

    // Initialise with the entering variable's own UB (bound flip candidate)
    const bool     entUBFin = std::isfinite(colUB[e]);
    double         best     = entUBFin ? colUB[e] : std::numeric_limits<double>::infinity();
    std::size_t    bestRow  = m;
    uint32_t       bestIdx  = entUBFin ? static_cast<uint32_t>(e)
                                       : std::numeric_limits<uint32_t>::max();
    bool           bflip    = entUBFin;
    bool           bestAtUB = false;

    for (std::size_t i = 0; i < m; ++i) {
        const double eta = sign * eta_orig[i];
        const double xBi = xB[i];
        double       ratio;
        bool         thisAtUB;

        if (eta > baguette::pivot_tol) {
            ratio    = xBi / eta;
            thisAtUB = false;
        } else if (eta < -baguette::pivot_tol) {
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

    if (bflip)        return {{m, true,  false}, std::move(eta_orig)};
    if (bestRow == m) return {{m, false, false}, std::move(eta_orig)};
    return {{bestRow, false, bestAtUB}, std::move(eta_orig)};
}

void LUTableau::pivotBV(std::size_t r, std::size_t j, bool leavingAtUB,
                         const std::vector<double>& eta_orig) {
    const uint32_t leavingCol = basicCols[r];

    // Un-complement entering column if AT_UB (uses precomputed eta_orig, O(m))
    if (atUB[j]) {
        const double uj = colUB[j];
        for (std::size_t i = 0; i < m; ++i)
            xB[i] += uj * eta_orig[i]; // AT_UB → AT_LB: xB += uj * eta
        const double old_rcj = rc[j];
        rc[j]  = -old_rcj;
        rc[n] -= uj * old_rcj;
        atUB[j] = false;
    }

    // Standard pivot with the un-complemented entering column
    pivot(r, j, eta_orig);
    atUB[j] = false; // entering var becomes basic (always AT_LB)

    // Complement leaving variable if it exits to its UB
    if (leavingAtUB)
        complement(leavingCol); // uses NEW B⁻¹ after pivot, O(m²)
    else
        atUB[leavingCol] = false;
}

std::vector<double> LUTableau::primalSolutionBV() const {
    std::vector<double> x(n, 0.0);
    for (std::size_t i = 0; i < m; ++i)
        x[basicCols[i]] = xB[i];
    for (std::size_t j = 0; j < n; ++j)
        if (atUB[j]) x[j] = colUB[j];
    return x;
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

std::pair<std::size_t, std::vector<double>>
LUTableau::selectLeavingWithEta(std::size_t j) const {
    auto        eta      = enteringColumn(j);
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
    return {leaving, std::move(eta)};
}

std::size_t LUTableau::selectLeaving(std::size_t j) const {
    return selectLeavingWithEta(j).first;
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

void LUTableau::pivot(std::size_t r, std::size_t j,
                       const std::vector<double>& eta) {
    double rho     = eta[r];
    double inv_rho = 1.0 / rho;
    double alpha   = rc[j];  // reduced cost before pivot

    // Save old row r of B⁻¹ — needed for incremental π and rc updates.
    std::vector<double> y(Binv.begin() + static_cast<std::ptrdiff_t>(r * m),
                           Binv.begin() + static_cast<std::ptrdiff_t>(r * m + m));

    // ── Update B⁻¹ — row-major loops (cache-friendly) ────────────────────────
    // Row r ← old row r / ρ
    for (std::size_t k = 0; k < m; ++k)
        Binv[r * m + k] = y[k] * inv_rho;
    // Row i (i≠r) ← row i − (η[i]/ρ) × old row r
    for (std::size_t i = 0; i < m; ++i) {
        if (i == r) continue;
        double coeff = eta[i] * inv_rho;
        if (coeff == 0.0) continue;
        for (std::size_t k = 0; k < m; ++k)
            Binv[i * m + k] -= coeff * y[k];
    }

    // ── Update xB ─────────────────────────────────────────────────────────────
    double old_xBr = xB[r];
    xB[r] = old_xBr * inv_rho;
    for (std::size_t i = 0; i < m; ++i) {
        if (i != r) xB[i] -= eta[i] * inv_rho * old_xBr;
    }

    basicCols[r] = static_cast<uint32_t>(j);

    // ── Incremental π: π_new[k] = π_old[k] + (α/ρ) × y[k]  — O(m) ──────────
    double ratio = alpha * inv_rho;
    for (std::size_t k = 0; k < m; ++k)
        pi[k] += ratio * y[k];

    // ── Incremental rc: rc_new[p] = rc_old[p] − (α/ρ) × (y · a_p) ───────────
    // For AT_UB (complemented) columns the stored rc[p] = −rc_orig[p], so the
    // update sign flips: rc_comp[p] += (α/ρ) × (y · a_p).
    const auto& Avec = *A_ptr;
    const bool hasBV = !atUB.empty();
    for (std::size_t k = 0; k < m; ++k) {
        double coeff = ratio * y[k];
        if (coeff == 0.0) continue;
        const double* rowA = &Avec[k * n];
        if (!hasBV) {
            for (std::size_t p = 0; p < n; ++p)
                rc[p] -= coeff * rowA[p];
        } else {
            for (std::size_t p = 0; p < n; ++p)
                rc[p] += atUB[p] ? coeff * rowA[p] : -coeff * rowA[p];
        }
    }
    rc[j] = 0.0;  // j is now basic — enforce numerically exact

    // ── Incremental objective: rc[n] = rc[n] − (α/ρ) × xB_old[r]  — O(1) ──
    // In BV mode xB[r] = B⁻¹ b_eff ≠ B⁻¹ b_orig, so using old_xBr is
    // correct for both BV (b_eff) and non-BV (b_eff = b_orig).
    rc[n] -= ratio * old_xBr;
}

void LUTableau::pivot(std::size_t r, std::size_t j) {
    pivot(r, j, enteringColumn(j));
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

    // Initialise with objective coefficients
    for (std::size_t j = 0; j < n; ++j)
        rc[j] = (j < c.size() ? c[j] : 0.0);

    // Subtract π^T A using row-major traversal over A (cache-friendly)
    for (std::size_t k = 0; k < m; ++k) {
        if (pi[k] == 0.0) continue;
        const double* rowA = &Avec[k * n];
        for (std::size_t j = 0; j < n; ++j)
            rc[j] -= pi[k] * rowA[j];
    }

    // rc[n] = −(π^T b)
    double piTb = 0.0;
    for (std::size_t k = 0; k < m; ++k)
        piTb += pi[k] * b[k];
    rc[n] = -piTb;
}

} // namespace baguette::internal
