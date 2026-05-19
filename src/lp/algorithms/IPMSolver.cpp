#include "IPMSolver.hpp"
#include "StandardForm.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

namespace baguette::internal {

// ── Dense LU solve: Ax = b (in-place, m×m, partial pivoting) ─────────────────
// Returns false if the matrix is numerically singular.
static bool luSolve(std::vector<double>& A, std::vector<double>& b, int m) {
    for (int k = 0; k < m; ++k) {
        int piv = k;
        for (int i = k + 1; i < m; ++i)
            if (std::abs(A[i*m+k]) > std::abs(A[piv*m+k])) piv = i;
        if (std::abs(A[piv*m+k]) < 1e-14) return false;
        if (piv != k) {
            std::swap(b[k], b[piv]);
            for (int j = 0; j < m; ++j) std::swap(A[k*m+j], A[piv*m+j]);
        }
        const double inv = 1.0 / A[k*m+k];
        for (int i = k + 1; i < m; ++i) {
            const double f = A[i*m+k] * inv;
            for (int j = k; j < m; ++j) A[i*m+j] -= f * A[k*m+j];
            b[i] -= f * b[k];
        }
    }
    for (int k = m - 1; k >= 0; --k) {
        for (int j = k + 1; j < m; ++j) b[k] -= A[k*m+j] * b[j];
        b[k] /= A[k*m+k];
    }
    return true;
}

// ── Build AAᵀ (m×m, symmetric) ───────────────────────────────────────────────
static std::vector<double> buildAAT(const std::vector<double>& A, int m, int n) {
    std::vector<double> M(m * m, 0.0);
    for (int i = 0; i < m; ++i)
        for (int j = i; j < m; ++j) {
            double v = 0.0;
            for (int k = 0; k < n; ++k) v += A[i*n+k] * A[j*n+k];
            M[i*m+j] = M[j*m+i] = v;
        }
    return M;
}

// ── Mehrotra starting-point heuristic ────────────────────────────────────────
// Produces (x⁰, y⁰, s⁰) with x⁰, s⁰ > 0.  The starting point is typically
// not primal/dual feasible; residuals shrink along with the duality gap μ.
static void initStartingPoint(
    const std::vector<double>& A, int m, int n,
    const std::vector<double>& b, const std::vector<double>& c,
    std::vector<double>& x, std::vector<double>& y, std::vector<double>& s)
{
    // Factor AAᵀ once; used for both x̄ and ȳ.
    auto AAT = buildAAT(A, m, n);

    // x̄ = Aᵀ (AAᵀ)⁻¹ b  (least-norm primal solution satisfying Ax̄ = b)
    std::vector<double> v = b;
    auto AAT_p = AAT;
    luSolve(AAT_p, v, m);
    x.assign(n, 0.0);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) x[j] += A[i*n+j] * v[i];

    // ȳ = (AAᵀ)⁻¹ Ac,  s̄ = c − Aᵀ ȳ  (least-norm dual solution)
    std::vector<double> Ac(m, 0.0);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) Ac[i] += A[i*n+j] * c[j];
    y = Ac;
    auto AAT_d = AAT;
    luSolve(AAT_d, y, m);
    s.assign(n, 0.0);
    for (int j = 0; j < n; ++j) {
        s[j] = c[j];
        for (int i = 0; i < m; ++i) s[j] -= A[i*n+j] * y[i];
    }

    // Shift x and s to be strictly positive (δ ≥ 1.5 × most-negative component)
    double dx = 1e-4, ds = 1e-4;
    for (double xi : x) dx = std::max(dx, -1.5 * xi);
    for (double si : s) ds = std::max(ds, -1.5 * si);
    for (double& xi : x) xi += dx;
    for (double& si : s) si += ds;

    // Centering correction: balance xᵀs across x and s uniformly
    double xdots = 0.0, sumx = 0.0, sums = 0.0;
    for (int j = 0; j < n; ++j) { xdots += x[j] * s[j]; sumx += x[j]; sums += s[j]; }
    const double corr_x = 0.5 * xdots / sums;
    const double corr_s = 0.5 * xdots / sumx;
    for (double& xi : x) xi += corr_x;
    for (double& si : s) si += corr_s;
}

// ── Newton direction (normal-equations form) ──────────────────────────────────
// Computes the Newton step toward the μ_target-center. In the short-step
// method the caller passes μ_target = (1−α)μ so each iterate contracts the
// duality gap by a factor (1−α²) and the residuals by (1−α).
//
// Normal equations: (A D Aᵀ + δI) Δy = rp + A S⁻¹(X rd + xs − μ_target e)
// where D = diag(x/s), then Δs = rd − Aᵀ Δy and Δx = (μ_target − xs − xΔs)/s.
// Returns false if the normal-equations matrix is singular.
static bool newtonDirection(
    const std::vector<double>& A, int m, int n,
    const std::vector<double>& x, const std::vector<double>& s,
    double mu_target,
    const std::vector<double>& rp, const std::vector<double>& rd,
    std::vector<double>& dx, std::vector<double>& dy, std::vector<double>& ds)
{
    // M = A D Aᵀ + δI  (δ = regularisation for numerical stability)
    constexpr double kReg = 1e-12;
    std::vector<double> M(m * m, 0.0);
    for (int i = 0; i < m; ++i) {
        for (int j = i; j < m; ++j) {
            double v = 0.0;
            for (int k = 0; k < n; ++k)
                v += A[i*n+k] * (x[k] / s[k]) * A[j*n+k];
            M[i*m+j] = M[j*m+i] = v;
        }
        M[i*m+i] += kReg;
    }

    // rhs_i = rp_i + Σ_j A_ij (x_j s_j + x_j rd_j − μ_target) / s_j
    std::vector<double> rhs(m, 0.0);
    for (int i = 0; i < m; ++i) {
        rhs[i] = rp[i];
        for (int j = 0; j < n; ++j)
            rhs[i] += A[i*n+j] * (x[j] * s[j] + x[j] * rd[j] - mu_target) / s[j];
    }

    dy = rhs;
    if (!luSolve(M, dy, m)) return false;

    // Δs = rd − Aᵀ Δy
    ds.assign(n, 0.0);
    for (int j = 0; j < n; ++j) {
        ds[j] = rd[j];
        for (int i = 0; i < m; ++i) ds[j] -= A[i*n+j] * dy[i];
    }

    // Δx = (μ_target − x s − x Δs) / s  (element-wise)
    dx.assign(n, 0.0);
    for (int j = 0; j < n; ++j)
        dx[j] = (mu_target - x[j] * s[j] - x[j] * ds[j]) / s[j];

    return true;
}

// ── Step size ─────────────────────────────────────────────────────────────────
// Returns 0.995 × min(alpha_cap, ratio test over x+αΔx≥0 and s+αΔs≥0).
static double computeStep(
    const std::vector<double>& x, const std::vector<double>& dx,
    const std::vector<double>& s, const std::vector<double>& ds,
    double alpha_cap)
{
    double alpha = alpha_cap;
    for (std::size_t j = 0; j < x.size(); ++j) {
        if (dx[j] < 0.0) alpha = std::min(alpha, -x[j] / dx[j]);
        if (ds[j] < 0.0) alpha = std::min(alpha, -s[j] / ds[j]);
    }
    return 0.995 * alpha;
}

// ── Primal/dual solution extraction ──────────────────────────────────────────
static LPDetailedResult extractResult(
    const LPStandardForm& sf,
    const std::vector<double>& x,
    const std::vector<double>& y,
    const std::vector<double>& s,
    int n,
    bool maximize)
{
    LPDetailedResult det;
    det.result.status = LPStatus::Optimal;

    // Primal values: un-shift each original variable
    det.result.primalValues.assign(sf.nOrig, 0.0);
    for (std::size_t j = 0; j < sf.nOrig; ++j) {
        double xj = x[j];
        if (sf.varFreeNegCol[j] != sf.nCols)
            xj -= x[sf.varFreeNegCol[j]];  // free-split: x = x⁺ − x⁻
        const uint32_t vid = sf.colOrigin[j];
        det.result.primalValues[vid] =
            (sf.varColSign[j] == +1) ? sf.varShiftVal[j] + xj
                                     : sf.varShiftVal[j] - xj;
    }

    // Objective: cᵀx in standard form + shift offset; flip sign for maximization.
    double obj = sf.objOffset;
    for (int j = 0; j < n; ++j) obj += sf.c[j] * x[j];
    det.result.objectiveValue = maximize ? -obj : obj;

    // Dual values: one per original constraint row, sign-corrected for negated rows
    det.dualValues.resize(sf.nOrigRows);
    for (std::size_t i = 0; i < sf.nOrigRows; ++i)
        det.dualValues[i] = sf.rowNegated[i] ? -y[i] : y[i];

    // Reduced costs for original variables (dual slack s_j, sign-corrected)
    det.reducedCosts.resize(sf.nOrig);
    for (std::size_t j = 0; j < sf.nOrig; ++j)
        det.reducedCosts[j] = s[j] * sf.varColSign[j];

    return det;
}

// ── Main solver ───────────────────────────────────────────────────────────────
LPDetailedResult solveShortStepIPM(
    const Model& model,
    uint32_t     maxIter,
    double       timeLimitS,
    std::chrono::steady_clock::time_point startTime)
{
    // Early lb > ub: empty domain → immediately infeasible
    {
        const auto& hot = model.getHot();
        for (std::size_t j = 0; j < model.numVars(); ++j) {
            if (hot.lb[j] > hot.ub[j]) {
                LPDetailedResult det;
                det.result.status      = LPStatus::Infeasible;
                det.farkas.infeasVarId = static_cast<int32_t>(j);
                return det;
            }
        }
    }

    LPStandardForm sf = toStandardForm(model);
    const bool maximize = (model.getObjSense() == ObjSense::Maximize);
    const int m = static_cast<int>(sf.nRows);
    const int n = static_cast<int>(sf.nCols);
    const auto& A = *sf.A;
    const auto& b = sf.b;
    const auto& c = sf.c;

    if (maxIter == 0) maxIter = 500;

    const auto timeUp = [&]() {
        return timeLimitS < std::numeric_limits<double>::infinity()
            && std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - startTime).count() >= timeLimitS;
    };
    if (timeUp()) {
        LPDetailedResult r; r.result.status = LPStatus::TimeLimit; return r;
    }

    // Starting point
    std::vector<double> x, y, s;
    initStartingPoint(A, m, n, b, c, x, y, s);

    // Fixed short step: α = 1/(1 + √n) — guarantees O(√n log(1/ε)) convergence.
    const double alpha_short = 1.0 / (1.0 + std::sqrt(static_cast<double>(n)));

    constexpr double kTol = 1e-8;

    for (uint32_t iter = 0; iter < maxIter; ++iter) {
        if (timeUp()) {
            LPDetailedResult r;
            r.result.status  = LPStatus::TimeLimit;
            r.iterationsUsed = iter;
            return r;
        }

        // Primal residual rp = b − Ax
        std::vector<double> rp(m, 0.0);
        for (int i = 0; i < m; ++i) {
            rp[i] = b[i];
            for (int j = 0; j < n; ++j) rp[i] -= A[i*n+j] * x[j];
        }

        // Dual residual rd = c − Aᵀy − s
        std::vector<double> rd(n, 0.0);
        for (int j = 0; j < n; ++j) {
            rd[j] = c[j] - s[j];
            for (int i = 0; i < m; ++i) rd[j] -= A[i*n+j] * y[i];
        }

        // Duality measure μ = xᵀs / n
        double mu = 0.0;
        for (int j = 0; j < n; ++j) mu += x[j] * s[j];
        mu /= n;

        // Convergence: gap and both residuals below tolerance
        double norm_rp = 0.0, norm_rd = 0.0;
        for (double v : rp) norm_rp = std::max(norm_rp, std::abs(v));
        for (double v : rd) norm_rd = std::max(norm_rd, std::abs(v));

        if (mu < kTol && norm_rp < kTol && norm_rd < kTol) {
            LPDetailedResult det = extractResult(sf, x, y, s, n, maximize);
            det.iterationsUsed   = iter;
            return det;
        }

        // Newton direction toward the (1−α)μ-center; reduces gap by (1−α²) per step.
        const double mu_target = (1.0 - alpha_short) * mu;
        std::vector<double> dx, dy, ds;
        if (!newtonDirection(A, m, n, x, s, mu_target, rp, rd, dx, dy, ds)) {
            LPDetailedResult r;
            r.result.status  = LPStatus::MaxIter;
            r.iterationsUsed = iter;
            return r;
        }

        // Step: capped at the short-step length α = 1/(1+√n)
        const double alpha = computeStep(x, dx, s, ds, alpha_short);
        if (alpha <= 0.0) {
            LPDetailedResult r;
            r.result.status  = LPStatus::MaxIter;
            r.iterationsUsed = iter;
            return r;
        }

        for (int j = 0; j < n; ++j) { x[j] += alpha * dx[j]; s[j] += alpha * ds[j]; }
        for (int i = 0; i < m; ++i)    y[i] += alpha * dy[i];
    }

    LPDetailedResult r;
    r.result.status  = LPStatus::MaxIter;
    r.iterationsUsed = maxIter;
    return r;
}

} // namespace baguette::internal