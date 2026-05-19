#include "MehrotraIPM.hpp"
#include "StandardForm.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

namespace baguette::internal {

// ── LU factorisation: partial pivoting, multipliers stored in lower triangle ──
// Returns false if the matrix is numerically singular.
static bool luFactor(std::vector<double>& A, int m, std::vector<int>& pivots) {
    pivots.resize(m);
    for (int k = 0; k < m; ++k) {
        int piv = k;
        for (int i = k + 1; i < m; ++i)
            if (std::abs(A[i*m+k]) > std::abs(A[piv*m+k])) piv = i;
        pivots[k] = piv;
        if (std::abs(A[piv*m+k]) < 1e-14) return false;
        if (piv != k)
            for (int j = 0; j < m; ++j) std::swap(A[k*m+j], A[piv*m+j]);
        const double inv = 1.0 / A[k*m+k];
        for (int i = k + 1; i < m; ++i) {
            const double f = A[i*m+k] * inv;
            A[i*m+k] = f;                                     // store L multiplier
            for (int j = k + 1; j < m; ++j) A[i*m+j] -= f * A[k*m+j];
        }
    }
    return true;
}

// ── Solve pre-factored LU · x = b in-place ───────────────────────────────────
static void luBack(const std::vector<double>& LU, int m,
                   const std::vector<int>& pivots, std::vector<double>& b) {
    for (int k = 0; k < m; ++k)
        if (pivots[k] != k) std::swap(b[k], b[pivots[k]]);
    for (int k = 0; k < m; ++k)
        for (int i = k + 1; i < m; ++i) b[i] -= LU[i*m+k] * b[k];
    for (int k = m - 1; k >= 0; --k) {
        for (int j = k + 1; j < m; ++j) b[k] -= LU[k*m+j] * b[j];
        b[k] /= LU[k*m+k];
    }
}

// ── Combined luSolve for the starting-point (factor + solve in one call) ──────
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

// ── Build and factor A D Aᵀ + δI where D = diag(x/s) ─────────────────────────
static bool buildAndFactor(const std::vector<double>& A, int m, int n,
                            const std::vector<double>& x, const std::vector<double>& s,
                            std::vector<double>& LU, std::vector<int>& pivots) {
    constexpr double kReg = 1e-12;
    LU.assign(m * m, 0.0);
    for (int i = 0; i < m; ++i) {
        for (int j = i; j < m; ++j) {
            double v = 0.0;
            for (int k = 0; k < n; ++k)
                v += A[i*n+k] * (x[k] / s[k]) * A[j*n+k];
            LU[i*m+j] = LU[j*m+i] = v;
        }
        LU[i*m+i] += kReg;
    }
    return luFactor(LU, m, pivots);
}

// ── Mehrotra starting-point heuristic ────────────────────────────────────────
static void buildAAT(const std::vector<double>& A, int m, int n,
                     std::vector<double>& M) {
    M.assign(m * m, 0.0);
    for (int i = 0; i < m; ++i)
        for (int j = i; j < m; ++j) {
            double v = 0.0;
            for (int k = 0; k < n; ++k) v += A[i*n+k] * A[j*n+k];
            M[i*m+j] = M[j*m+i] = v;
        }
}

static void initStartingPoint(
    const std::vector<double>& A, int m, int n,
    const std::vector<double>& b, const std::vector<double>& c,
    std::vector<double>& x, std::vector<double>& y, std::vector<double>& s)
{
    std::vector<double> AAT;
    buildAAT(A, m, n, AAT);

    std::vector<double> v = b;
    { auto M = AAT; luSolve(M, v, m); }
    x.assign(n, 0.0);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) x[j] += A[i*n+j] * v[i];

    std::vector<double> Ac(m, 0.0);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) Ac[i] += A[i*n+j] * c[j];
    y = Ac;
    { auto M = AAT; luSolve(M, y, m); }
    s.assign(n, 0.0);
    for (int j = 0; j < n; ++j) {
        s[j] = c[j];
        for (int i = 0; i < m; ++i) s[j] -= A[i*n+j] * y[i];
    }

    double dx = 1e-4, ds = 1e-4;
    for (double xi : x) dx = std::max(dx, -1.5 * xi);
    for (double si : s) ds = std::max(ds, -1.5 * si);
    for (double& xi : x) xi += dx;
    for (double& si : s) si += ds;

    double xdots = 0.0, sumx = 0.0, sums = 0.0;
    for (int j = 0; j < n; ++j) { xdots += x[j]*s[j]; sumx += x[j]; sums += s[j]; }
    const double cx = 0.5 * xdots / sums;
    const double cs = 0.5 * xdots / sumx;
    for (double& xi : x) xi += cx;
    for (double& si : s) si += cs;
}

// ── Newton direction from pre-factored normal equations ───────────────────────
// correction[j] = Δx_aff_j * Δs_aff_j (zero for the predictor step).
// Normal equations RHS: rp + A S⁻¹(x s + correction − μ_target e + x rd)
static void computeDirection(
    const std::vector<double>& A, int m, int n,
    const std::vector<double>& x, const std::vector<double>& s,
    const std::vector<double>& LU, const std::vector<int>& pivots,
    double mu_target,
    const std::vector<double>& rp, const std::vector<double>& rd,
    const std::vector<double>& correction,
    std::vector<double>& dx, std::vector<double>& dy, std::vector<double>& ds)
{
    std::vector<double> rhs(m, 0.0);
    for (int i = 0; i < m; ++i) {
        rhs[i] = rp[i];
        for (int j = 0; j < n; ++j)
            rhs[i] += A[i*n+j] * (x[j]*s[j] + correction[j] - mu_target + x[j]*rd[j]) / s[j];
    }
    dy = rhs;
    luBack(LU, m, pivots, dy);

    ds.assign(n, 0.0);
    for (int j = 0; j < n; ++j) {
        ds[j] = rd[j];
        for (int i = 0; i < m; ++i) ds[j] -= A[i*n+j] * dy[i];
    }

    dx.assign(n, 0.0);
    for (int j = 0; j < n; ++j)
        dx[j] = (mu_target - x[j]*s[j] - correction[j] - x[j]*ds[j]) / s[j];
}

// ── Ratio-test step length (0.99 × boundary distance) ────────────────────────
static double stepLen(const std::vector<double>& v, const std::vector<double>& dv) {
    double alpha = 1.0;
    for (std::size_t j = 0; j < v.size(); ++j)
        if (dv[j] < 0.0) alpha = std::min(alpha, -v[j] / dv[j]);
    return 0.99 * alpha;
}

// ── Solution extraction ───────────────────────────────────────────────────────
static LPDetailedResult extractResult(
    const LPStandardForm& sf,
    const std::vector<double>& x,
    const std::vector<double>& y,
    const std::vector<double>& s,
    int n, bool maximize)
{
    LPDetailedResult det;
    det.result.status = LPStatus::Optimal;

    det.result.primalValues.assign(sf.nOrig, 0.0);
    for (std::size_t j = 0; j < sf.nOrig; ++j) {
        double xj = x[j];
        if (sf.varFreeNegCol[j] != sf.nCols)
            xj -= x[sf.varFreeNegCol[j]];
        const uint32_t vid = sf.colOrigin[j];
        det.result.primalValues[vid] =
            (sf.varColSign[j] == +1) ? sf.varShiftVal[j] + xj
                                     : sf.varShiftVal[j] - xj;
    }

    double obj = sf.objOffset;
    for (int j = 0; j < n; ++j) obj += sf.c[j] * x[j];
    det.result.objectiveValue = maximize ? -obj : obj;

    det.dualValues.resize(sf.nOrigRows);
    for (std::size_t i = 0; i < sf.nOrigRows; ++i)
        det.dualValues[i] = sf.rowNegated[i] ? -y[i] : y[i];

    det.reducedCosts.resize(sf.nOrig);
    for (std::size_t j = 0; j < sf.nOrig; ++j)
        det.reducedCosts[j] = s[j] * sf.varColSign[j];

    return det;
}

// ── Main solver ───────────────────────────────────────────────────────────────
LPDetailedResult solveMehrotraIPM(
    const Model& model,
    uint32_t     maxIter,
    double       timeLimitS,
    std::chrono::steady_clock::time_point startTime)
{
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

    if (maxIter == 0) maxIter = 200;

    const auto timeUp = [&]() {
        return timeLimitS < std::numeric_limits<double>::infinity()
            && std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - startTime).count() >= timeLimitS;
    };
    if (timeUp()) {
        LPDetailedResult r; r.result.status = LPStatus::TimeLimit; return r;
    }

    std::vector<double> x, y, s;
    initStartingPoint(A, m, n, b, c, x, y, s);

    constexpr double kTol      = 1e-8;   // optimality tolerance
    constexpr double kGapSmall = 1e-6;   // μ threshold: gap has converged
    constexpr double kResLarge = 1e-3;   // residual threshold: not converging
    constexpr double kXBig     = 1e12;   // primal blow-up → unbounded
    constexpr double kStepTiny = 1e-8;   // step-stagnation threshold

    const std::vector<double> kZeros(n, 0.0);

    // Declared outside the loop so the post-loop infeasibility check can read them.
    double norm_rp = 0.0, norm_rd = 0.0;

    for (uint32_t iter = 0; iter < maxIter; ++iter) {
        if (timeUp()) {
            LPDetailedResult r;
            r.result.status    = LPStatus::TimeLimit;
            r.iterationsUsed   = iter;
            return r;
        }

        // Residuals and duality measure
        std::vector<double> rp(m, 0.0), rd(n, 0.0);
        for (int i = 0; i < m; ++i) {
            rp[i] = b[i];
            for (int j = 0; j < n; ++j) rp[i] -= A[i*n+j] * x[j];
        }
        for (int j = 0; j < n; ++j) {
            rd[j] = c[j] - s[j];
            for (int i = 0; i < m; ++i) rd[j] -= A[i*n+j] * y[i];
        }
        double mu = 0.0;
        for (int j = 0; j < n; ++j) mu += x[j] * s[j];
        mu /= n;

        norm_rp = 0.0; norm_rd = 0.0;
        for (double v : rp) norm_rp = std::max(norm_rp, std::abs(v));
        for (double v : rd) norm_rd = std::max(norm_rd, std::abs(v));

        // Optimality
        if (mu < kTol && norm_rp < kTol && norm_rd < kTol) {
            LPDetailedResult det = extractResult(sf, x, y, s, n, maximize);
            det.iterationsUsed   = iter;
            return det;
        }

        // Infeasibility: gap closed but primal residual didn't (early exit)
        if (mu < kGapSmall && norm_rp > kResLarge) {
            LPDetailedResult r;
            r.result.status  = LPStatus::Infeasible;
            r.iterationsUsed = iter;
            return r;
        }

        // Unboundedness: primal iterate blew up
        if (*std::max_element(x.begin(), x.end()) > kXBig) {
            LPDetailedResult r;
            r.result.status  = LPStatus::Unbounded;
            r.iterationsUsed = iter;
            return r;
        }

        // Factor A D Aᵀ + δI once; reuse for both predictor and corrector.
        std::vector<double> LU;
        std::vector<int>    pivots;
        if (!buildAndFactor(A, m, n, x, s, LU, pivots)) {
            LPDetailedResult r;
            r.result.status  = (norm_rp > kResLarge) ? LPStatus::Infeasible : LPStatus::MaxIter;
            r.iterationsUsed = iter;
            return r;
        }

        // ── Predictor (affine, μ_target = 0) ─────────────────────────────────
        std::vector<double> dx_aff, dy_aff, ds_aff;
        computeDirection(A, m, n, x, s, LU, pivots,
                         0.0, rp, rd, kZeros, dx_aff, dy_aff, ds_aff);

        // Affine step length (combined primal-dual), used only for σ.
        const double alpha_aff = [&]() {
            double a = 1.0;
            for (int j = 0; j < n; ++j) {
                if (dx_aff[j] < 0.0) a = std::min(a, -x[j] / dx_aff[j]);
                if (ds_aff[j] < 0.0) a = std::min(a, -s[j] / ds_aff[j]);
            }
            return 0.99 * a;
        }();

        // Adaptive centering: σ = (μ_aff / μ)³, clamped to [0, 1].
        double mu_aff = 0.0;
        for (int j = 0; j < n; ++j)
            mu_aff += (x[j] + alpha_aff * dx_aff[j]) * (s[j] + alpha_aff * ds_aff[j]);
        mu_aff /= n;
        const double sigma = std::min(1.0, std::pow(mu_aff / mu, 3.0));

        // ── Corrector (μ_target = σμ, Δx_aff ⊙ Δs_aff cross-term) ──────────
        std::vector<double> correction(n);
        for (int j = 0; j < n; ++j) correction[j] = dx_aff[j] * ds_aff[j];

        std::vector<double> dx, dy, ds;
        computeDirection(A, m, n, x, s, LU, pivots,
                         sigma * mu, rp, rd, correction, dx, dy, ds);

        // Separate primal and dual step lengths.
        const double alpha_p = stepLen(x, dx);
        const double alpha_d = stepLen(s, ds);

        if (alpha_p <= 0.0 || alpha_d <= 0.0) {
            LPDetailedResult r;
            r.result.status  = (norm_rp > kResLarge) ? LPStatus::Infeasible : LPStatus::MaxIter;
            r.iterationsUsed = iter;
            return r;
        }

        // Stagnation: primal step vanishes while ‖rp‖ stays large → infeasible.
        if (alpha_p < kStepTiny && norm_rp > kResLarge) {
            LPDetailedResult r;
            r.result.status  = LPStatus::Infeasible;
            r.iterationsUsed = iter;
            return r;
        }
        // Stagnation: dual step vanishes while ‖rd‖ stays large → unbounded.
        if (alpha_d < kStepTiny && norm_rd > kResLarge) {
            LPDetailedResult r;
            r.result.status  = LPStatus::Unbounded;
            r.iterationsUsed = iter;
            return r;
        }

        for (int j = 0; j < n; ++j) { x[j] += alpha_p * dx[j]; s[j] += alpha_d * ds[j]; }
        for (int i = 0; i < m; ++i)   y[i] += alpha_d * dy[i];
    }

    // ‖rp‖ is bounded below by the infeasibility gap and cannot reach 0.
    // If it is still large after maxIter iterations the problem is infeasible.
    LPDetailedResult r;
    r.result.status  = (norm_rp > kResLarge) ? LPStatus::Infeasible : LPStatus::MaxIter;
    r.iterationsUsed = maxIter;
    return r;
}

} // namespace baguette::internal