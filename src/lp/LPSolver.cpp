#include "baguette/lp/LPSolver.hpp"

#include <chrono>
#include <cmath>
#include <limits>

#include "baguette/core/Config.hpp"
#include "baguette/core/Sense.hpp"
#include "StandardForm.hpp"
#include "Tableau.hpp"

namespace baguette {

// ── Internal helpers ──────────────────────────────────────────────────────────

namespace {

using Clock = std::chrono::steady_clock;

/// Augmented standard form extended with artificial variables for phase I.
struct AugmentedForm {
    internal::LPStandardForm sf; ///< Extended with artificial columns.
    std::vector<uint32_t>    initialBasis;
    std::size_t              nArt = 0;
};

/// Build the phase-I augmented form from a standard form.
///
/// Natural initial basic variables:
///   - LessEq rows:       slack column (coeff +1) → natural basic.
///   - GreaterEq rows:    surplus column (coeff −1) → NOT a natural basic.
///   - Equal rows:        no slack → NOT a natural basic.
///   - Upper-bound rows:  UpperSlack column (coeff +1) → natural basic.
///
/// An artificial column (coeff +1) is appended for every non-natural row.
/// The phase-I objective is c = 1 for artificials, 0 otherwise.
AugmentedForm buildPhaseOne(const internal::LPStandardForm& sf,
                             const Model& model) {
    const auto& constraints = model.getConstraints();
    const std::size_t m    = sf.nRows;
    const std::size_t nOld = sf.nCols;

    std::vector<bool> needsArt(m, false);
    for (std::size_t i = 0; i < sf.nOrigRows; ++i) {
        Sense s = constraints[i].sense;
        needsArt[i] = (s == Sense::GreaterEq || s == Sense::Equal);
    }
    // Upper-bound rows have a natural UpperSlack — no artificial needed.

    std::size_t nArt = 0;
    for (bool b : needsArt) if (b) ++nArt;

    AugmentedForm aug;
    aug.nArt = nArt;

    // Copy only the fields that are reused as-is; A, c, colKind, colOrigin
    // are rebuilt below for the wider augmented matrix.
    aug.sf.nRows      = sf.nRows;
    aug.sf.nOrigRows  = sf.nOrigRows;
    aug.sf.nOrig      = sf.nOrig;
    aug.sf.nSlack     = sf.nSlack;
    aug.sf.b          = sf.b;
    aug.sf.rowSlackCol = sf.rowSlackCol;
    aug.sf.rowNegated  = sf.rowNegated;

    const std::size_t nNew = nOld + nArt;
    aug.sf.nCols = nNew;
    aug.sf.A.assign(m * nNew, 0.0);
    aug.sf.c.assign(nNew, 0.0);
    aug.sf.colKind.resize(nNew);
    aug.sf.colOrigin.resize(nNew);

    // Copy original A and column metadata into the wider matrix
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < nOld; ++j)
            aug.sf.A[i * nNew + j] = sf.A[i * nOld + j];

    for (std::size_t j = 0; j < nOld; ++j) {
        aug.sf.colKind[j]   = sf.colKind[j];
        aug.sf.colOrigin[j] = sf.colOrigin[j];
    }

    // Append artificial columns (phase-I objective = 1 for each)
    std::size_t artCol = nOld;
    for (std::size_t i = 0; i < m; ++i) {
        if (!needsArt[i]) continue;
        aug.sf.A[i * nNew + artCol] = 1.0;
        aug.sf.c[artCol]            = 1.0;
        aug.sf.colKind[artCol]      = ColumnKind::Slack; // internal marker
        aug.sf.colOrigin[artCol]    = 0;
        ++artCol;
    }

    // Build initial basis
    aug.initialBasis.resize(m);
    artCol = nOld;
    for (std::size_t i = 0; i < m; ++i) {
        if (needsArt[i]) {
            aug.initialBasis[i] = static_cast<uint32_t>(artCol++);
        } else if (i < sf.nOrigRows) {
            // LessEq row: slack column is natural basic
            aug.initialBasis[i] = sf.rowSlackCol[i];
        } else {
            // Upper-bound row: UpperSlack column
            std::size_t ubIdx = i - sf.nOrigRows;
            aug.initialBasis[i] = static_cast<uint32_t>(sf.nOrig + sf.nSlack + ubIdx);
        }
    }

    return aug;
}

/// Drive any artificial variables still in the basis (at value 0) out by
/// pivoting in a non-artificial column. Rows where no such pivot exists are
/// redundant (all original coefficients are zero) and are left unchanged.
void driveOutArtificials(internal::Tableau& tab,
                         const internal::LPStandardForm& sfOrig) {
    const std::size_t nOld = sfOrig.nCols;

    for (std::size_t i = 0; i < tab.m; ++i) {
        if (tab.basicCols[i] < nOld) continue; // not an artificial

        for (std::size_t j = 0; j < nOld; ++j) {
            if (std::abs(tab.tab[i * (tab.n + 1) + j]) > baguette::pivot_tol) {
                tab.pivot(i, j);
                break;
            }
        }
    }
}

/// Strip artificial columns and re-price the objective row for phase II.
void preparePhaseTwo(internal::Tableau& tab,
                     const internal::LPStandardForm& sfOrig) {
    const std::size_t nOld = sfOrig.nCols;
    const std::size_t m    = tab.m;

    std::vector<double> newTab(m * (nOld + 1));
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < nOld; ++j)
            newTab[i * (nOld + 1) + j] = tab.tab[i * (tab.n + 1) + j];
        newTab[i * (nOld + 1) + nOld] = tab.tab[i * (tab.n + 1) + tab.n];
    }

    tab.tab = std::move(newTab);
    tab.n   = nOld;

    // Re-price: rc_j = c_j − c_B * B⁻¹ a_j
    tab.rc.assign(nOld + 1, 0.0);
    for (std::size_t j = 0; j < nOld; ++j)
        tab.rc[j] = sfOrig.c[j];

    for (std::size_t i = 0; i < m; ++i) {
        double cb = sfOrig.c[tab.basicCols[i]];
        if (cb == 0.0) continue;
        for (std::size_t j = 0; j <= nOld; ++j)
            tab.rc[j] -= cb * tab.tab[i * (nOld + 1) + j];
    }
}

/// Extract the full LPDetailedResult from a solved phase-II tableau.
LPDetailedResult extractDetailed(const internal::Tableau& tab,
                                 const internal::LPStandardForm& sf,
                                 const Model& model,
                                 LPStatus status) {
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
    // For a LessEq row:   slack coeff = +1 → y_i = rc[slackCol]
    // For a GreaterEq row: surplus coeff = −1 → y_i = −rc[slackCol]
    // For an Equal row:   no slack → dual is read differently (via artificial's rc,
    //                     but artificials are stripped; use shadow from reinverted basis)
    // Row negation (b[i] < 0 flip) is corrected by sf.rowNegated.
    const auto& constraints = model.getConstraints();
    det.dualValues.resize(sf.nOrigRows);
    for (std::size_t i = 0; i < sf.nOrigRows; ++i) {
        uint32_t slackCol = sf.rowSlackCol[i];
        double raw = tab.rc[slackCol];

        // Dual sign derivation:
        //   LessEq slack (coeff +1):  rc[s] = c_s - y_i = -y_i  → y_i = -rc[s]
        //   GEQ surplus (coeff -1):   rc[s] = c_s + y_i = +y_i  → y_i = +rc[s]
        //   Equal row: no slack, dual not recoverable here → 0
        // For Maximize we solved min(-obj), so the standard-form dual has
        // the opposite sign from the max-problem shadow price.
        // Row negation (b[i] < 0 flip) adds another sign flip.
        double sign;
        if (constraints[i].sense == Sense::Equal) {
            det.dualValues[i] = 0.0; // not recoverable from slack rc
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
    // varColSign flips the column direction for ub-shifted vars, so their rc
    // must be sign-corrected to express the rate w.r.t. the original variable.
    det.reducedCosts.resize(nOrig);
    for (std::size_t j = 0; j < nOrig; ++j)
        det.reducedCosts[j] = sf.varColSign[j] * (maximize ? -tab.rc[j] : tab.rc[j]);

    // Basis record (in terms of the original sf column space, no artificials)
    det.basis.basicCols.assign(tab.basicCols.begin(), tab.basicCols.end());
    det.basis.colKind   = sf.colKind;
    det.basis.colOrigin = sf.colOrigin;

    return det;
}

/// Simplex loop shared by both phases.
LPStatus runSimplex(internal::Tableau& tab,
                    const internal::LPStandardForm& sf,
                    uint32_t maxIter,
                    double   timeLimitS,
                    Clock::time_point startTime) {
    uint32_t iter = 0;

    while (true) {
        if (maxIter > 0 && iter >= maxIter)
            return LPStatus::MaxIter;

        if (timeLimitS > 0.0) {
            double elapsed =
                std::chrono::duration<double>(Clock::now() - startTime).count();
            if (elapsed >= timeLimitS)
                return LPStatus::TimeLimit;
        }

        std::size_t entering = tab.selectEntering();
        if (entering == tab.n)
            return LPStatus::Optimal;

        std::size_t leaving = tab.selectLeaving(entering);
        if (leaving == tab.m)
            return LPStatus::Unbounded;

        tab.pivot(leaving, entering);
        ++iter;

        if (baguette::reinversion_period > 0 &&
            iter % baguette::reinversion_period == 0)
            tab.reinvert(sf);
    }
}

} // anonymous namespace

// ── Public API ────────────────────────────────────────────────────────────────

LPDetailedResult solveDetailed(const Model& model,
                               uint32_t maxIter,
                               double   timeLimitS) {
    auto startTime = Clock::now();

    // Early infeasibility: a variable with lb > ub has an empty domain.
    // This arises naturally in B&B after bound tightening.
    {
        const auto& hot = model.getHot();
        for (std::size_t j = 0; j < model.numVars(); ++j) {
            if (hot.lb[j] > hot.ub[j]) {
                LPDetailedResult det;
                det.result.status = LPStatus::Infeasible;
                return det;
            }
        }
    }

    // 1. Standard form
    internal::LPStandardForm sf = internal::toStandardForm(model);

    // 2. Phase I
    AugmentedForm aug = buildPhaseOne(sf, model);
    internal::Tableau tab;
    tab.init(aug.sf, aug.initialBasis);

    LPStatus p1Status = runSimplex(tab, aug.sf, maxIter, timeLimitS, startTime);

    if (p1Status == LPStatus::MaxIter || p1Status == LPStatus::TimeLimit) {
        LPDetailedResult det;
        det.result.status = p1Status;
        return det;
    }

    // Phase-I objective is always ≥ 0 (sum of non-negative artificials).
    // If it is still > lp_feasibility_tol after minimisation, the problem
    // has no feasible solution.
    if (tab.objectiveValue() > baguette::lp_feasibility_tol) {
        LPDetailedResult det;
        det.result.status = LPStatus::Infeasible;
        return det;
    }

    // 3. Drive remaining artificials out of the basis (degenerate exit)
    driveOutArtificials(tab, sf);

    // 4. Phase II
    preparePhaseTwo(tab, sf);

    LPStatus p2Status = runSimplex(tab, sf, maxIter, timeLimitS, startTime);

    // 5. Extract result
    LPDetailedResult det = extractDetailed(tab, sf, model, p2Status);

    if (p2Status == LPStatus::Unbounded)
        det.result.primalValues.clear();

    return det;
}

LPResult solve(const Model& model,
               uint32_t maxIter,
               double   timeLimitS) {
    return solveDetailed(model, maxIter, timeLimitS).result;
}

} // namespace baguette
