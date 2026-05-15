#include "PrimalSimplex.hpp"

#include <cassert>
#include <cmath>
#include <limits>

#include "Extractor.hpp"
#include "SimplexTableau.hpp"
#include "StandardForm.hpp"
#include "baguette/core/Sense.hpp"
#include "baguette/model/ModelEnums.hpp"

namespace baguette {

namespace {

// ── Augmented standard form (phase I) ────────────────────────────────────────

struct AugmentedForm {
    internal::LPStandardForm sf; ///< Extended with artificial columns.
    std::vector<uint32_t>    initialBasis;
    std::size_t              nArt = 0;

    /// For each original constraint row i (size == sf.nOrigRows):
    /// the column index of the artificial added for that Equal row, or
    /// sf.nCols (sentinel) if the row is not Equal.
    /// Used by extractDetailed() to read y_i = −rc[equalArtCol[i]] after phase II.
    std::vector<uint32_t> equalArtCol;
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
///
/// @note Complexity: O(m · nOrig) for copying A into the augmented matrix,
///   where m = sf.nRows and nOrig = sf.nCols. Appending nArt artificial columns
///   is O(nArt). Total O(m · (nOrig + nArt)).
AugmentedForm buildPhaseOne(const internal::LPStandardForm& sf,
                             const Model& model) {
    const auto& constraints = model.getLPConstraints();
    const std::size_t m    = sf.nRows;
    const std::size_t nOld = sf.nCols;

    std::vector<bool> needsArt(m, false);
    for (std::size_t i = 0; i < sf.nOrigRows; ++i) {
        Sense s = constraints[i].sense;
        // A negated LessEq row has its slack stored with coefficient -1 (not +1),
        // so it cannot serve as a natural positive basis column and needs an artificial.
        needsArt[i] = (s == Sense::GreaterEq || s == Sense::Equal || sf.rowNegated[i]);
    }
    // Upper-bound rows have a natural UpperSlack — no artificial needed.

    std::size_t nArt = 0;
    for (bool b : needsArt) if (b) ++nArt;

    AugmentedForm aug;
    aug.nArt = nArt;

    aug.sf.nRows      = sf.nRows;
    aug.sf.nOrigRows  = sf.nOrigRows;
    aug.sf.nOrig      = sf.nOrig;
    aug.sf.nSlack     = sf.nSlack;
    aug.sf.b          = sf.b;
    aug.sf.rowSlackCol = sf.rowSlackCol;
    aug.sf.rowNegated  = sf.rowNegated;

    // Initialise Equal-row artificial mapping with sentinel (= no artificial).
    aug.equalArtCol.assign(sf.nOrigRows, static_cast<uint32_t>(nOld + nArt));

    const std::size_t nNew = nOld + nArt;
    aug.sf.nCols = nNew;
    aug.sf.A = std::make_shared<std::vector<double>>(m * nNew, 0.0);
    aug.sf.c.assign(nNew, 0.0);
    aug.sf.colKind.resize(nNew);
    aug.sf.colOrigin.resize(nNew);

    // Copy original A and column metadata into the wider matrix
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < nOld; ++j)
            (*aug.sf.A)[i * nNew + j] = (*sf.A)[i * nOld + j];

    for (std::size_t j = 0; j < nOld; ++j) {
        aug.sf.colKind[j]   = sf.colKind[j];
        aug.sf.colOrigin[j] = sf.colOrigin[j];
    }

    // Append artificial columns (phase-I objective = 1 for each).
    std::size_t artCol = nOld;
    for (std::size_t i = 0; i < m; ++i) {
        if (i < sf.nOrigRows && !needsArt[i]) continue;
        if (i >= sf.nOrigRows) continue; // upper-bound rows: natural UpperSlack

        (*aug.sf.A)[i * nNew + artCol] = 1.0;
        aug.sf.c[artCol] = 1.0;
        aug.sf.colKind[artCol]   = ColumnKind::Slack; // artificial: treated like slack
        aug.sf.colOrigin[artCol] = static_cast<uint32_t>(i);

        if (constraints[i].sense == Sense::Equal)
            aug.equalArtCol[i] = static_cast<uint32_t>(artCol);

        ++artCol;
    }

    // Initial basis: natural columns for natural rows, artificials for the rest.
    aug.initialBasis.resize(m);
    std::size_t nextArt = nOld;
    for (std::size_t i = 0; i < m; ++i) {
        if (i >= sf.nOrigRows) {
            // Upper-bound row: natural UpperSlack column
            aug.initialBasis[i] = static_cast<uint32_t>(
                sf.nOrig + sf.nSlack + (i - sf.nOrigRows));
        } else if (!needsArt[i]) {
            aug.initialBasis[i] = sf.rowSlackCol[i]; // natural slack (LessEq, not negated)
        } else {
            aug.initialBasis[i] = static_cast<uint32_t>(nextArt++);
        }
    }

    return aug;
}

/// Drive any artificial variables still in the basis (at value 0) out by
/// pivoting in a non-artificial column. Rows where no such pivot exists are
/// redundant (all original coefficients are zero) and are left unchanged.
///
/// @note Complexity: O(nArt · m · n) where nArt = number of artificial basic
///   variables (≤ nOrigRows). Each candidate search costs O(nOld) and each
///   pivot costs O(m·n); at most nArt pivots are performed.
void driveOutArtificials(internal::SimplexTableau&       tab,
                          const internal::LPStandardForm& sfOrig) {
    const std::size_t nOld = sfOrig.nCols;

    for (std::size_t i = 0; i < tab.m; ++i) {
        if (tab.basicCols[i] < nOld) continue; // not an artificial

        for (std::size_t j = 0; j < nOld; ++j) {
            if (std::abs(tab.tab[i * (tab.n + 1) + j]) > tab.cfg.pivotTol) {
                tab.pivot(i, j);
                break;
            }
        }
    }
}

/// Fix basicCols entries that still point to artificial columns after
/// driveOutArtificials. This happens for redundant rows (all non-artificial
/// coefficients are zero, rhs = 0): no pivot was possible so the artificial
/// stayed in the basis. Assign any currently non-basic column instead; the
/// row is 0·x = 0 so the choice does not affect the primal solution.
///
/// @note Complexity: O(nRedundant · m · nOld).
void repairRedundantRows(internal::SimplexTableau& tab, std::size_t nOld) {
    std::vector<bool> inBasis(nOld, false);
    for (std::size_t i = 0; i < tab.m; ++i)
        if (tab.basicCols[i] < nOld) inBasis[tab.basicCols[i]] = true;

    for (std::size_t i = 0; i < tab.m; ++i) {
        if (tab.basicCols[i] < nOld) continue;
        bool found = false;

        // First pass: prefer all-zero columns (cannot enter the basis,
        // so assigning to a redundant row creates no future conflicts).
        for (std::size_t j = 0; j < nOld && !found; ++j) {
            if (inBasis[j]) continue;
            bool allZero = true;
            for (std::size_t r = 0; r < tab.m && allZero; ++r)
                allZero = (std::abs(tab.tab[r * (tab.n + 1) + j]) <= tab.cfg.pivotTol);
            if (allZero) {
                tab.basicCols[i] = static_cast<uint32_t>(j);
                inBasis[j] = true;
                found = true;
            }
        }

        // Fallback: any non-basic column (creates a duplicate that pivot() resolves
        // via hasRedundantRow).
        for (std::size_t j = 0; j < nOld && !found; ++j) {
            if (!inBasis[j]) {
                tab.basicCols[i] = static_cast<uint32_t>(j);
                inBasis[j] = true;
                tab.hasRedundantRow = true;
                found = true;
            }
        }
        assert(found && "repairRedundantRows: no free column — basis is over-complete");
    }
}

/// Transition the tableau from phase I to phase II.
///
/// Artificial columns are NOT stripped from the tableau.  Instead, their
/// phase-II cost is set to zero and they are excluded from pivot selection
/// via tab.nActive = sfOrig.nCols.  This keeps their rc entries up-to-date
/// through every phase-II pivot so that, at optimality:
///   rc[art_i] = 0 − y_i  →  y_i = −rc[art_i]
/// enabling dual-variable extraction for Equal constraints.
///
/// @note Complexity: O(m·n) for objective repricing.
void preparePhaseTwo(internal::SimplexTableau&       tab,
                      const internal::LPStandardForm& sfOrig) {
    const std::size_t nOld = sfOrig.nCols;
    const std::size_t m    = tab.m;
    const std::size_t w    = tab.n + 1;

    tab.nActive = nOld;

    repairRedundantRows(tab, nOld);

    // Re-price: rc_j = c_j − c_B * B⁻¹ a_j for ALL columns (including artificials).
    tab.rc.assign(w, 0.0);
    for (std::size_t j = 0; j < nOld; ++j)
        tab.rc[j] = sfOrig.c[j];

    for (std::size_t i = 0; i < m; ++i) {
        double cb = sfOrig.c[tab.basicCols[i]];
        if (cb == 0.0) continue;
        for (std::size_t j = 0; j < w; ++j)
            tab.rc[j] -= cb * tab.tab[i * w + j];
    }
}

/// Simplex loop shared by both phases.
///
/// @p iterConsumed  In/out shared pivot counter drawing from a single maxIter budget.
///
/// @note Complexity: O(K · m · n) total, where K = number of simplex pivots
///   and each pivot costs O(m·n). Periodic reinversion every reinversion_period
///   pivots adds O(m²·n) per cycle.
LPStatus runSimplex(internal::SimplexTableau&       tab,
                    const internal::LPStandardForm& sf,
                    uint32_t                        maxIter,
                    double                          timeLimitS,
                    std::chrono::steady_clock::time_point startTime,
                    uint32_t&                       iterConsumed) {
    uint32_t const timePeriod =
        tab.cfg.reinversionPeriod > 0 ? tab.cfg.reinversionPeriod : 64u;

    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count()
            >= timeLimitS)
        return LPStatus::TimeLimit;

    while (true) {
        if (maxIter > 0 && iterConsumed >= maxIter)
            return LPStatus::MaxIter;

        std::size_t entering = tab.selectEntering();
        if (entering == tab.n)
            return LPStatus::Optimal;

        std::size_t leaving = tab.selectLeaving(entering);
        if (leaving == tab.m)
            return LPStatus::Unbounded;

        tab.pivot(leaving, entering);
        ++iterConsumed;

        if (iterConsumed % timePeriod == 0) {
            if (tab.cfg.reinversionPeriod > 0)
                if (!tab.reinvert(sf)) return LPStatus::NumericalFailure;
            double elapsed =
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - startTime).count();
            if (elapsed >= timeLimitS)
                return LPStatus::TimeLimit;
        }
    }
}

/// Extract the Farkas infeasibility certificate from the phase-I tableau.
///
/// Called after the phase-I simplex terminates with objective > lp_feasibility_tol.
/// At optimality the phase-I dual y satisfies A^T y <= 0 and b^T y > 0.
/// The negated dual y_farkas = -y is the Farkas ray:
///   A_model^T y_farkas >= 0,   b_model^T y_farkas < 0
/// @note Complexity: O(nOrigRows).
FarkasRay extractFarkasPhaseI(const internal::SimplexTableau&  tab,
                               const internal::LPStandardForm& sf,
                               const AugmentedForm&             aug,
                               const Model&                    model) {
    FarkasRay ray;
    const auto& constraints = model.getLPConstraints();
    ray.y.resize(sf.nOrigRows, 0.0);

    for (std::size_t i = 0; i < sf.nOrigRows; ++i) {
        Sense s = constraints[i].sense;
        if (s == Sense::Equal) {
            if (!aug.equalArtCol.empty() && aug.equalArtCol[i] < tab.n)
                ray.y[i] = tab.rc[aug.equalArtCol[i]] - 1.0;
        } else {
            uint32_t slackCol = sf.rowSlackCol[i];
            if (slackCol >= sf.nCols) continue;
            ray.y[i] = (s == Sense::LessEq) ? tab.rc[slackCol] : -tab.rc[slackCol];
        }
    }
    return ray;
}

} // anonymous namespace

namespace internal {

LPDetailedResult solvePrimal(const Model&                          model,
                              uint32_t                              maxIter,
                              double                                timeLimitS,
                              std::chrono::steady_clock::time_point startTime,
                              bool                                  computeSensitivity,
                              bool                                  computeCutData,
                              const SimplexConfig&                  cfg) {
    // Early infeasibility: a variable with lb > ub has an empty domain.
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

    // 1. Standard form
    LPStandardForm sf = toStandardForm(model);

    // 2. Phase I
    AugmentedForm aug = buildPhaseOne(sf, model);
    SimplexTableau tab;
    tab.cfg = cfg;
    [[maybe_unused]] bool initOk = tab.init(aug.sf, aug.initialBasis);
    assert(initOk && "identity basis: cannot be singular");

    uint32_t iters = 0;
    LPStatus p1Status = runSimplex(tab, aug.sf, maxIter, timeLimitS, startTime, iters);

    if (p1Status == LPStatus::MaxIter || p1Status == LPStatus::TimeLimit ||
        p1Status == LPStatus::NumericalFailure) {
        LPDetailedResult det;
        det.result.status = p1Status;
        return det;
    }

    if (tab.objectiveValue() > tab.cfg.feasibilityTol) {
        LPDetailedResult det;
        det.result.status = LPStatus::Infeasible;
        det.farkas        = extractFarkasPhaseI(tab, sf, aug, model);
        return det;
    }

    // 3. Drive remaining artificials out of the basis (degenerate exit)
    driveOutArtificials(tab, sf);

    // 4. Phase II
    preparePhaseTwo(tab, sf);

    LPStatus p2Status = runSimplex(tab, sf, maxIter, timeLimitS, startTime, iters);

    if (p2Status == LPStatus::NumericalFailure) {
        LPDetailedResult det;
        det.result.status = LPStatus::NumericalFailure;
        return det;
    }

    // 5. Extract result
    LPDetailedResult det = extractDetailed(tab, sf, model, p2Status, aug.equalArtCol,
                                           computeSensitivity);

    if (p2Status == LPStatus::Unbounded)
        det.result.primalValues.clear();

    // Populate fractional rows for GMI cut generation.
    // Only the first sf.nCols columns are stored — artificials are excluded.
    if (computeCutData && p2Status == LPStatus::Optimal) {
        const auto& types = model.getCold().types;
        const std::size_t nSF = sf.nCols;
        const std::size_t np  = tab.n + 1;
        constexpr double kIntFeasTol = 1e-6;

        for (std::size_t r = 0; r < tab.m; ++r) {
            uint32_t col = tab.basicCols[r];
            if (col >= nSF) continue; // artificial column
            if (sf.colKind[col] != ColumnKind::Original) continue;
            if (sf.varFreeNegCol[col] < static_cast<uint32_t>(nSF)) continue;
            uint32_t varId = sf.colOrigin[col];
            if (types[varId] != VarType::Integer && types[varId] != VarType::Binary)
                continue;
            double sfBFS = tab.tab[r * np + tab.n];
            double frac  = sfBFS - std::floor(sfBFS);
            if (frac <= kIntFeasTol || frac >= 1.0 - kIntFeasTol) continue;

            FractionalRow fr;
            fr.origVarId = varId;
            fr.fracVal   = frac;
            fr.tabRow.assign(tab.tab.begin() + static_cast<std::ptrdiff_t>(r * np),
                             tab.tab.begin() + static_cast<std::ptrdiff_t>(r * np + nSF));
            det.fractionalRows.push_back(std::move(fr));
        }
    }

    return det;
}

} // namespace internal
} // namespace baguette
