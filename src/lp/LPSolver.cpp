#include "baguette/lp/LPSolver.hpp"

#include <cassert>
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

/// Augmented standard form extended with artificial variables for phase I.
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
    const auto& constraints = model.getConstraints();
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

    // Copy only the fields that are reused as-is; A, c, colKind, colOrigin
    // are rebuilt below for the wider augmented matrix.
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
    // Track the column index of each Equal-row artificial for dual extraction.
    std::size_t artCol = nOld;
    for (std::size_t i = 0; i < m; ++i) {
        if (!needsArt[i]) continue;
        (*aug.sf.A)[i * nNew + artCol] = 1.0;
        aug.sf.c[artCol]            = 1.0;
        aug.sf.colKind[artCol]      = ColumnKind::Slack; // internal marker
        aug.sf.colOrigin[artCol]    = 0;
        if (i < sf.nOrigRows && constraints[i].sense == Sense::Equal)
            aug.equalArtCol[i] = static_cast<uint32_t>(artCol);
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
///
/// @note Complexity: O(nArt · m · n) where nArt = number of artificial basic
///   variables (≤ nOrigRows). Each candidate search costs O(nOld) and each
///   pivot costs O(m·n); at most nArt pivots are performed.
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

/// Fix basicCols entries that still point to artificial columns after
/// driveOutArtificials. This happens for redundant rows (all non-artificial
/// coefficients are zero, rhs = 0): no pivot was possible so the artificial
/// stayed in the basis. Assign any currently non-basic column instead; the
/// row is 0·x = 0 so the choice does not affect the primal solution.
///
/// @note Complexity: O(nRedundant · m · nOld) where nRedundant ≤ m. For each
///   redundant row the all-zero check requires an inner O(m) scan per column,
///   giving O(m · nOld) per redundant row. Typically fast since redundant rows
///   are rare (only occur with linearly dependent constraints).
void repairRedundantRows(internal::Tableau& tab, std::size_t nOld) {
    std::vector<bool> inBasis(nOld, false);
    for (std::size_t i = 0; i < tab.m; ++i)
        if (tab.basicCols[i] < nOld) inBasis[tab.basicCols[i]] = true;

    for (std::size_t i = 0; i < tab.m; ++i) {
        if (tab.basicCols[i] < nOld) continue;
        bool found = false;

        // First pass: prefer all-zero columns. A column that is zero in every
        // tableau row has rc[j] = c[j] after re-pricing, and since a_ij = 0 for
        // all rows, no pivot can ever make rc[j] negative → it can never enter
        // the basis. Assigning it to a redundant row (0 = 0) prevents duplicate
        // basicCols entries that would corrupt primalSolution in phase II.
        for (std::size_t j = 0; j < nOld && !found; ++j) {
            if (inBasis[j]) continue;
            bool allZero = true;
            for (std::size_t r = 0; r < tab.m && allZero; ++r)
                allZero = (std::abs(tab.tab[r * (tab.n + 1) + j]) <= baguette::pivot_tol);
            if (allZero) {
                tab.basicCols[i] = static_cast<uint32_t>(j);
                inBasis[j] = true;
                found = true;
            }
        }

        // Fallback: any non-basic column. With Option A (no phantom slacks for
        // Equal rows) there may be no all-zero column. The assigned column will
        // create a duplicate in basicCols that pivot() resolves via hasRedundantRow.
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
/// @note Complexity: O(m·n) for objective repricing (m basic rows, each
///   updating the length-(n+1) rc vector). m = tab.m, n = tab.n (full width
///   including artificial columns). repairRedundantRows adds O(nRedundant · m · nOld).
void preparePhaseTwo(internal::Tableau& tab,
                     const internal::LPStandardForm& sfOrig) {
    const std::size_t nOld = sfOrig.nCols; // columns in the original (non-augmented) form
    const std::size_t m    = tab.m;
    const std::size_t w    = tab.n + 1;    // full tableau row width (includes artificials + rhs)

    // Restrict pivot selection to original columns only.
    tab.nActive = nOld;

    repairRedundantRows(tab, nOld);

    // Re-price: rc_j = c_j − c_B * B⁻¹ a_j for ALL columns (including artificials).
    // For j < nOld: c_j = sfOrig.c[j].
    // For j >= nOld (artificials): c_j = 0 (phase-II cost).
    tab.rc.assign(w, 0.0);
    for (std::size_t j = 0; j < nOld; ++j)
        tab.rc[j] = sfOrig.c[j];
    // rc[j] already 0 for j >= nOld (artificials) and for the rhs slot at j = tab.n.

    for (std::size_t i = 0; i < m; ++i) {
        // After driveOutArtificials + repairRedundantRows, basicCols[i] < nOld always.
        double cb = sfOrig.c[tab.basicCols[i]];
        if (cb == 0.0) continue;
        for (std::size_t j = 0; j < w; ++j)
            tab.rc[j] -= cb * tab.tab[i * w + j];
    }
}

/// Compute RHS and objective sensitivity ranges from an optimal tableau.
///
/// RHS ranging: for each model constraint i, the range [lo, hi] of b[i] for
/// which the current basis remains primal feasible.  Found by requiring that
/// x_B + Δ·(B⁻¹ e_i) ≥ 0 for all basic rows, where B⁻¹ e_i is read from
/// the slack/surplus column of row i (or the artificial column for Equal rows).
///
/// Objective ranging: for each original variable j, the range [lo, hi] of
/// c[j] for which the current basis remains dual feasible.
///   - Non-basic j: only rc[j] changes → rc[j] + Δc'_sf ≥ 0 → Δc'_sf ≥ −rc[j].
///   - Basic j in row r: rc[k] -= Δc'_sf · tab[r,k] for each non-basic k →
///     dual ratio test over all non-basic columns.
///
/// All ranges are expressed as actual parameter values (not deltas).
/// ±infinity entries indicate the bound is unlimited in that direction.
///
/// @note Complexity: O(m · n_eff) where m = tab.m (standard-form rows) and
///   n_eff is the number of active columns (sf.nCols on the primal path,
///   tab.n on the dual path).  RHS ranging: O(m · nOrigRows).
///   Objective ranging: O(nOrig · (m + n_eff)).  Dominant term: O(m · n_eff).
SensitivityResult extractSensitivity(const internal::Tableau&        tab,
                                     const internal::LPStandardForm& sf,
                                     const Model&                    model,
                                     const std::vector<uint32_t>&    equalArtCol) {
    using Lim = std::numeric_limits<double>;
    const bool   maximize    = (model.getObjSense() == ObjSense::Maximize);
    const auto&  constraints = model.getConstraints();
    const auto&  objCoeffs   = model.getHot().obj;
    const double inf         = Lim::infinity();

    const std::size_t m  = tab.m;
    const std::size_t n  = tab.n;
    const std::size_t np = n + 1; // row stride in tab.tab

    // Effective active columns: original SF cols only (excludes artificials on the
    // primal path, where tab.nActive == sf.nCols < tab.n).
    // On the dual path there are no artificials, so tab.nActive == 0 and we use tab.n.
    const std::size_t nEff = (tab.nActive > 0) ? tab.nActive : n;

    // Build a fast basic-column lookup over [0, nEff).
    std::vector<bool> isBasic(nEff, false);
    for (std::size_t r = 0; r < m; ++r)
        if (tab.basicCols[r] < nEff)
            isBasic[tab.basicCols[r]] = true;

    SensitivityResult sens;

    // ── RHS ranging ──────────────────────────────────────────────────────────────
    // For each model constraint i, find the standard-form delta interval [Δlo, Δhi]
    // such that all basic solution values remain non-negative, then translate to the
    // model b[i] value range.
    sens.rhsRange.resize(sf.nOrigRows);

    for (std::size_t i = 0; i < sf.nOrigRows; ++i) {
        uint32_t colI    = sf.rowSlackCol[i];
        double   dirSign = 1.0;

        // sf.rowSlackCol[i] == sf.nCols is the sentinel for Equal rows (no slack/surplus).
        if (colI < static_cast<uint32_t>(sf.nCols)) {
            // Slack (LessEq) or surplus (GreaterEq) column.
            // d = B⁻¹ e_i = dirSign · tab[:, colI], correcting for the coefficient of
            // colI in standard-form row i:
            //   LessEq   (!negated) : coeff = +1 → dirSign = +1
            //   LessEq   ( negated) : coeff = −1 → dirSign = −1
            //   GreaterEq(!negated) : coeff = −1 → dirSign = −1
            //   GreaterEq( negated) : coeff = +1 → dirSign = +1
            const bool negated = sf.rowNegated[i];
            if (constraints[i].sense == Sense::LessEq)
                dirSign = negated ? -1.0 : +1.0;
            else // GreaterEq
                dirSign = negated ? +1.0 : -1.0;
        } else {
            // Equal row: use the artificial column kept for dual extraction.
            // The artificial has coefficient +1 in the already-normalised row, so
            // tab[:, artCol] = B⁻¹ e_i with no additional sign correction.
            if (equalArtCol.empty() || equalArtCol[i] >= static_cast<uint32_t>(n)) {
                // Dual-simplex path or missing artificial: cannot range Equal row.
                sens.rhsRange[i] = {-inf, +inf};
                continue;
            }
            colI    = equalArtCol[i];
            dirSign = 1.0;
        }

        // Ratio test: largest Δ_sf interval such that x_B + Δ·d ≥ 0 for all rows.
        double deltaLo = -inf, deltaHi = +inf;
        for (std::size_t r = 0; r < m; ++r) {
            const double xBr = tab.tab[r * np + n];
            const double dr  = dirSign * tab.tab[r * np + colI];
            if (dr > baguette::pivot_tol)
                deltaLo = std::max(deltaLo, -xBr / dr);
            else if (dr < -baguette::pivot_tol)
                deltaHi = std::min(deltaHi, -xBr / dr);
        }

        // Translate from standard-form Δ to model-space b[i] range.
        // b'_i = (rowNegated ? −1 : +1) · (b_model[i] − varShift), so:
        //   Δ_model = (rowNegated ? −1 : +1) · Δ_sf
        const double modelRHS = constraints[i].rhs;
        if (!sf.rowNegated[i])
            sens.rhsRange[i] = {modelRHS + deltaLo, modelRHS + deltaHi};
        else
            sens.rhsRange[i] = {modelRHS - deltaHi, modelRHS - deltaLo};
    }

    // ── Objective ranging ─────────────────────────────────────────────────────────
    // For each original variable j, find the standard-form delta interval [Δlo, Δhi]
    // such that all reduced costs remain non-negative, then translate to c[j] range.
    sens.objRange.resize(sf.nOrig);

    for (std::size_t j = 0; j < sf.nOrig; ++j) {
        // Free variables are split as x⁺ − x⁻; coupled analysis not implemented.
        if (sf.varFreeNegCol[j] < static_cast<uint32_t>(sf.nCols)) {
            sens.objRange[j] = {-inf, +inf};
            continue;
        }

        // Conversion factor between the standard-form coefficient c'_j and the
        // model coefficient c_model[j]:  c'_j = factor · c_model[j].
        // factor ∈ {−1, +1}, so Δc_model = factor · Δc'_sf (self-inverse).
        const double factor = static_cast<double>(sf.varColSign[j]) * (maximize ? -1.0 : 1.0);
        const double cModel = objCoeffs[j];

        double deltaLoSF = -inf, deltaHiSF = +inf;

        if (!isBasic[j]) {
            // Non-basic: only rc[j] is affected by changing c'_j.
            // Dual feasibility: rc[j] + Δc'_sf ≥ 0  →  Δc'_sf ≥ −rc[j].
            deltaLoSF = -tab.rc[j];
            // No upper bound from dual feasibility alone.
        } else {
            // Basic in row r: changing c'_j shifts all non-basic reduced costs.
            // new rc[k] = rc[k] − Δc'_sf · tab[r, k] ≥ 0 for non-basic k.
            std::size_t r = 0;
            while (r < m && tab.basicCols[r] != static_cast<uint32_t>(j)) ++r;

            for (std::size_t k = 0; k < nEff; ++k) {
                if (isBasic[k]) continue;
                const double t   = tab.tab[r * np + k];
                const double rck = tab.rc[k];
                if (t > baguette::pivot_tol)
                    deltaHiSF = std::min(deltaHiSF, rck / t);
                else if (t < -baguette::pivot_tol)
                    deltaLoSF = std::max(deltaLoSF, rck / t);
            }
        }

        // Convert from standard-form Δ range to model coefficient range.
        // Δc_model = factor · Δc'_sf; if factor < 0 the direction flips.
        double lo, hi;
        if (factor > 0.0) {
            lo = cModel + deltaLoSF;
            hi = cModel + deltaHiSF;
        } else {
            lo = cModel - deltaHiSF;
            hi = cModel - deltaLoSF;
        }
        sens.objRange[j] = {lo, hi};
    }

    return sens;
}

/// Extract the full LPDetailedResult from a solved phase-II tableau.
///
/// @p equalArtCol  Optional mapping (size == sf.nOrigRows) from each Equal
///                 constraint row to the column index of its artificial variable
///                 in the augmented tableau.  When provided, dual values for
///                 Equal rows are read as  y_i = −rc[equalArtCol[i]]  instead
///                 of being left at zero.  Pass an empty vector when the tableau
///                 contains no artificial columns (e.g. dual-simplex cold start).
/// @note Complexity: O(nOrig + nOrigRows + nCols) for primal/dual/reduced-cost
///   extraction when computeSensitivity is false. When computeSensitivity is true,
///   dominated by extractSensitivity at O(m·n_eff).
LPDetailedResult extractDetailed(const internal::Tableau& tab,
                                 const internal::LPStandardForm& sf,
                                 const Model& model,
                                 LPStatus status,
                                 const std::vector<uint32_t>& equalArtCol = {},
                                 bool computeSensitivity = false) {
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
        double sign = 0.0;
        if (constraints[i].sense == Sense::Equal) {
            // Artificial column for this row was kept in the tableau with c_II = 0.
            // At optimality: rc[art_i] = 0 − y^T * e_i = −y_i  →  y_i = −rc[art_i].
            // Same sign convention as LessEq (slack is also a unit-vector column).
            if (!equalArtCol.empty() && equalArtCol[i] < tab.n) {
                raw  = tab.rc[equalArtCol[i]];
                sign = -1.0;
                if (sf.rowNegated[i]) sign = -sign;
                if (maximize)         sign = -sign;
                det.dualValues[i] = sign * raw;
            } else {
                det.dualValues[i] = 0.0; // dual simplex path: no artificial kept
            }
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

    if (computeSensitivity)
        det.sensitivity = extractSensitivity(tab, sf, model, equalArtCol);

    return det;
}

/// Simplex loop shared by both phases.
///
/// @p iterConsumed  In/out shared pivot counter.  On entry it holds the number
///                  of pivots already used (e.g. by phase I); on return it holds
///                  the updated total.  Passing the same variable to the phase-I
///                  and phase-II calls ensures both phases draw from a single
///                  maxIter budget.
///
/// @note Complexity: O(K · m · n) total, where K = number of simplex pivots
///   (problem-dependent; exponential worst case, polynomial in practice) and
///   each pivot costs O(m·n). Periodic reinversion every reinversion_period
///   pivots adds O(m²·n) per cycle.
LPStatus runSimplex(internal::Tableau& tab,
                    const internal::LPStandardForm& sf,
                    uint32_t maxIter,
                    double   timeLimitS,
                    SolverClock::time_point startTime,
                    uint32_t& iterConsumed) {
    // Check the time limit every reinversion_period pivots to avoid a syscall
    // on the hot path. Fall back to every 64 pivots when reinversion
    // is disabled so the time limit is still honoured.
    uint32_t const timePeriod =
        baguette::reinversion_period > 0 ? baguette::reinversion_period : 64u;

    // Pre-loop check: catches timeLimitS = 0 and exhausted budgets at B&B node entry.
    if (std::chrono::duration<double>(SolverClock::now() - startTime).count() >= timeLimitS)
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
            if (baguette::reinversion_period > 0)
                if (!tab.reinvert(sf)) return LPStatus::NumericalFailure;
            double elapsed =
                std::chrono::duration<double>(SolverClock::now() - startTime).count();
            if (elapsed >= timeLimitS)
                return LPStatus::TimeLimit;
        }
    }
}

/// Dual-simplex loop.
///
/// Precondition: the tableau is dual-feasible (all rc[j] ≥ 0).
/// The loop maintains dual feasibility and drives primal feasibility
/// until all rhs values are ≥ 0 (optimal) or infeasibility is detected.
///
/// @param outBlockingRow  If non-null, set to the leaving row index when
///                        infeasibility is detected (all entries >= 0, rhs < 0).
///                        Enables Farkas certificate extraction by the caller.
///
/// @note Complexity: O(K · m · n) total, where K = number of dual-simplex pivots
///   and each pivot costs O(m·n). Same periodic reinversion schedule as runSimplex.
LPStatus runDualSimplex(internal::Tableau& tab,
                        const internal::LPStandardForm& sf,
                        uint32_t maxIter,
                        double   timeLimitS,
                        SolverClock::time_point startTime,
                        std::size_t* outBlockingRow = nullptr) {
    uint32_t iter = 0;

    // Same batched time-limit check as runSimplex.
    uint32_t const timePeriod =
        baguette::reinversion_period > 0 ? baguette::reinversion_period : 64u;

    // Pre-loop check: catches timeLimitS = 0 and exhausted budgets at B&B node entry.
    if (std::chrono::duration<double>(SolverClock::now() - startTime).count() >= timeLimitS)
        return LPStatus::TimeLimit;

    while (true) {
        if (maxIter > 0 && iter >= maxIter)
            return LPStatus::MaxIter;

        std::size_t leaving = tab.selectLeavingDual();
        if (leaving == tab.m)
            return LPStatus::Optimal; // primal feasible + dual feasible → optimal

        std::size_t entering = tab.selectEnteringDual(leaving);
        if (entering == tab.n) {
            if (outBlockingRow) *outBlockingRow = leaving;
            return LPStatus::Infeasible; // no improving dual pivot → primal infeasible
        }

        tab.pivot(leaving, entering);
        ++iter;

        if (iter % timePeriod == 0) {
            if (baguette::reinversion_period > 0)
                if (!tab.reinvert(sf)) return LPStatus::NumericalFailure;
            double elapsed =
                std::chrono::duration<double>(SolverClock::now() - startTime).count();
            if (elapsed >= timeLimitS)
                return LPStatus::TimeLimit;
        }
    }
}

/// Build a dual-feasible initial basis for the dual simplex.
///
/// For each row, select the column of the natural basic variable:
///   - LessEq row i:     slack column sf.rowSlackCol[i]  (coeff +1, b ≥ 0)
///   - GreaterEq row i:  surplus column sf.rowSlackCol[i] (coeff −1)
///                       Gauss-Jordan will divide by −1, negating the row.
///                       The rhs in the tableau becomes −b[i] ≤ 0 (primal infeasible).
///   - Upper-bound row:  UpperSlack column (coeff +1, b = ub−lb ≥ 0)
///
/// Dual feasibility of the resulting basis requires rc[j] ≥ 0 for all
/// non-basic original columns, which holds when sf.c[j] ≥ 0 (all objective
/// coefficients non-negative after shifting / sign convention).
///
/// @returns The initial basis vector, or an empty vector if the model contains
///          Sense::Equal constraints (which have no natural basic variable).
/// @note Complexity: O(m), where m = sf.nRows. Includes an O(nOrigRows) scan
///   for Equal constraints and an O(nRows) pass to fill the basis vector.
std::vector<uint32_t> buildDualBasis(const internal::LPStandardForm& sf,
                                     const Model& model) {
    const auto& constraints = model.getConstraints();

    // Check: any Equal constraint → no natural basis available
    for (std::size_t i = 0; i < sf.nOrigRows; ++i)
        if (constraints[i].sense == Sense::Equal)
            return {}; // signal: not directly applicable

    std::vector<uint32_t> basis(sf.nRows);

    // Model-constraint rows
    for (std::size_t i = 0; i < sf.nOrigRows; ++i)
        basis[i] = sf.rowSlackCol[i]; // slack (LessEq) or surplus (GEQ)

    // Upper-bound rows (indices nOrigRows .. nRows-1)
    // UpperSlack columns: nOrig + nSlack, nOrig + nSlack + 1, ...
    for (std::size_t i = sf.nOrigRows; i < sf.nRows; ++i)
        basis[i] = static_cast<uint32_t>(sf.nOrig + sf.nSlack + (i - sf.nOrigRows));

    return basis;
}

/// Extract the Farkas infeasibility certificate from the dual-simplex blocking row.
///
/// Called when selectEnteringDual() returned n for @p leavingRow, meaning every
/// tableau entry in that row is >= 0 while the rhs is < 0.  The row represents
/// a linear combination of the original standard-form rows, and reading its
/// slack/surplus column entries recovers the Farkas multipliers y such that:
///   A_model^T y >= 0   (ensured by non-negativity of blocking row entries)
///   b_model^T y < 0    (ensured by negative rhs of blocking row)
///
/// Derivation (sign convention):
///   tab[r, rowSlackCol[i]] = (B^{-1} A_SF)_{r, slackCol_i}
///   For LessEq row i (A_SF[i, slackCol_i] = +/-1 due to possible negation):
///     y_model[i] = tab[r, rowSlackCol[i]]   (negation signs cancel)
///   For GEQ row i:
///     y_model[i] = -tab[r, rowSlackCol[i]]  (idem)
///   Equal rows have no slack column and do not arise on the dual-simplex path
///   (they force a fallback to primal), so y[i] = 0 for those rows.
/// @note Complexity: O(nOrigRows).
FarkasRay extractFarkasDualRow(const internal::Tableau&        tab,
                               const internal::LPStandardForm& sf,
                               const Model&                    model,
                               std::size_t                     leavingRow) {
    FarkasRay ray;
    const auto& constraints = model.getConstraints();
    ray.y.resize(sf.nOrigRows, 0.0);
    const std::size_t w = tab.n + 1;

    for (std::size_t i = 0; i < sf.nOrigRows; ++i) {
        uint32_t slackCol = sf.rowSlackCol[i];
        if (slackCol >= sf.nCols) continue; // Equal row — no slack, leave 0

        double entry = tab.tab[leavingRow * w + slackCol];
        ray.y[i] = (constraints[i].sense == Sense::LessEq) ? entry : -entry;
    }
    return ray;
}

/// Extract the Farkas infeasibility certificate from the phase-I tableau.
///
/// Called after the phase-I simplex terminates with objective > lp_feasibility_tol,
/// meaning the original system has no feasible solution.  At optimality the
/// phase-I dual variables y satisfy A^T y <= 0 and b^T y > 0 (dual feasible,
/// positive dual objective).  The negated dual, y_farkas = -y, is the Farkas ray:
///   A_model^T y_farkas >= 0,   b_model^T y_farkas < 0
///
/// Extraction from the reduced-cost row (same sign simplification as the dual case):
///   LessEq row i  : y_farkas[i] = rc[rowSlackCol[i]]
///   GEQ row i     : y_farkas[i] = -rc[rowSlackCol[i]]
///   Equal row i   : y_farkas[i] = rc[equalArtCol[i]] - 1
/// @note Complexity: O(nOrigRows).
FarkasRay extractFarkasPhaseI(const internal::Tableau&        tab,
                              const internal::LPStandardForm& sf,
                              const AugmentedForm&             aug,
                              const Model&                    model) {
    FarkasRay ray;
    const auto& constraints = model.getConstraints();
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

// ── Public API ────────────────────────────────────────────────────────────────

LPDetailedResult solveDetailed(const Model& model,
                               uint32_t maxIter,
                               double   timeLimitS,
                               SolverClock::time_point startTime,
                               bool     computeSensitivity,
                               bool     computeCutData) {

    // Early infeasibility: a variable with lb > ub has an empty domain.
    // This arises naturally in B&B after bound tightening.
    {
        const auto& hot = model.getHot();
        for (std::size_t j = 0; j < model.numVars(); ++j) {
            if (hot.lb[j] > hot.ub[j]) {
                LPDetailedResult det;
                det.result.status     = LPStatus::Infeasible;
                det.farkas.infeasVarId = static_cast<int32_t>(j);
                return det;
            }
        }
    }

    // 1. Standard form
    internal::LPStandardForm sf = internal::toStandardForm(model);

    // 2. Phase I
    AugmentedForm aug = buildPhaseOne(sf, model);
    internal::Tableau tab;
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

    // Phase-I objective is always ≥ 0 (sum of non-negative artificials).
    // If it is still > lp_feasibility_tol after minimisation, the problem
    // has no feasible solution.
    if (tab.objectiveValue() > baguette::lp_feasibility_tol) {
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

    // Populate fractional rows for GMI cut generation (primal path).
    // Only the first sf.nCols columns are stored — artificials are excluded.
    if (computeCutData && p2Status == LPStatus::Optimal) {
        const auto& types = model.getCold().types;
        const std::size_t nSF = sf.nCols; // original SF columns (no artificials)
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

LPResult solve(const Model&            model,
               uint32_t                maxIter,
               double                  timeLimitS,
               SolverClock::time_point startTime) {
    return solveDetailed(model, maxIter, timeLimitS, startTime).result;
}

LPDetailedResult solveDualDetailed(const Model&            model,
                                   uint32_t                maxIter,
                                   double                  timeLimitS,
                                   SolverClock::time_point startTime,
                                   const BasisRecord&      warmBasis,
                                   bool                    computeSensitivity,
                                   bool                    computeCutData) {

    // Early infeasibility: empty variable domain (including B&B bound tightening).
    {
        const auto& hot = model.getHot();
        for (std::size_t j = 0; j < model.numVars(); ++j) {
            if (hot.lb[j] > hot.ub[j]) {
                LPDetailedResult det;
                det.result.status     = LPStatus::Infeasible;
                det.farkas.infeasVarId = static_cast<int32_t>(j);
                return det;
            }
        }
    }

    // On the warm-start path, A and c are invariant between B&B nodes.
    // Reuse the cached A (shared_ptr copy is O(1)) and update only b,
    // varShiftVal, and objOffset via toStandardFormBoundsOnly.
    auto sfPtr = std::make_shared<internal::LPStandardForm>();
    if (!warmBasis.basicCols.empty() && warmBasis.sfCache) {
        *sfPtr = *warmBasis.sfCache; // shallow: A shared_ptr copied O(1)
        if (!internal::toStandardFormBoundsOnly(*sfPtr, model))
            *sfPtr = internal::toStandardForm(model); // structure changed: full rebuild
    } else {
        *sfPtr = internal::toStandardForm(model);
    }
    internal::LPStandardForm& sf = *sfPtr;
    internal::Tableau tab;

    // Helper: fall back to a primal solve and attach sfPtr so the next call
    // can use toStandardFormBoundsOnly instead of rebuilding A from scratch.
    // sfPtr is always a valid toStandardForm(model) at this point.
    auto fallback = [&]() -> LPDetailedResult {
        LPDetailedResult det = solveDetailed(model, maxIter, timeLimitS, startTime,
                                             computeSensitivity, computeCutData);
        if (det.result.status == LPStatus::Optimal)
            det.basis.sfCache = sfPtr;
        return det;
    };

    if (!warmBasis.basicCols.empty()) {
        // ── Warm-start path ──────────────────────────────────────────────────
        // A size mismatch means the bound-finiteness invariant was violated
        // (a variable gained or lost a finite bound, changing the number of
        // upper-bound rows or free-split columns). Fall back to cold primal.
        if (warmBasis.basicCols.size() != sf.nRows ||
            warmBasis.colKind.size()   != sf.nCols) {
            return fallback();
        }

        // Seed the tableau with the parent's basis and reinvert.
        // reinvert rebuilds B⁻¹A with the NEW b (updated bounds shift the RHS)
        // while A and c are unchanged → RC unchanged, dual feasibility preserved.
        tab.basicCols = warmBasis.basicCols;
        if (!tab.reinvert(sf))
            return fallback();
    } else {
        // ── Cold dual-start path ─────────────────────────────────────────────
        // Build a dual-feasible initial basis from natural slack/surplus columns.
        // Returns empty if any Equal constraint is present (no natural basic var).
        std::vector<uint32_t> coldBasis = buildDualBasis(sf, model);
        if (coldBasis.empty()) {
            // Fallback: Equal constraints prevent a natural dual-feasible basis.
            return fallback();
        }
        // Gauss-Jordan will negate GEQ rows (pivot on coeff −1), producing
        // negative rhs values for those rows (primal infeasible, dual feasible).
        [[maybe_unused]] bool coldOk = tab.init(sf, coldBasis);
        assert(coldOk && "slack/surplus basis: cannot be singular");
    }

    // Verify dual feasibility (shared by both paths).
    // Cold path: required for correctness (e.g. objective has negative coefficients).
    // Warm path: preserved by bound tightening but checked as safety net and
    //            to detect Maximize / negative-cost fallback cases.
    for (std::size_t j = 0; j < sf.nCols; ++j) {
        if (tab.rc[j] < -baguette::lp_optimality_tol) {
            return fallback();
        }
    }

    std::size_t blockingRow = tab.m; // sentinel: invalid
    LPStatus status = runDualSimplex(tab, sf, maxIter, timeLimitS, startTime, &blockingRow);

    if (status == LPStatus::NumericalFailure) {
        LPDetailedResult det;
        det.result.status = LPStatus::NumericalFailure;
        return det;
    }

    LPDetailedResult det = extractDetailed(tab, sf, model, status, {}, computeSensitivity);
    if (status != LPStatus::Optimal) {
        det.result.primalValues.clear();
        if (status == LPStatus::Infeasible && blockingRow < tab.m)
            det.farkas = extractFarkasDualRow(tab, sf, model, blockingRow);
    } else {
        det.basis.sfCache = sfPtr;

        if (computeCutData) {
            const auto& types = model.getCold().types;
            const std::size_t np = tab.n + 1;
            constexpr double kIntFeasTol = 1e-6;

            for (std::size_t r = 0; r < tab.m; ++r) {
                uint32_t col = tab.basicCols[r];
                if (sf.colKind[col] != ColumnKind::Original) continue;
                // Skip x⁺ of free-split variables; GMI formula requires a single column.
                if (sf.varFreeNegCol[col] < static_cast<uint32_t>(sf.nCols)) continue;
                uint32_t varId = sf.colOrigin[col];
                if (types[varId] != VarType::Integer && types[varId] != VarType::Binary)
                    continue;
                // Use the SF variable value (b̄_r) so that frac is frac(x'_j),
                // which is what the GMI formula requires. For lb-shifted integer
                // variables this equals frac(x_model); for ub-shifted it differs.
                double sfBFS = tab.tab[r * np + tab.n];
                double frac  = sfBFS - std::floor(sfBFS);
                if (frac <= kIntFeasTol || frac >= 1.0 - kIntFeasTol) continue;

                FractionalRow fr;
                fr.origVarId = varId;
                fr.fracVal   = frac;
                fr.tabRow.assign(tab.tab.begin() + static_cast<std::ptrdiff_t>(r * np),
                                 tab.tab.begin() + static_cast<std::ptrdiff_t>(r * np + tab.n));
                det.fractionalRows.push_back(std::move(fr));
            }
        }
    }

    return det;
}

LPResult solveDual(const Model&            model,
                   uint32_t                maxIter,
                   double                  timeLimitS,
                   SolverClock::time_point startTime,
                   const BasisRecord&      warmBasis) {
    return solveDualDetailed(model, maxIter, timeLimitS, startTime, warmBasis).result;
}

} // namespace baguette
