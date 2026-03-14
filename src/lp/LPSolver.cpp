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

    // Initialise Equal-row artificial mapping with sentinel (= no artificial).
    aug.equalArtCol.assign(sf.nOrigRows, static_cast<uint32_t>(nOld + nArt));

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

    // Append artificial columns (phase-I objective = 1 for each).
    // Track the column index of each Equal-row artificial for dual extraction.
    std::size_t artCol = nOld;
    for (std::size_t i = 0; i < m; ++i) {
        if (!needsArt[i]) continue;
        aug.sf.A[i * nNew + artCol] = 1.0;
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
/// enabling dual-variable extraction for Equal constraints (bug #18).
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

/// Extract the full LPDetailedResult from a solved phase-II tableau.
///
/// @p equalArtCol  Optional mapping (size == sf.nOrigRows) from each Equal
///                 constraint row to the column index of its artificial variable
///                 in the augmented tableau.  When provided, dual values for
///                 Equal rows are read as  y_i = −rc[equalArtCol[i]]  instead
///                 of being left at zero.  Pass an empty vector when the tableau
///                 contains no artificial columns (e.g. dual-simplex cold start).
LPDetailedResult extractDetailed(const internal::Tableau& tab,
                                 const internal::LPStandardForm& sf,
                                 const Model& model,
                                 LPStatus status,
                                 const std::vector<uint32_t>& equalArtCol = {}) {
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

    return det;
}

/// Simplex loop shared by both phases.
///
/// @p iterConsumed  In/out shared pivot counter.  On entry it holds the number
///                  of pivots already used (e.g. by phase I); on return it holds
///                  the updated total.  Passing the same variable to the phase-I
///                  and phase-II calls ensures both phases draw from a single
///                  maxIter budget (bug #19).
LPStatus runSimplex(internal::Tableau& tab,
                    const internal::LPStandardForm& sf,
                    uint32_t maxIter,
                    double   timeLimitS,
                    SolverClock::time_point startTime,
                    uint32_t& iterConsumed) {
    while (true) {
        if (maxIter > 0 && iterConsumed >= maxIter)
            return LPStatus::MaxIter;

        if (timeLimitS > 0.0) {
            double elapsed =
                std::chrono::duration<double>(SolverClock::now() - startTime).count();
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
        ++iterConsumed;

        if (baguette::reinversion_period > 0 &&
            iterConsumed % baguette::reinversion_period == 0)
            if (!tab.reinvert(sf)) return LPStatus::NumericalFailure;
    }
}

/// Dual-simplex loop.
///
/// Precondition: the tableau is dual-feasible (all rc[j] ≥ 0).
/// The loop maintains dual feasibility and drives primal feasibility
/// until all rhs values are ≥ 0 (optimal) or infeasibility is detected.
LPStatus runDualSimplex(internal::Tableau& tab,
                        const internal::LPStandardForm& sf,
                        uint32_t maxIter,
                        double   timeLimitS,
                        SolverClock::time_point startTime) {
    uint32_t iter = 0;

    while (true) {
        if (maxIter > 0 && iter >= maxIter)
            return LPStatus::MaxIter;

        if (timeLimitS > 0.0) {
            double elapsed =
                std::chrono::duration<double>(SolverClock::now() - startTime).count();
            if (elapsed >= timeLimitS)
                return LPStatus::TimeLimit;
        }

        std::size_t leaving = tab.selectLeavingDual();
        if (leaving == tab.m)
            return LPStatus::Optimal; // primal feasible + dual feasible → optimal

        std::size_t entering = tab.selectEnteringDual(leaving);
        if (entering == tab.n)
            return LPStatus::Infeasible; // no improving dual pivot → primal infeasible

        tab.pivot(leaving, entering);
        ++iter;

        if (baguette::reinversion_period > 0 &&
            iter % baguette::reinversion_period == 0)
            if (!tab.reinvert(sf)) return LPStatus::NumericalFailure;
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

} // anonymous namespace

// ── Public API ────────────────────────────────────────────────────────────────

LPDetailedResult solveDetailed(const Model& model,
                               uint32_t maxIter,
                               double   timeLimitS,
                               SolverClock::time_point startTime) {

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
    LPDetailedResult det = extractDetailed(tab, sf, model, p2Status, aug.equalArtCol);

    if (p2Status == LPStatus::Unbounded)
        det.result.primalValues.clear();

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
                                   const BasisRecord&      warmBasis) {

    // Early infeasibility: empty variable domain (including B&B bound tightening).
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

    internal::LPStandardForm sf = internal::toStandardForm(model);
    internal::Tableau tab;

    if (!warmBasis.basicCols.empty()) {
        // ── Warm-start path ──────────────────────────────────────────────────
        // A size mismatch means the bound-finiteness invariant was violated
        // (a variable gained or lost a finite bound, changing the number of
        // upper-bound rows or free-split columns). Fall back to cold primal.
        if (warmBasis.basicCols.size() != sf.nRows ||
            warmBasis.colKind.size()   != sf.nCols) {
            return solveDetailed(model, maxIter, timeLimitS, startTime);
        }

        // Seed the tableau with the parent's basis and reinvert.
        // reinvert rebuilds B⁻¹A with the NEW b (updated bounds shift the RHS)
        // while A and c are unchanged → RC unchanged, dual feasibility preserved.
        tab.basicCols = warmBasis.basicCols;
        if (!tab.reinvert(sf))
            return solveDetailed(model, maxIter, timeLimitS, startTime);
    } else {
        // ── Cold dual-start path ─────────────────────────────────────────────
        // Build a dual-feasible initial basis from natural slack/surplus columns.
        // Returns empty if any Equal constraint is present (no natural basic var).
        std::vector<uint32_t> coldBasis = buildDualBasis(sf, model);
        if (coldBasis.empty()) {
            // Fallback: Equal constraints prevent a natural dual-feasible basis.
            return solveDetailed(model, maxIter, timeLimitS, startTime);
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
            return solveDetailed(model, maxIter, timeLimitS, startTime);
        }
    }

    LPStatus status = runDualSimplex(tab, sf, maxIter, timeLimitS, startTime);

    if (status == LPStatus::NumericalFailure) {
        LPDetailedResult det;
        det.result.status = LPStatus::NumericalFailure;
        return det;
    }

    LPDetailedResult det = extractDetailed(tab, sf, model, status);
    if (status != LPStatus::Optimal)
        det.result.primalValues.clear();

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
