#include "RevisedSimplex.hpp"

#include <cassert>
#include <cmath>
#include <limits>

#include "LUTableau.hpp"
#include "StandardForm.hpp"
#include "baguette/core/Sense.hpp"
#include "baguette/model/ModelEnums.hpp"

namespace baguette {

namespace {

// ── Augmented standard form (phase I) ────────────────────────────────────────

struct AugmentedFormRev {
    internal::LPStandardForm sf;
    std::vector<uint32_t>    initialBasis;
    std::size_t              nArt = 0;
    /// For each original-row Equal constraint: column index of its artificial.
    /// Sentinel = sf.nCols (no artificial needed / not an Equal row).
    std::vector<uint32_t>    equalArtCol;
};

/// Build the phase-I augmented form.  Mirrors buildPhaseOne() in PrimalSimplex.cpp.
AugmentedFormRev buildPhaseOneRev(const internal::LPStandardForm& sf,
                                   const Model&                    model) {
    const auto& constraints = model.getLPConstraints();
    const std::size_t m    = sf.nRows;
    const std::size_t nOld = sf.nCols;

    std::vector<bool> needsArt(m, false);
    for (std::size_t i = 0; i < sf.nOrigRows; ++i) {
        Sense s = constraints[i].sense;
        needsArt[i] = (s == Sense::GreaterEq || s == Sense::Equal || sf.rowNegated[i]);
    }

    std::size_t nArt = 0;
    for (bool b : needsArt) if (b) ++nArt;

    AugmentedFormRev aug;
    aug.nArt = nArt;

    aug.sf.nRows      = sf.nRows;
    aug.sf.nOrigRows  = sf.nOrigRows;
    aug.sf.nOrig      = sf.nOrig;
    aug.sf.nSlack     = sf.nSlack;
    aug.sf.b          = sf.b;
    aug.sf.rowSlackCol = sf.rowSlackCol;
    aug.sf.rowNegated  = sf.rowNegated;
    aug.sf.varShiftVal = sf.varShiftVal;
    aug.sf.varColSign  = sf.varColSign;
    aug.sf.varFreeNegCol = sf.varFreeNegCol;
    aug.sf.objOffset   = sf.objOffset;

    aug.equalArtCol.assign(sf.nOrigRows, static_cast<uint32_t>(nOld + nArt));

    const std::size_t nNew = nOld + nArt;
    aug.sf.nCols = nNew;
    aug.sf.A     = std::make_shared<std::vector<double>>(m * nNew, 0.0);
    aug.sf.c.assign(nNew, 0.0);
    aug.sf.colKind.resize(nNew);
    aug.sf.colOrigin.resize(nNew);

    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < nOld; ++j)
            (*aug.sf.A)[i * nNew + j] = (*sf.A)[i * nOld + j];

    for (std::size_t j = 0; j < nOld; ++j) {
        aug.sf.colKind[j]   = sf.colKind[j];
        aug.sf.colOrigin[j] = sf.colOrigin[j];
    }

    std::size_t artCol = nOld;
    for (std::size_t i = 0; i < m; ++i) {
        if (i < sf.nOrigRows && !needsArt[i]) continue;
        if (i >= sf.nOrigRows) continue; // upper-bound rows have natural slack

        (*aug.sf.A)[i * nNew + artCol] = 1.0;
        aug.sf.c[artCol] = 1.0;
        aug.sf.colKind[artCol]   = ColumnKind::Slack;
        aug.sf.colOrigin[artCol] = static_cast<uint32_t>(i);

        if (constraints[i].sense == Sense::Equal)
            aug.equalArtCol[i] = static_cast<uint32_t>(artCol);

        ++artCol;
    }

    aug.initialBasis.resize(m);
    std::size_t nextArt = nOld;
    for (std::size_t i = 0; i < m; ++i) {
        if (i >= sf.nOrigRows) {
            aug.initialBasis[i] = static_cast<uint32_t>(
                sf.nOrig + sf.nSlack + (i - sf.nOrigRows));
        } else if (!needsArt[i]) {
            aug.initialBasis[i] = sf.rowSlackCol[i];
        } else {
            aug.initialBasis[i] = static_cast<uint32_t>(nextArt++);
        }
    }

    return aug;
}

/// Drive any phase-I artificials still in the basis (at value 0) out.
/// Uses tableau rows computed on demand.
void driveOutArtificialsRev(internal::LUTableau&             tab,
                             const internal::LPStandardForm& sfOrig) {
    const std::size_t nOld = sfOrig.nCols;

    for (std::size_t i = 0; i < tab.m; ++i) {
        if (tab.basicCols[i] < nOld) continue;
        // Find a non-zero entry in row i for an original column
        auto trow = tab.tableauRow(i);
        for (std::size_t j = 0; j < nOld; ++j) {
            if (std::abs(trow[j]) > tab.cfg.pivotTol) {
                tab.pivot(i, j);
                break;
            }
        }
    }
}

// ── Simplex loop (primal) ─────────────────────────────────────────────────────

LPStatus runSimplexRev(internal::LUTableau&                       tab,
                        const internal::LPStandardForm&            sf,
                        uint32_t                                   maxIter,
                        double                                     timeLimitS,
                        std::chrono::steady_clock::time_point      startTime,
                        uint32_t&                                  iterConsumed) {
    const uint32_t timePeriod =
        tab.cfg.reinversionPeriod > 0 ? tab.cfg.reinversionPeriod : 64u;

    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count()
            >= timeLimitS)
        return LPStatus::TimeLimit;

    while (true) {
        if (maxIter > 0 && iterConsumed >= maxIter)
            return LPStatus::MaxIter;

        std::size_t entering = tab.selectEntering();
        if (entering == tab.n) return LPStatus::Optimal;

        auto [leaving, eta] = tab.selectLeavingWithEta(entering);
        if (leaving == tab.m) return LPStatus::Unbounded;

        tab.pivot(leaving, entering, eta);
        ++iterConsumed;

        if (iterConsumed % timePeriod == 0) {
            if (tab.cfg.reinversionPeriod > 0)
                if (!tab.reinvert(sf)) return LPStatus::NumericalFailure;
            if (std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - startTime).count() >= timeLimitS)
                return LPStatus::TimeLimit;
        }
    }
}

// ── Farkas certificate from phase-I ──────────────────────────────────────────

FarkasRay extractFarkasPhaseIRev(const internal::LUTableau&       tab,
                                  const internal::LPStandardForm&  sf,
                                  const AugmentedFormRev&          aug,
                                  const Model&                     model) {
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

// ── Result extraction from LUTableau ─────────────────────────────────────────

/// Extract LPDetailedResult from an optimal or terminal LUTableau.
/// Sensitivity analysis uses on-demand tableau-row computation.
LPDetailedResult extractDetailedRev(const internal::LUTableau&       tab,
                                     const internal::LPStandardForm&  sf,
                                     const Model&                     model,
                                     LPStatus                         status,
                                     const std::vector<uint32_t>&     equalArtCol,
                                     bool                             computeSensitivity) {
    using Lim = std::numeric_limits<double>;

    LPDetailedResult det;
    det.result.status = status;

    const std::size_t nOrig   = sf.nOrig;
    const bool        maximize = (model.getObjSense() == ObjSense::Maximize);

    // Primal: un-shift using varShiftVal and varColSign
    std::vector<double> xPrime = tab.primalSolution();
    det.result.primalValues.resize(nOrig);
    for (std::size_t j = 0; j < nOrig; ++j) {
        double val = sf.varShiftVal[j] + sf.varColSign[j] * xPrime[j];
        uint32_t negCol = sf.varFreeNegCol[j];
        if (negCol < sf.nCols)
            val -= xPrime[negCol];
        det.result.primalValues[j] = val;
    }

    double obj = tab.objectiveValue() + sf.objOffset;
    det.result.objectiveValue = maximize ? -obj : obj;

    if (status != LPStatus::Optimal) return det;

    // Dual variables (from reduced costs of slack/artificial columns)
    const auto& constraints = model.getLPConstraints();
    det.dualValues.resize(sf.nOrigRows);
    for (std::size_t i = 0; i < sf.nOrigRows; ++i) {
        uint32_t slackCol = sf.rowSlackCol[i];
        double   raw      = 0.0;
        double   sign     = 0.0;

        if (constraints[i].sense == Sense::Equal) {
            if (!equalArtCol.empty() && equalArtCol[i] < tab.n) {
                raw  = tab.rc[equalArtCol[i]];
                sign = -1.0;
                if (sf.rowNegated[i]) sign = -sign;
                if (maximize)         sign = -sign;
                det.dualValues[i] = sign * raw;
            }
            continue;
        }
        raw  = tab.rc[slackCol];
        sign = (constraints[i].sense == Sense::LessEq) ? -1.0 : +1.0;
        if (sf.rowNegated[i]) sign = -sign;
        if (maximize)         sign = -sign;
        det.dualValues[i] = sign * raw;
    }

    // Reduced costs for original variables
    det.reducedCosts.resize(nOrig);
    for (std::size_t j = 0; j < nOrig; ++j)
        det.reducedCosts[j] = sf.varColSign[j] * (maximize ? -tab.rc[j] : tab.rc[j]);

    // Basis record
    det.basis.basicCols.assign(tab.basicCols.begin(), tab.basicCols.end());
    det.basis.colKind   = sf.colKind;
    det.basis.colOrigin = sf.colOrigin;

    // Sensitivity analysis via on-demand tableau rows
    if (computeSensitivity) {
        const double inf     = Lim::infinity();
        const std::size_t m  = tab.m;
        const std::size_t n  = tab.n;
        const std::size_t nEff = (tab.nActive > 0) ? tab.nActive : n;

        std::vector<bool> isBasic(nEff, false);
        for (std::size_t r = 0; r < m; ++r)
            if (tab.basicCols[r] < nEff) isBasic[tab.basicCols[r]] = true;

        SensitivityResult sens;

        // RHS ranging: for each model constraint i, compute B⁻¹ a_{slackCol}
        sens.rhsRange.resize(sf.nOrigRows);
        for (std::size_t i = 0; i < sf.nOrigRows; ++i) {
            uint32_t colI    = sf.rowSlackCol[i];
            double   dirSign = 1.0;

            if (colI < static_cast<uint32_t>(sf.nCols)) {
                const bool negated = sf.rowNegated[i];
                if (constraints[i].sense == Sense::LessEq)
                    dirSign = negated ? -1.0 : +1.0;
                else
                    dirSign = negated ? +1.0 : -1.0;
            } else {
                // Equal row: use artificial column
                if (equalArtCol.empty() || equalArtCol[i] >= static_cast<uint32_t>(n)) {
                    sens.rhsRange[i] = {-inf, +inf};
                    continue;
                }
                colI    = equalArtCol[i];
                dirSign = 1.0;
            }

            // Entering column d = B⁻¹ a_{colI} gives the direction in xB space
            auto d = tab.enteringColumn(colI);
            double deltaLo = -inf, deltaHi = +inf;
            for (std::size_t r = 0; r < m; ++r) {
                double dr = dirSign * d[r];
                if (dr > tab.cfg.pivotTol)
                    deltaLo = std::max(deltaLo, -tab.xB[r] / dr);
                else if (dr < -tab.cfg.pivotTol)
                    deltaHi = std::min(deltaHi, -tab.xB[r] / dr);
            }

            const double modelRHS = constraints[i].rhs;
            if (!sf.rowNegated[i])
                sens.rhsRange[i] = {modelRHS + deltaLo, modelRHS + deltaHi};
            else
                sens.rhsRange[i] = {modelRHS - deltaHi, modelRHS - deltaLo};
        }

        // Objective ranging
        const auto& objCoeffs = model.getHot().obj;
        sens.objRange.resize(sf.nOrig);
        for (std::size_t j = 0; j < sf.nOrig; ++j) {
            if (sf.varFreeNegCol[j] < static_cast<uint32_t>(sf.nCols)) {
                sens.objRange[j] = {-inf, +inf};
                continue;
            }
            const double factor = static_cast<double>(sf.varColSign[j]) * (maximize ? -1.0 : 1.0);
            const double cModel = objCoeffs[j];

            double deltaLoSF = -inf, deltaHiSF = +inf;

            if (!isBasic[j]) {
                deltaLoSF = -tab.rc[j];
            } else {
                std::size_t r = 0;
                while (r < m && tab.basicCols[r] != static_cast<uint32_t>(j)) ++r;

                auto trow = tab.tableauRow(r);
                for (std::size_t k = 0; k < nEff; ++k) {
                    if (isBasic[k]) continue;
                    double t   = trow[k];
                    double rck = tab.rc[k];
                    if (t > tab.cfg.pivotTol)
                        deltaHiSF = std::min(deltaHiSF, rck / t);
                    else if (t < -tab.cfg.pivotTol)
                        deltaLoSF = std::max(deltaLoSF, rck / t);
                }
            }

            double lo, hi;
            if (factor > 0.0) { lo = cModel + deltaLoSF; hi = cModel + deltaHiSF; }
            else               { lo = cModel - deltaHiSF; hi = cModel - deltaLoSF; }
            sens.objRange[j] = {lo, hi};
        }

        det.sensitivity = std::move(sens);
    }

    return det;
}

} // anonymous namespace

namespace internal {

LPDetailedResult solveRevised(const Model&                          model,
                               uint32_t                              maxIter,
                               double                                timeLimitS,
                               std::chrono::steady_clock::time_point startTime,
                               bool                                  computeSensitivity,
                               bool                                  computeCutData,
                               const SimplexConfig&                  cfg) {
    // Early infeasibility: empty variable domain
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

    // 2. Phase I: augment with artificial variables
    AugmentedFormRev aug = buildPhaseOneRev(sf, model);
    LUTableau tab;
    tab.cfg = cfg;
    [[maybe_unused]] bool initOk = tab.init(aug.sf, aug.initialBasis);
    assert(initOk && "identity artificial basis: cannot be singular");

    uint32_t iters = 0;
    LPStatus p1Status = runSimplexRev(tab, aug.sf, maxIter, timeLimitS, startTime, iters);

    if (p1Status == LPStatus::MaxIter || p1Status == LPStatus::TimeLimit ||
        p1Status == LPStatus::NumericalFailure) {
        LPDetailedResult det;
        det.result.status = p1Status;
        return det;
    }

    if (tab.objectiveValue() > tab.cfg.feasibilityTol) {
        LPDetailedResult det;
        det.result.status = LPStatus::Infeasible;
        det.farkas        = extractFarkasPhaseIRev(tab, sf, aug, model);
        return det;
    }

    // 3. Drive remaining artificials out of the basis (degenerate exit)
    driveOutArtificialsRev(tab, sf);

    // 4. Phase II: switch to real objective, restrict entering to original cols
    {
        // Build phase-II objective: c[j] = sfOrig.c[j] for j < nOld, 0 for artificials
        std::vector<double> c2(aug.sf.nCols, 0.0);
        for (std::size_t j = 0; j < sf.nCols; ++j)
            c2[j] = sf.c[j];
        tab.repriceObjective(c2, sf.nCols);
    }

    LPStatus p2Status = runSimplexRev(tab, aug.sf, maxIter, timeLimitS, startTime, iters);

    if (p2Status == LPStatus::NumericalFailure) {
        LPDetailedResult det;
        det.result.status = LPStatus::NumericalFailure;
        return det;
    }

    // 5. Extract result (using the augmented sf for column indices,
    //    but the original sf for metadata like rowSlackCol, varShiftVal, etc.)
    //    equalArtCol indices refer to aug.sf columns, which are valid in tab.rc.
    LPDetailedResult det = extractDetailedRev(tab, sf, model, p2Status,
                                               aug.equalArtCol, computeSensitivity);

    det.iterationsUsed = iters;

    if (p2Status == LPStatus::Unbounded)
        det.result.primalValues.clear();

    // Cut data for GMI generation
    if (computeCutData && p2Status == LPStatus::Optimal) {
        const auto& types = model.getCold().types;
        const std::size_t nSF = sf.nCols;
        constexpr double kIntFeasTol = 1e-6;

        for (std::size_t r = 0; r < tab.m; ++r) {
            uint32_t col = tab.basicCols[r];
            if (col >= nSF) continue;
            if (sf.colKind[col] != ColumnKind::Original) continue;
            if (sf.varFreeNegCol[col] < static_cast<uint32_t>(nSF)) continue;
            uint32_t varId = sf.colOrigin[col];
            if (types[varId] != VarType::Integer && types[varId] != VarType::Binary)
                continue;
            double sfBFS = tab.xB[r];
            double frac  = sfBFS - std::floor(sfBFS);
            if (frac <= kIntFeasTol || frac >= 1.0 - kIntFeasTol) continue;

            FractionalRow fr;
            fr.origVarId = varId;
            fr.fracVal   = frac;
            auto trow    = tab.tableauRow(r);
            fr.tabRow.assign(trow.begin(), trow.begin() + static_cast<std::ptrdiff_t>(nSF));
            det.fractionalRows.push_back(std::move(fr));
        }
    }

    return det;
}

} // namespace internal
} // namespace baguette
