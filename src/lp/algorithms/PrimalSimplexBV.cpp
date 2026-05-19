#include "PrimalSimplexBV.hpp"

#include <cassert>
#include <cmath>
#include <limits>

#include "SimplexTableauBV.hpp"
#include "StandardForm.hpp"
#include "baguette/core/Sense.hpp"
#include "baguette/model/ModelEnums.hpp"

namespace baguette {

namespace {

// ── Augmented form for BV Phase I ────────────────────────────────────────────

struct AugmentedFormBV {
    internal::LPStandardFormBV sfbv;
    std::vector<uint32_t>       initialBasis;
    std::size_t                 nArt = 0;
    std::vector<uint32_t>       equalArtCol; // sentinel = sfbv.nCols
};

AugmentedFormBV buildPhaseOneBV(const internal::LPStandardFormBV& sfbv,
                                 const Model&                      model) {
    const auto& constraints = model.getLPConstraints();
    const std::size_t m    = sfbv.nRows; // = nOrigRows only (no UB rows)
    const std::size_t nOld = sfbv.nCols;

    std::vector<bool> needsArt(m, false);
    for (std::size_t i = 0; i < m; ++i) {
        // After row negation the effective sense flips: GEQ→LEQ, LEQ→GEQ.
        // An artificial is needed iff the effective sense is GEQ or Equal
        // (natural slack coefficient is -1 or absent, giving an infeasible initial BFS).
        const bool neg = sfbv.rowNegated[i];
        const Sense s  = constraints[i].sense;
        needsArt[i] = (s == Sense::Equal) ||
                      (!neg && s == Sense::GreaterEq) ||
                      (neg  && s == Sense::LessEq);
    }

    std::size_t nArt = 0;
    for (bool b : needsArt) if (b) ++nArt;

    AugmentedFormBV aug;
    aug.nArt = nArt;

    aug.sfbv.nRows     = sfbv.nRows;
    aug.sfbv.nOrigRows = sfbv.nOrigRows;
    aug.sfbv.nOrig     = sfbv.nOrig;
    aug.sfbv.nSlack    = sfbv.nSlack;
    aug.sfbv.b          = sfbv.b;
    aug.sfbv.rowSlackCol = sfbv.rowSlackCol;
    aug.sfbv.rowNegated  = sfbv.rowNegated;

    aug.equalArtCol.assign(sfbv.nOrigRows, static_cast<uint32_t>(nOld + nArt));

    const std::size_t nNew  = nOld + nArt;
    aug.sfbv.nCols = nNew;
    aug.sfbv.A     = std::make_shared<std::vector<double>>(m * nNew, 0.0);
    aug.sfbv.c.assign(nNew, 0.0);
    aug.sfbv.colKind.resize(nNew);
    aug.sfbv.colOrigin.resize(nNew);
    aug.sfbv.colUB.assign(nNew, std::numeric_limits<double>::infinity());

    // Copy original A, column metadata and colUB
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < nOld; ++j)
            (*aug.sfbv.A)[i * nNew + j] = (*sfbv.A)[i * nOld + j];
    for (std::size_t j = 0; j < nOld; ++j) {
        aug.sfbv.colKind[j]   = sfbv.colKind[j];
        aug.sfbv.colOrigin[j] = sfbv.colOrigin[j];
        aug.sfbv.colUB[j]     = sfbv.colUB[j];
    }

    // Artificials (colUB = inf, phase-I objective = 1)
    std::size_t artCol = nOld;
    for (std::size_t i = 0; i < m; ++i) {
        if (!needsArt[i]) continue;
        (*aug.sfbv.A)[i * nNew + artCol] = 1.0;
        aug.sfbv.c[artCol]               = 1.0;
        aug.sfbv.colKind[artCol]         = ColumnKind::Slack;
        aug.sfbv.colOrigin[artCol]       = static_cast<uint32_t>(i);
        if (constraints[i].sense == Sense::Equal)
            aug.equalArtCol[i] = static_cast<uint32_t>(artCol);
        ++artCol;
    }

    // Initial basis
    aug.initialBasis.resize(m);
    std::size_t nextArt = nOld;
    for (std::size_t i = 0; i < m; ++i) {
        if (!needsArt[i])
            aug.initialBasis[i] = sfbv.rowSlackCol[i]; // natural LessEq slack
        else
            aug.initialBasis[i] = static_cast<uint32_t>(nextArt++);
    }

    return aug;
}

// ── Drive artificials out and repair redundant rows (same logic as PrimalSimplex) ──

void driveOutArtificialsBV(internal::SimplexTableauBV&        tab,
                            const internal::LPStandardFormBV&  sfbvOrig) {
    const std::size_t nOld = sfbvOrig.nCols;
    for (std::size_t i = 0; i < tab.m; ++i) {
        if (tab.basicCols[i] < nOld) continue;
        for (std::size_t j = 0; j < nOld; ++j) {
            if (std::abs(tab.tab[i * (tab.n + 1) + j]) > tab.cfg.pivotTol) {
                tab.pivotBV(i, j, /*leavingAtUB=*/false);
                break;
            }
        }
    }
}

void repairRedundantRowsBV(internal::SimplexTableauBV& tab, std::size_t nOld) {
    std::vector<bool> inBasis(nOld, false);
    for (std::size_t i = 0; i < tab.m; ++i)
        if (tab.basicCols[i] < nOld) inBasis[tab.basicCols[i]] = true;

    for (std::size_t i = 0; i < tab.m; ++i) {
        if (tab.basicCols[i] < nOld) continue;
        bool found = false;
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
        for (std::size_t j = 0; j < nOld && !found; ++j) {
            if (!inBasis[j]) {
                tab.basicCols[i] = static_cast<uint32_t>(j);
                inBasis[j] = true;
                tab.hasRedundantRow = true;
                found = true;
            }
        }
        assert(found && "repairRedundantRowsBV: no free column");
    }
}

void preparePhaseTwoBV(internal::SimplexTableauBV&        tab,
                        const internal::LPStandardFormBV&  sfbvOrig) {
    const std::size_t nOld = sfbvOrig.nCols;
    const std::size_t w    = tab.n + 1;

    tab.nActive = nOld;
    repairRedundantRowsBV(tab, nOld);

    // Re-price: use original objective (artificials get cost 0).
    // For AT_UB columns the complement invariant stores rc[j] = -c_j - c_B*B^{-1}*a_j
    // (note the sign flip on c_j vs. the LB formula). The tableau column for an AT_UB
    // variable j already stores B^{-1}*(-a_j), so the subtraction loop is the same for
    // both LB and AT_UB; only the initial cost term differs.
    tab.rc.assign(w, 0.0);
    for (std::size_t j = 0; j < nOld; ++j)
        tab.rc[j] = tab.atUB[j] ? -sfbvOrig.c[j] : sfbvOrig.c[j];
    for (std::size_t i = 0; i < tab.m; ++i) {
        double cb = (tab.basicCols[i] < nOld) ? sfbvOrig.c[tab.basicCols[i]] : 0.0;
        if (cb == 0.0) continue;
        for (std::size_t j = 0; j < w; ++j)
            tab.rc[j] -= cb * tab.tab[i * w + j];
    }
    // AT_UB non-basic variables contribute c_j*ub_j to the objective (rc[n] = -z).
    for (std::size_t j = 0; j < nOld; ++j)
        if (tab.atUB[j])
            tab.rc[w - 1] -= sfbvOrig.c[j] * sfbvOrig.colUB[j];
}

// ── BV simplex loop ───────────────────────────────────────────────────────────

LPStatus runSimplexBV(internal::SimplexTableauBV&        tab,
                       const internal::LPStandardFormBV&  sfbv,
                       uint32_t                            maxIter,
                       double                              timeLimitS,
                       std::chrono::steady_clock::time_point startTime,
                       uint32_t&                           iterConsumed) {
    const uint32_t timePeriod =
        tab.cfg.reinversionPeriod > 0 ? tab.cfg.reinversionPeriod : 64u;

    while (true) {
        if (maxIter > 0 && iterConsumed >= maxIter) return LPStatus::MaxIter;

        std::size_t entering = tab.selectEntering();
        if (entering == tab.n) return LPStatus::Optimal;

        auto [leavingRow, boundFlip, leavingAtUB] = tab.selectLeavingBV(entering);

        if (boundFlip) {
            tab.complement(entering);
        } else if (leavingRow == tab.m) {
            return LPStatus::Unbounded;
        } else {
            tab.pivotBV(leavingRow, entering, leavingAtUB);
        }
        ++iterConsumed;

        if (iterConsumed % timePeriod == 0) {
            if (tab.cfg.reinversionPeriod > 0)
                if (!tab.reinvert(sfbv)) return LPStatus::NumericalFailure;
            double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - startTime).count();
            if (elapsed >= timeLimitS) return LPStatus::TimeLimit;
        }
    }
}

// ── Farkas certificate (identical to PrimalSimplex path) ─────────────────────

FarkasRay extractFarkasBV(const internal::SimplexTableauBV&  tab,
                            const internal::LPStandardFormBV&  sfbv,
                            const AugmentedFormBV&              aug,
                            const Model&                        model) {
    FarkasRay ray;
    const auto& constraints = model.getLPConstraints();
    ray.y.resize(sfbv.nOrigRows, 0.0);
    for (std::size_t i = 0; i < sfbv.nOrigRows; ++i) {
        Sense s = constraints[i].sense;
        if (s == Sense::Equal) {
            if (!aug.equalArtCol.empty() && aug.equalArtCol[i] < tab.n)
                ray.y[i] = tab.rc[aug.equalArtCol[i]] - 1.0;
        } else {
            uint32_t slackCol = sfbv.rowSlackCol[i];
            if (slackCol >= sfbv.nCols) continue;
            ray.y[i] = (s == Sense::LessEq) ? tab.rc[slackCol] : -tab.rc[slackCol];
        }
    }
    return ray;
}

// ── Result extraction ─────────────────────────────────────────────────────────

LPDetailedResult extractBV(const internal::SimplexTableauBV&  tab,
                             const internal::LPStandardFormBV&  sfbv,
                             const Model&                        model,
                             LPStatus                            status,
                             const std::vector<uint32_t>&        equalArtCol,
                             bool                                computeCutData,
                             bool                                computeSensitivity = false) {
    LPDetailedResult det;
    det.result.status = status;

    const std::size_t nOrig  = sfbv.nOrig;
    const bool        maxObj = (model.getObjSense() == ObjSense::Maximize);

    // Primal un-shift (same logic as extractDetailed)
    std::vector<double> xp = tab.primalSolution();
    det.result.primalValues.resize(nOrig);
    for (std::size_t j = 0; j < nOrig; ++j) {
        double val = sfbv.varShiftVal[j] + sfbv.varColSign[j] * xp[j];
        uint32_t negCol = sfbv.varFreeNegCol[j];
        if (negCol < sfbv.nCols) val -= xp[negCol];
        det.result.primalValues[j] = val;
    }

    double obj = tab.objectiveValue() + sfbv.objOffset;
    det.result.objectiveValue = maxObj ? -obj : obj;

    if (status != LPStatus::Optimal) return det;

    // Dual values
    const auto& constraints = model.getLPConstraints();
    det.dualValues.resize(sfbv.nOrigRows);
    for (std::size_t i = 0; i < sfbv.nOrigRows; ++i) {
        uint32_t slackCol = sfbv.rowSlackCol[i];
        if (constraints[i].sense == Sense::Equal) {
            if (!equalArtCol.empty() && equalArtCol[i] < tab.n) {
                double raw  = tab.rc[equalArtCol[i]];
                double sign = sfbv.rowNegated[i] ? 1.0 : -1.0;
                if (maxObj) sign = -sign;
                det.dualValues[i] = sign * raw;
            }
            continue;
        }
        double sign = (constraints[i].sense == Sense::LessEq) ? -1.0 : +1.0;
        if (sfbv.rowNegated[i]) sign = -sign;
        if (maxObj)           sign = -sign;
        det.dualValues[i] = sign * tab.rc[slackCol];
    }

    // Reduced costs — un-complement for AT_UB non-basics
    det.reducedCosts.resize(nOrig);
    for (std::size_t j = 0; j < nOrig; ++j) {
        double rcj = tab.atUB[j] ? -tab.rc[j] : tab.rc[j];
        det.reducedCosts[j] = sfbv.varColSign[j] * (maxObj ? -rcj : rcj);
    }

    det.basis.basicCols.assign(tab.basicCols.begin(), tab.basicCols.end());
    det.basis.colKind   = sfbv.colKind;
    det.basis.colOrigin = sfbv.colOrigin;
    det.basis.atUBCache = tab.atUB;

    if (computeSensitivity)
        det.sensitivity = internal::extractSensitivityBV(tab, sfbv, model, equalArtCol);

    // GMI cut data
    if (computeCutData) {
        const auto& types = model.getCold().types;
        const std::size_t nSFBV = sfbv.nCols;
        const std::size_t np  = tab.n + 1;
        constexpr double kIntFeasTol = 1e-6;
        for (std::size_t r = 0; r < tab.m; ++r) {
            uint32_t col = tab.basicCols[r];
            if (col >= nSFBV) continue;
            if (sfbv.colKind[col] != ColumnKind::Original) continue;
            if (sfbv.varFreeNegCol[col] < static_cast<uint32_t>(nSFBV)) continue;
            uint32_t varId = sfbv.colOrigin[col];
            if (types[varId] != VarType::Integer && types[varId] != VarType::Binary)
                continue;
            double sfbvBFS = tab.tab[r * np + tab.n];
            double frac  = sfbvBFS - std::floor(sfbvBFS);
            if (frac <= kIntFeasTol || frac >= 1.0 - kIntFeasTol) continue;
            FractionalRow fr;
            fr.origVarId = varId;
            fr.fracVal   = frac;
            fr.tabRow.assign(tab.tab.begin() + static_cast<std::ptrdiff_t>(r * np),
                             tab.tab.begin() + static_cast<std::ptrdiff_t>(r * np + nSFBV));
            det.fractionalRows.push_back(std::move(fr));
        }
        // AT_UB non-basic integer variables with fractional UB: synthesize a virtual UB row.
        // The UB row's Gauss-Jordan leaves only one non-zero entry: 1.0 at column j itself
        // (the complement x'' = ub - x acts as an UpperSlack). generateGMICuts uses
        // atUBCache[j] to apply the correct UpperSlack substitution.
        for (std::size_t j = 0; j < nSFBV; ++j) {
            if (!tab.atUB[j]) continue;
            if (sfbv.colKind[j] != ColumnKind::Original) continue;
            if (sfbv.varFreeNegCol[j] < static_cast<uint32_t>(nSFBV)) continue;
            uint32_t varId = sfbv.colOrigin[j];
            if (types[varId] != VarType::Integer && types[varId] != VarType::Binary) continue;
            double ubSF = tab.colUB[j];
            double fr   = ubSF - std::floor(ubSF);
            if (fr <= kIntFeasTol || fr >= 1.0 - kIntFeasTol) continue;
            FractionalRow frow;
            frow.origVarId = varId;
            frow.fracVal   = fr;
            frow.tabRow.assign(nSFBV, 0.0);
            frow.tabRow[j] = 1.0;
            det.fractionalRows.push_back(std::move(frow));
        }
    }

    return det;
}

} // anonymous namespace

namespace internal {

LPDetailedResult solvePrimalBV(const Model&                          model,
                                uint32_t                              maxIter,
                                double                                timeLimitS,
                                std::chrono::steady_clock::time_point startTime,
                                bool                                  computeCutData,
                                bool                                  computeSensitivity,
                                const SimplexConfig&                  cfg) {
    // Early infeasibility: empty domain
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

    // 1. Compact standard form (no UB rows)
    LPStandardFormBV sfbv = toStandardFormBV(model);

    // 2. Phase I
    AugmentedFormBV aug = buildPhaseOneBV(sfbv, model);
    SimplexTableauBV tab;
    tab.cfg = cfg;
    [[maybe_unused]] bool ok = tab.init(aug.sfbv, aug.initialBasis);
    assert(ok && "identity-like basis: cannot be singular");

    uint32_t iters = 0;
    LPStatus p1 = runSimplexBV(tab, aug.sfbv, maxIter, timeLimitS, startTime, iters);

    if (p1 == LPStatus::MaxIter || p1 == LPStatus::TimeLimit ||
        p1 == LPStatus::NumericalFailure) {
        LPDetailedResult det;
        det.result.status = p1;
        return det;
    }

    if (tab.objectiveValue() > tab.cfg.feasibilityTol) {
        LPDetailedResult det;
        det.result.status = LPStatus::Infeasible;
        det.farkas        = extractFarkasBV(tab, sfbv, aug, model);
        return det;
    }

    // 3. Drive remaining artificials out (degenerate exit)
    driveOutArtificialsBV(tab, sfbv);

    // 4. Phase II
    preparePhaseTwoBV(tab, sfbv);

    LPStatus p2 = runSimplexBV(tab, sfbv, maxIter, timeLimitS, startTime, iters);

    if (p2 == LPStatus::NumericalFailure) {
        LPDetailedResult det;
        det.result.status = LPStatus::NumericalFailure;
        return det;
    }

    LPDetailedResult det = extractBV(tab, sfbv, model, p2, aug.equalArtCol,
                                      computeCutData, computeSensitivity);
    det.iterationsUsed = iters;
    if (p2 == LPStatus::Unbounded) det.result.primalValues.clear();
    return det;
}

} // namespace internal
} // namespace baguette
