#include "RevisedSimplexBV.hpp"

#include <cassert>
#include <cmath>
#include <limits>

#include "LUTableau.hpp"
#include "StandardForm.hpp"
#include "baguette/core/Sense.hpp"
#include "baguette/model/ModelEnums.hpp"

namespace baguette {

namespace {

// ── Augmented form for BV Phase I ────────────────────────────────────────────
// Mirrors buildPhaseOneBV() in PrimalSimplexBV.cpp.

struct AugmentedFormBV {
    internal::LPStandardFormBV sfbv;
    std::vector<uint32_t>       initialBasis;
    std::size_t                 nArt = 0;
    std::vector<uint32_t>       equalArtCol;
};

AugmentedFormBV buildPhaseOneBV(const internal::LPStandardFormBV& sfbv,
                                 const Model&                      model) {
    const auto& constraints = model.getLPConstraints();
    const std::size_t m    = sfbv.nRows;
    const std::size_t nOld = sfbv.nCols;

    std::vector<bool> needsArt(m, false);
    for (std::size_t i = 0; i < m; ++i) {
        const bool  neg = sfbv.rowNegated[i];
        const Sense s   = constraints[i].sense;
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
    aug.sfbv.varShiftVal = sfbv.varShiftVal;
    aug.sfbv.varColSign  = sfbv.varColSign;
    aug.sfbv.varFreeNegCol = sfbv.varFreeNegCol;
    aug.sfbv.objOffset   = sfbv.objOffset;

    aug.equalArtCol.assign(sfbv.nOrigRows, static_cast<uint32_t>(nOld + nArt));

    const std::size_t nNew = nOld + nArt;
    aug.sfbv.nCols = nNew;
    aug.sfbv.A     = std::make_shared<std::vector<double>>(m * nNew, 0.0);
    aug.sfbv.c.assign(nNew, 0.0);
    aug.sfbv.colKind.resize(nNew);
    aug.sfbv.colOrigin.resize(nNew);
    aug.sfbv.colUB.assign(nNew, std::numeric_limits<double>::infinity());

    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < nOld; ++j)
            (*aug.sfbv.A)[i * nNew + j] = (*sfbv.A)[i * nOld + j];

    for (std::size_t j = 0; j < nOld; ++j) {
        aug.sfbv.colKind[j]   = sfbv.colKind[j];
        aug.sfbv.colOrigin[j] = sfbv.colOrigin[j];
        aug.sfbv.colUB[j]     = sfbv.colUB[j];
    }

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

    aug.initialBasis.resize(m);
    std::size_t nextArt = nOld;
    for (std::size_t i = 0; i < m; ++i) {
        if (!needsArt[i])
            aug.initialBasis[i] = sfbv.rowSlackCol[i];
        else
            aug.initialBasis[i] = static_cast<uint32_t>(nextArt++);
    }

    return aug;
}

// ── Drive artificials out of basis ───────────────────────────────────────────

void driveOutArtificialsRevBV(internal::LUTableau&              tab,
                               const internal::LPStandardFormBV& sfbvOrig) {
    const std::size_t nOld = sfbvOrig.nCols;
    for (std::size_t i = 0; i < tab.m; ++i) {
        if (tab.basicCols[i] < nOld) continue;
        auto trow = tab.tableauRow(i);
        for (std::size_t j = 0; j < nOld; ++j) {
            // Skip AT_UB columns: pivoting on them would un-complement j first,
            // making xB[i] = colUB[j]*eta[i] ≠ 0 — a non-degenerate pivot that
            // puts j into the basis at its UB, violating the complement invariant.
            if (tab.atUB[j]) continue;
            if (std::abs(trow[j]) > tab.cfg.pivotTol) {
                auto [rr, eta] = tab.selectLeavingBVWithEta(j);
                // Drive out the artificial with a degenerate pivot; row i leaves at LB.
                tab.pivotBV(i, j, /*leavingAtUB=*/false, eta);
                break;
            }
        }
    }
}

// ── BV dual simplex loop (warm-start repair) ─────────────────────────────────

LPStatus runDualRevBV(internal::LUTableau&                       tab,
                       const internal::LPStandardFormBV&          sfbv,
                       uint32_t                                   maxIter,
                       double                                     timeLimitS,
                       std::chrono::steady_clock::time_point      startTime,
                       uint32_t&                                  iters) {
    const uint32_t timePeriod =
        tab.cfg.reinversionPeriod > 0 ? tab.cfg.reinversionPeriod : 64u;

    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count()
            >= timeLimitS)
        return LPStatus::TimeLimit;

    while (true) {
        if (maxIter > 0 && iters >= maxIter) return LPStatus::MaxIter;

        auto [leavingRow, exitsToUB] = tab.selectLeavingDualBV();
        if (leavingRow == tab.m) return LPStatus::Optimal;

        std::size_t entering = tab.selectEnteringDualBV(leavingRow, exitsToUB);
        if (entering == tab.n) return LPStatus::Infeasible;

        auto eta = tab.enteringColumn(entering);
        tab.pivotBV(leavingRow, entering, exitsToUB, eta);
        ++iters;

        if (iters % timePeriod == 0) {
            if (tab.cfg.reinversionPeriod > 0)
                if (!tab.reinvertBV(sfbv)) return LPStatus::NumericalFailure;
            if (std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - startTime).count() >= timeLimitS)
                return LPStatus::TimeLimit;
        }
    }
}

// ── BV simplex loop ───────────────────────────────────────────────────────────

LPStatus runSimplexRevBV(internal::LUTableau&                       tab,
                          const internal::LPStandardFormBV&          sfbv,
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
        if (maxIter > 0 && iterConsumed >= maxIter) return LPStatus::MaxIter;

        std::size_t entering = tab.selectEntering();
        if (entering == tab.n) return LPStatus::Optimal;

        auto [rr, eta] = tab.selectLeavingBVWithEta(entering);

        if (rr.boundFlip) {
            tab.complement(entering);
        } else if (rr.leavingRow == tab.m) {
            return LPStatus::Unbounded;
        } else {
            tab.pivotBV(rr.leavingRow, entering, rr.leavingAtUB, eta);
        }
        ++iterConsumed;

        if (iterConsumed % timePeriod == 0) {
            if (tab.cfg.reinversionPeriod > 0)
                if (!tab.reinvertBV(sfbv)) return LPStatus::NumericalFailure;
            if (std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - startTime).count() >= timeLimitS)
                return LPStatus::TimeLimit;
        }
    }
}

// ── Farkas certificate ────────────────────────────────────────────────────────

FarkasRay extractFarkasRevBV(const internal::LUTableau&        tab,
                               const internal::LPStandardFormBV& sfbv,
                               const AugmentedFormBV&            aug,
                               const Model&                      model) {
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

LPDetailedResult extractRevBV(const internal::LUTableau&        tab,
                                const internal::LPStandardFormBV& sfbv,
                                const Model&                      model,
                                LPStatus                          status,
                                const std::vector<uint32_t>&      equalArtCol,
                                bool                              computeCutData) {
    LPDetailedResult det;
    det.result.status = status;

    const std::size_t nOrig   = sfbv.nOrig;
    const bool        maximize = (model.getObjSense() == ObjSense::Maximize);

    std::vector<double> xp = tab.primalSolutionBV();
    det.result.primalValues.resize(nOrig);
    for (std::size_t j = 0; j < nOrig; ++j) {
        double val = sfbv.varShiftVal[j] + sfbv.varColSign[j] * xp[j];
        uint32_t negCol = sfbv.varFreeNegCol[j];
        if (negCol < sfbv.nCols) val -= xp[negCol];
        det.result.primalValues[j] = val;
    }

    double obj = tab.objectiveValue() + sfbv.objOffset;
    det.result.objectiveValue = maximize ? -obj : obj;

    if (status != LPStatus::Optimal) return det;

    const auto& constraints = model.getLPConstraints();
    det.dualValues.resize(sfbv.nOrigRows);
    for (std::size_t i = 0; i < sfbv.nOrigRows; ++i) {
        uint32_t slackCol = sfbv.rowSlackCol[i];
        if (constraints[i].sense == Sense::Equal) {
            if (!equalArtCol.empty() && equalArtCol[i] < tab.n) {
                double raw  = tab.rc[equalArtCol[i]];
                double sign = sfbv.rowNegated[i] ? 1.0 : -1.0;
                if (maximize) sign = -sign;
                det.dualValues[i] = sign * raw;
            }
            continue;
        }
        double sign = (constraints[i].sense == Sense::LessEq) ? -1.0 : +1.0;
        if (sfbv.rowNegated[i]) sign = -sign;
        if (maximize)           sign = -sign;
        det.dualValues[i] = sign * tab.rc[slackCol];
    }

    det.reducedCosts.resize(nOrig);
    for (std::size_t j = 0; j < nOrig; ++j) {
        double rcj = tab.atUB[j] ? -tab.rc[j] : tab.rc[j];
        det.reducedCosts[j] = sfbv.varColSign[j] * (maximize ? -rcj : rcj);
    }

    det.basis.basicCols.assign(tab.basicCols.begin(), tab.basicCols.end());
    det.basis.colKind   = sfbv.colKind;
    det.basis.colOrigin = sfbv.colOrigin;
    // Trim to sfbv.nCols: on the cold path tab.n = nNew (augmented with artificials),
    // which are always AT_LB and must be stripped so the size matches sfbv.nCols.
    det.basis.atUBCache.assign(tab.atUB.begin(),
                                tab.atUB.begin() + static_cast<std::ptrdiff_t>(sfbv.nCols));

    if (computeCutData) {
        const auto& types  = model.getCold().types;
        const std::size_t nSFBV = sfbv.nCols;
        constexpr double kIntFeasTol = 1e-6;

        for (std::size_t r = 0; r < tab.m; ++r) {
            uint32_t col = tab.basicCols[r];
            if (col >= nSFBV) continue;
            if (sfbv.colKind[col] != ColumnKind::Original) continue;
            if (sfbv.varFreeNegCol[col] < static_cast<uint32_t>(nSFBV)) continue;
            uint32_t varId = sfbv.colOrigin[col];
            if (types[varId] != VarType::Integer && types[varId] != VarType::Binary)
                continue;
            double frac = tab.xB[r] - std::floor(tab.xB[r]);
            if (frac <= kIntFeasTol || frac >= 1.0 - kIntFeasTol) continue;
            FractionalRow fr;
            fr.origVarId = varId;
            fr.fracVal   = frac;
            auto trow    = tab.tableauRow(r);
            // LUTableau::tableauRow returns original B⁻¹a_j for all columns.
            // generateGMICuts expects complemented values for AT_UB columns
            // (i.e., -t_orig[j]), matching SimplexTableauBV's explicit tableau.
            for (std::size_t j2 = 0; j2 < nSFBV; ++j2)
                if (tab.atUB[j2]) trow[j2] = -trow[j2];
            fr.tabRow.assign(trow.begin(), trow.begin() + static_cast<std::ptrdiff_t>(nSFBV));
            det.fractionalRows.push_back(std::move(fr));
        }

        // AT_UB non-basic integer variables with fractional UB
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

LPDetailedResult solveRevisedBV(const Model&                          model,
                                 uint32_t                              maxIter,
                                 double                                timeLimitS,
                                 std::chrono::steady_clock::time_point startTime,
                                 bool                                  computeCutData,
                                 const SimplexConfig&                  cfg,
                                 const BasisRecord&                    warmBasis) {
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

    // Standard form — shared_ptr so sfbvCache can propagate the A matrix O(1).
    const bool hasWarm = !warmBasis.basicCols.empty() && !warmBasis.atUBCache.empty();
    auto sfbvPtr = std::make_shared<LPStandardFormBV>();
    if (hasWarm && warmBasis.sfbvCache) {
        *sfbvPtr = *warmBasis.sfbvCache;        // shallow: A shared_ptr copied O(1)
        if (!toStandardFormBoundsOnlyBV(*sfbvPtr, model))
            *sfbvPtr = toStandardFormBV(model);
    } else {
        *sfbvPtr = toStandardFormBV(model);
    }
    LPStandardFormBV& sfbv = *sfbvPtr;

    LUTableau tab;
    tab.cfg = cfg;

    // ── Warm-start path ───────────────────────────────────────────────────────
    // Skip Phase I: initialise from the parent's basis (basicCols + atUBCache)
    // and run BV dual simplex to restore primal feasibility.
    if (hasWarm &&
        warmBasis.basicCols.size() == sfbv.nRows &&
        warmBasis.atUBCache.size() == sfbv.nCols) {

        // Use initBV (canonical full-init path) then re-apply complement states.
        // initBV sets m, n, colUB, resizes all vectors and calls doReinvert.
        std::vector<uint32_t> basisCopy = warmBasis.basicCols;
        if (tab.initBV(sfbv, std::move(basisCopy))) {
            for (std::size_t j = 0; j < sfbv.nCols; ++j)
                if (warmBasis.atUBCache[j]) tab.complement(j);
            bool dualFeasible = true;
            for (std::size_t j = 0; j < sfbv.nCols; ++j) {
                if (tab.rc[j] < -tab.cfg.optimalityTol) { dualFeasible = false; break; }
            }

            if (dualFeasible) {
                uint32_t iters = 0;
                LPStatus status = runDualRevBV(tab, sfbv, maxIter, timeLimitS, startTime, iters);

                LPDetailedResult det = extractRevBV(tab, sfbv, model, status, {}, computeCutData);
                det.iterationsUsed = iters;
                det.usedWarmStart  = true;

                if (status == LPStatus::Optimal) {
                    det.basis.sfbvCache = sfbvPtr;
                    det.basis.atUBCache = tab.atUB;
                }
                return det;
            }
        }
        // Warm basis is dual infeasible or singular — reset for cold start.
        tab = LUTableau{};
        tab.cfg = cfg;
    }

    // ── Cold-start path: Phase I + Phase II ───────────────────────────────────

    AugmentedFormBV aug = buildPhaseOneBV(sfbv, model);
    [[maybe_unused]] bool initOk = tab.initBV(aug.sfbv, aug.initialBasis);
    assert(initOk && "identity-like artificial basis: cannot be singular");

    uint32_t iters = 0;
    LPStatus p1 = runSimplexRevBV(tab, aug.sfbv, maxIter, timeLimitS, startTime, iters);

    if (p1 == LPStatus::MaxIter || p1 == LPStatus::TimeLimit ||
        p1 == LPStatus::NumericalFailure) {
        LPDetailedResult det;
        det.result.status = p1;
        return det;
    }

    if (tab.objectiveValue() > tab.cfg.feasibilityTol) {
        LPDetailedResult det;
        det.result.status = LPStatus::Infeasible;
        det.farkas        = extractFarkasRevBV(tab, sfbv, aug, model);
        return det;
    }

    driveOutArtificialsRevBV(tab, sfbv);

    // Phase II: switch to real objective, restrict entering to original cols.
    // Update aug.sfbv.c so reinvertBV(aug.sfbv) uses Phase II costs, not Phase I.
    {
        aug.sfbv.c.assign(aug.sfbv.nCols, 0.0);
        for (std::size_t j = 0; j < sfbv.nCols; ++j)
            aug.sfbv.c[j] = sfbv.c[j];
        tab.repriceBV(aug.sfbv.c, sfbv.nCols);
    }

    // Pass aug.sfbv (nCols = nNew) to reinvertBV to keep tab.n stable.
    LPStatus p2 = runSimplexRevBV(tab, aug.sfbv, maxIter, timeLimitS, startTime, iters);

    if (p2 == LPStatus::NumericalFailure) {
        LPDetailedResult det;
        det.result.status = LPStatus::NumericalFailure;
        return det;
    }

    LPDetailedResult det = extractRevBV(tab, sfbv, model, p2, aug.equalArtCol, computeCutData);
    det.iterationsUsed = iters;
    if (p2 == LPStatus::Unbounded) det.result.primalValues.clear();

    if (p2 == LPStatus::Optimal) {
        det.basis.sfbvCache = sfbvPtr;
        // extractRevBV already trimmed atUBCache to sfbv.nCols; no re-assignment needed.
    }

    return det;
}

} // namespace internal
} // namespace baguette
