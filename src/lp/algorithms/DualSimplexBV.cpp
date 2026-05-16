#include "DualSimplexBV.hpp"

#include <cassert>
#include <cmath>
#include <limits>
#include <memory>

#include "PrimalSimplexBV.hpp"
#include "SimplexTableauBV.hpp"
#include "StandardForm.hpp"
#include "baguette/core/Sense.hpp"
#include "baguette/model/ModelEnums.hpp"

namespace baguette {

namespace {

// ── Dual simplex loop ─────────────────────────────────────────────────────────

LPStatus runDualSimplexBV(internal::SimplexTableauBV&              tab,
                           const internal::LPStandardFormBV&        sfbv,
                           uint32_t                                  maxIter,
                           double                                    timeLimitS,
                           std::chrono::steady_clock::time_point     startTime,
                           std::size_t*                              outBlockingRow,
                           bool*                                     outBlockingExitsToUB) {
    uint32_t iter = 0;
    const uint32_t timePeriod =
        tab.cfg.reinversionPeriod > 0 ? tab.cfg.reinversionPeriod : 64u;

    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count()
            >= timeLimitS)
        return LPStatus::TimeLimit;

    while (true) {
        if (maxIter > 0 && iter >= maxIter)
            return LPStatus::MaxIter;

        auto [leavingRow, exitsToUB] = tab.selectLeavingDualBV();
        if (leavingRow == tab.m)
            return LPStatus::Optimal;

        std::size_t entering = tab.selectEnteringDualBV(leavingRow, exitsToUB);
        if (entering == tab.n) {
            if (outBlockingRow)      *outBlockingRow      = leavingRow;
            if (outBlockingExitsToUB) *outBlockingExitsToUB = exitsToUB;
            return LPStatus::Infeasible;
        }

        tab.pivotBV(leavingRow, entering, exitsToUB);
        ++iter;

        if (iter % timePeriod == 0) {
            if (tab.cfg.reinversionPeriod > 0)
                if (!tab.reinvert(sfbv)) return LPStatus::NumericalFailure;
            if (std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - startTime).count() >= timeLimitS)
                return LPStatus::TimeLimit;
        }
    }
}

// ── Cold-start dual basis ─────────────────────────────────────────────────────

// Slack/surplus basis (LessEq/GreaterEq only). Equal constraints have no natural
// basic variable — returns empty to signal fallback to primal.
std::vector<uint32_t> buildDualBasisBV(const internal::LPStandardFormBV& sfbv,
                                        const Model&                       model) {
    const auto& constraints = model.getLPConstraints();
    for (std::size_t i = 0; i < sfbv.nOrigRows; ++i)
        if (constraints[i].sense == Sense::Equal)
            return {};

    std::vector<uint32_t> basis(sfbv.nRows); // nRows == nOrigRows
    for (std::size_t i = 0; i < sfbv.nOrigRows; ++i)
        basis[i] = sfbv.rowSlackCol[i];
    return basis;
}

// ── Farkas certificate (Type-L blocking row) ──────────────────────────────────

FarkasRay extractFarkasDualBV(const internal::SimplexTableauBV& tab,
                               const internal::LPStandardFormBV& sfbv,
                               const Model&                       model,
                               std::size_t                        leavingRow) {
    FarkasRay ray;
    const auto& constraints = model.getLPConstraints();
    ray.y.resize(sfbv.nOrigRows, 0.0);
    const std::size_t w = tab.n + 1;

    for (std::size_t i = 0; i < sfbv.nOrigRows; ++i) {
        const uint32_t slackCol = sfbv.rowSlackCol[i];
        if (slackCol >= sfbv.nCols) continue; // Equal row — no slack
        const double entry = tab.tab[leavingRow * w + slackCol];
        ray.y[i] = (constraints[i].sense == Sense::LessEq) ? entry : -entry;
    }
    return ray;
}

// ── Result extraction ─────────────────────────────────────────────────────────

LPDetailedResult extractDualBV(const internal::SimplexTableauBV& tab,
                                const internal::LPStandardFormBV& sfbv,
                                const Model&                       model,
                                LPStatus                           status,
                                bool                               computeCutData,
                                bool                               computeSensitivity = false) {
    LPDetailedResult det;
    det.result.status = status;

    const std::size_t nOrig  = sfbv.nOrig;
    const bool        maxObj = (model.getObjSense() == ObjSense::Maximize);

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

    const auto& constraints = model.getLPConstraints();
    det.dualValues.resize(sfbv.nOrigRows);
    for (std::size_t i = 0; i < sfbv.nOrigRows; ++i) {
        const uint32_t slackCol = sfbv.rowSlackCol[i];
        if (constraints[i].sense == Sense::Equal) continue; // no slack → dual = 0
        double sign = (constraints[i].sense == Sense::LessEq) ? -1.0 : +1.0;
        if (sfbv.rowNegated[i]) sign = -sign;
        if (maxObj)             sign = -sign;
        det.dualValues[i] = sign * tab.rc[slackCol];
    }

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
        det.sensitivity = internal::extractSensitivityBV(tab, sfbv, model);

    if (computeCutData) {
        const auto& types = model.getCold().types;
        const std::size_t nSFBV = sfbv.nCols;
        const std::size_t np    = tab.n + 1;
        constexpr double kIntFeasTol = 1e-6;
        for (std::size_t r = 0; r < tab.m; ++r) {
            uint32_t col = tab.basicCols[r];
            if (col >= nSFBV) continue;
            if (sfbv.colKind[col] != ColumnKind::Original) continue;
            if (sfbv.varFreeNegCol[col] < static_cast<uint32_t>(nSFBV)) continue;
            uint32_t varId = sfbv.colOrigin[col];
            if (types[varId] != VarType::Integer && types[varId] != VarType::Binary) continue;
            double sfbvBFS = tab.tab[r * np + tab.n];
            double frac    = sfbvBFS - std::floor(sfbvBFS);
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

LPDetailedResult solveDualBV(const Model&                          model,
                              uint32_t                              maxIter,
                              double                                timeLimitS,
                              std::chrono::steady_clock::time_point startTime,
                              const BasisRecord&                    warmBasis,
                              bool                                  computeCutData,
                              bool                                  computeSensitivity,
                              const SimplexConfig&                  cfg) {
    if (std::chrono::duration<double>(
            std::chrono::steady_clock::now() - startTime).count() >= timeLimitS) {
        LPDetailedResult det;
        det.result.status = LPStatus::TimeLimit;
        return det;
    }

    // Early infeasibility: empty domain.
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

    // On the warm-start path, A and c are invariant between B&B nodes.
    // Reuse the cached A (shared_ptr copy is O(1)) and update only b,
    // varShiftVal, objOffset, and colUB via toStandardFormBoundsOnlyBV.
    const bool hasWarm = !warmBasis.basicCols.empty() && !warmBasis.atUBCache.empty();
    auto sfbvPtr = std::make_shared<LPStandardFormBV>();
    if (hasWarm && warmBasis.sfbvCache) {
        *sfbvPtr = *warmBasis.sfbvCache; // shallow: A shared_ptr copied O(1)
        if (!toStandardFormBoundsOnlyBV(*sfbvPtr, model))
            *sfbvPtr = toStandardFormBV(model);
    } else {
        *sfbvPtr = toStandardFormBV(model);
    }
    LPStandardFormBV& sfbv = *sfbvPtr;

    SimplexTableauBV tab;
    tab.cfg = cfg;

    auto fallback = [&]() -> LPDetailedResult {
        return solvePrimalBV(model, maxIter, timeLimitS, startTime,
                             computeCutData, computeSensitivity, cfg);
    };

    if (hasWarm) {
        // ── Warm-start path ──────────────────────────────────────────────────
        if (warmBasis.basicCols.size() != sfbv.nRows ||
            warmBasis.atUBCache.size() != sfbv.nCols)
            return fallback();

        tab.basicCols = warmBasis.basicCols;
        tab.atUB      = warmBasis.atUBCache;
        if (!tab.reinvert(sfbv))
            return fallback();
    } else {
        // ── Cold dual-start path ─────────────────────────────────────────────
        std::vector<uint32_t> coldBasis = buildDualBasisBV(sfbv, model);
        if (coldBasis.empty())
            return fallback();

        [[maybe_unused]] bool ok = tab.init(sfbv, coldBasis);
        assert(ok && "slack/surplus basis: cannot be singular");
    }

    // Verify dual feasibility (all rc[j] >= 0 under the complement invariant).
    for (std::size_t j = 0; j < sfbv.nCols; ++j) {
        if (tab.rc[j] < -tab.cfg.optimalityTol)
            return fallback();
    }

    std::size_t blockingRow      = tab.m;
    bool        blockingExitsToUB = false;
    LPStatus status = runDualSimplexBV(tab, sfbv, maxIter, timeLimitS, startTime,
                                       &blockingRow, &blockingExitsToUB);

    if (status == LPStatus::NumericalFailure) {
        LPDetailedResult det;
        det.result.status = LPStatus::NumericalFailure;
        return det;
    }

    LPDetailedResult det = extractDualBV(tab, sfbv, model, status,
                                          computeCutData, computeSensitivity);

    if (status == LPStatus::Optimal)
        det.basis.sfbvCache = sfbvPtr;

    if (status != LPStatus::Optimal) {
        det.result.primalValues.clear();
        // Farkas certificate only for Type-L blocking (below LB); Type-U is rare
        // (warm-start only) and has a more complex certificate — omitted for now.
        if (status == LPStatus::Infeasible &&
            blockingRow < tab.m && !blockingExitsToUB)
            det.farkas = extractFarkasDualBV(tab, sfbv, model, blockingRow);
    }

    det.usedWarmStart = hasWarm;
    return det;
}

} // namespace internal
} // namespace baguette
