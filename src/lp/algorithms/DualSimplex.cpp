#include "DualSimplex.hpp"

#include <cassert>
#include <cmath>
#include <limits>
#include <memory>

#include "Extractor.hpp"
#include "PrimalSimplex.hpp"
#include "SimplexTableau.hpp"
#include "StandardForm.hpp"
#include "baguette/core/Sense.hpp"
#include "baguette/model/ModelEnums.hpp"

namespace baguette {

namespace {

/// Dual-simplex loop.
///
/// Precondition: the tableau is dual-feasible (all rc[j] ≥ 0).
/// The loop maintains dual feasibility and drives primal feasibility
/// until all rhs values are ≥ 0 (optimal) or infeasibility is detected.
///
/// @param outBlockingRow  If non-null, set to the leaving row index when
///                        infeasibility is detected. Enables Farkas extraction.
///
/// @note Complexity: O(K · m · n) total, where K = number of dual-simplex pivots
///   and each pivot costs O(m·n). Same periodic reinversion schedule as runSimplex.
LPStatus runDualSimplex(internal::SimplexTableau&             tab,
                         const internal::LPStandardForm&       sf,
                         uint32_t                              maxIter,
                         double                                timeLimitS,
                         std::chrono::steady_clock::time_point startTime,
                         uint32_t&                             iter,
                         std::size_t*                          outBlockingRow = nullptr) {

    uint32_t const timePeriod =
        tab.cfg.reinversionPeriod > 0 ? tab.cfg.reinversionPeriod : 64u;

    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count()
            >= timeLimitS)
        return LPStatus::TimeLimit;

    while (true) {
        if (maxIter > 0 && iter >= maxIter)
            return LPStatus::MaxIter;

        std::size_t leaving = tab.selectLeavingDual();
        if (leaving == tab.m)
            return LPStatus::Optimal;

        std::size_t entering = tab.selectEnteringDual(leaving);
        if (entering == tab.n) {
            if (outBlockingRow) *outBlockingRow = leaving;
            return LPStatus::Infeasible;
        }

        tab.pivot(leaving, entering);
        ++iter;

        if (iter % timePeriod == 0) {
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

/// Build a dual-feasible initial basis for the dual simplex (cold start).
///
/// For each row, select the column of the natural basic variable:
///   - LessEq row i:     slack column sf.rowSlackCol[i]  (coeff +1, b ≥ 0)
///   - GreaterEq row i:  surplus column sf.rowSlackCol[i] (coeff −1)
///   - Upper-bound row:  UpperSlack column (coeff +1, b = ub−lb ≥ 0)
///
/// @returns The initial basis vector, or an empty vector if the model contains
///          Sense::Equal constraints (which have no natural basic variable).
/// @note Complexity: O(m), where m = sf.nRows.
std::vector<uint32_t> buildDualBasis(const internal::LPStandardForm& sf,
                                      const Model& model) {
    const auto& constraints = model.getLPConstraints();

    for (std::size_t i = 0; i < sf.nOrigRows; ++i)
        if (constraints[i].sense == Sense::Equal)
            return {};

    std::vector<uint32_t> basis(sf.nRows);

    for (std::size_t i = 0; i < sf.nOrigRows; ++i)
        basis[i] = sf.rowSlackCol[i];

    for (std::size_t i = sf.nOrigRows; i < sf.nRows; ++i)
        basis[i] = static_cast<uint32_t>(sf.nOrig + sf.nSlack + (i - sf.nOrigRows));

    return basis;
}

/// Extract the Farkas infeasibility certificate from the dual-simplex blocking row.
///
/// Called when selectEnteringDual() returned n for @p leavingRow, meaning every
/// tableau entry in that row is >= 0 while the rhs is < 0.
/// @note Complexity: O(nOrigRows).
FarkasRay extractFarkasDualRow(const internal::SimplexTableau& tab,
                                const internal::LPStandardForm& sf,
                                const Model&                    model,
                                std::size_t                     leavingRow) {
    FarkasRay ray;
    const auto& constraints = model.getLPConstraints();
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

} // anonymous namespace

namespace internal {

LPDetailedResult solveDual(const Model&                          model,
                            uint32_t                              maxIter,
                            double                                timeLimitS,
                            std::chrono::steady_clock::time_point startTime,
                            const BasisRecord&                    warmBasis,
                            bool                                  computeSensitivity,
                            bool                                  computeCutData,
                            const SimplexConfig&                  cfg) {
    // Early infeasibility: empty variable domain.
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
    // varShiftVal, and objOffset via toStandardFormBoundsOnly.
    auto sfPtr = std::make_shared<LPStandardForm>();
    if (!warmBasis.basicCols.empty() && warmBasis.sfCache) {
        *sfPtr = *warmBasis.sfCache; // shallow: A shared_ptr copied O(1)
        if (!toStandardFormBoundsOnly(*sfPtr, model))
            *sfPtr = toStandardForm(model);
    } else {
        *sfPtr = toStandardForm(model);
    }
    LPStandardForm& sf = *sfPtr;
    SimplexTableau tab;
    tab.cfg = cfg;

    // Fallback to primal: attaches sfPtr to the basis cache for the next call.
    auto fallback = [&]() -> LPDetailedResult {
        LPDetailedResult det = solvePrimal(model, maxIter, timeLimitS, startTime,
                                           computeSensitivity, computeCutData, cfg);
        if (det.result.status == LPStatus::Optimal)
            det.basis.sfCache = sfPtr;
        return det;
    };

    if (!warmBasis.basicCols.empty()) {
        // ── Warm-start path ──────────────────────────────────────────────────
        if (warmBasis.basicCols.size() != sf.nRows ||
            warmBasis.colKind.size()   != sf.nCols)
            return fallback();

        tab.basicCols = warmBasis.basicCols;
        if (!tab.reinvert(sf))
            return fallback();
    } else {
        // ── Cold dual-start path ─────────────────────────────────────────────
        std::vector<uint32_t> coldBasis = buildDualBasis(sf, model);
        if (coldBasis.empty())
            return fallback();

        [[maybe_unused]] bool coldOk = tab.init(sf, coldBasis);
        assert(coldOk && "slack/surplus basis: cannot be singular");
    }

    // Verify dual feasibility (shared by both paths).
    for (std::size_t j = 0; j < sf.nCols; ++j) {
        if (tab.rc[j] < -tab.cfg.optimalityTol)
            return fallback();
    }

    uint32_t    dualIters   = 0;
    std::size_t blockingRow = tab.m; // sentinel
    LPStatus status = runDualSimplex(tab, sf, maxIter, timeLimitS, startTime,
                                     dualIters, &blockingRow);

    if (status == LPStatus::NumericalFailure) {
        LPDetailedResult det;
        det.result.status  = LPStatus::NumericalFailure;
        det.iterationsUsed = dualIters;
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
                if (sf.varFreeNegCol[col] < static_cast<uint32_t>(sf.nCols)) continue;
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
                                 tab.tab.begin() + static_cast<std::ptrdiff_t>(r * np + tab.n));
                det.fractionalRows.push_back(std::move(fr));
            }
        }
    }

    det.iterationsUsed = dualIters;
    det.usedWarmStart  = !warmBasis.basicCols.empty();
    return det;
}

} // namespace internal
} // namespace baguette
