#include "baguette/lp/LPSolver.hpp"

#include "baguette/lp/Presolve.hpp"

#include "algorithms/DualSimplex.hpp"
#include "algorithms/DualSimplexBV.hpp"
#include "algorithms/IPMSolver.hpp"
#include "algorithms/MehrotraIPM.hpp"
#include "algorithms/PrimalSimplex.hpp"
#include "algorithms/PrimalSimplexBV.hpp"
#include "algorithms/RevisedSimplex.hpp"
#include "algorithms/NetworkSimplex.hpp"
#include "algorithms/RevisedSimplexBV.hpp"

namespace baguette {

LPResult solveLP(const Model& model, const LPOptions& opts) {
    return solveLPDetailed(model, opts).result;
}

LPDetailedResult solveLPDetailed(const Model& model, const LPOptions& opts) {
    // ── Presolve (opt-in) ──────────────────────────────────────────────────────
    if (opts.enablePresolve) {
        auto [presolved, pr] = presolve(model, opts.presolveMaxPasses, opts.timeLimitS, opts.startTime);
        if (pr.infeasible) {
            LPDetailedResult r;
            r.result.status = LPStatus::Infeasible;
            r.presolveStat  = pr;
            return r;
        }
        LPOptions inner      = opts;
        inner.enablePresolve = false;
        LPDetailedResult r   = solveLPDetailed(presolved, inner);
        r.presolveStat       = pr;
        return r;
    }
    if (opts.method == LPMethod::PrimalSimplex) {
        return internal::solvePrimal(model, opts.maxIter, opts.timeLimitS,
                                     opts.startTime, opts.computeSensitivity,
                                     opts.computeCutData);
    }
    if (opts.method == LPMethod::RevisedSimplex) {
        return internal::solveRevised(model, opts.maxIter, opts.timeLimitS,
                                      opts.startTime, opts.computeSensitivity,
                                      opts.computeCutData);
    }
    if (opts.method == LPMethod::RevisedSimplexBV) {
        return internal::solveRevisedBV(model, opts.maxIter, opts.timeLimitS,
                                        opts.startTime, opts.computeCutData);
    }
    if (opts.method == LPMethod::ShortStepIPM) {
        return internal::solveShortStepIPM(model, opts.maxIter, opts.timeLimitS,
                                           opts.startTime);
    }
    if (opts.method == LPMethod::MehrotraIPM) {
        return internal::solveMehrotraIPM(model, opts.maxIter, opts.timeLimitS,
                                          opts.startTime);
    }
    if (opts.method == LPMethod::PrimalSimplexBV) {
        return internal::solvePrimalBV(model, opts.maxIter, opts.timeLimitS,
                                       opts.startTime, opts.computeCutData,
                                       opts.computeSensitivity);
    }
    if (opts.method == LPMethod::DualSimplexBV) {
        return internal::solveDualBV(model, opts.maxIter, opts.timeLimitS,
                                     opts.startTime, opts.warmBasis,
                                     opts.computeCutData, opts.computeSensitivity);
    }
    if (opts.method == LPMethod::NetworkSimplex) {
        return internal::solveNetworkSimplex(model, opts.maxIter, opts.timeLimitS,
                                             opts.startTime, opts.computeCutData);
    }
    if (opts.method == LPMethod::DualSimplex) {
        return internal::solveDual(model, opts.maxIter, opts.timeLimitS,
                                   opts.startTime, opts.warmBasis,
                                   opts.computeSensitivity, opts.computeCutData);
    }
    // Auto: DualSimplexBV with fallback to PrimalSimplexBV.
    return internal::solveDualBV(model, opts.maxIter, opts.timeLimitS,
                                 opts.startTime, opts.warmBasis,
                                 opts.computeCutData, opts.computeSensitivity);
}

} // namespace baguette
