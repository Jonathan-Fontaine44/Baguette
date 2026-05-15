#include "baguette/lp/LPSolver.hpp"

#include "baguette/lp/Presolve.hpp"

#include "algorithms/SimplexConfig.hpp"
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
    // ── Bound-tightening presolve (TB) ─────────────────────────────────────────
    if (opts.enablePresolve) {
        auto [presolved, pr] = presolveTB(model, opts.presolveMaxPasses,
                                          opts.timeLimitS, opts.startTime);
        if (pr.infeasible) {
            LPDetailedResult r;
            r.result.status = LPStatus::Infeasible;
            r.presolveStat  = pr;
            return r;
        }
        LPOptions inner      = opts;
        inner.enablePresolve = false;

        if (inner.enableElimination) {
            inner.enableElimination = false;
            EliminationRecord rec;
            Model reduced = presolveElim(presolved, rec);
            if (rec.infeasible) {
                LPDetailedResult r;
                r.result.status = LPStatus::Infeasible;
                r.presolveStat  = pr;
                return r;
            }
            LPDetailedResult r = solveLPDetailed(reduced, inner);
            postsolveElim(r, rec);
            r.presolveStat = pr;
            return r;
        }

        LPDetailedResult r = solveLPDetailed(presolved, inner);
        r.presolveStat = pr;
        return r;
    }

    // ── Method dispatch ────────────────────────────────────────────────────────
    const internal::SimplexConfig simplexCfg{
        .feasibilityTol    = opts.feasibilityTol,
        .optimalityTol     = opts.optimalityTol,
        .reinversionPeriod = opts.reinversionPeriod,
    };

    if (opts.method == LPMethod::PrimalSimplex) {
        return internal::solvePrimal(model, opts.maxIter, opts.timeLimitS,
                                     opts.startTime, opts.computeSensitivity,
                                     opts.computeCutData, simplexCfg);
    }
    if (opts.method == LPMethod::RevisedSimplex) {
        return internal::solveRevised(model, opts.maxIter, opts.timeLimitS,
                                      opts.startTime, opts.computeSensitivity,
                                      opts.computeCutData, simplexCfg);
    }
    if (opts.method == LPMethod::RevisedSimplexBV) {
        return internal::solveRevisedBV(model, opts.maxIter, opts.timeLimitS,
                                        opts.startTime, opts.computeCutData,
                                        simplexCfg);
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
                                       opts.computeSensitivity, simplexCfg);
    }
    if (opts.method == LPMethod::DualSimplexBV) {
        return internal::solveDualBV(model, opts.maxIter, opts.timeLimitS,
                                     opts.startTime, opts.warmBasis,
                                     opts.computeCutData, opts.computeSensitivity,
                                     simplexCfg);
    }
    if (opts.method == LPMethod::NetworkSimplex) {
        return internal::solveNetworkSimplex(model, opts.maxIter, opts.timeLimitS,
                                             opts.startTime, opts.computeCutData,
                                             simplexCfg);
    }
    if (opts.method == LPMethod::DualSimplex) {
        return internal::solveDual(model, opts.maxIter, opts.timeLimitS,
                                   opts.startTime, opts.warmBasis,
                                   opts.computeSensitivity, opts.computeCutData,
                                   simplexCfg);
    }
    // Auto: DualSimplexBV with fallback to PrimalSimplexBV.
    return internal::solveDualBV(model, opts.maxIter, opts.timeLimitS,
                                 opts.startTime, opts.warmBasis,
                                 opts.computeCutData, opts.computeSensitivity,
                                 simplexCfg);
}

} // namespace baguette
