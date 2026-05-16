#include "baguette/lp/LPSolver.hpp"

#include <cassert>

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

    switch (opts.method) {
        case LPMethod::PrimalSimplex:
            return internal::solvePrimal(model, opts.maxIter, opts.timeLimitS,
                                         opts.startTime, opts.computeSensitivity,
                                         opts.computeCutData, simplexCfg);
        case LPMethod::RevisedSimplex:
            return internal::solveRevised(model, opts.maxIter, opts.timeLimitS,
                                          opts.startTime, opts.computeSensitivity,
                                          opts.computeCutData, simplexCfg);
        case LPMethod::RevisedSimplexBV:
            return internal::solveRevisedBV(model, opts.maxIter, opts.timeLimitS,
                                            opts.startTime, opts.computeCutData,
                                            simplexCfg);
        case LPMethod::ShortStepIPM:
            return internal::solveShortStepIPM(model, opts.maxIter, opts.timeLimitS,
                                               opts.startTime);
        case LPMethod::MehrotraIPM:
            return internal::solveMehrotraIPM(model, opts.maxIter, opts.timeLimitS,
                                              opts.startTime);
        case LPMethod::PrimalSimplexBV:
            return internal::solvePrimalBV(model, opts.maxIter, opts.timeLimitS,
                                           opts.startTime, opts.computeCutData,
                                           opts.computeSensitivity, simplexCfg);
        case LPMethod::DualSimplexBV:
            return internal::solveDualBV(model, opts.maxIter, opts.timeLimitS,
                                         opts.startTime, opts.warmBasis,
                                         opts.computeCutData, opts.computeSensitivity,
                                         simplexCfg);
        case LPMethod::NetworkSimplex:
            return internal::solveNetworkSimplex(model, opts.maxIter, opts.timeLimitS,
                                                 opts.startTime, opts.computeCutData,
                                                 simplexCfg);
        case LPMethod::DualSimplex:
            return internal::solveDual(model, opts.maxIter, opts.timeLimitS,
                                       opts.startTime, opts.warmBasis,
                                       opts.computeSensitivity, opts.computeCutData,
                                       simplexCfg);
        case LPMethod::Auto:
            return internal::solveDualBV(model, opts.maxIter, opts.timeLimitS,
                                         opts.startTime, opts.warmBasis,
                                         opts.computeCutData, opts.computeSensitivity,
                                         simplexCfg);
        default:
            // A new LPMethod was added to the enum but not handled here.
            assert(false && "Unhandled LPMethod — add a case for each new enum value");
            return internal::solveDualBV(model, opts.maxIter, opts.timeLimitS,
                                         opts.startTime, opts.warmBasis,
                                         opts.computeCutData, opts.computeSensitivity,
                                         simplexCfg); // unreachable
    }
}

} // namespace baguette
