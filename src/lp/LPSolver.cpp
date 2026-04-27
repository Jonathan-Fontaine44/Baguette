#include "baguette/lp/LPSolver.hpp"

#include "algorithms/DualSimplex.hpp"
#include "algorithms/PrimalSimplex.hpp"
#include "algorithms/RevisedSimplex.hpp"

namespace baguette {

LPResult solveLP(const Model& model, const LPOptions& opts) {
    return solveLPDetailed(model, opts).result;
}

LPDetailedResult solveLPDetailed(const Model& model, const LPOptions& opts) {
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
    // Auto or DualSimplex: attempt dual simplex with primal fallback.
    return internal::solveDual(model, opts.maxIter, opts.timeLimitS,
                               opts.startTime, opts.warmBasis,
                               opts.computeSensitivity, opts.computeCutData);
}

} // namespace baguette
