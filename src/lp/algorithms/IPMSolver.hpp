#pragma once

#include <chrono>
#include <cstdint>

#include "baguette/lp/LPResult.hpp"
#include "baguette/model/Model.hpp"

namespace baguette::internal {

/// Short-step feasible path-following interior-point method.
///
/// Primal-dual path-following with fixed step α = 1/(1+√n), which keeps the
/// iterate in the N₂(θ) neighbourhood of the central path and guarantees
/// convergence in O(√n log(1/ε)) iterations.  Starting point computed via
/// the Mehrotra heuristic (least-norm primal/dual solutions + positivity shift
/// + centering correction).
///
/// @par Complexity
///   O(K · m² · n) total, where K = number of iterations ≤ O(√n log(1/ε)),
///   m = number of constraints, n = number of standard-form variables.
///   Each iteration: O(m²n) to build the normal-equations matrix ADAᵀ,
///   O(m³) to factor it, O(mn) for the remaining back-substitutions.
LPDetailedResult solveShortStepIPM(const Model& model,
                                   uint32_t     maxIter,
                                   double       timeLimitS,
                                   std::chrono::steady_clock::time_point startTime);

} // namespace baguette::internal
