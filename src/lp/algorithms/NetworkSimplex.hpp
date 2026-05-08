#pragma once

#include <chrono>
#include <cstdint>

#include "baguette/lp/LPResult.hpp"
#include "baguette/model/Model.hpp"

namespace baguette::internal {

/// Network simplex for min-cost flow LPs.
///
/// Detects if @p model is a pure min-cost flow problem (equality constraints,
/// node-arc incidence matrix with ±1 coefficients). If yes, runs the primal
/// network simplex: basis = rooted spanning tree, pivots in O(n).
/// Falls back to DualSimplexBV when the model is not a pure network.
///
/// @note Sensitivity analysis and warm-start are not supported.
/// @note Complexity O(K·n) total, where K = pivot count and n = node count,
///   versus O(K·m²) for the general revised simplex on the same problem.
LPDetailedResult solveNetworkSimplex(const Model&                          model,
                                     uint32_t                              maxIter,
                                     double                                timeLimitS,
                                     std::chrono::steady_clock::time_point startTime,
                                     bool                                  computeCutData);

} // namespace baguette::internal
