#pragma once

#include <chrono>
#include <cstdint>

#include "baguette/lp/LPResult.hpp"
#include "baguette/model/Model.hpp"

namespace baguette::internal {

/// Two-phase primal revised simplex using an explicit m×m basis inverse (LUTableau).
///
/// Produces the same LPDetailedResult as solvePrimal / solveDual, but maintains
/// B⁻¹ (m×m) rather than the full m×n tableau, reducing peak memory when m ≪ n.
/// Sensitivity analysis is supported via on-demand tableau-row computation.
///
/// @note Complexity: O(K·m·n) total pivots (K pivots, each O(m·n) with full
///   repricing), plus O(m³) per reinversion every reinversion_period pivots.
LPDetailedResult solveRevised(const Model&                          model,
                               uint32_t                              maxIter,
                               double                                timeLimitS,
                               std::chrono::steady_clock::time_point startTime,
                               bool                                  computeSensitivity,
                               bool                                  computeCutData);

} // namespace baguette::internal
