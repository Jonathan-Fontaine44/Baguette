#pragma once

#include <chrono>
#include <cstdint>

#include "baguette/lp/LPResult.hpp"
#include "baguette/model/Model.hpp"

namespace baguette::internal {

/// Two-phase primal bounded-variable revised simplex using an explicit m×m basis
/// inverse (LUTableau + BV complement invariant).
///
/// Uses LPStandardFormBV (m = nOrigRows, no explicit UB rows).  Variable upper
/// bounds are enforced via the complement invariant, keeping the working set at
/// O(m²) instead of O((m+n_UB)²) as in RevisedSimplex.
///
/// @note Sensitivity analysis is not supported on this path.
/// @note Complexity O(K·m·n) total pivots (K pivots), plus O(m³) per reinversion
///   every reinversion_period pivots.
LPDetailedResult solveRevisedBV(const Model&                          model,
                                 uint32_t                              maxIter,
                                 double                                timeLimitS,
                                 std::chrono::steady_clock::time_point startTime,
                                 bool                                  computeCutData);

} // namespace baguette::internal
