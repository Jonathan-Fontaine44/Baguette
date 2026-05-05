#pragma once

#include <chrono>
#include <cstdint>

#include "baguette/lp/LPResult.hpp"
#include "baguette/model/Model.hpp"

namespace baguette::internal {

/// Two-phase primal simplex using the bounded-variable (BV) technique.
///
/// Variable upper bounds are enforced via the complement invariant in the ratio
/// test — no explicit upper-bound rows are added to the constraint matrix.
/// This keeps m = nOrigRows, eliminating the O(n) row inflation of solvePrimal().
///
/// Warm-start is not supported on this path.
LPDetailedResult solvePrimalBV(const Model&                          model,
                                uint32_t                              maxIter,
                                double                                timeLimitS,
                                std::chrono::steady_clock::time_point startTime,
                                bool                                  computeCutData,
                                bool                                  computeSensitivity = false);

} // namespace baguette::internal
