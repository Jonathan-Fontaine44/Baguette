#pragma once

#include <chrono>
#include <cstdint>

#include "baguette/lp/LPResult.hpp"
#include "baguette/model/Model.hpp"

namespace baguette::internal {

/// Run the BV dual simplex (cold or warm start, primal fallback to PrimalSimplexBV).
///
/// Uses LPStandardFormBV (no UB rows — m = nOrigRows) with the complement
/// invariant. Cold-start basis: natural slack/surplus columns; dual feasibility
/// requires all objective coefficients ≥ 0 after shifting (falls back otherwise).
///
/// Warm-start path: reads BasisRecord::basicCols + BasisRecord::atUBCache from a
/// previous BV solve. On success, populates those fields in the returned result.
///
/// @node Complexity
/// Standard-form setup O(m·n), then O(K·m·n) for K dual pivots.
/// Falls back to solvePrimalBV() complexity when dual feasibility fails.
LPDetailedResult solveDualBV(const Model&                          model,
                              uint32_t                              maxIter,
                              double                                timeLimitS,
                              std::chrono::steady_clock::time_point startTime,
                              const BasisRecord&                    warmBasis,
                              bool                                  computeCutData,
                              bool                                  computeSensitivity = false);

} // namespace baguette::internal
