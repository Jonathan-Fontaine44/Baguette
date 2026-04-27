#pragma once

#include <chrono>
#include <cstdint>

#include "baguette/lp/LPResult.hpp"
#include "baguette/model/Model.hpp"

namespace baguette::internal {

/// Run the dual simplex (with warm-start support and primal fallback).
///
/// Attempts a dual-feasible start from natural slack/surplus columns (cold path)
/// or from @p warmBasis (warm path).  Falls back to solvePrimal() when:
///   - Equal constraints prevent a natural dual-feasible basis (cold path).
///   - Any objective coefficient is negative after shifting (e.g. Maximize).
///   - Warm-basis dimensions mismatch the current standard form.
///   - Reinversion fails or the warm basis is not dual-feasible.
///
/// @note Complexity: O(m·n) amortised for standard-form setup (O(1) with
///   sfCache), then O(K·m·n) for the dual simplex. Falls back to solvePrimal()
///   complexity when dual feasibility cannot be established.
LPDetailedResult solveDual(const Model&                          model,
                            uint32_t                              maxIter,
                            double                                timeLimitS,
                            std::chrono::steady_clock::time_point startTime,
                            const BasisRecord&                    warmBasis,
                            bool                                  computeSensitivity,
                            bool                                  computeCutData);

} // namespace baguette::internal
