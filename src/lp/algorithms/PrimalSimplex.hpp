#pragma once

#include <chrono>
#include <cstdint>

#include "SimplexConfig.hpp"
#include "baguette/lp/LPResult.hpp"
#include "baguette/model/Model.hpp"

namespace baguette::internal {

/// Run the primal two-phase simplex and return the full detailed result.
///
/// Performs standard-form conversion, phase-I (artificial variables), and
/// phase-II (optimisation).  Handles early lb > ub infeasibility detection.
///
/// @note Complexity: O(m·n) for standard-form setup, then O(K·m·n) for the
///   two-phase simplex where K = total pivot count (problem-dependent).
LPDetailedResult solvePrimal(const Model&                          model,
                              uint32_t                              maxIter,
                              double                                timeLimitS,
                              std::chrono::steady_clock::time_point startTime,
                              bool                                  computeSensitivity,
                              bool                                  computeCutData,
                              const SimplexConfig&                  cfg = {});

} // namespace baguette::internal
