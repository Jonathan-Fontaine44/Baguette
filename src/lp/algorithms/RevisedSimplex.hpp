#pragma once

#include <chrono>
#include <cstdint>

#include "SimplexConfig.hpp"
#include "baguette/lp/LPResult.hpp"
#include "baguette/model/Model.hpp"

namespace baguette::internal {

/// Two-phase primal revised simplex using an explicit m×m basis inverse (LUTableau).
///
/// Produces the same LPDetailedResult as solvePrimal / solveDual, but maintains
/// B⁻¹ (m×m) rather than the full m×n tableau, reducing peak memory when m ≪ n.
/// Sensitivity analysis is supported via on-demand tableau-row computation.
///
/// When @p warmBasis is non-empty, Phase I is skipped: the solver initialises
/// directly from the parent node's basis and runs dual simplex to restore primal
/// feasibility (dual feasibility is preserved across bound-only changes).  Falls
/// back to the full Phase I + II cold start if the warm basis is incompatible or
/// dual infeasible after init.
///
/// @note Complexity: O(K·m·n) total pivots (K pivots, each O(m·n) with full
///   repricing), plus O(m³) per reinversion every reinversion_period pivots.
LPDetailedResult solveRevised(const Model&                          model,
                               uint32_t                              maxIter,
                               double                                timeLimitS,
                               std::chrono::steady_clock::time_point startTime,
                               bool                                  computeSensitivity,
                               bool                                  computeCutData,
                               const SimplexConfig&                  cfg      = {},
                               const BasisRecord&                    warmBasis = {});

} // namespace baguette::internal
