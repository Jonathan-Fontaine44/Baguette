#pragma once

#include <chrono>
#include <cstdint>

#include "SimplexConfig.hpp"
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
/// When @p warmBasis is non-empty (basicCols + atUBCache), Phase I is skipped:
/// the solver initialises directly from the parent's basis and runs dual simplex
/// (BV-aware) to restore primal feasibility.  Falls back to cold start if the
/// warm basis is incompatible or dual infeasible after init.
///
/// @note Sensitivity analysis is not supported on this path.
/// @note Complexity O(K·m·n) total pivots (K pivots), plus O(m³) per reinversion
///   every reinversion_period pivots.
LPDetailedResult solveRevisedBV(const Model&                          model,
                                 uint32_t                              maxIter,
                                 double                                timeLimitS,
                                 std::chrono::steady_clock::time_point startTime,
                                 bool                                  computeCutData,
                                 const SimplexConfig&                  cfg,
                                 const BasisRecord&                    warmBasis);

} // namespace baguette::internal
