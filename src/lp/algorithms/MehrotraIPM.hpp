#pragma once

#include <chrono>
#include <cstdint>

#include "baguette/lp/LPResult.hpp"
#include "baguette/model/Model.hpp"

namespace baguette::internal {

/// Primal-dual infeasible-start IPM with Mehrotra predictor-corrector.
///
/// No feasibility requirement on the starting point. Each iteration:
///   1. Predictor (affine) direction with μ_target = 0.
///   2. Compute α_aff, μ_aff; adaptive centering σ = (μ_aff/μ)³.
///   3. Corrector direction with μ_target = σμ and Δx_aff⊙Δs_aff cross-term.
///   4. Separate primal/dual step lengths (0.99 × ratio test).
///   5. Detect infeasibility (μ → 0, ‖rp‖ stays large) and
///      unboundedness (x diverges).
///
/// @node Complexity
///   O(K · m²n) total, K ≤ maxIter. Each iteration: one O(m²n) matrix build,
///   one O(m³) LU factorisation (reused for predictor and corrector), O(mn)
///   back-substitutions. Typically K = 15–50 in practice.
LPDetailedResult solveMehrotraIPM(const Model& model,
                                  uint32_t     maxIter,
                                  double       timeLimitS,
                                  std::chrono::steady_clock::time_point startTime);

} // namespace baguette::internal