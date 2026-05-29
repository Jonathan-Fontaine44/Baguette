#pragma once

#include <cstdint>
#include <vector>

#include "baguette/lp/LPResult.hpp"
#include "baguette/milp/CuttingPlanes.hpp"
#include "baguette/model/Model.hpp"

namespace baguette {

/// Generate Mixed-Integer Rounding (MIR) cuts from LessEq model constraints.
///
/// For each LessEq constraint `Σ aⱼxⱼ ≤ b` with fractional RHS (after
/// shifting integer variables by their lower bounds), applies the MIR formula:
///
///   For integer vars:    α̃ⱼ = ⌊aⱼ⌋ + max(0, frac(aⱼ) − f) / (1 − f)
///   For continuous vars: α̃ⱼ = aⱼ / f  (positive coeff only; negative dropped)
///   Cut:                 Σ α̃ⱼ xⱼ ≤ ⌊b'⌋ + Σ α̃ⱼ lbⱼ
///
/// where `f = frac(b')` and `b' = b − Σ aⱼ lbⱼ` is the shifted RHS.
/// Only cuts that are violated by the current LP solution are returned.
///
/// @note Variables with negative shifted coefficients are skipped (conservative).
///   Standard MIR would complement them via `x'ⱼ = ubⱼ − xⱼ`, but this requires
///   finite upper bounds and is not implemented — cuts are valid but potentially weaker.
///
/// Internal — activated by BBOptions::enableMIR. Not part of the public API.
///
/// @note Complexity: O(C × K) where C = number of LP constraints and
///   K = average number of non-zeros per constraint.
std::vector<Cut> generateMIRCuts(const LPDetailedResult& lp,
                                  const Model&            model,
                                  uint32_t                maxCuts    = 0,
                                  double                  intFeasTol = 1e-6);

/// Generate Continuous MIR (CMIR) cuts from GreaterEq model constraints.
///
/// Applies complementation `x'ⱼ = ubⱼ − xⱼ` to convert a GEQ constraint
/// `Σ aⱼxⱼ ≥ b` (with aⱼ > 0, finite ubⱼ) into a LessEq form in the
/// complemented variables, then applies MIR on that form.
///
/// The resulting cut in original-variable space is: Σ α̃ⱼ xⱼ ≥ Σ α̃ⱼ ubⱼ − ⌊b̄⌋
/// where `b̄ = Σ aⱼ ubⱼ − b`.
/// Only cuts that are violated by the current LP solution are returned.
///
/// Internal — activated by BBOptions::enableMIR. Not part of the public API.
///
/// @note Complexity: O(C × K) where C = number of LP constraints and
///   K = average number of non-zeros per constraint.
std::vector<Cut> generateCMIRCuts(const LPDetailedResult& lp,
                                   const Model&            model,
                                   uint32_t                maxCuts    = 0,
                                   double                  intFeasTol = 1e-6);

} // namespace baguette
