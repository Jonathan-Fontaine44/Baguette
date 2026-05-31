#pragma once

#include <cstdint>

namespace baguette::internal {

/// Internal numerical parameters for simplex algorithms.
///
/// Not exposed in public LPOptions. Each solve constructs one SimplexConfig
/// from LPOptions (user-configurable fields) plus fixed internal defaults
/// (pivotTol, reinversionPeriod). Stored by value in SimplexTableau / LUTableau
/// so hot-path member functions access per-solve values, not global state.
///
/// @note Complexity
///   All fields are read O(1) per pivot. No dynamic allocation.
struct SimplexConfig {
    /// Primal feasibility tolerance - from LPOptions::feasibilityTol.
    double feasibilityTol = 1e-9;

    /// Dual optimality tolerance - from LPOptions::optimalityTol.
    double optimalityTol = 1e-9;

    /// Minimum absolute pivot magnitude. Internal constant, not user-facing.
    /// Pivot candidates with |a_ij| <= pivotTol are skipped to avoid
    /// numerical blow-up from near-zero pivots.
    double pivotTol = 1e-9;

    /// Reinversion period. Internal constant, not user-facing.
    /// B⁻¹ is rebuilt from scratch every this many pivots to cap floating-point
    /// drift. 0 disables reinversion.
    uint32_t reinversionPeriod = 50;

    /// If true, use Dantzig's most-negative-rc entering rule instead of Bland's.
    /// Applies to primal-simplex phases only (no effect on dual or IPM).
    bool useDantzig = false;
};

} // namespace baguette::internal
