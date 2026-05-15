#pragma once

#include <cstdint>

namespace baguette {

// ── LinearExpr ──────────────────────────────────────────────────────────────

/// Zero threshold for coefficient cancellation in LinearExpr.
///
/// A coefficient whose absolute value is ≤ `cancellation_tol` is treated as
/// zero and removed from a LinearExpr (in addTerm and operator+).
///
/// Default: 1e-9.
inline double cancellation_tol = 1e-9;

/// Set the zero threshold used when cancelling coefficients in LinearExpr.
/// @param tol New threshold value (must be ≥ 0).
inline void set_cancellation_tol(double tol) { cancellation_tol = tol; }

// ── LP solver — global defaults ──────────────────────────────────────────────

/// Primal feasibility tolerance.
/// Used by Domain::isFixed() and bound-tightening presolve.
/// Per-solve configuration: LPOptions::feasibilityTol.
/// Default: 1e-9.
inline double lp_feasibility_tol = 1e-9;

/// Set the LP feasibility tolerance.
/// @param tol New threshold (must be > 0).
inline void set_lp_feasibility_tol(double tol) { lp_feasibility_tol = tol; }

/// Dual optimality tolerance.
/// Per-solve configuration: LPOptions::optimalityTol.
/// Default: 1e-9.
inline double lp_optimality_tol = 1e-9;

/// Set the LP optimality tolerance.
/// @param tol New threshold (must be > 0).
inline void set_lp_optimality_tol(double tol) { lp_optimality_tol = tol; }

} // namespace baguette
