#pragma once

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

// ── LP solver ───────────────────────────────────────────────────────────────

/// Feasibility tolerance for the LP solver.
/// Used in the ratio test (skip near-zero pivot candidates) and to decide
/// whether the phase-I objective is zero (problem is feasible).
/// Default: 1e-9.
inline double lp_feasibility_tol = 1e-9;

/// Set the LP feasibility tolerance.
/// @param tol New threshold (must be > 0).
inline void set_lp_feasibility_tol(double tol) { lp_feasibility_tol = tol; }

/// Optimality tolerance for the LP solver.
/// A reduced cost is considered negative (improving) only if it is below
/// −lp_optimality_tol.
/// Default: 1e-9.
inline double lp_optimality_tol = 1e-9;

/// Set the LP optimality tolerance.
/// @param tol New threshold (must be > 0).
inline void set_lp_optimality_tol(double tol) { lp_optimality_tol = tol; }

/// Minimum absolute value of a pivot element.
/// Candidates with |a_ij| ≤ pivot_tol are skipped in the ratio test to
/// avoid numerical blow-up from near-zero pivots.
/// Default: 1e-9.
inline double pivot_tol = 1e-9;

/// Set the minimum pivot magnitude threshold.
/// @param tol New threshold (must be > 0).
inline void set_pivot_tol(double tol) { pivot_tol = tol; }

/// Reinversion period: the tableau is rebuilt from scratch every this many
/// pivots to prevent floating-point drift from accumulating.
/// Default: 50.
inline int reinversion_period = 50;

/// Set the reinversion period.
/// @param period Number of pivots between reinversions (must be ≥ 1).
inline void set_reinversion_period(int period) { reinversion_period = period; }

} // namespace baguette
