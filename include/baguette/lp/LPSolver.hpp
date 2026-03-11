#pragma once

#include <chrono>
#include <cstdint>

#include "baguette/lp/LPResult.hpp"
#include "baguette/model/Model.hpp"

namespace baguette {

/// Shared clock type for all solver time limits.
using SolverClock = std::chrono::steady_clock;

/// Solve the LP relaxation of @p model.
///
/// Returns status, objective value, and primal solution.
/// Integer and Binary variables are treated as continuous (LP relaxation).
///
/// @param model      The model to solve.
/// @param maxIter    Maximum number of simplex pivots. 0 = unlimited.
/// @param timeLimitS Wall-clock time limit in seconds. 0.0 = unlimited.
///                   Must be ≥ 0.0 (no unsigned floating-point type in C++).
LPResult solve(const Model& model,
               uint32_t maxIter    = 0,
               double   timeLimitS = 0.0);

/// Solve the LP relaxation of @p model and return the full detailed result.
///
/// In addition to the basic LPResult, provides dual variables, reduced costs,
/// and a BasisRecord for B&B warm-starting.
/// Integer and Binary variables are treated as continuous (LP relaxation).
///
/// @note Dual variables, reduced costs, and basis are computed from the
///       final tableau state. They cannot be recovered from an LPResult after
///       the fact — call this function directly if you need them.
///
/// @note Dual variables for `Sense::Equal` constraints are always 0.
///       Artificial variables are stripped before phase II, so their shadow
///       price cannot be recovered from the tableau's reduced-cost row.
///
/// @param model      The model to solve.
/// @param maxIter    Maximum number of simplex pivots. 0 = unlimited.
/// @param timeLimitS Wall-clock time limit in seconds. 0.0 = unlimited.
/// @param startTime  Reference point for the time limit. Defaults to now().
///                   Pass a B&B root startTime to share the budget across nodes.
LPDetailedResult solveDetailed(const Model& model,
                               uint32_t maxIter    = 0,
                               double   timeLimitS = 0.0,
                               SolverClock::time_point startTime = SolverClock::now());

// ── Dual simplex ───────────────────────────────────────────────────────────────

/// Solve @p model using the dual simplex algorithm where applicable,
/// otherwise fall back to the primal two-phase simplex.
///
/// The dual simplex starts from a dual-feasible basis constructed from the
/// natural slack / surplus columns of the standard form.  The standard form is
/// always a minimisation (Maximize is handled by negating the objective), so
/// dual feasibility of the natural basis requires every standard-form objective
/// coefficient sf.c[j] ≥ 0.  In practice this holds for Minimize problems with
/// non-negative costs, and for B&B warm-starts where the parent's optimal basis
/// is passed via @p startTime (the main intended use case).
/// GEQ rows are handled natively: the surplus column (coeff −1) is negated by
/// Gauss-Jordan, giving a primal-infeasible but dual-feasible start.
///
/// Fallback to the primal simplex occurs when:
///   - Any constraint has `Sense::Equal` (no natural basic variable exists).
///   - Any standard-form objective coefficient is negative after the lb-shift
///     (e.g. Maximize with positive costs, or Minimize with negative costs).
///
/// @param model      The model to solve.
/// @param maxIter    Maximum simplex pivots (0 = unlimited).
/// @param timeLimitS Wall-clock limit in seconds (0.0 = unlimited).
/// @param startTime  Reference point for the time limit. Defaults to now().
///                   Pass a B&B root startTime to share the budget across nodes.
LPResult solveDual(const Model& model,
                   uint32_t maxIter    = 0,
                   double   timeLimitS = 0.0);

/// Same as solveDual() but returns the full LPDetailedResult.
LPDetailedResult solveDualDetailed(const Model& model,
                                   uint32_t maxIter    = 0,
                                   double   timeLimitS = 0.0,
                                   SolverClock::time_point startTime = SolverClock::now());

} // namespace baguette
