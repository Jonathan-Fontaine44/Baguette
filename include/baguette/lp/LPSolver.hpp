#pragma once

#include <chrono>
#include <cstdint>
#include <limits>

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
/// @param timeLimitS Wall-clock time limit in seconds. infinity() = unlimited.
/// @param startTime  Reference point for the time limit. Defaults to now().
///                   Pass a B&B root startTime to share the budget across nodes.
/// @note Complexity: O(m·n) for standard-form setup, then O(K·m·n) for the
///   two-phase simplex where K = total pivot count (problem-dependent).
///   See solveDetailed() for full details.
LPResult solve(const Model&            model,
               uint32_t                maxIter    = 0,
               double                  timeLimitS = std::numeric_limits<double>::infinity(),
               SolverClock::time_point startTime  = SolverClock::now());

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
/// @note When @p computeSensitivity is false (default), the
///       `LPDetailedResult::sensitivity` field is left empty.  Pass true only
///       when RHS / objective ranging is needed (e.g. root LP analysis); avoid
///       it in B&B hot loops where the O(m·n) cost is paid at every node.
///
/// @note When @p computeCutData is false (default), the
///       `LPDetailedResult::fractionalRows` field is left empty.  Pass true to
///       enable Gomory cut generation at this node.  Only the first sf.nCols
///       columns of the phase-II tableau row are stored (artificials excluded).
///
/// @param model              The model to solve.
/// @param maxIter            Maximum number of simplex pivots. 0 = unlimited.
/// @param timeLimitS         Wall-clock time limit in seconds. infinity() = unlimited.
/// @param startTime          Reference point for the time limit. Defaults to now().
///                           Pass a B&B root startTime to share the budget across nodes.
/// @param computeSensitivity If true, fills LPDetailedResult::sensitivity. Default false.
/// @param computeCutData     If true, fills LPDetailedResult::fractionalRows. Default false.
/// @note Complexity: O(m·n) for standard-form setup, then O(K·m·n) for the two-phase
///   primal simplex, where K = total pivot count (problem-dependent; exponential worst
///   case, polynomial in practice). m = constraint rows (model + UB rows), n = SF columns.
///   Periodic reinversion adds O(m²·n) every reinversion_period pivots.
LPDetailedResult solveDetailed(const Model& model,
                               uint32_t maxIter            = 0,
                               double   timeLimitS         = std::numeric_limits<double>::infinity(),
                               SolverClock::time_point startTime  = SolverClock::now(),
                               bool     computeSensitivity = false,
                               bool     computeCutData     = false);

// ── Dual simplex ───────────────────────────────────────────────────────────────

/// Solve @p model using the dual simplex algorithm where applicable,
/// otherwise fall back to the primal two-phase simplex.
///
/// **Cold start** (default, @p warmBasis empty): builds a dual-feasible basis
/// from the natural slack / surplus columns of the standard form.  The standard
/// form is always a minimisation (Maximize is handled by negating the objective),
/// so dual feasibility of the natural basis requires every standard-form
/// objective coefficient sf.c[j] ≥ 0.  GEQ rows are handled natively: the
/// surplus column (coeff −1) is negated by Gauss-Jordan, giving a
/// primal-infeasible but dual-feasible start.
///
/// **Warm start** (@p warmBasis non-empty, B&B use case): reinverts the
/// tableau directly from the parent node's BasisRecord, then runs the dual
/// simplex to restore primal feasibility after bound tightening.  The parent's
/// basis is still dual-feasible because bound tightening only changes the RHS
/// vector b, not A or c, so RC = c − c_B B⁻¹ A is unchanged.
///
/// **Standard-form caching**: when @p warmBasis carries a `sfCache` (populated
/// by a previous solveDualDetailed() call), the constraint matrix A is reused
/// via shared_ptr (O(1)) and only b, varShiftVal, and objOffset are recomputed
/// for the new bounds.  This avoids the O(m·n) zero-fill and re-fill of the
/// full standard form on every B&B node.  The returned BasisRecord always
/// carries an updated `sfCache` when status == Optimal.
///
/// **Bound-finiteness invariant for warm start** — the finiteness (finite vs.
/// infinite) of every variable bound must be the same in @p model as in the
/// parent model that produced @p warmBasis.  Changing finiteness alters the
/// standard-form structure (upper-bound rows, free-split columns), making the
/// basis incompatible.  See Model::withVarBounds() for details.
///
/// Automatic fallbacks to a cold primal two-phase solve occur when:
///   - Cold path: any constraint has `Sense::Equal`, or any standard-form
///     objective coefficient is negative (e.g. Maximize with positive costs).
///   - Warm path: @p warmBasis dimensions are incompatible with @p model's
///     standard form (bound-finiteness invariant violated), or the warm basis
///     is not dual-feasible after reinversion, or reinversion fails numerically.
///   All fallback exits also populate `sfCache` in the returned BasisRecord.
///
/// @param model      The model to solve.
/// @param maxIter    Maximum dual-simplex pivots (0 = unlimited).
/// @param timeLimitS Wall-clock limit in seconds (infinity() = unlimited).
/// @param startTime  Reference point for the time limit. Defaults to now().
///                   Pass a B&B root startTime to share the budget across nodes.
/// @param warmBasis  Parent node's BasisRecord for warm start. Default {} = cold start.
/// @note Complexity: O(m·n) amortised for standard-form setup (O(1) with sfCache),
///   then O(K·m·n) for the dual simplex. Falls back to solveDetailed() complexity
///   when dual feasibility cannot be established. See solveDualDetailed() for full details.
LPResult solveDual(const Model&            model,
                   uint32_t                maxIter    = 0,
                   double                  timeLimitS = std::numeric_limits<double>::infinity(),
                   SolverClock::time_point startTime  = SolverClock::now(),
                   const BasisRecord&      warmBasis  = {});

/// Same as solveDual() but returns the full LPDetailedResult, including a
/// new BasisRecord suitable for passing to the next level of the B&B tree.
///
/// @note When @p computeSensitivity is false (default), the
///       `LPDetailedResult::sensitivity` field is left empty.  In B&B warm-start
///       loops the O(m·n) sensitivity cost would be paid at every node; pass true
///       only for the root LP or other one-shot solves that need ranging.
///
/// @note When @p computeCutData is false (default), the
///       `LPDetailedResult::fractionalRows` field is left empty.  Pass true only
///       at B&B nodes where cut generation is desired.  Populating the raw tableau
///       rows costs O(m·n) additional memory.  Not populated on the fallback primal
///       path.
///
/// @param model              The model to solve.
/// @param maxIter            Maximum dual-simplex pivots (0 = unlimited).
/// @param timeLimitS         Wall-clock limit in seconds (infinity() = unlimited).
/// @param startTime          Reference point for the time limit. Defaults to now().
///                           Pass the B&B root startTime to share the budget across nodes.
/// @param warmBasis          Parent node's BasisRecord for warm start. Default {} = cold start.
/// @param computeSensitivity If true, fills LPDetailedResult::sensitivity. Default false.
/// @param computeCutData     If true, fills LPDetailedResult::fractionalRows with raw
///                           tableau rows for basic fractional integer variables.
///                           Populated only on the dual simplex path when status == Optimal.
/// @note Complexity: O(m·n) amortised for standard-form setup (O(1) matrix reuse with
///   sfCache), then O(K·m·n) for the dual simplex where K = total pivot count.
///   Warm-start reinversion is O(m²·n). Falls back to solveDetailed() complexity when
///   dual feasibility cannot be established. m = SF rows, n = SF columns.
LPDetailedResult solveDualDetailed(const Model&            model,
                                   uint32_t                maxIter            = 0,
                                   double                  timeLimitS         = std::numeric_limits<double>::infinity(),
                                   SolverClock::time_point startTime          = SolverClock::now(),
                                   const BasisRecord&      warmBasis          = {},
                                   bool                    computeSensitivity = false,
                                   bool                    computeCutData     = false);

} // namespace baguette
