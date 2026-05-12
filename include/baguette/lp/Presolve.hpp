#pragma once

#include <chrono>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "baguette/model/Model.hpp"

namespace baguette {

// Forward declarations to avoid circular includes.
// LPResult.hpp and MILPResult.hpp both #include this header via PresolveResult.
struct LPDetailedResult;
struct MILPResult;

// ── Bound-Tightening Presolve (TB) ────────────────────────────────────────────

/// Result of the bound-tightening presolve phase.
struct PresolveResult {
    /// True if presolve detected an infeasible domain (lb > ub implied by
    /// the constraints). When true, the model must not be solved.
    bool infeasible = false;

    /// True if the pass loop was cut short by the time limit.
    bool timeLimitReached = false;

    /// Total number of variable bound updates applied across all passes.
    uint32_t boundsTightened = 0;

    /// Number of variables with lb == ub (fixed) at the end of presolve.
    uint32_t fixedVars = 0;

    /// Number of bound-propagation passes completed.
    uint32_t passesRun = 0;
};

/// Apply bound-tightening presolve to @p model **in place**.
///
/// Propagates variable bounds through each constraint to narrow [lb, ub]
/// intervals. Iterates to fixpoint or until @p maxPasses is reached (or the
/// time limit expires). Detects infeasibility when a variable's domain becomes
/// empty (lb > ub).
///
/// **Techniques applied:**
///   1. Constraint-based bound tightening — for each (in)equality, compute
///      the activity bounds and tighten each variable's bound individually.
///   2. Fixed-variable detection — variables with lb == ub are counted.
///
/// @p model is modified directly. Use presolveTB() for a copy-based variant
/// that preserves the original model.
///
/// @return  Presolve statistics. If result.infeasible is true, @p model has
///          an empty domain and must not be solved. If result.timeLimitReached
///          is true, the fixpoint may not have been reached.
///
/// @param model      Model to presolve; bounds are updated in place.
/// @param maxPasses  Maximum number of full constraint passes. 0 = unlimited
///                   (runs until fixpoint). Default: 0.
/// @param timeLimitS Wall-clock budget in seconds shared with the outer solve.
///                   The pass loop exits early (timeLimitReached = true) if the
///                   elapsed time since @p startTime exceeds this value.
///                   Default: no limit.
/// @param startTime  Reference clock. Defaults to now() when omitted.
///
/// @par Complexity
/// O(P × C × N) where P = passes run, C = number of constraints, and
/// N = average number of variables per constraint.
PresolveResult presolveTBInPlace(
    Model& model,
    uint32_t maxPasses  = 0,
    double   timeLimitS = std::numeric_limits<double>::infinity(),
    std::chrono::steady_clock::time_point startTime =
        std::chrono::steady_clock::now());

/// Apply bound-tightening presolve to a **copy** of @p model.
///
/// The original @p model is NOT modified. Returns the presolved copy
/// together with the presolve statistics, so callers can compare the
/// original and the presolved model for debugging.
///
/// Internally copies @p model and calls presolveTBInPlace() on the copy.
/// Prefer this variant at the top level (e.g. before a single LP/MILP
/// solve) when the caller may need to inspect the original bounds later.
/// Use presolveTBInPlace() for inner loops where the copy overhead matters.
///
/// @param model     Original model; not modified.
/// @param maxPasses See presolveTBInPlace().
/// @param timeLimitS See presolveTBInPlace().
/// @param startTime  See presolveTBInPlace().
/// @return          {presolved_copy, statistics}.
///
/// @par Complexity
/// O(m + P × C × N) — O(m) copy plus the in-place cost.
std::pair<Model, PresolveResult> presolveTB(
    const Model& model,
    uint32_t     maxPasses   = 0,
    double       timeLimitS  = std::numeric_limits<double>::infinity(),
    std::chrono::steady_clock::time_point startTime =
        std::chrono::steady_clock::now());

// ── Elimination Presolve ───────────────────────────────────────────────────────

/// Data produced by presolveElim() required to reconstruct the full-model
/// solution from a reduced-model LP/MILP result via postsolveElim().
struct EliminationRecord {
    uint32_t origVarCount        = 0; ///< Variable count in the original model.
    uint32_t origConstraintCount = 0; ///< Constraint count in the original model.

    /// varMap[orig_id] = reduced_id, or UINT32_MAX if the variable was fixed.
    std::vector<uint32_t> varMap;
    /// reducedToOrig[reduced_id] = orig_id (inverse of varMap for non-fixed vars).
    std::vector<uint32_t> reducedToOrig;
    /// Fixed variables: {orig_id, fixed_value} for each eliminated variable.
    std::vector<std::pair<uint32_t, double>> fixedVars;

    /// conMap[orig_idx] = reduced_idx, or UINT32_MAX if the row was eliminated.
    std::vector<uint32_t> conMap;
    /// reducedToOrigCon[reduced_idx] = orig_idx (inverse of conMap for kept rows).
    std::vector<uint32_t> reducedToOrigCon;

    /// Objective constant offset: sum(c_j * v_j for all fixed variables j).
    /// Added back to the reduced-model objective at postsolveElim().
    double objAdjustment = 0.0;

    uint32_t varsEliminated = 0; ///< Number of fixed variables removed.
    uint32_t rowsEliminated = 0; ///< Number of always-satisfied rows removed.
};

/// Build a reduced model by eliminating fixed variables and redundant rows.
///
/// **Fixed-variable elimination (column reduction):**
///   A variable x_j is fixed when lb_j == ub_j (within lp_feasibility_tol).
///   Its value v_j = lb_j is substituted into every constraint
///   (adjustedRHS -= a_ij * v_j) and into the objective constant
///   (rec.objAdjustment += c_j * v_j). The variable is removed from the
///   reduced model.
///
/// **Redundant-row elimination (row reduction):**
///   A constraint is always satisfied when, given the current bounds of the
///   remaining (non-fixed) variables, the constraint can never be violated:
///     - LEQ: maxActivity <= rhs  (never tight → always slack)
///     - GEQ: minActivity >= rhs  (never tight → always slack)
///     - EQ:  both conditions hold (trivially determined)
///   Redundant rows carry no information and are dropped. Their dual variables
///   are 0 by complementary slackness; postsolveElim() inserts zeros at the
///   correct positions.
///
/// The original @p model is NOT modified.
///
/// @param model  Input model (ideally after presolveTBInPlace() for best effect,
///               since TB fixes variables that elimination then removes).
/// @param rec    Output: mapping data needed by postsolveElim(). Overwritten.
/// @return  Reduced model with fewer variables and/or constraints.
///
/// @par Complexity
/// O(V + C × N) where V = variables, C = constraints, N = avg terms/constraint.
Model presolveElim(const Model& model, EliminationRecord& rec);

/// Reconstruct the full-model LP solution from a reduced-model result.
///
/// Adjusts @p r in place:
///   - objectiveValue  : += rec.objAdjustment  (when primal is available)
///   - primalValues    : expanded to origVarCount; fixed vars inserted at
///                       their fixed values (status Optimal/MaxIter/TimeLimit).
///   - dualValues      : expanded to origConstraintCount; 0 for eliminated rows
///                       (status Optimal only).
///   - reducedCosts    : expanded to origVarCount; 0 for fixed variables
///                       (status Optimal only).
///
/// Sensitivity analysis (rhsRange, objRange) is left sized for the reduced
/// model — its semantics are undefined after elimination.
void postsolveElim(LPDetailedResult& r, const EliminationRecord& rec);

/// Reconstruct the full-model MILP solution from a reduced-model result.
///
/// Adjusts @p r in place:
///   - objectiveValue  : += rec.objAdjustment  (only when primalValues non-empty)
///   - primalValues    : expanded to origVarCount; fixed vars inserted at
///                       their fixed values.
void postsolveElim(MILPResult& r, const EliminationRecord& rec);

} // namespace baguette
