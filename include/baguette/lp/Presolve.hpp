#pragma once

#include <chrono>
#include <cstdint>
#include <limits>
#include <utility>

#include "baguette/model/Model.hpp"

namespace baguette {

/// Result of the presolve phase.
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
/// @p model is modified directly. Use presolve() for a copy-based variant
/// that preserves the original model.
///
/// @return  Presolve statistics. If result.infeasible is true, @p model has
///          an empty domain and must not be solved. If result.timeLimitReached
///          is true, the fixpoint may not have been reached.
///
/// @param model      Model to presolve; bounds are updated in place.
/// @param maxPasses  Maximum number of full constraint passes. 0 = unlimited
///                   (runs until fixpoint). Default: 10.
/// @param timeLimitS Wall-clock budget in seconds shared with the outer solve.
///                   The pass loop exits early (timeLimitReached = true) if the
///                   elapsed time since @p startTime exceeds this value.
///                   Default: no limit.
/// @param startTime  Reference clock. Defaults to now() when omitted.
///
/// @par Complexity
/// O(P × C × N) where P = passes run, C = number of constraints, and
/// N = average number of variables per constraint.
PresolveResult presolveInPlace(
    Model& model,
    uint32_t maxPasses = 10,
    double   timeLimitS = std::numeric_limits<double>::infinity(),
    std::chrono::steady_clock::time_point startTime =
        std::chrono::steady_clock::now());

/// Apply bound-tightening presolve to a **copy** of @p model.
///
/// The original @p model is NOT modified. Returns the presolved copy
/// together with the presolve statistics, so callers can compare the
/// original and the presolved model for debugging.
///
/// Internally copies @p model and calls presolveInPlace() on the copy.
/// Prefer this variant at the top level (e.g. before a single LP/MILP
/// solve) when the caller may need to inspect the original bounds later.
/// Use presolveInPlace() for inner loops where the copy overhead matters.
///
/// @param model     Original model; not modified.
/// @param maxPasses See presolveInPlace().
/// @param timeLimitS See presolveInPlace().
/// @param startTime  See presolveInPlace().
/// @return          {presolved_copy, statistics}.
///
/// @par Complexity
/// O(m + P × C × N) — O(m) copy plus the in-place cost.
std::pair<Model, PresolveResult> presolve(
    const Model& model,
    uint32_t     maxPasses   = 10,
    double       timeLimitS  = std::numeric_limits<double>::infinity(),
    std::chrono::steady_clock::time_point startTime =
        std::chrono::steady_clock::now());

} // namespace baguette
