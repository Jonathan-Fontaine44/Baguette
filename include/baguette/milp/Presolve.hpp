#pragma once

#include <chrono>
#include <limits>

#include "baguette/milp/MILPResult.hpp"
#include "baguette/model/Model.hpp"
#include "baguette/model/Presolve.hpp"

namespace baguette {

/// Reconstruct the full-model MILP solution from a reduced-model result.
///
/// Adjusts @p r in place:
///   - objectiveValue  : += rec.objAdjustment  (only when primalValues non-empty)
///   - primalValues    : expanded to origVarCount; fixed vars inserted at
///                       their fixed values.
void postsolveElim(MILPResult& r, const EliminationRecord& rec);

/// Apply MILP-specific presolve to @p model in place.
///
/// Interleaves LP bound-tightening with integrality rounding: for each
/// Integer/Binary variable, lb is snapped to ceil(lb) and ub to floor(ub).
/// Infeasibility is detected when lb > ub after rounding.  The two-phase loop
/// (LP-propagation to its own fixed point, then integer rounding) repeats
/// until no bound changes or @p maxPasses outer iterations are reached.
///
/// Must not be called for LP-relaxation presolve — use presolveTBInPlace.
///
/// @param maxPasses  Maximum outer (LP + round) cycles; 0 = until fixed point.
///
/// @par Complexity O(P × (C × N + V)) where P = outer iterations, C = LP
///   constraints, N = max variables per constraint, V = integer variable count.
MILPPresolveResult presolveMILPInPlace(
    Model&   model,
    uint32_t maxPasses  = 0,
    double   timeLimitS = std::numeric_limits<double>::infinity(),
    std::chrono::steady_clock::time_point startTime =
        std::chrono::steady_clock::now());

} // namespace baguette
