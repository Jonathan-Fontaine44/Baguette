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
/// until no bound changes or @p maxCycles outer iterations are reached.
///
/// Before the outer loop, constraint RHS values are tightened (PR1):
/// for every constraint whose variables are all Integer/Binary and whose
/// coefficients are all integer-valued, the RHS is rounded down (floor for ≤)
/// or up (ceil for ≥).  This is valid because the LHS is always integer at any
/// feasible integer point, allowing tighter LP propagation without loss of
/// integer solutions.  The count of tightened RHS values is reported in
/// MILPPresolveResult::rhsRounded.
///
/// Must not be called for LP-relaxation presolve — use presolveTBInPlace.
///
/// @param maxCycles   Maximum outer (LP-fixpoint + integrality round) cycles.
///                    0 = run until fixed point.  Independent of
///                    LPOptions::presolveMaxPasses, which controls LP pass count
///                    inside each LP solve node.  Configure via
///                    BBOptions::milpPresolveMaxCycles.
/// @param intFeasTol  Tolerance for snapping integer bounds: lb → ceil(lb - tol),
///                    ub → floor(ub + tol).  Must match BBOptions::intFeasTol so
///                    that the presolve and the B&B tree use the same definition
///                    of "integer-feasible".
///
/// @note Complexity O(C × N + P × (C × N + V)) where C = constraint count,
///   N = max variables per constraint, P = outer iterations,
///   V = integer variable count.  The leading C × N term is the one-time
///   PR1 RHS scan; P × (...) is the iterative LP + rounding loop.
MILPPresolveResult presolveMILPInPlace(
    Model&   model,
    uint32_t maxCycles  = 0,
    double   intFeasTol = 1e-6,
    double   timeLimitS = std::numeric_limits<double>::infinity(),
    std::chrono::steady_clock::time_point startTime =
        std::chrono::steady_clock::now());

} // namespace baguette
