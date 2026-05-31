#pragma once

#include "baguette/lp/LPResult.hpp"
#include "baguette/model/Presolve.hpp"

namespace baguette {

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
/// model - its semantics are undefined after elimination.
void postsolveElim(LPDetailedResult& r, const EliminationRecord& rec);

} // namespace baguette
