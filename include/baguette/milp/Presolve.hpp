#pragma once

#include "baguette/milp/MILPResult.hpp"
#include "baguette/model/Presolve.hpp"

namespace baguette {

/// Reconstruct the full-model MILP solution from a reduced-model result.
///
/// Adjusts @p r in place:
///   - objectiveValue  : += rec.objAdjustment  (only when primalValues non-empty)
///   - primalValues    : expanded to origVarCount; fixed vars inserted at
///                       their fixed values.
void postsolveElim(MILPResult& r, const EliminationRecord& rec);

} // namespace baguette
