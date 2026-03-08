#pragma once

#include <cstdint>

#include "baguette/lp/LPResult.hpp"
#include "baguette/model/Model.hpp"

namespace baguette {

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
LPDetailedResult solveDetailed(const Model& model,
                               uint32_t maxIter    = 0,
                               double   timeLimitS = 0.0);

} // namespace baguette
