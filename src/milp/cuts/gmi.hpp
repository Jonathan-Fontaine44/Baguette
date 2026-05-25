#pragma once

#include <cstdint>
#include <vector>

#include "baguette/lp/LPResult.hpp"
#include "baguette/milp/CuttingPlanes.hpp"
#include "baguette/model/Model.hpp"

namespace baguette {

/// Generate Gomory Mixed-Integer (GMI) cuts from fractional tableau rows.
///
/// Internal — called by the B&C loop in solveMILP(). Not part of the public API.
/// Users activate GMI via BBOptions::enableCuts.
///
/// @param rows        Fractional rows from LPDetailedResult::fractionalRows.
/// @param basis       BasisRecord from the same solve (colKind / colOrigin metadata).
/// @param model       The current B&B-node model (variable bounds and constraints).
/// @param maxCuts     Maximum number of cuts to return. 0 = return all.
/// @param intFeasTol  Integer feasibility tolerance (same as BBOptions::intFeasTol).
///
/// @note Complexity: O(|rows| × n × nnz_avg) where n = number of SF columns and
///   nnz_avg = average non-zeros per model constraint. Dominated by slack substitution.
std::vector<Cut> generateGMICuts(const std::vector<FractionalRow>& rows,
                                  const BasisRecord&                basis,
                                  const Model&                      model,
                                  uint32_t                          maxCuts    = 0,
                                  double                            intFeasTol = 1e-6);

} // namespace baguette
