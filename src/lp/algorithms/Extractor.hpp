#pragma once

#include <cstdint>
#include <vector>

#include "baguette/lp/LPResult.hpp"
#include "baguette/model/Model.hpp"
#include "SimplexTableau.hpp"
#include "StandardForm.hpp"

namespace baguette::internal {

/// Extract the full LPDetailedResult from a solved phase-II tableau.
///
/// @p equalArtCol  Optional mapping (size == sf.nOrigRows) from each Equal
///                 constraint row to the column index of its artificial variable
///                 in the augmented tableau.  When provided, dual values for
///                 Equal rows are read as  y_i = −rc[equalArtCol[i]]  instead
///                 of being left at zero.  Pass an empty vector on the dual path
///                 (no artificial columns kept).
/// @note Complexity: O(nOrig + nOrigRows + nCols) when computeSensitivity is
///   false. When true, dominated by sensitivity analysis at O(m·n_eff).
LPDetailedResult extractDetailed(const SimplexTableau&        tab,
                                  const LPStandardForm&        sf,
                                  const Model&                 model,
                                  LPStatus                     status,
                                  const std::vector<uint32_t>& equalArtCol    = {},
                                  bool                         computeSensitivity = false);

} // namespace baguette::internal
