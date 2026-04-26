#pragma once

#include <cstdint>
#include <vector>

namespace baguette {

/// Status returned by a CP propagation call.
enum class CPStatus {
    Feasible,   ///< Constraint(s) consistent; bounds may have been tightened.
    Infeasible, ///< Domain wipe-out detected; current B&B node is provably infeasible.
};

/// Result of one CP propagation call.
struct PropagationResult {
    CPStatus              status       = CPStatus::Feasible;
    std::vector<uint32_t> changedVarIds; ///< Sorted IDs of variables whose bounds were tightened.
};

} // namespace baguette
