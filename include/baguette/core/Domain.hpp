#pragma once

#include <cmath>

#include "Config.hpp"

namespace baguette {

/// Bounds domain `[lb, ub]` for a single variable.
///
/// Represents the set of values a variable may take during propagation.
/// Used by DomainStore to track current bounds at each B&B node.
struct Domain {
    double lb; ///< Lower bound (inclusive).
    double ub; ///< Upper bound (inclusive).

    /// @return `true` if the domain is empty (`lb > ub`).
    bool isEmpty() const { return lb > ub; }

    /// @return `true` if the variable is fixed to a single value.
    /// Uses `lp_feasibility_tol` to account for ULP-level drift introduced by
    /// floor/ceil operations in B&B bound tightening.
    bool isFixed() const { return std::abs(ub - lb) <= lp_feasibility_tol; }

    /// @return `true` if @p v belongs to `[lb, ub]`.
    bool contains(double v) const { return lb <= v && v <= ub; }
};

} // namespace baguette
