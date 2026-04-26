#pragma once

#include <cstdint>
#include <vector>

#include "baguette/cp/CPTypes.hpp"

namespace baguette {

class Model;

/// Abstract base for user-defined CP constraints (open extension path).
///
/// Built-in constraints (AllDiff, Cumulative) use std::variant for zero-overhead
/// dispatch.  Subclass CPConstraint for user-defined or exotic constraints —
/// virtual dispatch occurs once per constraint per B&B node, negligible versus
/// the O(K²) propagation work inside.
///
/// Constraints must be immutable after construction.  Store via
/// shared_ptr<const CPConstraint> so Model remains copyable without clone().
class CPConstraint {
public:
    virtual ~CPConstraint() = default;

    /// @note Complexity  Defined by the subclass.
    virtual PropagationResult propagate(Model& model) const = 0;

    virtual bool     cpFeasible(const std::vector<double>& sol, double tol) const = 0;
    virtual uint32_t cpViolatedVar(const std::vector<double>& sol, double tol) const = 0;
};

} // namespace baguette
