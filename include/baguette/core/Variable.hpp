    #pragma once

#include <cstdint>

namespace baguette {

/// Pure handle identifying a decision variable within a Model.
///
/// A Variable is just a typed integer ID — 4 bytes, trivially copyable,
/// with no pointer or reference to the Model. All variable data (bounds,
/// type, label) lives in the Model, indexed by `id`.
///
/// This design allows variables to be freely copied into sub-problems
/// during Branch & Bound without ownership or lifetime issues.
struct Variable {
    std::uint32_t id; ///< Unique index within the owning Model.

    /// Two variables are equal iff they share the same ID.
    bool operator==(const Variable&) const = default;

    /// Strict ordering by ID, used to maintain sorted invariants in LinearExpr.
    bool operator<(const Variable& other) const { return id < other.id; }
};

} // namespace baguette
