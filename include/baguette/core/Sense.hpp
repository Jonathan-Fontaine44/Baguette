#pragma once

namespace baguette {

/// Sense of a linear constraint: the relation between the left-hand side
/// and the right-hand side (e.g. `expr <= rhs`).
enum class Sense {
    LessEq,    ///< Less-than-or-equal: `lhs <= rhs`.
    Equal,     ///< Equality: `lhs = rhs`.
    GreaterEq  ///< Greater-than-or-equal: `lhs >= rhs`.
};

} // namespace baguette
