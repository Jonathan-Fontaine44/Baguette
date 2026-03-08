#pragma once

namespace baguette {

/// Type of a decision variable.
enum class VarType {
    Continuous, ///< x ∈ [lb, ub] ⊆ ℝ
    Integer,    ///< x ∈ [lb, ub] ∩ ℤ
    Binary      ///< x ∈ {0, 1}
};

/// Optimization direction.
enum class ObjSense {
    Minimize, ///< Minimize the objective.
    Maximize  ///< Maximize the objective.
};

} // namespace baguette
