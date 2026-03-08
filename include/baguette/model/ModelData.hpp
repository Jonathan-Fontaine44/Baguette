#pragma once

#include <string>
#include <vector>

#include "baguette/core/LinearExpr.hpp"
#include "baguette/core/Sense.hpp"

namespace baguette {

/// Type of a decision variable.
enum class VarType {
    Continuous, ///< x ∈ [lb, ub] ⊆ ℝ
    Integer,    ///< x ∈ [lb, ub] ∩ ℤ
    Binary      ///< x ∈ {0, 1}
};

/// Hot data — accessed at every simplex iteration.
/// Stored as Structure of Arrays (SoA) for SIMD-friendly access patterns.
/// The simplex reads lb/ub/obj densely; label and type are never needed there.
struct ModelHot {
    std::vector<double> lb;   ///< Lower bounds, indexed by VarID.
    std::vector<double> ub;   ///< Upper bounds, indexed by VarID.
    std::vector<double> obj;  ///< Objective coefficients, indexed by VarID.
};

/// Cold data — accessed only during model construction and output.
struct ModelCold {
    std::vector<std::string> labels; ///< Variable names, indexed by VarID.
    std::vector<VarType>     types;  ///< Variable types, indexed by VarID.
};

/// A linear constraint: `lhs sense rhs`.
struct Constraint {
    LinearExpr lhs;   ///< Left-hand side expression.
    Sense      sense; ///< Relation between lhs and rhs.
    double     rhs;   ///< Right-hand side scalar.
};

/// Optimization direction.
enum class ObjSense {
    Minimize, ///< Minimize the objective.
    Maximize  ///< Maximize the objective.
};

} // namespace baguette
