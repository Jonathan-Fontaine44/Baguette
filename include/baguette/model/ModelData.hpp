#pragma once

#include <string>
#include <vector>

#include "baguette/core/LinearExpr.hpp"
#include "baguette/core/Sense.hpp"
#include "baguette/model/ModelEnums.hpp"

namespace baguette {

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

} // namespace baguette
