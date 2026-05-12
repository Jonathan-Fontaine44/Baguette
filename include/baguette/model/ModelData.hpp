#pragma once

#include <cstdint>
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

/// One entry in the variable→LP-constraint reverse index.
struct VarLPEntry {
    uint32_t conIdx;  ///< Index into Model::constraints.
    uint32_t termIdx; ///< Position within Constraint::lhs.varIds/coeffs.
};

/// Cold data — accessed only during model construction and output.
struct ModelCold {
    std::vector<std::string> labels; ///< Variable names, indexed by VarID.
    std::vector<VarType>     types;  ///< Variable types, indexed by VarID.

    /// varToLP[j] = all LP constraints in which variable j appears,
    /// with the index of its term within the constraint's LinearExpr.
    /// Updated by Model::addLPConstraint(). O(1) coefficient lookup.
    std::vector<std::vector<VarLPEntry>> varToLP;

    /// varToCP[j] = indices of builtin CP constraints in which variable j appears.
    /// Updated by Model::addCPConstraint(BuiltinConstraint).
    /// Custom CPConstraint objects are not indexed (no varIds() virtual method).
    std::vector<std::vector<uint32_t>> varToCP;
};

/// A linear constraint: `lhs sense rhs`.
struct Constraint {
    LinearExpr lhs;   ///< Left-hand side expression.
    Sense      sense; ///< Relation between lhs and rhs.
    double     rhs;   ///< Right-hand side scalar.
};

} // namespace baguette
