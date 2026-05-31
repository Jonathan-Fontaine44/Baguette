#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "baguette/core/LinearExpr.hpp"
#include "baguette/core/Sense.hpp"
#include "baguette/model/ModelEnums.hpp"

namespace baguette {

/// Hot data - accessed at every simplex iteration.
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
    uint32_t termIdx; ///< Position within LPConstraint::lhs.varIds/coeffs.
};

/// Cold data - accessed only during model construction and output.
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

/// Opaque handle identifying a constraint added via Model::addLPConstraint().
using ConstraintId = uint32_t;

/// A linear constraint in user-facing form: `lhs sense rhs + rhsConst`.
///
/// Two representations share this type:
///
/// - **Original** (possibly two-sided): returned by Model::getLPConstraint(id).
///   `rhs` may contain variable terms (e.g. from `expr1 <= expr2`).
///   Preserves the form the user built for debugging and inspection.
///
/// - **Normalized** (solver-ready): returned by Model::getLPConstraints().
///   `rhs` is always empty - all variable terms are on `lhs`.
///   Use `isNormalized()` to distinguish the two forms.
///
/// The full constraint semantics: `lhs  sense  (rhs + rhsConst)`.
struct LPConstraint {
    LinearExpr lhs;
    Sense      sense;
    LinearExpr rhs;              ///< Variable terms on the RHS; empty when normalized.
    double     rhsConst = 0.0;  ///< Scalar part of the RHS.

    /// @return true when rhs has no variable terms (solver-ready form).
    bool isNormalized() const { return rhs.empty(); }

    /// Return the normalized form: all variable terms moved to lhs, scalar on rhsConst.
    /// Any constant stored in lhs or rhs is absorbed into rhsConst.
    /// The returned constraint always satisfies isNormalized() and lhs.constant == 0.
    ///
    /// @note Complexity: O(n+m) where n = lhs.size(), m = rhs.size().
    LPConstraint normalize() const {
        LPConstraint n;
        n.lhs   = lhs;
        n.lhs  -= rhs;                            // move rhs variable terms to lhs
        n.sense = sense;
        // n.rhs stays default-constructed (empty)
        n.rhsConst = rhsConst - n.lhs.constant;  // absorb lhs.constant into scalar
        n.lhs.constant = 0.0;
        return n;
    }
};

// ── Constraint-building operators ─────────────────────────────────────────────
// Return LPConstraint by value.  Cold-loop construction - no hot-path penalty.
// operator== intentionally returns LPConstraint, not bool: models are built, not compared.

/// Build `lhs <= rhs`.
inline LPConstraint operator<=(LinearExpr lhs, double rhs) {
    return {std::move(lhs), Sense::LessEq, {}, rhs};
}
/// Build `lhs >= rhs`.
inline LPConstraint operator>=(LinearExpr lhs, double rhs) {
    return {std::move(lhs), Sense::GreaterEq, {}, rhs};
}
/// Build `lhs == rhs`.
inline LPConstraint operator==(LinearExpr lhs, double rhs) {
    return {std::move(lhs), Sense::Equal, {}, rhs};
}

/// Build `lhs <= rhs` (two-sided; rhs.constant absorbed into rhsConst).
inline LPConstraint operator<=(LinearExpr lhs, LinearExpr rhs) {
    double c = rhs.constant; rhs.constant = 0.0;
    return {std::move(lhs), Sense::LessEq, std::move(rhs), c};
}
/// Build `lhs >= rhs` (two-sided; rhs.constant absorbed into rhsConst).
inline LPConstraint operator>=(LinearExpr lhs, LinearExpr rhs) {
    double c = rhs.constant; rhs.constant = 0.0;
    return {std::move(lhs), Sense::GreaterEq, std::move(rhs), c};
}
/// Build `lhs == rhs` (two-sided; rhs.constant absorbed into rhsConst).
inline LPConstraint operator==(LinearExpr lhs, LinearExpr rhs) {
    double c = rhs.constant; rhs.constant = 0.0;
    return {std::move(lhs), Sense::Equal, std::move(rhs), c};
}

/// Build `var <= rhs`.
inline LPConstraint operator<=(Variable lhs, double rhs) { return 1.0 * lhs <= rhs; }
/// Build `var >= rhs`.
inline LPConstraint operator>=(Variable lhs, double rhs) { return 1.0 * lhs >= rhs; }
/// Build `var == rhs`.
inline LPConstraint operator==(Variable lhs, double rhs) { return 1.0 * lhs == rhs; }

} // namespace baguette
