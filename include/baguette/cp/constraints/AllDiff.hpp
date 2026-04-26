#pragma once

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <vector>

#include "baguette/core/Variable.hpp"
#include "baguette/cp/CPTypes.hpp"

namespace baguette {

class Model; // forward declaration (propagate definition needs full type)

/// AllDifferent constraint: all variables must take distinct integer values.
///
/// Enforced via Bounds Consistency (BC): fixed-value elimination repeated to
/// fixpoint, then range feasibility check.  Tightens variable domains without
/// full value enumeration.
struct AllDiffConstraint {
    std::vector<Variable> vars; ///< Variables that must all differ.

    AllDiffConstraint() = default;
    AllDiffConstraint(std::initializer_list<Variable> vs) : vars(vs) {}
    explicit AllDiffConstraint(std::vector<Variable> vs) : vars(std::move(vs)) {}
};

/// Propagate one AllDiff constraint (Bounds Consistency).
///
/// Fixed-value elimination: if x_i is fixed to v, raise lb_j or lower ub_j to
/// exclude v for all j ≠ i.  Repeated until fixpoint.  Range check: if
/// max(ub) − min(lb) + 1 < k the constraint is infeasible.
///
/// @note Complexity  O(K² × I) where K = vars.size(), I = fixpoint iterations (≤ K).
PropagationResult propagate(const AllDiffConstraint& con, Model& model);

/// Check whether a given integer solution satisfies AllDiff (all values distinct).
/// Returns true iff all sol[vars[i].id] values differ by more than @p tol.
bool cpFeasible(const AllDiffConstraint& con, const std::vector<double>& sol, double tol);

/// Return the ID of the first variable involved in an AllDiff violation in @p sol,
/// or std::numeric_limits<uint32_t>::max() if no violation is found.
uint32_t cpViolatedVar(const AllDiffConstraint& con, const std::vector<double>& sol, double tol);

} // namespace baguette
