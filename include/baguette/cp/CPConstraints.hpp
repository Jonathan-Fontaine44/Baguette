#pragma once

#include <cstdint>
#include <limits>
#include <variant>
#include <vector>

#include "baguette/cp/CPTypes.hpp"
#include "baguette/cp/constraints/AllDiff.hpp"
#include "baguette/cp/constraints/Cumulative.hpp"
#include "baguette/model/Model.hpp"

namespace baguette {

/// Discriminated union of all supported CP constraint types.
///
/// To add a new constraint type:
///   1. Define MyConstraint in a new header under cp/constraints/.
///   2. Implement  PropagationResult propagate(const MyConstraint&, Model&).
///   3. Add  MyConstraint  to this alias — propagateCP() needs no further change.
using AnyConstraint = std::variant<AllDiffConstraint, CumulativeConstraint>;

/// Set of CP constraints enforced at every B&B node inside solveMILP().
///
/// An empty CPConstraints (the default) adds zero overhead to the B&B loop.
struct CPConstraints {
    std::vector<AnyConstraint> constraints;

    bool empty() const noexcept { return constraints.empty(); }

    /// Add any CP constraint (AllDiff, Cumulative, …).
    void add(AnyConstraint c) { constraints.push_back(std::move(c)); }
};

/// Propagate all CP constraints against the current model bounds (Bounds Consistency).
///
/// Dispatches to the correct typed  propagate()  overload via  std::visit.
/// Returns CPStatus::Infeasible immediately on the first domain wipe-out.
/// On return, changedVarIds lists (sorted, deduplicated) every variable whose
/// bounds were tightened — the B&B caller must push these into dirtyVars so
/// that restoreBounds() cleans them up before the next node.
///
/// @note Complexity  O(Σ cost(propagate_i)) summed over all constraints.
///   Zero overhead when cp.empty() (caller short-circuits before calling).
///
/// @param cp    The set of CP constraints to enforce.
/// @param model The model whose bounds are read and potentially tightened.
/// @return      Feasible (with changedVarIds) or Infeasible.
PropagationResult propagateCP(const CPConstraints& cp, Model& model);

/// Check whether a given solution satisfies all CP constraints.
///
/// Used at integer-feasible B&B leaves to catch LP solutions that satisfy all LP
/// constraints and integrality but still violate a CP constraint (e.g., AllDiff
/// with x = y = 1 because both are at their lower bound in the LP optimal).
///
/// Dispatches to  cpFeasible(constraint, sol, tol)  for each constraint type.
bool cpFeasible(const CPConstraints& cp, const std::vector<double>& sol, double tol);

/// Return the ID of the first variable involved in a CP constraint violation in @p sol,
/// or std::numeric_limits<uint32_t>::max() if no violation exists.
///
/// Used to select a branch variable when an integer-feasible LP solution violates
/// a CP constraint.  The B&B then branches on this variable — left child forces
/// x ≤ val − 1, right child forces x ≥ val + 1 — to exclude the conflicting value.
uint32_t cpViolatedVar(const CPConstraints& cp, const std::vector<double>& sol, double tol);

} // namespace baguette
