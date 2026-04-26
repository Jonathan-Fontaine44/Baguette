#pragma once

#include <cstdint>
#include <limits>
#include <vector>

#include "baguette/core/Variable.hpp"
#include "baguette/cp/CPTypes.hpp"

namespace baguette {

class Model; // forward declaration

/// One task in a Cumulative constraint.
struct Task {
    Variable varStart;    ///< Integer start-time variable.
    int32_t  duration;    ///< Fixed duration (≥ 1 time unit).
    int32_t  consumption; ///< Resource consumption during execution (≥ 0).
};

/// Cumulative constraint: the summed resource consumption of active tasks must
/// not exceed @p capacity at any time point.
///
/// BC propagation tightens the earliest start of each task by checking whether
/// the compulsory resource load from other tasks would cause an overload during
/// any window in which task i could start.  Compulsory region of task j:
/// [ub(start_j), lb(start_j) + duration_j) — the interval that task j must
/// occupy regardless of its chosen start time.
struct CumulativeConstraint {
    std::vector<Task> tasks;
    int32_t           capacity; ///< Global resource capacity per time unit.
};

/// Propagate one Cumulative constraint (Bounds Consistency, earliest-start direction).
///
/// For each task i, the earliest start est_i is advanced past any time t where
/// the compulsory resource load from other tasks at some time point in
/// [t, t + duration_i) would exceed capacity − consumption_i.
///
/// @note Complexity  O(N² × D × I) where N = tasks.size(),
///   D = max range of any start variable, I = fixpoint iterations.
///   Tight B&B bounds keep D small in practice.
PropagationResult propagate(const CumulativeConstraint& con, Model& model);

/// Check whether a given integer solution satisfies the Cumulative constraint
/// (no time point is overloaded).  @p sol[task.varStart.id] is rounded to integer.
bool cpFeasible(const CumulativeConstraint& con, const std::vector<double>& sol, double tol);

/// Return the start-variable ID of the first task in a Cumulative overload, or
/// std::numeric_limits<uint32_t>::max() if no violation is found.
uint32_t cpViolatedVar(const CumulativeConstraint& con, const std::vector<double>& sol, double tol);

} // namespace baguette
