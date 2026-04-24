#pragma once

#include <limits>
#include <vector>

namespace baguette {

/// Status returned by the MILP Branch & Bound solver.
enum class MILPStatus {
    Optimal,    ///< Optimal integer solution found and proven (queue exhausted).
    Infeasible, ///< No integer-feasible solution exists.
    Unbounded,  ///< LP relaxation is unbounded; the MILP is also unbounded.
    TimeLimit,  ///< Time limit reached; best solution found so far (may be empty).
    MaxNodes    ///< Node limit reached; best solution found so far (may be empty).
};

/// Result returned by solveMILP().
///
/// When status is Optimal, objectiveValue and primalValues hold the proven
/// optimal integer solution.  When status is TimeLimit or MaxNodes, they hold
/// the best integer-feasible solution found (may be empty if none was found).
/// When status is Infeasible or Unbounded, primalValues is empty and
/// objectiveValue is +∞ (Minimize) or −∞ (Maximize).
struct MILPResult {
    MILPStatus status = MILPStatus::Infeasible;

    /// Objective value of the best integer-feasible solution found.
    double objectiveValue = 0.0;

    /// Primal solution indexed by Variable::id (size == Model::numVars()).
    /// Empty when no integer-feasible solution was found.
    std::vector<double> primalValues;

    /// Total number of B&B nodes explored (including the root).
    int nodesExplored = 0;
};

} // namespace baguette
