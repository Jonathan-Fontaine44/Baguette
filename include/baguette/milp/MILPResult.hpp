#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <ostream>
#include <string_view>
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

inline std::string_view to_string(MILPStatus s) {
    switch (s) {
        case MILPStatus::Optimal:     return "Optimal";
        case MILPStatus::Infeasible:  return "Infeasible";
        case MILPStatus::Unbounded:   return "Unbounded";
        case MILPStatus::TimeLimit:   return "TimeLimit";
        case MILPStatus::MaxNodes:    return "MaxNodes";
    }
    return "Unknown";
}

inline std::ostream& operator<<(std::ostream& os, MILPStatus s) {
    return os << to_string(s);
}

/// Granular diagnostics collected during solveMILP() when BBOptions::collectStats
/// is true.  All counters are zero-initialised.  Nodes that hit NumericalFailure
/// or MaxIter are not counted in any pruning bucket.
struct BBStats {
    /// Total number of B&B nodes explored (including the root).
    uint32_t nodesExplored = 0;

    /// Total number of GMI cuts added to the model across all nodes.
    uint32_t cutsAdded = 0;

    /// Nodes that generated at least one GMI cut.
    uint32_t nodesWithCuts = 0;

    /// Nodes pruned because their LP bound could not improve the incumbent.
    uint32_t nodesPrunedByBound = 0;

    /// Nodes pruned because the LP or CP reported infeasibility.
    uint32_t nodesPrunedByInfeasibility = 0;

    /// Nodes that had a warm-start basis from their parent but fell back to a
    /// cold primal solve (e.g., because a previously added GMI cut invalidated
    /// the cached standard-form dimension in BasisRecord::sfCache).
    uint32_t warmStartFallbacks = 0;

    /// Total LP solves across all nodes.  At most 2 per node (1 base + 1 after
    /// cut addition); always 1 per node when enableCuts is false.
    uint32_t lpSolvesTotal = 0;

    /// Histogram of GMI cuts generated per tree depth.
    /// cutsPerDepth[d] = total cuts generated at depth d.
    /// Empty when no cuts were generated (enableCuts false or no fractional rows).
    std::vector<uint32_t> cutsPerDepth;
};

/// Diagnostics produced by presolveMILPInPlace().
struct MILPPresolveResult {
    bool     infeasible       = false; ///< True if presolve proved infeasibility.
    bool     timeLimitReached = false; ///< True if time limit was hit before fixpoint.
    uint32_t boundsTightened  = 0;     ///< LP-based bound updates applied.
    uint32_t boundsRounded    = 0;     ///< Integer variables whose bounds were snapped (ceil/floor).
    uint32_t rhsRounded       = 0;     ///< Constraint RHS values rounded (floor for ≤, ceil for ≥).
    uint32_t fixedVars        = 0;     ///< Variables with lb == ub at termination.
    uint32_t passesRun        = 0;     ///< LP propagation passes executed.
    uint32_t varsProbed       = 0;     ///< Binary variables probed (levels 3 and 6).
    uint32_t varsProbedFixed  = 0;     ///< Variables fixed as a result of probing.
    uint32_t impliedRowsAdded = 0;     ///< Binary implication rows added to the model (level 5).
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

    /// Optional diagnostics; populated only when BBOptions::collectStats is true.
    std::optional<BBStats> stats;

    /// Presolve statistics. Populated only when BBOptions::enablePresolve is true.
    std::optional<MILPPresolveResult> presolveStat;
};

} // namespace baguette
