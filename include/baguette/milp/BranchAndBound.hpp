#pragma once

#include <chrono>
#include <cstdint>
#include <limits>

#include "baguette/milp/MILPResult.hpp"
#include "baguette/model/Model.hpp"

namespace baguette {

/// Variable selection strategy used when branching on a fractional variable.
enum class BranchStrategy {
    /// Branch on the first fractional integer variable (by Variable::id).
    /// Cheapest selection — O(n) scan, stops at first fractional.
    FirstFractional,

    /// Branch on the most fractional integer variable (fractional part closest to 0.5).
    /// Maximises the minimum of (frac, 1−frac) over all fractional integer variables.
    MostFractional,

    /// Branch on the variable with the best pseudo-cost product score.
    /// Falls back to MostFractional for variables with no branching history.
    PseudoCost,
};

/// Node selection strategy for the B&B queue.
enum class NodeSelection {
    /// Explore the node with the best LP bound first.
    /// Minimises the number of nodes needed to prove optimality.
    BestBound,

    /// Explore the deepest node first (stack discipline).
    /// Finds a feasible integer solution faster; smaller basis changes → better warm-start.
    DepthFirst,

    /// Plunge depth-first until the first integer-feasible incumbent is found,
    /// then switch automatically to BestBound to prove optimality.
    /// Combines fast incumbent finding (DFS) with tight bound convergence (BestBound).
    HybridPlunge,
};

/// Options for solveMILP().
struct BBOptions {
    /// Variable selection strategy for branching. Default: MostFractional.
    BranchStrategy branchStrat = BranchStrategy::MostFractional;

    /// Node selection strategy. Default: BestBound.
    NodeSelection nodeSelect = NodeSelection::BestBound;

    /// Maximum number of B&B nodes to explore. 0 = no limit.
    uint32_t maxNodes = 1'000'000;

    /// Wall-clock time limit in seconds (shared with LP solves via startTime).
    double timeLimitS = 3600.0;

    /// Maximum simplex pivots per LP node solve. 0 = unlimited.
    uint32_t maxIterLP = 0;

    /// Absolute tolerance for declaring a variable value integer-feasible.
    /// A variable x_i is considered integer if |x_i − round(x_i)| ≤ intFeasTol.
    double intFeasTol = 1e-6;

    /// Absolute MIP gap tolerance. A node is pruned when its LP bound cannot
    /// improve the incumbent by more than mipGapAbs:
    ///   Minimize: prune if lpBound ≥ incumbent − mipGapAbs
    ///   Maximize: prune if lpBound ≤ incumbent + mipGapAbs
    /// Propagated to every LP bound comparison against the incumbent so that
    /// near-optimal nodes are not explored unnecessarily.
    /// The returned solution may be suboptimal by at most mipGapAbs.
    double mipGapAbs = 1e-6;

    /// If true, generate Gomory Mixed-Integer (GMI) cuts at each node where
    /// the LP relaxation is fractional and the dual simplex solved optimally.
    /// Cuts are globally valid and added permanently to the model copy, so
    /// all subsequent nodes benefit. Default: false.
    bool enableCuts = false;

    /// Maximum number of GMI cuts generated per node. 0 = unlimited.
    /// Has no effect when enableCuts is false.
    uint32_t maxCutsPerNode = 10;
};

/// Shared clock type (same as LPSolver).
using SolverClock = std::chrono::steady_clock;

/// Solve a MILP using Branch & Bound with LP relaxation at each node.
///
/// Integer and Binary variables are branched on; Continuous variables are
/// relaxed.  If no integer/binary variables are present, the LP relaxation
/// is solved directly and returned as a single-node result.
///
/// Each child node is warm-started from its parent's BasisRecord via the
/// dual simplex, which avoids Phase I and reuses the cached constraint
/// matrix A (O(1) shared_ptr copy).  The LP solver falls back to a cold
/// primal solve automatically when the warm basis is incompatible.
///
/// The @p startTime parameter is shared with every LP node solve so that
/// the time budget is consumed globally across the entire B&B tree.
///
/// @param model     The model to solve. Integer/Binary vars are branched on.
/// @param opts      Solver options (branching strategy, node selection, limits).
/// @param startTime Reference point for the time limit. Defaults to now().
///                  Pass an external startTime to share a budget with outer code.
/// @note Complexity: O(N × K × m × n) where N = nodes explored, K = average
///   simplex pivots per node, m = SF rows, n = SF columns.  Warm-start
///   reinversion is O(m²·n) per node; standard-form matrix reuse via
///   sfCache reduces standard-form setup from O(m·n) to O(1) per node.
///
/// @note Cut addition and warm-start: when GMI cuts are enabled, cuts are added
///   permanently to the working model copy via Model::addConstraint().  After
///   each cut addition the local LP is re-solved from a cold start (the
///   pre-cut BasisRecord is structurally incompatible with the enriched model).
///   Sibling nodes that were queued before the cut was generated also carry
///   stale BasisRecords; solveDualDetailed() detects the sfCache dimension
///   mismatch and falls back to a cold primal solve for those nodes too.
///   This is correct but means that cut generation sacrifices warm-start reuse
///   for all queued siblings — a known trade-off of the global-cut strategy.
MILPResult solveMILP(const Model&            model,
                     const BBOptions&        opts      = {},
                     SolverClock::time_point startTime = SolverClock::now());

} // namespace baguette