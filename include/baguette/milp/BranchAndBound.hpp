#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <limits>
#include <vector>

#include "baguette/lp/LPSolver.hpp"
#include "baguette/milp/CuttingPlanes.hpp"
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

/// User-supplied cut generator callback.
///
/// Called at each B&B node where the LP relaxation is Optimal, after the
/// built-in GMI generator (if enabled).  The callback receives the full LP
/// result and a read-only view of the current node model (with tightened
/// variable bounds applied).
///
/// Return zero or more cuts to add permanently to the model.  Each cut may
/// use any Sense (GreaterEq, LessEq, Equal).  Returning an empty vector is
/// valid and incurs no overhead beyond the call itself.
///
/// @note The model passed to the callback reflects current node bounds
///   (set via restoreBounds before the LP solve) but not necessarily the
///   root bounds — variable lb/ub have been tightened by branching.
/// @note Cuts are added to the global model copy shared across all nodes.
///   They are globally valid (must hold for every feasible integer point)
///   and remain in the model for all subsequent nodes.
/// @note The callback is not called when the LP is Infeasible, Unbounded,
///   or has hit a limit — only on Optimal results.
using CutGenerator =
    std::function<std::vector<Cut>(const LPDetailedResult&, const Model&)>;

/// Options for solveMILP().
struct BBOptions {
    /// Variable selection strategy for branching. Default: MostFractional.
    BranchStrategy branchStrat = BranchStrategy::MostFractional;

    /// Node selection strategy. Default: HybridPlunge.
    NodeSelection nodeSelect = NodeSelection::HybridPlunge;

    /// Maximum nodes explored in the DFS plunge phase of HybridPlunge before
    /// aborting the plunge and switching to BestBound. 0 = no limit.
    ///
    /// Without a cap, HybridPlunge may explore thousands of nodes in a DFS
    /// sub-tree where no integer solution exists (e.g., PrimalSimplex on
    /// degenerate LPs with Dantzig pivot), because the plunge-to-BestBound
    /// switch only fires on the first incumbent. A cap of ~200 aborts the
    /// unproductive plunge early and lets BestBound find an incumbent via a
    /// different exploration order. Has no effect unless nodeSelect is
    /// HybridPlunge.
    uint32_t maxPlungeNodes = 200;

    /// Maximum number of B&B nodes to explore. 0 = no limit.
    uint32_t maxNodes = 1'000'000;

    /// Wall-clock time limit in seconds (shared with LP solves via startTime).
    double timeLimitS = 3600.0;

    /// Absolute tolerance for declaring a variable value integer-feasible.
    /// A variable x_i is considered integer if |x_i − round(x_i)| ≤ intFeasTol.
    double intFeasTol = 1e-6;

    /// Absolute MIP gap tolerance. A node is pruned when its LP bound cannot
    /// improve the incumbent by more than max(mipGapAbs, mipGapRel*|incumbent|).
    ///   Minimize: prune if lpBound ≥ incumbent − gap
    ///   Maximize: prune if lpBound ≤ incumbent + gap
    /// Default 1e-6. For an exact proof-logged solver, keep at 1e-6 and set
    /// mipGapRel = 0.0 (default).
    double mipGapAbs = 1e-6;

    /// Relative MIP gap tolerance. The effective gap used for pruning is
    ///   gap = max(mipGapAbs, mipGapRel × |incumbent|)
    /// Default 0.0 — exact solver: only mipGapAbs applies. Set to e.g. 1e-4
    /// (0.01 %) for large-objective problems where a small absolute error is
    /// acceptable. Only applied when a finite incumbent exists.
    double mipGapRel = 0.0;

    /// If true, generate Gomory Mixed-Integer (GMI) cuts at each node where
    /// the LP relaxation is fractional and the dual simplex solved optimally.
    /// Cuts are globally valid and added permanently to the model copy, so
    /// all subsequent nodes benefit. Default: false.
    bool enableCuts = false;

    /// If true, generate Mixed-Integer Rounding (MIR) cuts from LessEq
    /// model constraints, and CMIR cuts from GreaterEq constraints via
    /// complementation.  Cuts are added after GMI (if enabled) and before
    /// user CutGenerators.  Share the maxCutsPerNode / maxTotalCuts budget
    /// with GMI.  Default: false.
    bool enableMIR = false;

    /// Maximum number of GMI cuts generated per node. 0 = unlimited.
    /// Has no effect when enableCuts is false.
    uint32_t maxCutsPerNode = 10;

    /// Maximum total number of GMI cuts added across all nodes.
    /// When the budget is exhausted, cut generation stops for all subsequent nodes.
    /// Prevents unbounded LP growth: with maxCutsPerNode=10 and a 10 000-node tree,
    /// the model can otherwise accumulate up to 100 000 extra rows.
    /// Default 500. Set to 0 for unlimited (not recommended on large trees).
    uint32_t maxTotalCuts = 500;

    /// Maximum GMI cuts generated at the root node specifically.
    /// 0 (default) = use maxCutsPerNode (same as all other nodes).
    /// Set to a larger value (e.g. maxTotalCuts / 10) to allow more aggressive
    /// cut generation at the root: each root cut is globally valid and tightens
    /// the bound for the entire B&B tree.
    /// Has no effect when enableCuts is false.
    uint32_t maxRootCuts = 0;

    /// User-supplied cut generators, called at each node after GMI (if enabled).
    /// Each generator receives the current LP result and model; it returns a
    /// (possibly empty) list of globally-valid cuts to add permanently.
    ///
    /// Cuts from all generators are collected, capped against maxTotalCuts
    /// (shared with GMI), added to the model, and trigger one cold LP re-solve
    /// (same as GMI).  Generators are called in order; all generators that fit
    /// within the remaining budget are invoked before the re-solve.
    ///
    /// The LP is solved with computeCutData = true whenever at least one
    /// generator is registered, so generators can inspect fractionalRows.
    ///
    /// @note Cuts must be globally valid (hold for every feasible integer point),
    ///   not just for the current node.  Adding locally-valid cuts corrupts the
    ///   search and produces incorrect results.
    std::vector<CutGenerator> cutGenerators;

    /// LP method used at the root node.
    /// Auto (default): falls back to lpOpts.method.
    /// Use a stronger method here (e.g. MehrotraIPM) when a tighter root bound
    /// matters more than solve speed.
    LPMethod rootMethod = LPMethod::Auto;

    /// LP method used at all B&B nodes other than the root.
    /// Auto (default): uses rootMethod (which itself falls back to lpOpts.method).
    /// Set to DualSimplexBV for fast warm-started node solves after an IPM root.
    LPMethod nodeMethod = LPMethod::Auto;

    /// LP solver options forwarded to every node's LP solve.
    ///
    /// Configure @p maxIter (pivot limit) and other per-solve settings here.
    /// @p method is overridden per-node by rootMethod / nodeMethod above.
    /// The following fields are managed internally by solveMILP() and any
    /// user-supplied value is silently overridden:
    ///   - @p method      ← from rootMethod / nodeMethod
    ///   - @p timeLimitS  ← from BBOptions::timeLimitS
    ///   - @p startTime   ← shared clock set at the B&B root
    ///   - @p warmBasis   ← from the parent node's BasisRecord
    ///   - @p computeCutData ← from BBOptions::enableCuts
    LPOptions lpOpts;

    /// If true, populate MILPResult::stats with granular diagnostics.
    /// When false (default), no counters are maintained — zero overhead on
    /// production runs.  Enable for cut-effectiveness and warm-start diagnosis.
    bool collectStats = false;

    /// MILP presolve level applied once before the B&B root node.
    /// Operates on the working model copy; the original model is unchanged.
    /// Populates MILPResult::presolveStat.
    ///
    /// | Level | Technique                                                    |
    /// |-------|--------------------------------------------------------------|
    /// |   0   | None — skip presolve entirely                                |
    /// |   1   | LP bound-tightening + integrality rounding + PR1 (default)  |
    /// |   2   | + CP fixpoint propagation at root before the B&B tree        |
    /// |   3   | + Weak probing (binary fix + propagation + bounds intersect)  |
    /// |   4   | + Root LP solve (LP infeasibility detection)                  |
    /// |   5   | + Binary implication rows injected from probing               |
    /// |   6   | + Strong probing (LP solve per binary fix)                    |
    ///
    /// Levels are cumulative: level N implies all levels 1 … N-1.
    uint32_t presolveLevel = 1;

    /// Maximum outer MILP presolve cycles (LP-fixpoint + integrality round).
    /// Independent of LPOptions::presolveMaxPasses (LP pass count per node).
    /// 0 = run to fixpoint (default).
    uint32_t milpPresolveMaxCycles = 0;

    /// If true, apply elimination presolve (presolveElim) after bound tightening.
    /// Removes fixed variables (lb == ub) and always-satisfied constraints before
    /// the B&B loop, reducing the size of every LP solve in the tree.
    /// postsolveElim() restores the full solution after B&B terminates.
    /// Default true.  Skipped when presolveLevel == 0.
    bool enableElimination = true;

    /// Maximum binary variables probed per probing pass (levels 3 and 6).
    /// 0 = probe all binary variables in the model.
    uint32_t probingMaxVars = 50;

    /// Maximum binary implication rows injected into the model (level 5).
    uint32_t maxImpliedRows = 100;

    /// LP method used for each LP solve in strong probing (level 6).
    /// DualSimplexBV (default) handles a single bound change cheaply.
    LPMethod probingLPMethod = LPMethod::DualSimplexBV;

    /// If true, iterate CP propagation at each B&B node until no further bounds
    /// change (fixpoint). If false (default), a single propagation pass is done.
    ///
    /// Fixpoint iteration cascades bound changes across constraints: tightening one
    /// variable re-triggers all constraints sharing that variable, potentially fixing
    /// more cells without branching. This is critical for pure-CP models (e.g.,
    /// Sudoku with AllDiff only) where a single pass propagates too little.
    ///
    /// Cost: O(C × K) per fixpoint iteration, where C = number of CP constraints
    /// and K = constraint size. The loop terminates in at most O(V) iterations
    /// (each iteration must fix at least one variable bound to continue).
    bool cpPropagateToFixpoint = true;
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
///   permanently to the working model copy via Model::addLPConstraint().  After
///   each cut addition the local LP is re-solved from a cold start (the
///   pre-cut BasisRecord is structurally incompatible with the enriched model).
///   Sibling nodes that were queued before the cut was generated also carry
///   stale BasisRecords; solveDualDetailed() detects the sfCache dimension
///   mismatch and falls back to a cold primal solve for those nodes too.
///   This is correct but means that cut generation sacrifices warm-start reuse
///   for all queued siblings — a known trade-off of the global-cut strategy.
///
/// @note CP integration: if the model has CP constraints (added via
///   Model::addCPConstraint()), propagateCP() is called after restoreBounds()
///   and before the first LP solve at each node.  Bounds tightened by CP
///   propagation are tracked in dirtyVars and reset automatically by the next
///   restoreBounds() call — no extra bookkeeping required.  If CP reports
///   Infeasible the node is pruned without an LP solve.
MILPResult solveMILP(const Model&            model,
                     const BBOptions&        opts      = {},
                     SolverClock::time_point startTime = SolverClock::now());

} // namespace baguette