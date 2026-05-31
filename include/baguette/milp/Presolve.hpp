#pragma once

#include <chrono>
#include <limits>

#include "baguette/lp/LPSolver.hpp"
#include "baguette/milp/MILPResult.hpp"
#include "baguette/model/Model.hpp"
#include "baguette/model/Presolve.hpp"

namespace baguette {

/// Reconstruct the full-model MILP solution from a reduced-model result.
///
/// Adjusts @p r in place:
///   - objectiveValue  : += rec.objAdjustment  (only when primalValues non-empty)
///   - primalValues    : expanded to origVarCount; fixed vars inserted at
///                       their fixed values.
void postsolveElim(MILPResult& r, const EliminationRecord& rec);

// ── Multi-level MILP presolve ──────────────────────────────────────────────────

/// Options for presolveMILPInPlace().
///
/// Levels are cumulative: level N implies all levels 1 … N-1 also run.
///
/// | Level | Added technique                                          | Cost      |
/// |-------|----------------------------------------------------------|-----------|
/// |   0   | None - skip presolve entirely                            | -         |
/// |   1   | LP bound-tightening + integrality rounding + PR1         | O(P·C·N)  |
/// |   2   | + CP fixpoint propagation at root (before B&B tree)      | O(I·C·K)  |
/// |   3   | + Weak probing: fix each binary var, propagate, intersect | O(k·L2)  |
/// |   4   | + Root LP relaxation solve (LP infeasibility detection)   | O(1 LP)   |
/// |   5   | + Inject binary implication rows discovered by probing    | O(k·L2)   |
/// |   6   | + Strong probing: LP solve per fix (LP infeasibility)     | O(k·LP)   |
struct MILPPresolveOpts {
    /// Active presolve level (0–6, default 1).
    uint32_t level = 1;

    /// Maximum outer LP+round cycles for level 1 (0 = run to fixpoint).
    uint32_t maxCycles = 0;

    /// Integrality tolerance: lb ← ⌈lb − tol⌉, ub ← ⌊ub + tol⌋.
    double intFeasTol = 1e-6;

    /// Wall-clock time budget shared across all sub-operations.
    double timeLimitS = std::numeric_limits<double>::infinity();

    /// Maximum binary variables probed per probing pass (levels 3 and 6).
    /// 0 = probe all binary variables.
    uint32_t probingMaxVars = 50;

    /// Maximum binary implication rows injected into the model (level 5).
    uint32_t maxImpliedRows = 100;

    /// LP solver options forwarded to the root LP solve (level 4) and to
    /// each LP solve in strong probing (level 6).  The @p method field is
    /// overridden for strong probing by @p probingLPMethod.
    LPOptions lpOpts = {};

    /// LP method used for each fix/solve in strong probing (level 6).
    /// DualSimplexBV (default) warm-starts cheaply from a single bound change.
    LPMethod probingLPMethod = LPMethod::DualSimplexBV;
};

/// Apply multi-level MILP presolve to @p model in place.
///
/// **Level 1** (default): Interleaves LP bound-tightening with integrality
/// rounding.  Before the outer loop, RHS values of all-integer constraints are
/// rounded (PR1).  The LP+round cycle repeats until fixpoint or @p opts.maxCycles.
///
/// **Level 2**: After level 1, runs CP fixpoint propagation over the model's
/// CP constraints (propagateCP to fixpoint), then re-runs the LP+round loop if
/// any bounds changed.  No-op when the model has no CP constraints.
///
/// **Level 3**: Weak probing - for each binary variable x_i (up to
/// @p opts.probingMaxVars): fix x_i = lb, run level-2 presolve on a copy;
/// fix x_i = ub, run level-2 presolve on another copy; intersect the resulting
/// bounds (new_lb[j] = max(cur_lb[j], min(lb0[j], lb1[j]))).  If one fixation
/// is infeasible, x_i is forced to the other value.  If both are infeasible,
/// the model is infeasible.
///
/// **Level 4**: Root LP solve - solves the LP relaxation of the presolved model
/// once.  Detects LP infeasibility (which implies MILP infeasibility) before
/// the B&B tree starts.
///
/// **Level 5**: Implication rows - during the level-3 weak probing, binary-to-
/// binary implications (x_i = v → x_j = w) are recorded and injected as LP
/// constraints (e.g. x_i + x_j ≥ 1).  Capped at @p opts.maxImpliedRows.
/// Rows are added before the level-4 root LP solve, so the LP benefits from
/// the tighter model.
///
/// **Level 6**: Strong probing - for each binary variable x_i (up to
/// @p opts.probingMaxVars): solve the full LP relaxation with x_i fixed to lb,
/// then with x_i fixed to ub.  LP infeasibility for one value fixes x_i to the
/// other.  Detects infeasibilities that constraint propagation (level 3) cannot.
/// Runs after levels 1–5; any variables fixed here trigger a final LP+round pass.
///
/// @note Complexity O(level) × O(P·C·N) for propagation,
///   O(k · LP) for levels 3 and 6 where k = binary vars probed.
///
/// @note Complexity
///   Level 1: O(P × C × N). Level 2: + O(I × C_CP × K). Level 3: O(k × L2).
///   Level 4: O(1 LP solve). Level 5: O(k × L2). Level 6: O(k × LP).
MILPPresolveResult presolveMILPInPlace(
    Model&                  model,
    const MILPPresolveOpts& opts      = {},
    std::chrono::steady_clock::time_point startTime =
        std::chrono::steady_clock::now());

} // namespace baguette
