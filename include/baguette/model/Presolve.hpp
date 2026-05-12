#pragma once

#include <chrono>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "baguette/cp/CPConstraints.hpp"
#include "baguette/model/Model.hpp"

namespace baguette {

// ── Bound-Tightening Presolve (TB) ────────────────────────────────────────────

/// Result of the bound-tightening presolve phase.
struct PresolveResult {
    bool     infeasible       = false; ///< True if presolve detected lb > ub.
    bool     timeLimitReached = false; ///< True if time limit was hit before fixpoint.
    uint32_t boundsTightened  = 0;     ///< Total bound updates applied.
    uint32_t fixedVars        = 0;     ///< Variables with lb == ub at end.
    uint32_t passesRun        = 0;     ///< Bound-propagation passes completed.
};

/// Apply bound-tightening presolve to @p model **in place**.
///
/// @note Complexity  O(P × C × N)
PresolveResult presolveTBInPlace(
    Model& model,
    uint32_t maxPasses  = 0,
    double   timeLimitS = std::numeric_limits<double>::infinity(),
    std::chrono::steady_clock::time_point startTime =
        std::chrono::steady_clock::now());

/// Apply bound-tightening presolve to a **copy** of @p model.
///
/// @note Complexity  O(m + P × C × N)
std::pair<Model, PresolveResult> presolveTB(
    const Model& model,
    uint32_t     maxPasses   = 0,
    double       timeLimitS  = std::numeric_limits<double>::infinity(),
    std::chrono::steady_clock::time_point startTime =
        std::chrono::steady_clock::now());

// ── Elimination Presolve ───────────────────────────────────────────────────────

/// Mapping data produced by presolveElim(), required to reconstruct the full
/// solution via postsolveElim() / postsolveElimCP().
struct EliminationRecord {
    uint32_t origVarCount        = 0;
    uint32_t origConstraintCount = 0;

    /// varMap[orig_id] = reduced_id, or UINT32_MAX if the variable was fixed.
    std::vector<uint32_t> varMap;
    /// reducedToOrig[reduced_id] = orig_id.
    std::vector<uint32_t> reducedToOrig;
    /// Fixed variables: {orig_id, fixed_value}.
    std::vector<std::pair<uint32_t, double>> fixedVars;

    /// conMap[orig_idx] = reduced_idx, or UINT32_MAX if the row was eliminated.
    std::vector<uint32_t> conMap;
    /// reducedToOrigCon[reduced_idx] = orig_idx.
    std::vector<uint32_t> reducedToOrigCon;

    /// Objective constant: sum(c_j * v_j for fixed variables j).
    double objAdjustment = 0.0;

    uint32_t varsEliminated = 0;
    uint32_t rowsEliminated = 0;
};

/// Build a reduced model by eliminating fixed variables and redundant rows.
///
/// LP constraints only — call presolveElimCP() afterwards to transfer CP
/// constraints with remapped variable IDs.
///
/// @note Complexity  O(V + C × N)
Model presolveElim(const Model& model, EliminationRecord& rec);

// ── CP reduction ──────────────────────────────────────────────────────────────

/// Result of reducing a single built-in CP constraint against an elimination
/// record.
struct ReduceResult {
    std::optional<BuiltinConstraint> constraint; ///< nullopt if trivially satisfied.
    bool infeasible = false; ///< Fixed values already violate the constraint.
};

/// Reduce a single built-in CP constraint against @p rec.
///
/// Fixed variables (rec.varMap[j] == UINT32_MAX) are removed; their values
/// are checked for constraint violations (e.g. two AllDiff vars fixed equal).
/// Do not act on ReduceResult::infeasible during LP relaxation — only in MILP.
///
/// @note Complexity  O(|vars in constraint|)
ReduceResult reduce(const BuiltinConstraint& bc, const EliminationRecord& rec);

/// Transfer CP constraints from @p cpOrig to @p reduced, remapping variable
/// IDs via @p rec.  Call immediately after presolveElim().
///
/// Returns true if a constraint detected infeasibility from fixed values.
/// Custom CPConstraint objects delegate to CPConstraint::reduce().
///
/// @note Complexity  O(Σ |constraint vars|) over all CP constraints.
bool presolveElimCP(const CPConstraints& cpOrig,
                    const EliminationRecord& rec,
                    Model& reduced);

} // namespace baguette
