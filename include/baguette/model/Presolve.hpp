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

    /// Ghost (CP-only) variables occupy indices [lpVarCount, total).
    uint32_t lpVarCount = 0;

    /// varMap[orig_id] = reduced_id (LP var or ghost — never UINT32_MAX).
    std::vector<uint32_t> varMap;
    /// reducedToOrig[reduced_LP_id] = orig_id  (LP vars only, size == lpVarCount).
    std::vector<uint32_t> reducedToOrig;
    /// Fixed variables: {orig_id, fixed_value} — kept for diagnostics / postsolve.
    std::vector<std::pair<uint32_t, double>> fixedVars;

    /// conMap[orig_idx] = reduced_idx, or UINT32_MAX if the row was eliminated.
    std::vector<uint32_t> conMap;
    /// reducedToOrigCon[reduced_idx] = orig_idx.
    std::vector<uint32_t> reducedToOrigCon;

    /// Objective constant: sum(c_j * v_j for fixed variables j).
    double objAdjustment = 0.0;

    uint32_t varsEliminated = 0;
    uint32_t rowsEliminated = 0;

    /// True if a constraint whose every variable was fixed is itself infeasible
    /// (e.g. all variables sum to a value that violates the constraint).
    /// Callers should short-circuit and return Infeasible without solving the LP.
    bool infeasible = false;
};

/// Build a reduced model by eliminating fixed variables and redundant rows.
///
/// LP constraints only — call presolveElimCP() afterwards to transfer CP
/// constraints with remapped variable IDs.
///
/// @note Complexity  O(V + C × N)
Model presolveElim(const Model& model, EliminationRecord& rec);

// ── CP reduction ──────────────────────────────────────────────────────────────

/// Reduce a single built-in CP constraint against @p rec.
///
/// All variable IDs are remapped via rec.varMap.  Fixed variables become ghost
/// variables in the reduced model (lb == ub == fixedValue); their presence in
/// the reduced constraint lets CP propagation enforce their values naturally.
/// Returns nullopt if the constraint becomes trivially satisfied (0 or 1 var).
///
/// @note Complexity  O(|vars in constraint|)
std::optional<BuiltinConstraint> reduce(const BuiltinConstraint& bc,
                                        const EliminationRecord& rec);

/// Transfer CP constraints from @p cpOrig to @p reduced, remapping variable
/// IDs via @p rec.  Call immediately after presolveElim().
///
/// Ghost variables (lb == ub) must already be present in @p reduced (added by
/// presolveElim).  Custom CPConstraint objects delegate to CPConstraint::reduce().
///
/// @note Complexity  O(Σ |constraint vars|) over all CP constraints.
void presolveElimCP(const CPConstraints& cpOrig,
                    const EliminationRecord& rec,
                    Model& reduced);

} // namespace baguette
