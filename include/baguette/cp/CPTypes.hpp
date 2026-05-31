#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace baguette {

/// Status returned by a CP propagation call.
enum class CPStatus {
    Feasible,   ///< Constraint(s) consistent; bounds may have been tightened.
    Infeasible, ///< Domain wipe-out detected; current B&B node is provably infeasible.
};

/// Machine-readable certificate for a CP infeasibility, written into the proof log.
///
/// Generic across all constraint types: each failing constraint writes its name
/// (with relevant parameters) and the minimal set of variables that witness the
/// infeasibility together with their domain bounds at the time of failure.
///
/// A verifier re-runs the constraint check on the reported (varId, lb, ub) triples
/// to confirm infeasibility without re-solving the full LP.
///
/// @par Populated by
///   propagate(AllDiffConstraint, ...) - one or more conflict variables.
///   propagate(CumulativeConstraint, ...) - the overloaded task variable.
///   propagateCP() - attaches constraintIdx after the per-constraint call.
///   User-defined CPConstraint::propagate() may populate witness directly.
struct CPFailureWitness {
    /// Human-readable constraint identifier: "AllDiff", "Cumulative(cap=N)", "Custom(idx=N)".
    std::string constraintDesc;

    /// Zero-based index of the failing constraint in CPConstraints::builtins or customs.
    uint32_t constraintIdx = 0;

    /// IDs of the variables that together witness the infeasibility.
    std::vector<uint32_t> varIds;

    /// Lower bounds of the conflict variables at the moment of failure (parallel to varIds).
    std::vector<double> varLb;

    /// Upper bounds of the conflict variables at the moment of failure (parallel to varIds).
    std::vector<double> varUb;
};

/// Result of one CP propagation call.
struct PropagationResult {
    CPStatus              status       = CPStatus::Feasible;
    std::vector<uint32_t> changedVarIds; ///< Sorted IDs of variables whose bounds were tightened.

    /// Failure certificate; set only when status == Infeasible.
    std::optional<CPFailureWitness> witness;
};

} // namespace baguette
