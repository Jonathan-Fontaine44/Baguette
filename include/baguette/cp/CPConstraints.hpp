#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <variant>
#include <vector>

#include "baguette/cp/CPConstraint.hpp"
#include "baguette/cp/CPTypes.hpp"
#include "baguette/cp/constraints/AllDiff.hpp"
#include "baguette/cp/constraints/Cumulative.hpp"

namespace baguette {

class Model;

/// Fast-path discriminated union for built-in CP constraint types.
///
/// Dispatch via std::visit — zero virtual calls.
/// To register a new built-in: add its type here and include its header above.
using BuiltinConstraint = std::variant<AllDiffConstraint, CumulativeConstraint>;

/// Container for all CP constraints attached to a model.
///
/// Two-tier storage:
///   builtins_: BuiltinConstraint (std::variant) — zero-overhead std::visit dispatch.
///   customs_:  shared_ptr<const CPConstraint>   — virtual dispatch for user-defined constraints.
///
/// @note shared_ptr<const> allows Model to remain copyable without clone().
struct CPConstraints {
    void add(BuiltinConstraint c)                   { builtins_.push_back(std::move(c)); }
    void add(std::shared_ptr<const CPConstraint> c) { customs_.push_back(std::move(c)); }

    bool empty() const noexcept { return builtins_.empty() && customs_.empty(); }

    void merge(const CPConstraints& other) {
        builtins_.insert(builtins_.end(), other.builtins_.begin(), other.builtins_.end());
        customs_.insert(customs_.end(),   other.customs_.begin(),  other.customs_.end());
    }

    const std::vector<BuiltinConstraint>&                   builtins() const { return builtins_; }
    const std::vector<std::shared_ptr<const CPConstraint>>& customs()  const { return customs_;  }

private:
    std::vector<BuiltinConstraint>                   builtins_;
    std::vector<std::shared_ptr<const CPConstraint>> customs_;
};

/// Propagate all CP constraints against the current model bounds.
///
/// Built-ins dispatch via std::visit, customs via virtual propagate().
/// Returns CPStatus::Infeasible on the first domain wipe-out (fail-fast).
/// changedVarIds lists (sorted, deduplicated) every variable whose bounds were
/// tightened — caller must push into dirtyVars for restoreBounds().
///
/// @note Complexity  O(Σ cost(propagate_i)) summed over all constraints.
PropagationResult propagateCP(const CPConstraints& cp, Model& model);

/// Check whether @p sol satisfies all CP constraints.
bool cpFeasible(const CPConstraints& cp, const std::vector<double>& sol, double tol);

/// Return the ID of the first variable involved in a CP violation in @p sol,
/// or std::numeric_limits<uint32_t>::max() if no violation exists.
uint32_t cpViolatedVar(const CPConstraints& cp, const std::vector<double>& sol, double tol);

} // namespace baguette
