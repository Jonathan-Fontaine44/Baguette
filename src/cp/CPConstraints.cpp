#include "baguette/cp/CPConstraints.hpp"

#include <algorithm>
#include <limits>

namespace baguette {

PropagationResult propagateCP(const CPConstraints& cp, Model& model) {
    PropagationResult result;

    for (const AnyConstraint& con : cp.constraints) {
        PropagationResult r = std::visit(
            [&](const auto& c) { return propagate(c, model); }, con);

        result.changedVarIds.insert(result.changedVarIds.end(),
                                    r.changedVarIds.begin(), r.changedVarIds.end());

        if (r.status == CPStatus::Infeasible) {
            result.status = CPStatus::Infeasible;
            return result; // fail-fast: first infeasible constraint stops propagation
        }
    }

    // Merge multiple constraints may touch the same variable.
    std::sort(result.changedVarIds.begin(), result.changedVarIds.end());
    result.changedVarIds.erase(
        std::unique(result.changedVarIds.begin(), result.changedVarIds.end()),
        result.changedVarIds.end());

    return result;
}

bool cpFeasible(const CPConstraints& cp, const std::vector<double>& sol, double tol) {
    for (const AnyConstraint& con : cp.constraints)
        if (!std::visit([&](const auto& c) { return cpFeasible(c, sol, tol); }, con))
            return false;
    return true;
}

uint32_t cpViolatedVar(const CPConstraints& cp, const std::vector<double>& sol, double tol) {
    constexpr uint32_t kNone = std::numeric_limits<uint32_t>::max();
    for (const AnyConstraint& con : cp.constraints) {
        uint32_t id = std::visit([&](const auto& c) { return cpViolatedVar(c, sol, tol); }, con);
        if (id != kNone) return id;
    }
    return kNone;
}

} // namespace baguette
