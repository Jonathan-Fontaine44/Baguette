#include "baguette/cp/CPConstraints.hpp"

#include <algorithm>
#include <limits>

#include "baguette/cp/CPConstraint.hpp"
#include "baguette/model/Model.hpp"

namespace baguette {

PropagationResult propagateCP(const CPConstraints& cp, Model& model) {
    PropagationResult result;

    for (const BuiltinConstraint& con : cp.builtins()) {
        PropagationResult r = std::visit(
            [&](const auto& c) { return propagate(c, model); }, con);

        result.changedVarIds.insert(result.changedVarIds.end(),
                                    r.changedVarIds.begin(), r.changedVarIds.end());

        if (r.status == CPStatus::Infeasible) {
            result.status = CPStatus::Infeasible;
            return result;
        }
    }

    for (const std::shared_ptr<const CPConstraint>& con : cp.customs()) {
        PropagationResult r = con->propagate(model);

        result.changedVarIds.insert(result.changedVarIds.end(),
                                    r.changedVarIds.begin(), r.changedVarIds.end());

        if (r.status == CPStatus::Infeasible) {
            result.status = CPStatus::Infeasible;
            return result;
        }
    }

    std::sort(result.changedVarIds.begin(), result.changedVarIds.end());
    result.changedVarIds.erase(
        std::unique(result.changedVarIds.begin(), result.changedVarIds.end()),
        result.changedVarIds.end());

    return result;
}

bool cpFeasible(const CPConstraints& cp, const std::vector<double>& sol, double tol) {
    for (const BuiltinConstraint& con : cp.builtins())
        if (!std::visit([&](const auto& c) { return cpFeasible(c, sol, tol); }, con))
            return false;
    for (const std::shared_ptr<const CPConstraint>& con : cp.customs())
        if (!con->cpFeasible(sol, tol))
            return false;
    return true;
}

uint32_t cpViolatedVar(const CPConstraints& cp, const std::vector<double>& sol, double tol) {
    constexpr uint32_t kNone = std::numeric_limits<uint32_t>::max();
    for (const BuiltinConstraint& con : cp.builtins()) {
        uint32_t id = std::visit([&](const auto& c) { return cpViolatedVar(c, sol, tol); }, con);
        if (id != kNone) return id;
    }
    for (const std::shared_ptr<const CPConstraint>& con : cp.customs()) {
        uint32_t id = con->cpViolatedVar(sol, tol);
        if (id != kNone) return id;
    }
    return kNone;
}

} // namespace baguette
