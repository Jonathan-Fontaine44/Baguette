#include "baguette/cp/constraints/AllDiff.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "baguette/model/Model.hpp"

namespace baguette {

namespace {

// Convert floating-point model bounds to effective integer bounds.
// Uses 1e-9 tolerance to handle near-integer doubles after LP pivots.
inline int64_t iLb(double lb) { return static_cast<int64_t>(std::ceil(lb  - 1e-9)); }
inline int64_t iUb(double ub) { return static_cast<int64_t>(std::floor(ub + 1e-9)); }

enum class StepResult { NoChange, Changed, Infeasible };

// One pass of fixed-value elimination.
// For each fixed x_i (lb_i == ub_i), exclude its value from every other x_j:
//   lb_j == v → raise lb_j to v+1
//   ub_j == v → lower ub_j to v-1
// Stops on the first empty domain (lb_j > ub_j).
StepResult fixedValueElimination(const std::vector<Variable>& vars,
                                  Model&                       model,
                                  std::vector<uint32_t>&       changedOut) {
    const auto& hot    = model.getHot(); // live reference — updated by setVarBounds
    StepResult  result = StepResult::NoChange;

    for (const Variable vi : vars) {
        int64_t li = iLb(hot.lb[vi.id]);
        int64_t ui = iUb(hot.ub[vi.id]);
        if (li != ui) continue; // not fixed — skip
        const int64_t v = li;

        for (const Variable vj : vars) {
            if (vj.id == vi.id) continue;
            int64_t lj    = iLb(hot.lb[vj.id]);
            int64_t uj    = iUb(hot.ub[vj.id]);
            int64_t newLj = lj;
            int64_t newUj = uj;
            if (lj == v) newLj = v + 1;
            if (uj == v) newUj = v - 1;
            if (newLj > newUj) return StepResult::Infeasible;
            if (newLj != lj || newUj != uj) {
                model.setVarBounds(vj, static_cast<double>(newLj), static_cast<double>(newUj));
                changedOut.push_back(vj.id);
                result = StepResult::Changed;
            }
        }
    }
    return result;
}

// Range feasibility: k variables must take k distinct integer values.
// Necessary condition: max(ub_i) − min(lb_i) + 1 ≥ k.
bool rangeFeasible(const std::vector<Variable>& vars, const Model& model) {
    const auto& hot = model.getHot();
    int64_t minLb = std::numeric_limits<int64_t>::max();
    int64_t maxUb = std::numeric_limits<int64_t>::min();
    for (const Variable v : vars) {
        int64_t li = iLb(hot.lb[v.id]);
        int64_t ui = iUb(hot.ub[v.id]);
        if (li < minLb) minLb = li;
        if (ui > maxUb) maxUb = ui;
    }
    return (maxUb - minLb + 1) >= static_cast<int64_t>(vars.size());
}

} // namespace

PropagationResult propagate(const AllDiffConstraint& con, Model& model) {
    PropagationResult result;
    if (con.vars.size() < 2) return result; // trivially satisfied

    // Fixpoint: repeat fixed-value elimination until no bound changes.
    bool anyChange = true;
    while (anyChange) {
        StepResult step = fixedValueElimination(con.vars, model, result.changedVarIds);
        if (step == StepResult::Infeasible) {
            result.status = CPStatus::Infeasible;
            return result;
        }
        anyChange = (step == StepResult::Changed);
    }

    if (!rangeFeasible(con.vars, model)) {
        result.status = CPStatus::Infeasible;
    }

    return result;
}

bool cpFeasible(const AllDiffConstraint& con, const std::vector<double>& sol, double tol) {
    for (std::size_t i = 0; i < con.vars.size(); ++i)
        for (std::size_t j = i + 1; j < con.vars.size(); ++j)
            if (std::abs(sol[con.vars[i].id] - sol[con.vars[j].id]) <= tol)
                return false;
    return true;
}

uint32_t cpViolatedVar(const AllDiffConstraint& con, const std::vector<double>& sol, double tol) {
    for (std::size_t i = 0; i < con.vars.size(); ++i)
        for (std::size_t j = i + 1; j < con.vars.size(); ++j)
            if (std::abs(sol[con.vars[i].id] - sol[con.vars[j].id]) <= tol)
                return con.vars[i].id;
    return std::numeric_limits<uint32_t>::max();
}

} // namespace baguette
