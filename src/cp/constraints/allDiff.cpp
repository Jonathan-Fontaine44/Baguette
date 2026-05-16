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
    const std::size_t prevSize = changedOut.size();

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

    // Remove duplicates introduced in this call (a variable may be affected by
    // multiple fixed siblings in the same pass, producing O(K²) redundant entries).
    auto newBegin = changedOut.begin() + static_cast<std::ptrdiff_t>(prevSize);
    std::sort(newBegin, changedOut.end());
    changedOut.erase(std::unique(newBegin, changedOut.end()), changedOut.end());

    return result;
}

// Hall interval propagation.
// For each interval [u, v] formed by pairs of variable bounds:
//   count := |{xi : lb_i >= u AND ub_i <= v}|
//   if count > v-u+1: infeasible (Hall's theorem violated)
//   if count == v-u+1: Hall interval — clip other domains at u-1 or v+1
//
// Also subsumes the global range feasibility check (u=min_lb, v=max_ub).
StepResult hallPropagation(const std::vector<Variable>& vars,
                            Model&                       model,
                            std::vector<uint32_t>&       changedOut) {
    const auto&       hot = model.getHot();
    const std::size_t K   = vars.size();

    // Candidate interval endpoints: all distinct lb and ub values
    std::vector<int64_t> uCands, vCands;
    uCands.reserve(K);
    vCands.reserve(K);
    for (const Variable v : vars) {
        uCands.push_back(iLb(hot.lb[v.id]));
        vCands.push_back(iUb(hot.ub[v.id]));
    }
    std::sort(uCands.begin(), uCands.end());
    uCands.erase(std::unique(uCands.begin(), uCands.end()), uCands.end());
    std::sort(vCands.begin(), vCands.end());
    vCands.erase(std::unique(vCands.begin(), vCands.end()), vCands.end());

    StepResult        result   = StepResult::NoChange;
    const std::size_t prevSize = changedOut.size();

    for (const int64_t u : uCands) {
        for (const int64_t v : vCands) {
            if (v < u) continue;
            const int64_t capacity = v - u + 1;

            int64_t count = 0;
            for (const Variable vi : vars)
                if (iLb(hot.lb[vi.id]) >= u && iUb(hot.ub[vi.id]) <= v) ++count;

            if (count > capacity) return StepResult::Infeasible;
            if (count < capacity) continue;

            // Hall interval: prune variables whose domain overlaps but extends outside [u, v]
            for (const Variable vj : vars) {
                const int64_t lj = iLb(hot.lb[vj.id]);
                const int64_t uj = iUb(hot.ub[vj.id]);
                if (lj >= u && uj <= v) continue; // inside Hall set

                int64_t newLj = lj;
                int64_t newUj = uj;
                if (lj < u && uj >= u && uj <= v) newUj = u - 1; // overlaps from below
                if (lj >= u && lj <= v && uj > v) newLj = v + 1; // overlaps from above

                if (newLj > newUj) return StepResult::Infeasible;
                if (newLj != lj || newUj != uj) {
                    model.setVarBounds(vj, static_cast<double>(newLj), static_cast<double>(newUj));
                    changedOut.push_back(vj.id);
                    result = StepResult::Changed;
                }
            }
        }
    }

    auto newBegin = changedOut.begin() + static_cast<std::ptrdiff_t>(prevSize);
    std::sort(newBegin, changedOut.end());
    changedOut.erase(std::unique(newBegin, changedOut.end()), changedOut.end());

    return result;
}

} // namespace

PropagationResult propagate(const AllDiffConstraint& con, Model& model) {
    PropagationResult result;
    if (con.vars.size() < 2) return result; // trivially satisfied

    // Fixpoint: alternate fixed-value elimination and Hall interval propagation.
    // Hall subsumes the global range feasibility check.
    bool anyChange = true;
    while (anyChange) {
        anyChange = false;

        StepResult fve = fixedValueElimination(con.vars, model, result.changedVarIds);
        if (fve == StepResult::Infeasible) { result.status = CPStatus::Infeasible; return result; }
        if (fve == StepResult::Changed) anyChange = true;

        StepResult hall = hallPropagation(con.vars, model, result.changedVarIds);
        if (hall == StepResult::Infeasible) { result.status = CPStatus::Infeasible; return result; }
        if (hall == StepResult::Changed) anyChange = true;
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
