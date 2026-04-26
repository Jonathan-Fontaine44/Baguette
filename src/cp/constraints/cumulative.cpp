#include "baguette/cp/constraints/Cumulative.hpp"

#include <cmath>
#include <limits>

#include "baguette/model/Model.hpp"

namespace baguette {

namespace {

inline int64_t iLb(double lb) { return static_cast<int64_t>(std::ceil(lb  - 1e-9)); }
inline int64_t iUb(double ub) { return static_cast<int64_t>(std::floor(ub + 1e-9)); }

struct TaskBounds {
    int64_t  est; // earliest start  = lb(varStart)
    int64_t  lst; // latest start    = ub(varStart)
    int64_t  ect; // earliest compl. = est + duration
    Variable var;
    int32_t  dur;
    int32_t  cns;
};

// Compulsory region of task j: [lst_j, ect_j).
// Returns true iff time point tp falls inside task j's compulsory region.
inline bool isCompulsoryAt(const TaskBounds& j, int64_t tp) {
    return j.lst < j.ect && j.lst <= tp && tp < j.ect;
}

// Check whether task i can start at time t without exceeding capacity at any
// time point in [t, t + dur_i): a time point is infeasible if the compulsory
// load from other tasks plus consumption_i exceeds capacity.
bool canStart(int i, int64_t t, const std::vector<TaskBounds>& tb, int32_t capacity) {
    const int64_t  end = t + tb[i].dur;
    for (int64_t tp = t; tp < end; ++tp) {
        int32_t load = tb[i].cns;
        for (int j = 0; j < static_cast<int>(tb.size()); ++j) {
            if (j == i) continue;
            if (isCompulsoryAt(tb[j], tp)) load += tb[j].cns;
        }
        if (load > capacity) return false;
    }
    return true;
}

} // namespace

PropagationResult propagate(const CumulativeConstraint& con, Model& model) {
    PropagationResult result;
    const int n = static_cast<int>(con.tasks.size());
    if (n < 2) return result;

    // Snapshot bounds (hot reference stays live; tb mirrors initial state).
    const auto& hot = model.getHot();
    std::vector<TaskBounds> tb;
    tb.reserve(n);
    for (const Task& t : con.tasks) {
        int64_t est = iLb(hot.lb[t.varStart.id]);
        int64_t lst = iUb(hot.ub[t.varStart.id]);
        tb.push_back({est, lst, est + t.duration, t.varStart, t.duration, t.consumption});
    }

    // Fixpoint: advance each est_i past any start where an overload would occur.
    bool changed = true;
    while (changed) {
        changed = false;
        for (int i = 0; i < n; ++i) {
            int64_t t = tb[i].est;
            while (t <= tb[i].lst && !canStart(i, t, tb, con.capacity))
                ++t;

            if (t > tb[i].lst) {
                result.status = CPStatus::Infeasible;
                return result;
            }

            if (t > tb[i].est) {
                tb[i].est = t;
                tb[i].ect = t + tb[i].dur;
                model.setVarBounds(tb[i].var,
                                   static_cast<double>(t),
                                   static_cast<double>(tb[i].lst));
                result.changedVarIds.push_back(tb[i].var.id);
                changed = true;
            }
        }
    }

    return result;
}

bool cpFeasible(const CumulativeConstraint& con, const std::vector<double>& sol, double /*tol*/) {
    for (const Task& ti : con.tasks) {
        int64_t si = static_cast<int64_t>(std::round(sol[ti.varStart.id]));
        for (int64_t tp = si; tp < si + ti.duration; ++tp) {
            int32_t load = ti.consumption;
            for (const Task& tj : con.tasks) {
                if (tj.varStart.id == ti.varStart.id) continue;
                int64_t sj = static_cast<int64_t>(std::round(sol[tj.varStart.id]));
                if (sj <= tp && tp < sj + tj.duration) load += tj.consumption;
            }
            if (load > con.capacity) return false;
        }
    }
    return true;
}

uint32_t cpViolatedVar(const CumulativeConstraint& con, const std::vector<double>& sol, double /*tol*/) {
    for (const Task& ti : con.tasks) {
        int64_t si = static_cast<int64_t>(std::round(sol[ti.varStart.id]));
        for (int64_t tp = si; tp < si + ti.duration; ++tp) {
            int32_t load = ti.consumption;
            for (const Task& tj : con.tasks) {
                if (tj.varStart.id == ti.varStart.id) continue;
                int64_t sj = static_cast<int64_t>(std::round(sol[tj.varStart.id]));
                if (sj <= tp && tp < sj + tj.duration) load += tj.consumption;
            }
            if (load > con.capacity) return ti.varStart.id;
        }
    }
    return std::numeric_limits<uint32_t>::max();
}

} // namespace baguette
