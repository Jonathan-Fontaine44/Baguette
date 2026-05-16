#include "baguette/cp/constraints/Cumulative.hpp"

#include <cmath>
#include <deque>
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

} // namespace

PropagationResult propagate(const CumulativeConstraint& con, Model& model) {
    PropagationResult result;
    const int n = static_cast<int>(con.tasks.size());
    if (n < 2) return result;

    const auto& hot = model.getHot();
    std::vector<TaskBounds> tb;
    tb.reserve(n);
    for (const Task& t : con.tasks) {
        int64_t est = iLb(hot.lb[t.varStart.id]);
        int64_t lst = iUb(hot.ub[t.varStart.id]);
        tb.push_back({est, lst, est + t.duration, t.varStart, t.duration, t.consumption});
    }

    bool changed = true;
    while (changed) {
        changed = false;

        // Time horizon [tMin, tMax): covers all possible task windows.
        int64_t tMin = tb[0].est;
        int64_t tMax = tb[0].lst + tb[0].dur;
        for (int i = 1; i < n; ++i) {
            if (tb[i].est < tMin)          tMin = tb[i].est;
            int64_t end_i = tb[i].lst + tb[i].dur;
            if (end_i > tMax)              tMax = end_i;
        }
        const std::size_t D = static_cast<std::size_t>(tMax - tMin);
        if (D == 0) break;

        // Compulsory load profile via difference array.
        // P[k] = total compulsory load at time (tMin + k), for k in [0, D).
        std::vector<int32_t> P(D + 1, 0);
        for (int i = 0; i < n; ++i) {
            if (tb[i].lst >= tb[i].ect) continue; // no compulsory region
            const std::size_t lo = static_cast<std::size_t>(tb[i].lst - tMin);
            const std::size_t hi = static_cast<std::size_t>(tb[i].ect - tMin);
            P[lo] += tb[i].cns;
            P[hi] -= tb[i].cns;
        }
        for (std::size_t k = 1; k < D; ++k) P[k] += P[k - 1];
        P.resize(D);

        for (int i = 0; i < n; ++i) {
            // Temporarily remove task i's compulsory contribution from P.
            const bool        hasComp = tb[i].lst < tb[i].ect;
            const std::size_t lo_i   = hasComp ? static_cast<std::size_t>(tb[i].lst - tMin) : 0;
            const std::size_t hi_i   = hasComp ? static_cast<std::size_t>(tb[i].ect - tMin) : 0;
            if (hasComp)
                for (std::size_t k = lo_i; k < hi_i; ++k) P[k] -= tb[i].cns;

            // Find the earliest t in [est_i, lst_i] where
            // max(P[t .. t+dur_i)) + cns_i ≤ capacity.
            // Uses a deque-based sliding window max in O(lst_i − est_i + dur_i).
            const int32_t    limit    = con.capacity - tb[i].cns;
            const int64_t    est_i    = tb[i].est;
            const int64_t    lst_i    = tb[i].lst;
            const std::size_t w       = static_cast<std::size_t>(tb[i].dur);
            const std::size_t k_start = static_cast<std::size_t>(est_i - tMin);

            std::deque<std::size_t> dq;
            for (std::size_t k = k_start; k < k_start + w; ++k) {
                while (!dq.empty() && P[dq.back()] <= P[k]) dq.pop_back();
                dq.push_back(k);
            }

            int64_t newEst = lst_i + 1; // sentinel: no valid start found
            for (int64_t t = est_i; t <= lst_i; ++t) {
                const std::size_t kt = static_cast<std::size_t>(t - tMin);
                if (dq.empty() || P[dq.front()] <= limit) { newEst = t; break; }
                while (!dq.empty() && dq.front() <= kt) dq.pop_front();
                const std::size_t k_enter = kt + w;
                if (k_enter < D) {
                    while (!dq.empty() && P[dq.back()] <= P[k_enter]) dq.pop_back();
                    dq.push_back(k_enter);
                }
            }

            if (hasComp)
                for (std::size_t k = lo_i; k < hi_i; ++k) P[k] += tb[i].cns;

            if (newEst > lst_i) {
                result.status = CPStatus::Infeasible;
                return result;
            }
            if (newEst > tb[i].est) {
                tb[i].est = newEst;
                tb[i].ect = newEst + tb[i].dur;
                model.setVarBounds(tb[i].var,
                                   static_cast<double>(newEst),
                                   static_cast<double>(lst_i));
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
