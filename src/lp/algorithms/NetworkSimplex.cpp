#include "NetworkSimplex.hpp"

#include "DualSimplexBV.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include "baguette/core/Config.hpp"
#include "baguette/core/Sense.hpp"
#include "baguette/model/ModelEnums.hpp"

namespace baguette {
namespace {

static constexpr double kBigM  = 1e9;
static constexpr double kInf   = std::numeric_limits<double>::infinity();
static constexpr int    kNoArc = -1;
static constexpr int    kNoNode = -1;

enum class NS : int8_t { L = -1, T = 0, U = 1 };

// ── Network data ──────────────────────────────────────────────────────────────

struct Net {
    int nNodes;  // number of original nodes (= number of constraints)
    int nReal;   // number of real arcs (= number of model variables)
    int nArcs;   // total arcs (nReal + nNodes artificial)
    int root;    // artificial root node index (= nNodes)
    int nTotal;  // nNodes + 1

    std::vector<int>    tail, head;
    std::vector<double> cost;
    std::vector<double> cap;   // upper bound in shifted space (kInf for unbounded)

    std::vector<int>    arcVar;    // arcVar[j] = variable index for arc j, -1 for artificials
    std::vector<double> varShift;  // x[j] = varShift[j] + x'[j]
    double              objOffset = 0.0;

    std::vector<double> flow;
    std::vector<NS>     state;

    // Tree (size nTotal)
    std::vector<int>    parent;
    std::vector<int>    parentArc;
    std::vector<bool>   parentFwd;  // true if tree arc goes parent → child
    std::vector<int>    depth;
    std::vector<double> pi;         // node potentials

    // Tree adjacency: arc indices incident to each node that are in T
    std::vector<std::vector<int>> treeAdj;
};

// ── Detection and construction ────────────────────────────────────────────────

bool tryBuildNetwork(const Model& model, Net& g) {
    const auto& constraints = model.getLPConstraints();
    const auto& hot = model.getHot();
    const int m = (int)constraints.size();
    const int n = (int)model.numVars();

    if (m == 0 || n == 0) return false;

    for (const auto& c : constraints)
        if (c.sense != Sense::Equal) return false;

    // Detect node-arc incidence: each variable must appear exactly once with +1
    // and once with -1 (one arc in, one arc out, arbitrary nodes)
    std::vector<int> arcTail(n, kNoNode), arcHead(n, kNoNode);

    for (int i = 0; i < m; ++i) {
        const auto& lhs = constraints[i].lhs;
        for (int k = 0; k < (int)lhs.size(); ++k) {
            int    j = (int)lhs.varIds[k];
            double c = lhs.coeffs[k];
            if (j >= n) return false;
            if (std::abs(std::abs(c) - 1.0) > 1e-9) return false;
            if (c > 0.0) {
                if (arcTail[j] != kNoNode) return false;
                arcTail[j] = i;
            } else {
                if (arcHead[j] != kNoNode) return false;
                arcHead[j] = i;
            }
        }
    }
    for (int j = 0; j < n; ++j)
        if (arcTail[j] == kNoNode || arcHead[j] == kNoNode) return false;

    // Build network
    g.nNodes = m;
    g.nReal  = n;
    g.nArcs  = n + m;
    g.root   = m;
    g.nTotal = m + 1;

    g.tail.resize(g.nArcs);
    g.head.resize(g.nArcs);
    g.cost.resize(g.nArcs);
    g.cap.resize(g.nArcs);
    g.arcVar.resize(g.nArcs, -1);
    g.varShift.resize(n);

    const bool   maximize = (model.getObjSense() == ObjSense::Maximize);
    const double objSign  = maximize ? -1.0 : 1.0;

    double shiftObj = 0.0;
    for (int j = 0; j < n; ++j) {
        double lb = hot.lb[j];
        double ub = hot.ub[j];
        double cj = objSign * hot.obj[j];

        double shift = std::isfinite(lb) ? lb : 0.0;
        g.varShift[j] = shift;
        shiftObj     += cj * shift;

        g.tail[j]   = arcTail[j];
        g.head[j]   = arcHead[j];
        g.cost[j]   = cj;
        g.cap[j]    = (std::isfinite(ub) && std::isfinite(lb)) ? ub - lb : kInf;
        g.arcVar[j] = j;
    }
    // objOffset: when we multiply out objSign we'll un-negate at extraction
    g.objOffset = shiftObj + objSign * model.getObjConstant();

    // Effective supply (after lb-shift)
    std::vector<double> supply(m);
    for (int i = 0; i < m; ++i) supply[i] = constraints[i].rhs;
    for (int j = 0; j < n; ++j) {
        double sh = g.varShift[j];
        if (sh != 0.0) {
            supply[g.tail[j]] -= sh;
            supply[g.head[j]] += sh;
        }
    }

    // Artificial arcs: arc nReal+i connects node i to root
    for (int i = 0; i < m; ++i) {
        int a = n + i;
        g.arcVar[a] = -1;
        g.cost[a]   = kBigM;
        g.cap[a]    = kInf;
        if (supply[i] >= 0.0) {
            g.tail[a] = i;        // excess: send out to root
            g.head[a] = g.root;
        } else {
            g.tail[a] = g.root;   // deficit: receive from root
            g.head[a] = i;
        }
    }

    // Initial flows: real arcs at LB (flow=0), artificial arcs carry |supply|
    g.flow.assign(g.nArcs, 0.0);
    g.state.assign(g.nArcs, NS::L);

    for (int i = 0; i < m; ++i) {
        int a       = n + i;
        g.flow[a]   = std::abs(supply[i]);
        g.state[a]  = NS::T;
    }

    // Initial spanning tree: root connected to every original node via art arcs
    g.parent.assign(g.nTotal, kNoNode);
    g.parentArc.assign(g.nTotal, kNoArc);
    g.parentFwd.assign(g.nTotal, false);
    g.depth.assign(g.nTotal, 0);
    g.pi.assign(g.nTotal, 0.0);
    g.treeAdj.assign(g.nTotal, {});

    for (int i = 0; i < m; ++i) {
        int a          = n + i;
        g.parent[i]    = g.root;
        g.parentArc[i] = a;
        g.depth[i]     = 1;
        if (supply[i] >= 0.0) {
            // arc (i→root): going child→parent = NOT parentFwd
            g.parentFwd[i] = false;
            g.pi[i]        = kBigM;   // rc(i,root)=kBigM-pi[i]+0=0 → pi[i]=kBigM
        } else {
            // arc (root→i): going parent→child = IS parentFwd
            g.parentFwd[i] = true;
            g.pi[i]        = -kBigM;  // rc(root,i)=kBigM-0+pi[i]=0 → pi[i]=-kBigM
        }
        g.treeAdj[g.root].push_back(a);
        g.treeAdj[i].push_back(a);
    }

    return true;
}

// ── Tree BFS rebuild ──────────────────────────────────────────────────────────

void rebuildTree(Net& g) {
    std::fill(g.parent.begin(),    g.parent.end(),    kNoNode);
    std::fill(g.parentArc.begin(), g.parentArc.end(), kNoArc);
    std::fill(g.depth.begin(),     g.depth.end(),     0);

    std::vector<bool> visited(g.nTotal, false);
    visited[g.root] = true;

    std::vector<int> q;
    q.reserve(g.nTotal);
    q.push_back(g.root);
    for (int qi = 0; qi < (int)q.size(); ++qi) {
        int u = q[qi];
        for (int a : g.treeAdj[u]) {
            int v = (g.tail[a] == u) ? g.head[a] : g.tail[a];
            if (visited[v]) continue;
            visited[v]     = true;
            g.parent[v]    = u;
            g.parentArc[v] = a;
            g.parentFwd[v] = (g.tail[a] == u);  // arc goes parent→child?
            g.depth[v]     = g.depth[u] + 1;
            q.push_back(v);
        }
    }
}

// ── Potential recomputation (BFS from root) ───────────────────────────────────

void recomputePotentials(Net& g) {
    g.pi[g.root] = 0.0;
    std::vector<bool> visited(g.nTotal, false);
    visited[g.root] = true;
    std::vector<int> q;
    q.reserve(g.nTotal);
    q.push_back(g.root);
    for (int qi = 0; qi < (int)q.size(); ++qi) {
        int u = q[qi];
        for (int a : g.treeAdj[u]) {
            int v = (g.tail[a] == u) ? g.head[a] : g.tail[a];
            if (visited[v]) continue;
            visited[v] = true;
            // Tree arc rc = 0: c - pi[tail] + pi[head] = 0
            // parentFwd[v]=true  → arc (u→v), tail=u, head=v: pi[v] = pi[u] - cost
            // parentFwd[v]=false → arc (v→u), tail=v, head=u: pi[v] = pi[u] + cost
            g.pi[v] = g.parentFwd[v] ? g.pi[u] - g.cost[a]
                                      : g.pi[u] + g.cost[a];
            q.push_back(v);
        }
    }
}

// ── Entering arc selection ────────────────────────────────────────────────────

int selectEntering(const Net& g) {
    int    best     = kNoArc;
    double bestViol = lp_optimality_tol;
    for (int a = 0; a < g.nArcs; ++a) {
        if (g.state[a] == NS::T) continue;
        double rc = g.cost[a] - g.pi[g.tail[a]] + g.pi[g.head[a]];
        if (g.state[a] == NS::L && rc < -bestViol) {
            bestViol = -rc;
            best     = a;
        } else if (g.state[a] == NS::U && rc > bestViol) {
            bestViol = rc;
            best     = a;
        }
    }
    return best;
}

// ── Cycle path ────────────────────────────────────────────────────────────────

struct ArcDir { int arc; bool fwd; };

// Returns arcs traversed from node u to node v in the current spanning tree,
// with fwd=true meaning the arc is traversed in its tail→head direction.
std::vector<ArcDir> findTreePath(const Net& g, int u, int v) {
    std::vector<ArcDir> pathU, pathV;
    int a = u, b = v;
    // Walk both sides up to equal depth
    while (g.depth[a] > g.depth[b]) {
        // going a → parent[a]: fwd = !parentFwd[a]
        pathU.push_back({g.parentArc[a], !g.parentFwd[a]});
        a = g.parent[a];
    }
    while (g.depth[b] > g.depth[a]) {
        // going b → parent[b] (will be reversed later)
        pathV.push_back({g.parentArc[b], !g.parentFwd[b]});
        b = g.parent[b];
    }
    // Walk both up to LCA
    while (a != b) {
        pathU.push_back({g.parentArc[a], !g.parentFwd[a]});
        a = g.parent[a];
        pathV.push_back({g.parentArc[b], !g.parentFwd[b]});
        b = g.parent[b];
    }
    // a == b == LCA
    // Result: pathU (u→LCA) + reversed pathV with flipped direction (LCA→v)
    std::vector<ArcDir> result = pathU;
    for (auto it = pathV.rbegin(); it != pathV.rend(); ++it)
        result.push_back({it->arc, !it->fwd});
    return result;
}

// ── Pivot ─────────────────────────────────────────────────────────────────────

struct PivotInfo {
    int    leaving;
    double delta;
    bool   leavingAtUB;   // new state of leaving arc
    bool   augFwd;        // augmentation direction: true = forward on entering arc
    std::vector<ArcDir> cyclePath;  // tree arcs u→v (not including entering arc)
};

PivotInfo findPivot(const Net& g, int entering) {
    PivotInfo p;
    p.augFwd   = (g.state[entering] == NS::L);
    int etail  = p.augFwd ? g.tail[entering] : g.head[entering];
    int ehead  = p.augFwd ? g.head[entering] : g.tail[entering];

    // Residual on entering arc
    double resE    = p.augFwd ? (g.cap[entering] - g.flow[entering]) : g.flow[entering];
    p.delta        = resE;
    p.leaving      = entering;
    p.leavingAtUB  = p.augFwd;  // entering arc at L hitting cap → U; at U hitting 0 → L

    // Tree path from ehead back to etail
    p.cyclePath = findTreePath(g, ehead, etail);

    for (const auto& ad : p.cyclePath) {
        double res = ad.fwd ? (g.cap[ad.arc] - g.flow[ad.arc]) : g.flow[ad.arc];
        if (res < p.delta - 1e-12) {
            p.delta       = res;
            p.leaving     = ad.arc;
            p.leavingAtUB = ad.fwd;  // forward traversal → hits cap (UB)
        }
    }
    return p;
}

void applyPivot(Net& g, int entering, const PivotInfo& p) {
    double delta = p.delta;

    // Update entering arc flow
    if (p.augFwd) g.flow[entering] += delta;
    else          g.flow[entering] -= delta;

    // Update cycle arc flows
    for (const auto& ad : p.cyclePath) {
        if (ad.fwd) g.flow[ad.arc] += delta;
        else        g.flow[ad.arc] -= delta;
    }

    int leaving = p.leaving;

    // If entering arc is also the leaving arc: it just changes state, no tree update
    if (leaving == entering) {
        g.state[entering] = p.leavingAtUB ? NS::U : NS::L;
        return;
    }

    // Update states
    g.state[leaving]  = p.leavingAtUB ? NS::U : NS::L;
    g.state[entering] = NS::T;

    // Remove leaving arc from treeAdj
    int lt = g.tail[leaving], lh = g.head[leaving];
    auto& adjlt = g.treeAdj[lt];
    auto& adjlh = g.treeAdj[lh];
    adjlt.erase(std::find(adjlt.begin(), adjlt.end(), leaving));
    adjlh.erase(std::find(adjlh.begin(), adjlh.end(), leaving));

    // Add entering arc to treeAdj
    int et = g.tail[entering], eh = g.head[entering];
    g.treeAdj[et].push_back(entering);
    g.treeAdj[eh].push_back(entering);

    rebuildTree(g);
    recomputePotentials(g);
}

// ── Infeasibility check ───────────────────────────────────────────────────────

bool hasArtificialFlow(const Net& g) {
    for (int i = 0; i < g.nNodes; ++i) {
        int a = g.nReal + i;
        double f = 0.0;
        if      (g.state[a] == NS::T) f = g.flow[a];
        else if (g.state[a] == NS::U) f = g.cap[a];
        if (f > lp_feasibility_tol) return true;
    }
    return false;
}

// ── Result extraction ─────────────────────────────────────────────────────────

LPDetailedResult extractResult(const Net& g, const Model& model, LPStatus status) {
    LPDetailedResult det;
    det.result.status = status;

    const int    n        = model.numVars();
    const bool   maximize = (model.getObjSense() == ObjSense::Maximize);
    const auto&  hot      = model.getHot();

    det.result.primalValues.resize(n);
    for (int j = 0; j < n; ++j) {
        double xp = (g.state[j] == NS::U) ? g.cap[j] : g.flow[j];
        det.result.primalValues[j] = g.varShift[j] + xp;
    }

    double obj = 0.0;
    for (int j = 0; j < n; ++j)
        obj += hot.obj[j] * det.result.primalValues[j];
    obj += model.getObjConstant();
    det.result.objectiveValue = obj;

    if (status != LPStatus::Optimal) return det;

    const auto& constraints = model.getLPConstraints();
    det.dualValues.resize(g.nNodes);
    for (int i = 0; i < g.nNodes; ++i) {
        // LP dual y satisfies A^T y = c for tree arcs.
        // For arc (tail,head): y[tail] - y[head] = c_arc = π[tail] - π[head]
        // → y[i] = π[i] (for minimization).
        // For maximization: internal cost = -c_orig, so π satisfies -c_orig convention
        // → LP dual in original units: y[i] = -π[i].
        det.dualValues[i] = maximize ? -g.pi[i] : g.pi[i];
    }

    det.reducedCosts.resize(n);
    for (int j = 0; j < n; ++j) {
        // rc_network = cost[j] - π[tail] + π[head] (zero for tree arcs)
        // For min: cost[j] = c_j → LP rc = rc_network
        // For max: cost[j] = -c_j_orig → LP rc in original = -rc_network
        double rc = g.cost[j] - g.pi[g.tail[j]] + g.pi[g.head[j]];
        det.reducedCosts[j] = maximize ? -rc : rc;
    }

    return det;
}

// ── Main solver ───────────────────────────────────────────────────────────────

LPDetailedResult runNetworkSimplex(const Model&                          model,
                                   uint32_t                              maxIter,
                                   double                                timeLimitS,
                                   std::chrono::steady_clock::time_point startTime) {
    // Early infeasibility: empty domain
    {
        const auto& hot = model.getHot();
        for (std::size_t j = 0; j < model.numVars(); ++j) {
            if (hot.lb[j] > hot.ub[j] + lp_feasibility_tol) {
                LPDetailedResult det;
                det.result.status      = LPStatus::Infeasible;
                det.farkas.infeasVarId = (int32_t)j;
                return det;
            }
        }
    }

    Net g;
    if (!tryBuildNetwork(model, g)) {
        return internal::solveDualBV(model, maxIter, timeLimitS, startTime,
                                     {}, false, false);
    }

    uint32_t iters = 0;

    while (true) {
        if (maxIter > 0 && iters >= maxIter) {
            LPDetailedResult det;
            det.result.status = LPStatus::MaxIter;
            return det;
        }
        if (std::chrono::duration<double>(
                std::chrono::steady_clock::now() - startTime).count() >= timeLimitS) {
            LPDetailedResult det;
            det.result.status = LPStatus::TimeLimit;
            return det;
        }

        int entering = selectEntering(g);
        if (entering == kNoArc) {
            LPStatus st = hasArtificialFlow(g) ? LPStatus::Infeasible : LPStatus::Optimal;
            if (st == LPStatus::Infeasible) {
                LPDetailedResult det;
                det.result.status = LPStatus::Infeasible;
                return det;
            }
            return extractResult(g, model, LPStatus::Optimal);
        }

        PivotInfo piv = findPivot(g, entering);

        if (piv.delta >= kInf * 0.5) {
            LPDetailedResult det;
            det.result.status = LPStatus::Unbounded;
            return det;
        }

        applyPivot(g, entering, piv);
        ++iters;
    }
}

} // anonymous namespace

namespace internal {

LPDetailedResult solveNetworkSimplex(const Model&                          model,
                                     uint32_t                              maxIter,
                                     double                                timeLimitS,
                                     std::chrono::steady_clock::time_point startTime,
                                     bool /*computeCutData*/) {
    return runNetworkSimplex(model, maxIter, timeLimitS, startTime);
}

} // namespace internal
} // namespace baguette
