#include "baguette/milp/BranchAndBound.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "baguette/core/Sense.hpp"
#include "baguette/lp/LPResult.hpp"
#include "baguette/lp/LPSolver.hpp"
#include "baguette/milp/CuttingPlanes.hpp"
#include "baguette/model/ModelEnums.hpp"

namespace baguette {

namespace {

// ── Internal node ──────────────────────────────────────────────────────────────

// One variable bound change accumulated along the path from the root to a node.
// newLb / newUb replace the root-level bounds for varId at this depth.
struct BoundChange {
    uint32_t varId;
    double   newLb;
    double   newUb;
};

struct Node {
    std::vector<BoundChange> changes;  // accumulated deltas from root (one per branch depth)
    BasisRecord              basis;    // warm-start data from the parent LP solve
    double                   lpBound;  // LP objective at the parent (pruning + pseudo-costs)
    int                      depth;

    // Pseudo-cost bookkeeping: records the branching decision that created this node.
    int    parentBranchVar = -1; // -1 = root node (no parent branch)
    bool   branchedUp      = false;
    double parentFrac      = 0.0;
};

// ── Helpers ────────────────────────────────────────────────────────────────────

std::vector<uint32_t> collectIntVarIds(const Model& model) {
    const auto& types = model.getCold().types;
    std::vector<uint32_t> ids;
    ids.reserve(types.size());
    for (uint32_t i = 0; i < static_cast<uint32_t>(types.size()); ++i) {
        if (types[i] == VarType::Integer || types[i] == VarType::Binary)
            ids.push_back(i);
    }
    return ids;
}

int selectBranchVar(const std::vector<double>&   sol,
                    const std::vector<uint32_t>& intVarIds,
                    BranchStrategy               strat,
                    double                       intFeasTol) {
    int    bestId    = -1;
    double bestScore = -1.0;

    for (uint32_t id : intVarIds) {
        double x    = sol[id];
        double frac = x - std::floor(x);

        if (frac <= intFeasTol || frac >= 1.0 - intFeasTol)
            continue;

        if (strat == BranchStrategy::FirstFractional)
            return static_cast<int>(id);

        double score = std::min(frac, 1.0 - frac);
        if (score > bestScore) {
            bestScore = score;
            bestId    = static_cast<int>(id);
        }
    }
    return bestId;
}

// ── Pseudo-cost branching ──────────────────────────────────────────────────────
// Per-variable accumulators estimating the objective change per unit of
// fractionality when branching up (lb→ceil) or down (ub→floor).
// Only used inside solveMILP; not part of the public CuttingPlanes API.

struct PseudoCosts {
    std::vector<double>   upCost;
    std::vector<uint32_t> upCount;
    std::vector<double>   downCost;
    std::vector<uint32_t> downCount;
};

PseudoCosts initPseudoCosts(uint32_t numVars) {
    PseudoCosts pc;
    pc.upCost.assign(numVars, 0.0);
    pc.upCount.assign(numVars, 0);
    pc.downCost.assign(numVars, 0.0);
    pc.downCount.assign(numVars, 0);
    return pc;
}

// Product score: max(dUp, eps) × max(dDown, eps).
// Falls back to min(frac, 1−frac) for variables with no branching history.
int selectBranchVarPseudoCost(const std::vector<double>&   sol,
                               const std::vector<uint32_t>& intVarIds,
                               const PseudoCosts&           pc,
                               double                       intFeasTol) {
    constexpr double kEps = 1e-10;
    int    bestId    = -1;
    double bestScore = -1.0;

    for (uint32_t id : intVarIds) {
        double x  = sol[id];
        double fr = x - std::floor(x);
        if (fr <= intFeasTol || fr >= 1.0 - intFeasTol) continue;

        double dUp   = (pc.upCount[id]   > 0)
                     ? (pc.upCost[id]   / static_cast<double>(pc.upCount[id]))   * (1.0 - fr)
                     : std::min(fr, 1.0 - fr);
        double dDown = (pc.downCount[id] > 0)
                     ? (pc.downCost[id] / static_cast<double>(pc.downCount[id])) * fr
                     : std::min(fr, 1.0 - fr);

        double score = std::max(dUp, kEps) * std::max(dDown, kEps);
        if (score > bestScore) {
            bestScore = score;
            bestId    = static_cast<int>(id);
        }
    }
    return bestId;
}

void updatePseudoCosts(PseudoCosts& pc,
                       uint32_t    varId,
                       double      frac,
                       double      parentObj,
                       double      childObj,
                       bool        branchedUp) {
    double delta = std::abs(childObj - parentObj);
    double denom = branchedUp ? (1.0 - frac) : frac;
    if (denom <= 0.0) return;
    double rate = delta / denom;
    if (branchedUp) { pc.upCost[varId]   += rate; ++pc.upCount[varId]; }
    else            { pc.downCost[varId] += rate; ++pc.downCount[varId]; }
}

} // namespace

// ── solveMILP ──────────────────────────────────────────────────────────────────

MILPResult solveMILP(const Model&            modelRef,
                     const BBOptions&        opts,
                     SolverClock::time_point startTime) {
    Model model = modelRef;

    const bool   minimize = (model.getObjSense() == ObjSense::Minimize);
    const double inf      = std::numeric_limits<double>::infinity();

    const std::vector<uint32_t> intIds = collectIntVarIds(model);

    // ── Pseudo-costs ───────────────────────────────────────────────────────────
    PseudoCosts pc = initPseudoCosts(static_cast<uint32_t>(model.numVars()));

    // ── Incumbent tracking ─────────────────────────────────────────────────────
    double              incumbent    = minimize ? inf : -inf;
    std::vector<double> incumbentSol;

    uint32_t nodesExplored = 0;
    uint32_t cutsAdded     = 0;
    bool     timeLimitHit  = false;
    bool     maxNodesHit   = false;
    bool     unboundedHit  = false;

    // ── Time helpers ───────────────────────────────────────────────────────────
    auto elapsedS = [&]() -> double {
        using Dur = std::chrono::duration<double>;
        return std::chrono::duration_cast<Dur>(SolverClock::now() - startTime).count();
    };

    // ── Pruning predicate ──────────────────────────────────────────────────────
    auto canPrune = [&](double lpBound) -> bool {
        return minimize ? (lpBound >= incumbent - opts.mipGapAbs)
                        : (lpBound <= incumbent + opts.mipGapAbs);
    };

    // ── Node queue comparator (BestBound / post-plunge mode) ──────────────────
    auto cmpNodes = [minimize](const Node& a, const Node& b) -> bool {
        return minimize ? (a.lpBound > b.lpBound) : (a.lpBound < b.lpBound);
    };

    // ── HybridPlunge phase flag ────────────────────────────────────────────────
    // Starts true (DFS plunge) and flips to false after the first incumbent is
    // found, at which point the queue is heapified and BestBound takes over.
    bool plunging = (opts.nodeSelect == NodeSelection::HybridPlunge);

    // Returns true when heap ordering should be used.
    auto isHeapMode = [&]() -> bool {
        return opts.nodeSelect == NodeSelection::BestBound ||
               (opts.nodeSelect == NodeSelection::HybridPlunge && !plunging);
    };

    // ── Queue (vector used as heap or stack depending on mode) ─────────────────
    std::vector<Node> queue;

    auto pushNode = [&](Node n) {
        queue.push_back(std::move(n));
        if (isHeapMode())
            std::push_heap(queue.begin(), queue.end(), cmpNodes);
    };

    auto popNode = [&]() -> Node {
        if (isHeapMode())
            std::pop_heap(queue.begin(), queue.end(), cmpNodes);
        Node n = std::move(queue.back());
        queue.pop_back();
        return n;
    };

    // ── Root bounds cache + dirty-variable tracking ────────────────────────────
    // Root bounds are the model's initial bounds (before any branching).
    // dirtyVars tracks which variables currently differ from rootLb/rootUb so
    // that restoreBounds only touches O(prev_depth + curr_depth) variables
    // instead of O(n) for every node.
    const std::vector<double> rootLb = model.getHot().lb;
    const std::vector<double> rootUb = model.getHot().ub;
    std::vector<uint32_t>     dirtyVars; // varIds that currently differ from root bounds

    // ── Restore model bounds from a node's accumulated delta trail ─────────────
    auto restoreBounds = [&](const Node& node) {
        // Reset previously modified variables to root bounds.
        for (uint32_t id : dirtyVars)
            model.setVarBounds(Variable{id}, rootLb[id], rootUb[id]);
        // Apply this node's changes and rebuild the dirty set.
        dirtyVars.clear();
        for (const BoundChange& bc : node.changes) {
            model.setVarBounds(Variable{bc.varId}, bc.newLb, bc.newUb);
            dirtyVars.push_back(bc.varId);
        }
        // Deduplicate: the same variable may appear at multiple depths.
        std::sort(dirtyVars.begin(), dirtyVars.end());
        dirtyVars.erase(std::unique(dirtyVars.begin(), dirtyVars.end()), dirtyVars.end());
    };

    // ── Root node ──────────────────────────────────────────────────────────────
    {
        Node root;
        root.changes  = {};               // no changes from root bounds
        root.lpBound  = minimize ? -inf : inf;
        root.depth    = 0;
        // root.basis is default-constructed (empty) → cold start
        pushNode(std::move(root));
    }

    // ── Main B&C loop ──────────────────────────────────────────────────────────
    while (!queue.empty()) {
        if (elapsedS() >= opts.timeLimitS) { timeLimitHit = true; break; }
        if (opts.maxNodes > 0 && nodesExplored >= opts.maxNodes) { maxNodesHit = true; break; }

        Node node = popNode();
        ++nodesExplored;

        // Apply this node's accumulated bound deltas to the shared model.
        restoreBounds(node);

        // ── First LP solve ─────────────────────────────────────────────────────
        LPDetailedResult lp = solveDualDetailed(
            model,
            opts.maxIterLP,
            opts.timeLimitS,
            startTime,
            node.basis,
            /*computeSensitivity=*/false,
            /*computeCutData=*/opts.enableCuts);

        // ── Handle LP outcome ──────────────────────────────────────────────────
        switch (lp.result.status) {
            case LPStatus::Infeasible:      continue;
            case LPStatus::Unbounded:       unboundedHit = true; goto done;
            case LPStatus::TimeLimit:       timeLimitHit = true; goto done;
            case LPStatus::NumericalFailure: continue;
            case LPStatus::MaxIter:         continue;
            case LPStatus::Optimal:         break;
        }

        // ── Pseudo-cost update (branching improvement, pre-cut) ────────────────
        if (opts.branchStrat == BranchStrategy::PseudoCost &&
            node.parentBranchVar >= 0) {
            updatePseudoCosts(pc,
                              static_cast<uint32_t>(node.parentBranchVar),
                              node.parentFrac,
                              node.lpBound,
                              lp.result.objectiveValue,
                              node.branchedUp);
        }

        // ── Prune by bound (pre-cut) ───────────────────────────────────────────
        if (canPrune(lp.result.objectiveValue))
            continue;

        // ── GMI cut generation ─────────────────────────────────────────────────
        if (opts.enableCuts && !lp.fractionalRows.empty()) {
            std::vector<Cut> cuts = generateGMICuts(
                lp.fractionalRows, lp.basis, model,
                opts.maxCutsPerNode, opts.intFeasTol);

            if (!cuts.empty()) {
                for (const Cut& c : cuts)
                    model.addConstraint(c.expr, Sense::GreaterEq, c.rhs);
                cutsAdded += static_cast<uint32_t>(cuts.size());

                // Re-solve with the new cuts.  The warm basis is passed as {} (cold
                // start) because addConstraint() changed the model structure: the
                // sfCache stored in lp.basis refers to the pre-cut standard form and
                // its dimension no longer matches the current model.
                //
                // Side-effect on sibling nodes: nodes already queued (created before
                // this cut was added) also carry a pre-cut BasisRecord.  When they are
                // eventually popped, solveDualDetailed() detects the sfCache dimension
                // mismatch and silently falls back to a cold primal solve as well.
                // This is correct — their basis is simply stale — but it means that
                // cut addition degrades warm-start reuse for all queued siblings.
                lp = solveDualDetailed(
                    model,
                    opts.maxIterLP,
                    opts.timeLimitS,
                    startTime,
                    /*warmBasis=*/{},
                    /*computeSensitivity=*/false,
                    /*computeCutData=*/false);

                switch (lp.result.status) {
                    case LPStatus::Infeasible:       continue;
                    case LPStatus::Unbounded:        unboundedHit = true; goto done;
                    case LPStatus::TimeLimit:        timeLimitHit = true; goto done;
                    case LPStatus::NumericalFailure: continue;
                    case LPStatus::MaxIter:          continue;
                    case LPStatus::Optimal:          break;
                }

                if (canPrune(lp.result.objectiveValue))
                    continue;
            }
        }

        // ── Check integer feasibility ──────────────────────────────────────────
        int branchId;
        if (opts.branchStrat == BranchStrategy::PseudoCost) {
            branchId = selectBranchVarPseudoCost(
                lp.result.primalValues, intIds, pc, opts.intFeasTol);
        } else {
            branchId = selectBranchVar(
                lp.result.primalValues, intIds, opts.branchStrat, opts.intFeasTol);
        }

        if (branchId == -1) {
            double obj    = lp.result.objectiveValue;
            bool   better = minimize ? (obj < incumbent) : (obj > incumbent);
            if (better) {
                incumbent    = obj;
                incumbentSol = lp.result.primalValues;
                // HybridPlunge: first incumbent found — heapify and switch to BestBound.
                if (plunging) {
                    plunging = false;
                    std::make_heap(queue.begin(), queue.end(), cmpNodes);
                }
            }
            continue;
        }

        // ── Branch ────────────────────────────────────────────────────────────
        const double xj     = lp.result.primalValues[static_cast<uint32_t>(branchId)];
        const double floorX = std::floor(xj);
        const double ceilX  = std::ceil(xj);
        const double fracXj = xj - floorX;
        const double bound  = lp.result.objectiveValue;

        // Read the variable's current bounds from the model (already restored for this node).
        const double curLb = model.getHot().lb[static_cast<uint32_t>(branchId)];
        const double curUb = model.getHot().ub[static_cast<uint32_t>(branchId)];

        // Left child:  x_j ≤ floor(x_j)
        if (!canPrune(bound)) {
            Node left;
            left.changes = node.changes; // copy parent's trail
            left.changes.push_back({static_cast<uint32_t>(branchId), curLb, floorX});
            left.lpBound         = bound;
            left.depth           = node.depth + 1;
            left.basis           = lp.basis;
            left.parentBranchVar = branchId;
            left.branchedUp      = false;
            left.parentFrac      = fracXj;
            pushNode(std::move(left));
        }

        // Right child: x_j ≥ ceil(x_j)
        if (!canPrune(bound)) {
            Node right;
            right.changes = node.changes; // copy parent's trail
            right.changes.push_back({static_cast<uint32_t>(branchId), ceilX, curUb});
            right.lpBound         = bound;
            right.depth           = node.depth + 1;
            right.basis           = lp.basis;
            right.parentBranchVar = branchId;
            right.branchedUp      = true;
            right.parentFrac      = fracXj;
            pushNode(std::move(right));
        }
    }

done:
    MILPResult result;
    result.nodesExplored = nodesExplored;
    result.cutsAdded     = cutsAdded;

    if (unboundedHit) {
        result.status         = MILPStatus::Unbounded;
        result.objectiveValue = minimize ? -inf : inf;
        return result;
    }

    if (!incumbentSol.empty()) {
        if (timeLimitHit)
            result.status = MILPStatus::TimeLimit;
        else if (maxNodesHit)
            result.status = MILPStatus::MaxNodes;
        else
            result.status = MILPStatus::Optimal;
        result.objectiveValue = incumbent;
        result.primalValues   = std::move(incumbentSol);
    } else {
        if (timeLimitHit)
            result.status = MILPStatus::TimeLimit;
        else if (maxNodesHit)
            result.status = MILPStatus::MaxNodes;
        else
            result.status = MILPStatus::Infeasible;
        result.objectiveValue = minimize ? inf : -inf;
    }

    return result;
}

} // namespace baguette
