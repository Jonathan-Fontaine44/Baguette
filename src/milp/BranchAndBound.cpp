#include "baguette/milp/BranchAndBound.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>
#include <cstdint>

#include "baguette/core/Sense.hpp"
#include "baguette/cp/CPConstraints.hpp"
#include "baguette/lp/LPResult.hpp"
#include "baguette/lp/LPSolver.hpp"
#include "baguette/milp/Presolve.hpp"
#include "milp/cuts/gmi.hpp"
#include "milp/cuts/mir.hpp"
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

// Persistent linked-list node for the bound-change trail.
// Children share the parent's prefix; only their own BoundChange is appended.
// O(1) child creation; O(depth) pointer traversal in restoreBounds (no copy).
struct ChangesNode {
    std::shared_ptr<ChangesNode> parent;
    BoundChange change;

    ~ChangesNode() {
        // Iterative cleanup prevents stack overflow on deep B&B trees where the
        // recursive default destructor would overflow the call stack.
        auto p = std::move(parent);
        while (p && p.use_count() == 1) {
            auto next = std::move(p->parent);
            p = std::move(next);
        }
    }
};

struct Node {
    std::shared_ptr<ChangesNode> changesHead; // nullptr = root (no changes)
    BasisRecord                        basis;
    double                             lpBound;
    int                                depth;

    // Pseudo-cost bookkeeping: records the branching decision that created this node.
    int    parentBranchVar = -1; // -1 = root node (no parent branch)
    bool   branchedUp      = false;
    double parentFrac      = 0.0;
};

// ── Helpers ────────────────────────────────────────────────────────────────────

std::vector<uint32_t> collectIntVarIds(const Model& model) {
    const auto& types = model.getCold().types;
    const uint32_t lpVarCount = static_cast<uint32_t>(model.numVars());
    std::vector<uint32_t> ids;
    ids.reserve(lpVarCount);
    for (uint32_t i = 0; i < lpVarCount; ++i) {
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

    // ── MILP presolve (opt-in, applied once at the root) ──────────────────────
    // Interleaves LP bound-tightening with integer bound rounding — not called
    // for LP relaxation nodes; only at the B&B root on the integer model.
    std::optional<MILPPresolveResult> presolveStat;
    if (opts.enablePresolve) {
        MILPPresolveResult pr = presolveMILPInPlace(model, opts.milpPresolveMaxCycles,
                                                    opts.intFeasTol,
                                                    opts.timeLimitS, startTime);
        presolveStat = pr;
        if (pr.infeasible) {
            MILPResult result;
            result.status       = MILPStatus::Infeasible;
            result.presolveStat = presolveStat;
            return result;
        }
    }

    // ── Elimination presolve (opt-in; skipped when enablePresolve is false) ───
    EliminationRecord elimRec;
    const bool elimApplied = opts.enablePresolve && opts.enableElimination;
    if (elimApplied) {
        const CPConstraints cpSaved = model.getCPConstraints();
        model = presolveElim(model, elimRec);
        if (elimRec.infeasible) {
            MILPResult result;
            result.status       = MILPStatus::Infeasible;
            result.presolveStat = presolveStat;
            return result;
        }
        presolveElimCP(cpSaved, elimRec, model);
    }

    const CPConstraints& cp = model.getCPConstraints();

    const bool   minimize = (model.getObjSense() == ObjSense::Minimize);
    const double inf      = std::numeric_limits<double>::infinity();

    // When all objective coefficients are integer-valued, the IP optimal is
    // integer too. LP bounds that land slightly below (e.g. 9.9999997 instead
    // of 10.0) due to FP noise then defeat canPrune after the first incumbent
    // is found. Round them toward the integer optimum before every comparison.
    const auto& objVec = model.getHot().obj;
    const bool integerObj = std::all_of(objVec.begin(), objVec.end(), [&](double c) {
        return std::abs(c - std::round(c)) <= opts.intFeasTol;
    });
    auto effectiveBound = [&](double v) -> double {
        if (!integerObj) return v;
        return minimize ? std::ceil(v - opts.intFeasTol)
                        : std::floor(v + opts.intFeasTol);
    };

    const std::vector<uint32_t> intIds = collectIntVarIds(model);

    // ── Pseudo-costs ───────────────────────────────────────────────────────────
    PseudoCosts pc = initPseudoCosts(static_cast<uint32_t>(model.numVars()));

    // ── Incumbent tracking ─────────────────────────────────────────────────────
    double              incumbent    = minimize ? inf : -inf;
    std::vector<double> incumbentSol;

    uint32_t nodesExplored = 0;
    uint32_t cutsAdded     = 0;
    BBStats  stats_acc;
    bool     timeLimitHit  = false;
    bool     maxNodesHit   = false;
    bool     unboundedHit  = false;

    // ── Time helpers ───────────────────────────────────────────────────────────
    auto elapsedS = [&]() -> double {
        using Dur = std::chrono::duration<double>;
        return std::chrono::duration_cast<Dur>(SolverClock::now() - startTime).count();
    };

    // ── Pruning predicate ──────────────────────────────────────────────────────
    // gap = max(mipGapAbs, mipGapRel × |incumbent|).
    // The relative term is only applied when both mipGapRel > 0 and the
    // incumbent is finite (avoids 0 × ∞ = NaN before the first incumbent).
    auto canPrune = [&](double lpBound) -> bool {
        const double gap = (opts.mipGapRel > 0.0 && std::isfinite(incumbent))
            ? std::max(opts.mipGapAbs, opts.mipGapRel * std::abs(incumbent))
            : opts.mipGapAbs;
        return minimize ? (lpBound >= incumbent - gap)
                        : (lpBound <= incumbent + gap);
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
    // dirtyVars lists variables that currently differ from rootLb/rootUb.
    // isDirty[id] is the O(1) membership guard that prevents duplicates without
    // sorting: restoreBounds is O(d) and CP merge is O(|changed|).
    const std::vector<double> rootLb = model.getHot().lb;
    const std::vector<double> rootUb = model.getHot().ub;
    std::vector<uint32_t>     dirtyVars;
    std::vector<bool>         isDirty(model.numTotalVars(), false);

    // ── Restore model bounds from a node's accumulated delta trail ─────────────
    auto restoreBounds = [&](const Node& node) {
        for (uint32_t id : dirtyVars) {
            model.setVarBounds(Variable{id}, rootLb[id], rootUb[id]);
            isDirty[id] = false;
        }
        dirtyVars.clear();
        // Collect path leaf→root, apply root→leaf so the deepest change wins.
        std::vector<const BoundChange*> path;
        for (const ChangesNode* p = node.changesHead.get(); p; p = p->parent.get())
            path.push_back(&p->change);
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            const uint32_t id = (*it)->varId;
            model.setVarBounds(Variable{id}, (*it)->newLb, (*it)->newUb);
            if (!isDirty[id]) { isDirty[id] = true; dirtyVars.push_back(id); }
        }
    };

    // ── Root node ──────────────────────────────────────────────────────────────
    {
        Node root;
        root.changesHead = nullptr;
        root.lpBound     = minimize ? -inf : inf;
        root.depth       = 0;
        pushNode(std::move(root));
    }

    // ── Main B&C loop ──────────────────────────────────────────────────────────
    while (!queue.empty()) {
        if (elapsedS() >= opts.timeLimitS) { timeLimitHit = true; break; }
        if (opts.maxNodes > 0 && nodesExplored >= opts.maxNodes) { maxNodesHit = true; break; }

        // ── HybridPlunge cap ───────────────────────────────────────────────────
        // If DFS has explored maxPlungeNodes without finding an incumbent,
        // abort the plunge and switch to BestBound. Prevents indefinite DFS
        // in degenerate LP sub-trees where no integer vertex is reachable via
        // the current pivot path (e.g. PrimalSimplex+Dantzig on TSP-like LPs).
        if (plunging && opts.maxPlungeNodes > 0 && nodesExplored >= opts.maxPlungeNodes) {
            plunging = false;
            std::make_heap(queue.begin(), queue.end(), cmpNodes);
        }

        Node node = popNode();

        // ── Pre-LP bound prune ─────────────────────────────────────────────────
        // The child LP bound >= parent LP bound (branching only tightens the
        // feasible region). If the parent bound already certifies pruning, skip
        // the LP solve entirely. Critical for problems where LP relaxation value
        // equals the IP optimal: after the first incumbent is found, every
        // pending node with lpBound == incumbent is pruned without an LP solve.
        if (canPrune(effectiveBound(node.lpBound))) {
            if (opts.collectStats) ++stats_acc.nodesPrunedByBound;
            continue;
        }

        ++nodesExplored;

        // Apply this node's accumulated bound deltas to the shared model.
        restoreBounds(node);

        // ── CP propagation (before LP: tighten bounds, prune without LP solve) ──
        if (!cp.empty()) {
            bool cpInfeasible = false;
            bool anyChanged   = true;
            while (anyChanged) {
                PropagationResult pr = propagateCP(cp, model);
                anyChanged = !pr.changedVarIds.empty() && opts.cpPropagateToFixpoint;

                // Add CP-tightened vars to dirtyVars so restoreBounds() resets them.
                // isDirty guards against duplicates in O(1) per var — no sort needed.
                for (uint32_t id : pr.changedVarIds)
                    if (!isDirty[id]) { isDirty[id] = true; dirtyVars.push_back(id); }
                if (pr.status == CPStatus::Infeasible) {
                    cpInfeasible = true;
                    break;
                }
            }
            if (cpInfeasible) {
                if (opts.collectStats) ++stats_acc.nodesPrunedByInfeasibility;
                continue; // node pruned by CP — no LP solve needed
            }
        }

        // ── Lightweight integer bound rounding O(V_int) — no LP needed ────────
        // After branching (and CP propagation), integer variable bounds may be
        // fractional (e.g., CP tightened x.lb to 2.3 → ceil to 3).  Snap them
        // here to detect cheap infeasibility and tighten the LP relaxation
        // before the solve.  New dirty vars are merged into dirtyVars so that
        // restoreBounds() resets them at the next node.
        {
            bool intInfeasible = false;
            const std::size_t oldSize = dirtyVars.size();
            for (uint32_t id : intIds) {
                const double lb    = model.getHot().lb[id];
                const double ub    = model.getHot().ub[id];
                const double newLb = (lb == -inf) ? lb : std::ceil(lb  - opts.intFeasTol);
                const double newUb = (ub ==  inf) ? ub : std::floor(ub + opts.intFeasTol);
                if (newLb > newUb + opts.intFeasTol) { intInfeasible = true; break; }
                if (newLb != lb || newUb != ub) {
                    model.setVarBounds(Variable{id}, newLb, newUb);
                    dirtyVars.push_back(id);
                }
            }
            if (intInfeasible) {
                if (opts.collectStats) ++stats_acc.nodesPrunedByInfeasibility;
                continue;
            }
            if (dirtyVars.size() > oldSize) {
                // intIds is sorted → the new suffix is already sorted; merge.
                std::inplace_merge(dirtyVars.begin(),
                                   dirtyVars.begin() + static_cast<std::ptrdiff_t>(oldSize),
                                   dirtyVars.end());
                dirtyVars.erase(std::unique(dirtyVars.begin(), dirtyVars.end()),
                                dirtyVars.end());
            }
        }

        // ── First LP solve ─────────────────────────────────────────────────────
        if (opts.collectStats) ++stats_acc.lpSolvesTotal;
        // Effective root method: rootMethod if set, else lpOpts.method.
        // Effective node method: nodeMethod if set, else effective root method.
        const LPMethod effRoot = (opts.rootMethod != LPMethod::Auto)
                                     ? opts.rootMethod : opts.lpOpts.method;
        const LPMethod effNode = (opts.nodeMethod != LPMethod::Auto)
                                     ? opts.nodeMethod : effRoot;
        LPOptions lpOpts           = opts.lpOpts;
        lpOpts.method              = (node.depth == 0) ? effRoot : effNode;
        lpOpts.timeLimitS          = opts.timeLimitS;
        lpOpts.startTime           = startTime;
        lpOpts.warmBasis           = node.basis;
        lpOpts.computeCutData      = opts.enableCuts || !opts.cutGenerators.empty();
        lpOpts.enablePresolve      = false; // root presolve was already applied
        LPDetailedResult lp = solveLPDetailed(model, lpOpts);

        // ── Handle LP outcome ──────────────────────────────────────────────────
        switch (lp.result.status) {
            case LPStatus::Infeasible:
                if (opts.collectStats) ++stats_acc.nodesPrunedByInfeasibility;
                continue;
            case LPStatus::Unbounded:       unboundedHit = true; goto done;
            case LPStatus::TimeLimit:       timeLimitHit = true; goto done;
            case LPStatus::NumericalFailure: continue;
            case LPStatus::MaxIter:         continue;
            case LPStatus::Optimal:         break;
        }

        // Warm-start fallback: parent provided a basis but dual simplex fell
        // back to cold primal (sfCache mismatch or dual-feasibility failure).
        if (opts.collectStats && !node.basis.basicCols.empty() && !lp.usedWarmStart)
            ++stats_acc.warmStartFallbacks;

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
        if (canPrune(effectiveBound(lp.result.objectiveValue))) {
            if (opts.collectStats) ++stats_acc.nodesPrunedByBound;
            continue;
        }

        // ── GMI cut generation ─────────────────────────────────────────────────
        // Skip if the global budget (maxTotalCuts) is exhausted.
        const bool budgetOk = opts.maxTotalCuts == 0 || cutsAdded < opts.maxTotalCuts;
        if (opts.enableCuts && !lp.fractionalRows.empty() && budgetOk) {
            // At the root use maxRootCuts (default: maxTotalCuts/10); elsewhere
            // use maxCutsPerNode.  Then cap both against the remaining global budget.
            const bool isRootCut = (node.depth == 0);
            uint32_t perNode = (isRootCut && opts.maxRootCuts > 0)
                                   ? opts.maxRootCuts
                                   : opts.maxCutsPerNode;
            if (opts.maxTotalCuts > 0) {
                const uint32_t remaining = opts.maxTotalCuts - cutsAdded;
                if (perNode == 0 || perNode > remaining) perNode = remaining;
            }
            std::vector<Cut> cuts = generateGMICuts(
                lp.fractionalRows, lp.basis, model,
                perNode, opts.intFeasTol);

            if (!cuts.empty()) {
                for (const Cut& c : cuts)
                    model.addLPConstraint(c.expr, c.sense, c.rhs);
                cutsAdded += static_cast<uint32_t>(cuts.size());

                if (opts.collectStats) {
                    ++stats_acc.nodesWithCuts;
                    const auto d = static_cast<uint32_t>(node.depth);
                    if (d >= stats_acc.cutsPerDepth.size())
                        stats_acc.cutsPerDepth.resize(d + 1, 0);
                    stats_acc.cutsPerDepth[d] += static_cast<uint32_t>(cuts.size());
                }

                // Re-solve with the new cuts.  The warm basis is passed as {} (cold
                // start) because addLPConstraint() changed the model structure: the
                // sfCache stored in lp.basis refers to the pre-cut standard form and
                // its dimension no longer matches the current model.
                //
                // Side-effect on sibling nodes: nodes already queued (created before
                // this cut was added) also carry a pre-cut BasisRecord.  When they are
                // eventually popped, solveDualDetailed() detects the sfCache dimension
                // mismatch and silently falls back to a cold primal solve as well.
                // This is correct — their basis is simply stale — but it means that
                // cut addition degrades warm-start reuse for all queued siblings.
                if (opts.collectStats) ++stats_acc.lpSolvesTotal;
                LPOptions lpOptsCold       = opts.lpOpts;
                lpOptsCold.method          = (node.depth == 0) ? effRoot : effNode;
                lpOptsCold.timeLimitS      = opts.timeLimitS;
                lpOptsCold.startTime       = startTime;
                lpOptsCold.enablePresolve  = false;
                lp = solveLPDetailed(model, lpOptsCold);

                switch (lp.result.status) {
                    case LPStatus::Infeasible:
                        if (opts.collectStats) ++stats_acc.nodesPrunedByInfeasibility;
                        continue;
                    case LPStatus::Unbounded:        unboundedHit = true; goto done;
                    case LPStatus::TimeLimit:        timeLimitHit = true; goto done;
                    case LPStatus::NumericalFailure: continue;
                    case LPStatus::MaxIter:          continue;
                    case LPStatus::Optimal:          break;
                }

                if (canPrune(effectiveBound(lp.result.objectiveValue))) {
                    if (opts.collectStats) ++stats_acc.nodesPrunedByBound;
                    continue;
                }
            }
        }

        // ── MIR / CMIR cuts ───────────────────────────────────────────────────
        if (opts.enableMIR && lp.result.status == LPStatus::Optimal) {
            const bool budgetOk = opts.maxTotalCuts == 0 || cutsAdded < opts.maxTotalCuts;
            if (budgetOk) {
                uint32_t remaining = (opts.maxTotalCuts > 0)
                                         ? opts.maxTotalCuts - cutsAdded : 0;
                uint32_t perNode = (opts.maxCutsPerNode > 0)
                                       ? opts.maxCutsPerNode : 0;
                uint32_t cap = (remaining > 0 && (perNode == 0 || perNode > remaining))
                                   ? remaining : perNode;

                std::vector<Cut> mirCuts = generateMIRCuts(lp, model, cap, opts.intFeasTol);
                const uint32_t mirCap = (cap > 0 && mirCuts.size() > cap)
                                            ? cap : static_cast<uint32_t>(mirCuts.size());
                mirCuts.resize(mirCap);

                if (cap > mirCap) {
                    uint32_t cmirCap = cap - mirCap;
                    auto cmir = generateCMIRCuts(lp, model, cmirCap, opts.intFeasTol);
                    for (auto& c : cmir) mirCuts.push_back(std::move(c));
                    if (mirCuts.size() > cap) mirCuts.resize(cap);
                }

                if (!mirCuts.empty()) {
                    for (const Cut& c : mirCuts)
                        model.addLPConstraint(c.expr, c.sense, c.rhs);
                    cutsAdded += static_cast<uint32_t>(mirCuts.size());

                    if (opts.collectStats) {
                        ++stats_acc.nodesWithCuts;
                        const auto d = static_cast<uint32_t>(node.depth);
                        if (d >= stats_acc.cutsPerDepth.size())
                            stats_acc.cutsPerDepth.resize(d + 1, 0);
                        stats_acc.cutsPerDepth[d] += static_cast<uint32_t>(mirCuts.size());
                    }

                    if (opts.collectStats) ++stats_acc.lpSolvesTotal;
                    LPOptions lpOptsCold      = opts.lpOpts;
                    lpOptsCold.method         = (node.depth == 0) ? effRoot : effNode;
                    lpOptsCold.timeLimitS     = opts.timeLimitS;
                    lpOptsCold.startTime      = startTime;
                    lpOptsCold.enablePresolve = false;
                    lp = solveLPDetailed(model, lpOptsCold);

                    switch (lp.result.status) {
                        case LPStatus::Infeasible:
                            if (opts.collectStats) ++stats_acc.nodesPrunedByInfeasibility;
                            continue;
                        case LPStatus::Unbounded:        unboundedHit = true; goto done;
                        case LPStatus::TimeLimit:        timeLimitHit = true; goto done;
                        case LPStatus::NumericalFailure: continue;
                        case LPStatus::MaxIter:          continue;
                        case LPStatus::Optimal:          break;
                    }
                    if (canPrune(effectiveBound(lp.result.objectiveValue))) {
                        if (opts.collectStats) ++stats_acc.nodesPrunedByBound;
                        continue;
                    }
                }
            }
        }

        // ── User cut generators ────────────────────────────────────────────────
        // Called after GMI (if any) on Optimal LPs, independent of enableCuts.
        // All generators are queried first; their cuts are added in one batch
        // and trigger a single cold LP re-solve.
        if (!opts.cutGenerators.empty() && lp.result.status == LPStatus::Optimal) {
            const bool budgetOk = opts.maxTotalCuts == 0 || cutsAdded < opts.maxTotalCuts;
            if (budgetOk) {
                std::vector<Cut> userCuts;
                for (const auto& gen : opts.cutGenerators) {
                    auto batch = gen(lp, model);
                    for (auto& c : batch)
                        userCuts.push_back(std::move(c));
                }
                // Cap against remaining global cut budget.
                if (opts.maxTotalCuts > 0) {
                    const uint32_t remaining = opts.maxTotalCuts - cutsAdded;
                    if (userCuts.size() > remaining)
                        userCuts.resize(remaining);
                }
                if (!userCuts.empty()) {
                    for (const Cut& c : userCuts)
                        model.addLPConstraint(c.expr, c.sense, c.rhs);
                    cutsAdded += static_cast<uint32_t>(userCuts.size());

                    if (opts.collectStats) {
                        ++stats_acc.nodesWithCuts;
                        const auto d = static_cast<uint32_t>(node.depth);
                        if (d >= stats_acc.cutsPerDepth.size())
                            stats_acc.cutsPerDepth.resize(d + 1, 0);
                        stats_acc.cutsPerDepth[d] += static_cast<uint32_t>(userCuts.size());
                    }

                    if (opts.collectStats) ++stats_acc.lpSolvesTotal;
                    LPOptions lpOptsCold      = opts.lpOpts;
                    lpOptsCold.method         = (node.depth == 0) ? effRoot : effNode;
                    lpOptsCold.timeLimitS     = opts.timeLimitS;
                    lpOptsCold.startTime      = startTime;
                    lpOptsCold.enablePresolve = false;
                    lp = solveLPDetailed(model, lpOptsCold);

                    switch (lp.result.status) {
                        case LPStatus::Infeasible:
                            if (opts.collectStats) ++stats_acc.nodesPrunedByInfeasibility;
                            continue;
                        case LPStatus::Unbounded:        unboundedHit = true; goto done;
                        case LPStatus::TimeLimit:        timeLimitHit = true; goto done;
                        case LPStatus::NumericalFailure: continue;
                        case LPStatus::MaxIter:          continue;
                        case LPStatus::Optimal:          break;
                    }
                    if (canPrune(effectiveBound(lp.result.objectiveValue))) {
                        if (opts.collectStats) ++stats_acc.nodesPrunedByBound;
                        continue;
                    }
                }
            }
        }

        // ── Check integer feasibility ──────────────────────────────────────────
        // Extend LP solution with ghost var values (lb==ub) so CP functions can
        // index any variable ID including those beyond the LP variable range.
        std::vector<double> fullSol = lp.result.primalValues;
        if (const std::size_t total = model.numTotalVars(); fullSol.size() < total) {
            const auto& hot = model.getHot();
            fullSol.resize(total);
            for (std::size_t i = lp.result.primalValues.size(); i < total; ++i)
                fullSol[i] = hot.lb[i]; // ghost: lb == ub
        }

        int  branchId;
        bool cpBranch = false; // true: branching to resolve a CP violation, not LP fractionality
        if (opts.branchStrat == BranchStrategy::PseudoCost) {
            branchId = selectBranchVarPseudoCost(
                lp.result.primalValues, intIds, pc, opts.intFeasTol);
        } else {
            branchId = selectBranchVar(
                lp.result.primalValues, intIds, opts.branchStrat, opts.intFeasTol);
        }

        // If the LP solution is all-integer, verify CP constraints before accepting.
        // An LP-optimal point may satisfy variable bounds and LP constraints yet
        // violate a CP constraint (e.g., AllDiff returning x = y = 1 at both lower bounds).
        if (branchId == -1 && !cp.empty()) {
            const uint32_t vid = cpViolatedVar(cp, fullSol, opts.intFeasTol);
            if (vid != std::numeric_limits<uint32_t>::max()) {
                // First-fail: among all unfixed integer variables, branch on the one
                // with the smallest remaining domain (ub − lb). This is the standard
                // MRV heuristic for pure CP problems: tight domains prune faster.
                const auto& hot     = model.getHot();
                int         bestVar = static_cast<int>(vid);
                double      bestSz  = hot.ub[vid] - hot.lb[vid];
                for (uint32_t id : intIds) {
                    double sz = hot.ub[id] - hot.lb[id];
                    if (sz > opts.intFeasTol && sz < bestSz) {
                        bestSz  = sz;
                        bestVar = static_cast<int>(id);
                    }
                }
                branchId = bestVar;
                cpBranch = true; // branch to exclude the conflicting integer value
            }
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
        const double xj = fullSol[static_cast<uint32_t>(branchId)];

        // Read the variable's current bounds from the model (already restored for this node).
        const double curLb = model.getHot().lb[static_cast<uint32_t>(branchId)];
        const double curUb = model.getHot().ub[static_cast<uint32_t>(branchId)];

        // LP-fractional branch: split at floor/ceil of the LP value.
        // CP-violated branch: xj is integer but CP-infeasible. Use domain bisection
        // (mid = floor((lb+ub)/2)) instead of value-exclusion (xj±1): value-exclusion
        // degenerates when the LP is trivial (no constraints, xj==curLb always), because
        // floorX = xj-1 < curLb makes the left child always empty, collapsing the tree
        // into a linear chain that terminates as Infeasible without exploring the domain.
        const double mid    = std::floor((curLb + curUb) / 2.0);
        const double floorX = cpBranch ? mid         : std::floor(xj);
        const double ceilX  = cpBranch ? (mid + 1.0) : std::ceil(xj);
        const double fracXj = cpBranch ? 0.5 : (xj - std::floor(xj));
        const double bound  = effectiveBound(lp.result.objectiveValue);

        // Left child:  x_j ≤ floor(x_j)  [or domain lower half for CP branch]
        // Guard: floorX >= curLb ensures the domain [curLb, floorX] is non-empty.
        if (!canPrune(bound) && floorX >= curLb) {
            auto chg    = std::make_shared<ChangesNode>();
            chg->parent = node.changesHead;
            chg->change = {static_cast<uint32_t>(branchId), curLb, floorX};
            Node left;
            left.changesHead     = std::move(chg);
            left.lpBound         = bound;
            left.depth           = node.depth + 1;
            left.basis           = lp.basis;
            left.parentBranchVar = branchId;
            left.branchedUp      = false;
            left.parentFrac      = fracXj;
            pushNode(std::move(left));
        }

        // Right child: x_j ≥ ceil(x_j)  [or domain upper half for CP branch]
        // Guard: ceilX <= curUb ensures the domain [ceilX, curUb] is non-empty.
        if (!canPrune(bound) && ceilX <= curUb) {
            auto chg    = std::make_shared<ChangesNode>();
            chg->parent = node.changesHead;
            chg->change = {static_cast<uint32_t>(branchId), ceilX, curUb};
            Node right;
            right.changesHead     = std::move(chg);
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

    if (opts.collectStats) {
        stats_acc.nodesExplored = nodesExplored;
        stats_acc.cutsAdded     = cutsAdded;
        result.stats = std::move(stats_acc);
    }

    result.presolveStat = presolveStat;

    if (unboundedHit) {
        result.status         = MILPStatus::Unbounded;
        result.objectiveValue = minimize ? -inf : inf;
        // primalValues is empty → postsolveElim is a no-op, but call for uniformity.
        if (elimApplied) postsolveElim(result, elimRec);
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

    if (elimApplied) postsolveElim(result, elimRec);
    return result;
}

} // namespace baguette
