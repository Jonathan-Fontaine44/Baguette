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

struct Node {
    std::vector<double> lb;      // complete bounds snapshot for all variables
    std::vector<double> ub;
    BasisRecord         basis;   // warm-start data from the parent LP solve
    double              lpBound; // LP objective at the parent (for pruning + pseudo-costs)
    int                 depth;

    // Pseudo-cost bookkeeping: records the branching decision that created this node.
    int    parentBranchVar = -1; // -1 = root node (no parent branch)
    bool   branchedUp      = false;
    double parentFrac      = 0.0;
};

// ── Helpers ────────────────────────────────────────────────────────────────────

// Collect ids of Integer and Binary variables.
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

// Return the id of the variable to branch on using FirstFractional or MostFractional.
// Returns -1 if all integer vars are at integer values (|x_i - round(x_i)| <= intFeasTol).
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
            continue; // integer-feasible, skip

        if (strat == BranchStrategy::FirstFractional)
            return static_cast<int>(id);

        // MostFractional: maximise min(frac, 1-frac) → closest to 0.5
        double score = std::min(frac, 1.0 - frac);
        if (score > bestScore) {
            bestScore = score;
            bestId    = static_cast<int>(id);
        }
    }
    return bestId;
}

} // namespace

// ── solveMILP ──────────────────────────────────────────────────────────────────

MILPResult solveMILP(const Model&            modelRef,
                     const BBOptions&        opts,
                     SolverClock::time_point startTime) {
    // Work on a mutable copy so that addConstraint() (for cuts) and setVarBounds()
    // can be called in the hot loop.
    Model model = modelRef;

    const bool   minimize = (model.getObjSense() == ObjSense::Minimize);
    const double inf      = std::numeric_limits<double>::infinity();

    const std::vector<uint32_t> intIds = collectIntVarIds(model);

    // ── Pseudo-costs ───────────────────────────────────────────────────────────
    PseudoCosts pc = initPseudoCosts(static_cast<uint32_t>(model.numVars()));

    // ── Incumbent tracking ─────────────────────────────────────────────────────
    double              incumbent    = minimize ? inf : -inf;
    std::vector<double> incumbentSol;

    int      nodesExplored = 0;
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

    // ── Node queue comparator (BestBound mode) ─────────────────────────────────
    auto cmpNodes = [minimize](const Node& a, const Node& b) -> bool {
        return minimize ? (a.lpBound > b.lpBound) : (a.lpBound < b.lpBound);
    };

    // ── Queue (vector used as heap or stack depending on nodeSelect) ───────────
    std::vector<Node> queue;

    auto pushNode = [&](Node n) {
        queue.push_back(std::move(n));
        if (opts.nodeSelect == NodeSelection::BestBound)
            std::push_heap(queue.begin(), queue.end(), cmpNodes);
    };

    auto popNode = [&]() -> Node {
        if (opts.nodeSelect == NodeSelection::BestBound)
            std::pop_heap(queue.begin(), queue.end(), cmpNodes);
        Node n = std::move(queue.back());
        queue.pop_back();
        return n;
    };

    // ── Restore model bounds from a node snapshot ──────────────────────────────
    auto restoreBounds = [&](const Node& node) {
        for (uint32_t i = 0; i < static_cast<uint32_t>(node.lb.size()); ++i)
            model.setVarBounds(Variable{i}, node.lb[i], node.ub[i]);
    };

    // ── Root node ──────────────────────────────────────────────────────────────
    {
        const auto& hot = model.getHot();
        Node root;
        root.lb      = hot.lb;
        root.ub      = hot.ub;
        root.lpBound = minimize ? -inf : inf;
        root.depth   = 0;
        // root.basis is empty → solveDualDetailed will cold-start
        pushNode(std::move(root));
    }

    // ── Main B&C loop ──────────────────────────────────────────────────────────
    while (!queue.empty()) {
        if (elapsedS() >= opts.timeLimitS) { timeLimitHit = true; break; }
        if (opts.maxNodes > 0 && nodesExplored >= opts.maxNodes) { maxNodesHit = true; break; }

        Node node = popNode();
        ++nodesExplored;

        // Apply this node's bounds to the shared model.
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
            case LPStatus::Infeasible:
                continue;
            case LPStatus::Unbounded:
                unboundedHit = true;
                goto done;
            case LPStatus::TimeLimit:
                timeLimitHit = true;
                goto done;
            case LPStatus::NumericalFailure:
                continue;
            case LPStatus::MaxIter:
                continue;
            case LPStatus::Optimal:
                break;
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

                // Re-solve with the new cuts (warm basis incompatible: cold start).
                lp = solveDualDetailed(
                    model,
                    opts.maxIterLP,
                    opts.timeLimitS,
                    startTime,
                    /*warmBasis=*/{},
                    /*computeSensitivity=*/false,
                    /*computeCutData=*/false);

                switch (lp.result.status) {
                    case LPStatus::Infeasible:  continue;
                    case LPStatus::Unbounded:   unboundedHit = true; goto done;
                    case LPStatus::TimeLimit:   timeLimitHit = true; goto done;
                    case LPStatus::NumericalFailure: continue;
                    case LPStatus::MaxIter:     continue;
                    case LPStatus::Optimal:     break;
                }

                // Prune again with the tightened (post-cut) bound.
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
            // All integer variables at integer values → feasible solution.
            double obj    = lp.result.objectiveValue;
            bool   better = minimize ? (obj < incumbent) : (obj > incumbent);
            if (better) {
                incumbent    = obj;
                incumbentSol = lp.result.primalValues;
            }
            continue;
        }

        // ── Branch ────────────────────────────────────────────────────────────
        const double xj      = lp.result.primalValues[static_cast<uint32_t>(branchId)];
        const double floorX  = std::floor(xj);
        const double ceilX   = std::ceil(xj);
        const double fracXj  = xj - floorX;
        const double bound   = lp.result.objectiveValue;

        // Left child:  x_j ≤ floor(x_j)
        if (!canPrune(bound)) {
            Node left;
            left.lb              = node.lb;
            left.ub              = node.ub;
            left.ub[static_cast<uint32_t>(branchId)] = floorX;
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
            right.lb              = node.lb;
            right.ub              = node.ub;
            right.lb[static_cast<uint32_t>(branchId)] = ceilX;
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
    // ── Build result ───────────────────────────────────────────────────────────
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
