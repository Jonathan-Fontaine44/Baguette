#include "baguette/milp/BranchAndBound.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "baguette/lp/LPResult.hpp"
#include "baguette/lp/LPSolver.hpp"
#include "baguette/model/ModelEnums.hpp"

namespace baguette {

namespace {

// ── Internal node ──────────────────────────────────────────────────────────────

struct Node {
    std::vector<double> lb;      // complete bounds snapshot for all variables
    std::vector<double> ub;
    BasisRecord         basis;   // warm-start data from the parent LP solve
    double              lpBound; // LP objective at the parent (lower/upper bound)
    int                 depth;
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

// Return the id of the variable to branch on, or -1 if all integer vars are
// at integer values (|x_i - round(x_i)| <= intFeasTol).
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
    // Work on a mutable copy so that setVarBounds() can be called in the hot loop.
    Model model = modelRef;

    const bool   minimize = (model.getObjSense() == ObjSense::Minimize);
    const double inf      = std::numeric_limits<double>::infinity();

    const std::vector<uint32_t> intIds = collectIntVarIds(model);

    // ── Incumbent tracking ─────────────────────────────────────────────────────
    double              incumbent    = minimize ? inf : -inf;
    std::vector<double> incumbentSol;

    int  nodesExplored = 0;
    bool timeLimitHit  = false;
    bool maxNodesHit   = false;
    bool unboundedHit  = false;

    // ── Time helpers ───────────────────────────────────────────────────────────
    auto elapsedS = [&]() -> double {
        using Dur = std::chrono::duration<double>;
        return std::chrono::duration_cast<Dur>(SolverClock::now() - startTime).count();
    };

    // ── Pruning predicate ──────────────────────────────────────────────────────
    // True when an LP bound cannot improve on the current incumbent.
    // Minimize: prune if lpBound >= incumbent (LP lower bound ≥ best known).
    // Maximize: prune if lpBound <= incumbent (LP upper bound ≤ best known).
    auto canPrune = [&](double lpBound) -> bool {
        return minimize ? (lpBound >= incumbent) : (lpBound <= incumbent);
    };

    // ── Node queue comparator (BestBound mode) ─────────────────────────────────
    // std::push_heap / pop_heap use a max-heap: the element for which cmp(x, top)
    // is false for all x is placed at the front (popped first).
    // For Minimize: smallest lpBound first → cmp(a,b) = (a.lpBound > b.lpBound).
    // For Maximize: largest  lpBound first → cmp(a,b) = (a.lpBound < b.lpBound).
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
        root.lpBound = minimize ? -inf : inf; // no LP bound yet for the root
        root.depth   = 0;
        // root.basis is empty → solveDualDetailed will cold-start
        pushNode(std::move(root));
    }

    // ── Main B&B loop ──────────────────────────────────────────────────────────
    while (!queue.empty()) {
        if (elapsedS() >= opts.timeLimitS) { timeLimitHit = true; break; }
        if (opts.maxNodes > 0 && nodesExplored >= opts.maxNodes) { maxNodesHit = true; break; }

        Node node = popNode();
        ++nodesExplored;

        // Apply this node's bounds to the shared model.
        restoreBounds(node);

        // Solve LP relaxation at this node.
        // Pass opts.timeLimitS + startTime so the LP solver shares the global budget.
        LPDetailedResult lp = solveDualDetailed(
            model,
            opts.maxIterLP,
            opts.timeLimitS,
            startTime,
            node.basis);

        // ── Handle LP outcome ──────────────────────────────────────────────────
        switch (lp.result.status) {
            case LPStatus::Infeasible:
                continue; // node is infeasible → prune

            case LPStatus::Unbounded:
                unboundedHit = true;
                goto done; // MILP is also unbounded → abort immediately

            case LPStatus::TimeLimit:
                timeLimitHit = true;
                goto done;

            case LPStatus::NumericalFailure:
                // Skip this node conservatively; do not abort the whole search.
                continue;

            case LPStatus::MaxIter:
                // LP did not converge; skip node (cannot use an unproven bound).
                continue;

            case LPStatus::Optimal:
                break; // handled below
        }

        // ── Prune by bound ─────────────────────────────────────────────────────
        if (canPrune(lp.result.objectiveValue))
            continue;

        // ── Check integer feasibility ──────────────────────────────────────────
        int branchId = selectBranchVar(
            lp.result.primalValues, intIds, opts.branchStrat, opts.intFeasTol);

        if (branchId == -1) {
            // All integer variables are at integer values → feasible solution.
            double obj    = lp.result.objectiveValue;
            bool   better = minimize ? (obj < incumbent) : (obj > incumbent);
            if (better) {
                incumbent    = obj;
                incumbentSol = lp.result.primalValues;
            }
            continue; // nothing to branch on
        }

        // ── Branch ────────────────────────────────────────────────────────────
        const double xj     = lp.result.primalValues[static_cast<uint32_t>(branchId)];
        const double floorX = std::floor(xj);
        const double ceilX  = std::ceil(xj);
        const double bound  = lp.result.objectiveValue;

        // Left child:  x_j ≤ floor(x_j)
        if (!canPrune(bound)) {
            Node left;
            left.lb      = node.lb;
            left.ub      = node.ub;
            left.ub[static_cast<uint32_t>(branchId)] = floorX;
            left.lpBound = bound;
            left.depth   = node.depth + 1;
            left.basis   = lp.basis; // warm-start from current LP solution
            pushNode(std::move(left));
        }

        // Right child: x_j ≥ ceil(x_j)
        if (!canPrune(bound)) {
            Node right;
            right.lb      = node.lb;
            right.ub      = node.ub;
            right.lb[static_cast<uint32_t>(branchId)] = ceilX;
            right.lpBound = bound;
            right.depth   = node.depth + 1;
            right.basis   = lp.basis;
            pushNode(std::move(right));
        }
    }

done:
    // ── Build result ───────────────────────────────────────────────────────────
    MILPResult result;
    result.nodesExplored = nodesExplored;

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
            result.status = MILPStatus::Optimal; // queue exhausted
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
