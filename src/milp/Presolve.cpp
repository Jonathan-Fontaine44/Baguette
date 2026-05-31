#include "baguette/milp/Presolve.hpp"

#include <chrono>
#include <cmath>
#include <limits>
#include <unordered_set>
#include <vector>

#include "baguette/core/LinearExpr.hpp"
#include "baguette/core/Sense.hpp"
#include "baguette/cp/CPConstraints.hpp"
#include "baguette/lp/LPSolver.hpp"
#include "baguette/model/ModelData.hpp"
#include "baguette/model/ModelEnums.hpp"

namespace baguette {

void postsolveElim(MILPResult& r, const EliminationRecord& rec) {
    if (rec.varsEliminated == 0 && rec.rowsEliminated == 0) return;
    if (r.primalValues.empty()) return;

    r.objectiveValue += rec.objAdjustment;

    std::vector<double> full(rec.origVarCount, 0.0);
    for (uint32_t k = 0; k < static_cast<uint32_t>(rec.reducedToOrig.size()); ++k)
        if (k < r.primalValues.size())
            full[rec.reducedToOrig[k]] = r.primalValues[k];
    for (const auto& [origId, val] : rec.fixedVars)
        full[origId] = val;
    r.primalValues = std::move(full);
}

// ── presolveMILPInPlace ────────────────────────────────────────────────────────

MILPPresolveResult presolveMILPInPlace(
    Model&                  model,
    const MILPPresolveOpts& opts,
    std::chrono::steady_clock::time_point startTime)
{
    MILPPresolveResult res;
    if (opts.level == 0) return res;

    const double   kIntTol = opts.intFeasTol;
    constexpr double kInf  = std::numeric_limits<double>::infinity();
    const auto& types      = model.getCold().types;
    const uint32_t n       = static_cast<uint32_t>(model.numVars());

    auto elapsed = [&]() -> double {
        return std::chrono::duration<double>(
            std::chrono::steady_clock::now() - startTime).count();
    };

    // ── Snap integer/binary bounds to nearest integer ──────────────────────────
    // Returns false if any domain becomes empty (infeasible).
    auto roundIntBounds = [&]() -> bool {
        for (uint32_t i = 0; i < n; ++i) {
            const VarType t = types[i];
            if (t != VarType::Integer && t != VarType::Binary) continue;
            const double lb    = model.getHot().lb[i];
            const double ub    = model.getHot().ub[i];
            const double newLb = (lb == -kInf) ? lb : std::ceil(lb  - kIntTol);
            const double newUb = (ub ==  kInf) ? ub : std::floor(ub + kIntTol);
            if (newLb > newUb + kIntTol) return false;
            if (newLb != lb || newUb != ub) {
                model.setVarBounds(Variable{i}, newLb, newUb);
                ++res.boundsRounded;
            }
        }
        return true;
    };

    // ── One LP-propagation pass + integer rounding ─────────────────────────────
    // Returns false if infeasible or time limit hit (res.infeasible /
    // res.timeLimitReached are set accordingly).
    auto runLPRoundCycle = [&]() -> bool {
        PresolveResult pr = presolveTBInPlace(model, 1, opts.timeLimitS, startTime);
        res.passesRun       += pr.passesRun;
        res.boundsTightened += pr.boundsTightened;
        if (pr.infeasible)       { res.infeasible = true;       return false; }
        if (pr.timeLimitReached) { res.timeLimitReached = true;  return false; }
        if (!roundIntBounds())   { res.infeasible = true;       return false; }
        return true;
    };

    // ── LP+round outer loop to fixpoint ───────────────────────────────────────
    // Used by multiple levels; factored as a lambda to avoid repetition.
    auto runToFixpoint = [&]() -> bool {
        for (uint32_t outer = 0; opts.maxCycles == 0 || outer < opts.maxCycles; ++outer) {
            if (elapsed() >= opts.timeLimitS) { res.timeLimitReached = true; return false; }
            const uint32_t prevTightened = res.boundsTightened;
            const uint32_t prevRounded   = res.boundsRounded;
            if (!runLPRoundCycle()) return false;
            if (res.boundsTightened == prevTightened && res.boundsRounded == prevRounded) break;
        }
        return true;
    };

    // ── Count fixed variables (lb == ub) ──────────────────────────────────────
    auto countFixed = [&]() {
        const auto& hot = model.getHot();
        for (uint32_t i = 0; i < n; ++i)
            if (hot.ub[i] - hot.lb[i] <= kIntTol) ++res.fixedVars;
    };

    // ── Level 1: PR1 + LP bound-tightening + integer rounding ─────────────────

    // Initial integer rounding before LP propagation.
    if (!roundIntBounds()) { res.infeasible = true; return res; }

    // PR1 - round RHS of all-integer constraints once before the outer loop.
    // @note Complexity O(C × N) over all constraints.
    {
        const auto& cons = model.getLPConstraints();
        for (uint32_t ci = 0; ci < static_cast<uint32_t>(cons.size()); ++ci) {
            const LPConstraint& con = cons[ci];
            if (con.sense == Sense::Equal) continue;
            if (con.lhs.varIds.empty()) continue;
            bool eligible = true;
            for (std::size_t k = 0; k < con.lhs.varIds.size(); ++k) {
                const VarType t = types[con.lhs.varIds[k]];
                if (t != VarType::Integer && t != VarType::Binary) { eligible = false; break; }
                if (std::abs(con.lhs.coeffs[k] - std::round(con.lhs.coeffs[k])) > kIntTol) {
                    eligible = false; break;
                }
            }
            if (!eligible) continue;
            const double rhs    = con.rhsConst;
            const double newRhs = (con.sense == Sense::LessEq)
                ? std::floor(rhs + kIntTol) : std::ceil(rhs - kIntTol);
            if (std::abs(newRhs - rhs) > kIntTol) {
                model.setConstraintRHS(ci, newRhs);
                ++res.rhsRounded;
            }
        }
    }

    if (!runToFixpoint()) return res;

    if (opts.level == 1) { countFixed(); return res; }

    // ── Level 2: CP fixpoint propagation at root ───────────────────────────────

    {
        const CPConstraints& cp = model.getCPConstraints();
        bool cpChanged = false;
        if (!cp.empty()) {
            bool anyChanged = true;
            while (anyChanged) {
                PropagationResult pr = propagateCP(cp, model);
                anyChanged  = !pr.changedVarIds.empty();
                cpChanged   = cpChanged || anyChanged;
                res.boundsTightened += static_cast<uint32_t>(pr.changedVarIds.size());
                if (pr.status == CPStatus::Infeasible) { res.infeasible = true; return res; }
            }
        }
        if (cpChanged && !runToFixpoint()) return res;
    }

    if (opts.level == 2) { countFixed(); return res; }

    // ── Collect binary-domain variables for probing (levels 3 and 6) ──────────
    std::vector<uint32_t> probeVarIds;
    {
        const auto& hot = model.getHot();
        for (uint32_t i = 0; i < n; ++i) {
            const VarType t = types[i];
            if (t != VarType::Binary && t != VarType::Integer) continue;
            if (hot.ub[i] - hot.lb[i] <= kIntTol) continue;         // already fixed
            if (hot.ub[i] - hot.lb[i] > 1.0 + kIntTol) continue;   // not binary-domain
            probeVarIds.push_back(i);
        }
        if (opts.probingMaxVars > 0 &&
            static_cast<uint32_t>(probeVarIds.size()) > opts.probingMaxVars)
            probeVarIds.resize(opts.probingMaxVars);
    }

    // ── Implication row deduplication (level 5) ───────────────────────────────
    // Encoding: bits 34–63 = min(vi,vj) for symmetric types (0,3), vi for others;
    //           bits  2–33 = max(vi,vj) / vj; bits 0–1 = type.
    std::unordered_set<uint64_t> seenImplRows;
    const bool collectImpl = (opts.level >= 5);

    // Attempt to add one implication row to the model; deduplicates by key.
    auto tryAddImpl = [&](uint32_t vi, uint32_t vj, int type,
                          LinearExpr lhs, Sense sense, double rhs) {
        if (res.impliedRowsAdded >= opts.maxImpliedRows) return;
        // Symmetric types (0 = x_i+x_j>=1, 3 = x_i+x_j<=1): normalise key.
        const uint32_t a = (type == 0 || type == 3) ? std::min(vi, vj) : vi;
        const uint32_t b = (type == 0 || type == 3) ? std::max(vi, vj) : vj;
        const uint64_t key = (static_cast<uint64_t>(a) << 34) |
                             (static_cast<uint64_t>(b) <<  2) |
                             static_cast<uint64_t>(type & 3);
        if (!seenImplRows.insert(key).second) return;
        model.addLPConstraint(std::move(lhs), sense, rhs);
        ++res.impliedRowsAdded;
    };

    // ── Level 3: Weak probing ─────────────────────────────────────────────────
    // Sub-problem options: run levels 1 and 2 only (no further probing).
    MILPPresolveOpts subOpts;
    subOpts.level      = 2;
    subOpts.maxCycles  = opts.maxCycles;
    subOpts.intFeasTol = opts.intFeasTol;
    subOpts.timeLimitS = opts.timeLimitS;

    bool anyProbingFixed = false;

    for (uint32_t vi : probeVarIds) {
        if (elapsed() >= opts.timeLimitS) { res.timeLimitReached = true; break; }
        const double lb_i = model.getHot().lb[vi];
        const double ub_i = model.getHot().ub[vi];
        if (ub_i - lb_i <= kIntTol) continue; // fixed by an earlier probe

        ++res.varsProbed;

        // Probe x_vi = lb_i
        Model copy0 = model;
        copy0.setVarBounds(Variable{vi}, lb_i, lb_i);
        MILPPresolveResult r0 = presolveMILPInPlace(copy0, subOpts, startTime);

        // Probe x_vi = ub_i
        Model copy1 = model;
        copy1.setVarBounds(Variable{vi}, ub_i, ub_i);
        MILPPresolveResult r1 = presolveMILPInPlace(copy1, subOpts, startTime);

        if (r0.infeasible && r1.infeasible) { res.infeasible = true; return res; }

        if (r0.infeasible) {
            model.setVarBounds(Variable{vi}, ub_i, ub_i);
            ++res.varsProbedFixed; anyProbingFixed = true; continue;
        }
        if (r1.infeasible) {
            model.setVarBounds(Variable{vi}, lb_i, lb_i);
            ++res.varsProbedFixed; anyProbingFixed = true; continue;
        }

        // Both feasible: intersect bounds.
        // new_lb[j] = max(cur_lb[j], min(lb0[j], lb1[j]))  (union of feasible sets)
        // new_ub[j] = min(cur_ub[j], max(ub0[j], ub1[j]))
        const auto& hot0 = copy0.getHot();
        const auto& hot1 = copy1.getHot();
        for (uint32_t vj = 0; vj < n; ++vj) {
            const double cur_lb = model.getHot().lb[vj];
            const double cur_ub = model.getHot().ub[vj];
            const double new_lb = std::max(cur_lb, std::min(hot0.lb[vj], hot1.lb[vj]));
            const double new_ub = std::min(cur_ub, std::max(hot0.ub[vj], hot1.ub[vj]));
            if (new_lb > cur_lb + kIntTol || new_ub < cur_ub - kIntTol) {
                if (new_lb > new_ub + kIntTol) { res.infeasible = true; return res; }
                model.setVarBounds(Variable{vj}, new_lb, new_ub);
                ++res.boundsTightened;
            }
        }

        // Level 5: collect binary implication rows from this probe pair.
        if (collectImpl) {
            for (uint32_t vj = 0; vj < n; ++vj) {
                if (vj == vi) continue;
                const VarType tj = types[vj];
                if (tj != VarType::Binary && tj != VarType::Integer) continue;
                const double cur_lb_j = model.getHot().lb[vj];
                const double cur_ub_j = model.getHot().ub[vj];
                // Only binary-domain {lb_j, lb_j+1} variables with lb_j ≈ 0, ub_j ≈ 1.
                if (cur_ub_j - cur_lb_j <= kIntTol) continue;
                if (std::abs(cur_lb_j) > kIntTol || std::abs(cur_ub_j - 1.0) > kIntTol) continue;

                // x_vi = lb_i → x_vj forced to ub_j (= 1): x_vi + x_vj >= lb_i + 1
                if (hot0.lb[vj] >= 1.0 - kIntTol)
                    tryAddImpl(vi, vj, 0,
                        1.0*Variable{vi} + 1.0*Variable{vj}, Sense::GreaterEq, lb_i + 1.0);

                // x_vi = lb_i → x_vj forced to lb_j (= 0): x_vj - x_vi <= -lb_i
                if (hot0.ub[vj] <= kIntTol)
                    tryAddImpl(vi, vj, 1,
                        1.0*Variable{vj} - 1.0*Variable{vi}, Sense::LessEq, -lb_i);

                // x_vi = ub_i → x_vj forced to ub_j (= 1): x_vj - x_vi >= 1 - ub_i
                if (hot1.lb[vj] >= 1.0 - kIntTol)
                    tryAddImpl(vi, vj, 2,
                        1.0*Variable{vj} - 1.0*Variable{vi}, Sense::GreaterEq, 1.0 - ub_i);

                // x_vi = ub_i → x_vj forced to lb_j (= 0): x_vi + x_vj <= ub_i
                if (hot1.ub[vj] <= kIntTol)
                    tryAddImpl(vi, vj, 3,
                        1.0*Variable{vi} + 1.0*Variable{vj}, Sense::LessEq, ub_i);
            }
        }
    }

    // Propagate any variables fixed by probing.
    if (anyProbingFixed && !runToFixpoint()) return res;

    if (opts.level <= 3) { countFixed(); return res; }

    // ── Level 4: Root LP relaxation solve ─────────────────────────────────────
    // Detects LP infeasibility (⟹ MILP infeasibility) before the B&B tree.
    {
        LPOptions lpO      = opts.lpOpts;
        lpO.timeLimitS     = opts.timeLimitS;
        lpO.startTime      = startTime;
        lpO.enablePresolve = false;
        LPDetailedResult lp = solveLPDetailed(model, lpO);
        if (lp.result.status == LPStatus::Infeasible) {
            res.infeasible = true; return res;
        }
    }

    if (opts.level == 4) { countFixed(); return res; }

    // Level 5: implication rows were already injected during the probing loop.
    if (opts.level == 5) { countFixed(); return res; }

    // ── Level 6: Strong probing (LP solve per binary fix) ─────────────────────
    {
        LPOptions spOpts      = opts.lpOpts;
        spOpts.method         = opts.probingLPMethod;
        spOpts.timeLimitS     = opts.timeLimitS;
        spOpts.startTime      = startTime;
        spOpts.enablePresolve = false;

        bool anyStrongFixed = false;

        for (uint32_t vi : probeVarIds) {
            if (elapsed() >= opts.timeLimitS) { res.timeLimitReached = true; break; }
            const double lb_i = model.getHot().lb[vi];
            const double ub_i = model.getHot().ub[vi];
            if (ub_i - lb_i <= kIntTol) continue; // already fixed by earlier levels

            // LP solve with x_vi = lb_i
            Model copy0 = model;
            copy0.setVarBounds(Variable{vi}, lb_i, lb_i);
            LPDetailedResult lp0 = solveLPDetailed(copy0, spOpts);
            const bool infeas0 = (lp0.result.status == LPStatus::Infeasible);

            // LP solve with x_vi = ub_i
            Model copy1 = model;
            copy1.setVarBounds(Variable{vi}, ub_i, ub_i);
            LPDetailedResult lp1 = solveLPDetailed(copy1, spOpts);
            const bool infeas1 = (lp1.result.status == LPStatus::Infeasible);

            if (infeas0 && infeas1) { res.infeasible = true; return res; }

            if (infeas0) {
                model.setVarBounds(Variable{vi}, ub_i, ub_i);
                ++res.varsProbedFixed; anyStrongFixed = true;
            } else if (infeas1) {
                model.setVarBounds(Variable{vi}, lb_i, lb_i);
                ++res.varsProbedFixed; anyStrongFixed = true;
            }
        }

        if (anyStrongFixed && !runToFixpoint()) return res;
    }

    countFixed();
    return res;
}

} // namespace baguette
