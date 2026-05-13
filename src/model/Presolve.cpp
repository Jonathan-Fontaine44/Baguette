#include "baguette/model/Presolve.hpp"

#include <chrono>
#include <cmath>
#include <limits>
#include <optional>
#include <variant>

#include "baguette/core/Config.hpp"
#include "baguette/core/LinearExpr.hpp"
#include "baguette/core/Sense.hpp"
#include "baguette/core/Variable.hpp"
#include "baguette/cp/constraints/AllDiff.hpp"
#include "baguette/cp/constraints/Cumulative.hpp"
#include "baguette/model/ModelData.hpp"

namespace baguette {

namespace {

const double kInf = std::numeric_limits<double>::infinity();

bool singlePass(Model& model, uint32_t& boundsTightened, bool& infeasible) {
    const double tol = lp_feasibility_tol;
    bool changed     = false;

    for (const auto& con : model.getLPConstraints()) {
        const auto&       varIds = con.lhs.varIds;
        const auto&       coeffs = con.lhs.coeffs;
        const std::size_t n      = varIds.size();
        if (n == 0) continue;

        const ModelHot& hot = model.getHot();

        double minFin = 0.0, maxFin = 0.0;
        int    minInf = 0,   maxInf = 0;

        for (std::size_t i = 0; i < n; ++i) {
            const double c  = coeffs[i];
            const double lb = hot.lb[varIds[i]];
            const double ub = hot.ub[varIds[i]];
            if (c > 0.0) {
                if (lb == -kInf) ++minInf; else minFin += c * lb;
                if (ub ==  kInf) ++maxInf; else maxFin += c * ub;
            } else if (c < 0.0) {
                if (ub ==  kInf) ++minInf; else minFin += c * ub;
                if (lb == -kInf) ++maxInf; else maxFin += c * lb;
            }
        }

        if ((con.sense == Sense::LessEq || con.sense == Sense::Equal) &&
            minInf == 0 && minFin > con.rhs + tol) {
            infeasible = true;
            return false;
        }
        if ((con.sense == Sense::GreaterEq || con.sense == Sense::Equal) &&
            maxInf == 0 && maxFin < con.rhs - tol) {
            infeasible = true;
            return false;
        }

        for (std::size_t i = 0; i < n; ++i) {
            const uint32_t id = varIds[i];
            const double   c  = coeffs[i];
            const double lb = hot.lb[id];
            const double ub = hot.ub[id];

            if (c == 0.0) continue;

            if (con.sense == Sense::LessEq || con.sense == Sense::Equal) {
                if (c > 0.0) {
                    const int excl_inf = minInf - (lb == -kInf ? 1 : 0);
                    if (excl_inf == 0) {
                        const double excl  = minFin - (lb == -kInf ? 0.0 : c * lb);
                        const double newUb = (con.rhs - excl) / c;
                        if (newUb < ub - tol) {
                            if (newUb < lb - tol) { infeasible = true; return false; }
                            model.setVarBounds(Variable{id}, lb, newUb);
                            ++boundsTightened;
                            changed = true;
                        }
                    }
                } else {
                    const int excl_inf = minInf - (ub == kInf ? 1 : 0);
                    if (excl_inf == 0) {
                        const double excl  = minFin - (ub == kInf ? 0.0 : c * ub);
                        const double newLb = (con.rhs - excl) / c;
                        if (newLb > lb + tol) {
                            if (newLb > ub + tol) { infeasible = true; return false; }
                            model.setVarBounds(Variable{id}, newLb, ub);
                            ++boundsTightened;
                            changed = true;
                        }
                    }
                }
            }

            if (con.sense == Sense::GreaterEq || con.sense == Sense::Equal) {
                const double lb2 = model.getHot().lb[id];
                const double ub2 = model.getHot().ub[id];

                if (c > 0.0) {
                    const int excl_inf = maxInf - (ub == kInf ? 1 : 0);
                    if (excl_inf == 0) {
                        const double excl  = maxFin - (ub == kInf ? 0.0 : c * ub);
                        const double newLb = (con.rhs - excl) / c;
                        if (newLb > lb2 + tol) {
                            if (newLb > ub2 + tol) { infeasible = true; return false; }
                            model.setVarBounds(Variable{id}, newLb, ub2);
                            ++boundsTightened;
                            changed = true;
                        }
                    }
                } else {
                    const int excl_inf = maxInf - (lb == -kInf ? 1 : 0);
                    if (excl_inf == 0) {
                        const double excl  = maxFin - (lb == -kInf ? 0.0 : c * lb);
                        const double newUb = (con.rhs - excl) / c;
                        if (newUb < ub2 - tol) {
                            if (newUb < lb2 - tol) { infeasible = true; return false; }
                            model.setVarBounds(Variable{id}, lb2, newUb);
                            ++boundsTightened;
                            changed = true;
                        }
                    }
                }
            }
        }
    }

    return changed;
}

} // namespace

// ── Bound-Tightening Presolve ──────────────────────────────────────────────────

PresolveResult presolveTBInPlace(Model& model, uint32_t maxPasses,
                                 double timeLimitS,
                                 std::chrono::steady_clock::time_point startTime) {
    PresolveResult result;
    bool infeasible = false;

    for (uint32_t pass = 0; maxPasses == 0 || pass < maxPasses; ++pass) {
        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - startTime).count();
        if (elapsed >= timeLimitS) {
            result.timeLimitReached = true;
            break;
        }
        ++result.passesRun;
        const bool changed = singlePass(model, result.boundsTightened, infeasible);
        if (infeasible) {
            result.infeasible = true;
            return result;
        }
        if (!changed) break;
    }

    const auto& hot = model.getHot();
    for (std::size_t i = 0; i < hot.lb.size(); ++i) {
        if (hot.ub[i] - hot.lb[i] <= lp_feasibility_tol)
            ++result.fixedVars;
    }

    return result;
}

std::pair<Model, PresolveResult> presolveTB(const Model& model, uint32_t maxPasses,
                                            double timeLimitS,
                                            std::chrono::steady_clock::time_point startTime) {
    Model copy        = model;
    PresolveResult pr = presolveTBInPlace(copy, maxPasses, timeLimitS, startTime);
    return {std::move(copy), pr};
}

// ── Elimination Presolve ───────────────────────────────────────────────────────

Model presolveElim(const Model& orig, EliminationRecord& rec) {
    const ModelHot&  hot  = orig.getHot();
    const ModelCold& cold = orig.getCold();
    const double     tol  = lp_feasibility_tol;

    const auto nVars = static_cast<uint32_t>(orig.numVars());
    const auto nCons = static_cast<uint32_t>(orig.numConstraints());

    rec.origVarCount        = nVars;
    rec.origConstraintCount = nCons;
    rec.varMap.assign(nVars, UINT32_MAX);
    rec.conMap.assign(nCons, UINT32_MAX);
    rec.objAdjustment  = 0.0;
    rec.varsEliminated = 0;
    rec.rowsEliminated = 0;
    rec.fixedVars.clear();
    rec.reducedToOrig.clear();
    rec.reducedToOrigCon.clear();

    // ── Step 1: Identify fixed variables ──────────────────────────────────────
    std::vector<double> fixedVal(nVars, std::numeric_limits<double>::quiet_NaN());

    uint32_t nextReducedVar = 0;
    for (uint32_t j = 0; j < nVars; ++j) {
        if (hot.ub[j] - hot.lb[j] <= tol) {
            double v    = hot.lb[j];
            fixedVal[j] = v;
            rec.fixedVars.push_back({j, v});
            rec.objAdjustment += hot.obj[j] * v;
            ++rec.varsEliminated;
        } else {
            rec.varMap[j] = nextReducedVar++;
            rec.reducedToOrig.push_back(j);
        }
    }

    // ── Step 2: Add non-fixed variables to the reduced model ──────────────────
    Model reduced;
    std::vector<Variable> newVars;
    newVars.reserve(rec.reducedToOrig.size());
    for (uint32_t origId : rec.reducedToOrig) {
        Variable v = reduced.addVar(hot.lb[origId], hot.ub[origId],
                                    cold.types[origId], cold.labels[origId]);
        newVars.push_back(v);
    }

    // ── Step 3: Pre-compute adjusted RHS using the variable→constraint index ─────
    std::vector<double> adjustedRHSVec(nCons);
    for (uint32_t i = 0; i < nCons; ++i)
        adjustedRHSVec[i] = orig.getLPConstraints()[i].rhs;

    for (const auto& [j, val] : rec.fixedVars) {
        for (const VarLPEntry& e : cold.varToLP[j]) {
            const double c = orig.getLPConstraints()[e.conIdx].lhs.coeffs[e.termIdx];
            adjustedRHSVec[e.conIdx] -= c * val;
        }
    }

    // ── Build reduced constraints, eliminate redundant rows ────────────────────
    for (uint32_t i = 0; i < nCons; ++i) {
        const Constraint& con = orig.getLPConstraints()[i];
        const double adjustedRHS = adjustedRHSVec[i];

        double minFin = 0.0, maxFin = 0.0;
        int    minInf = 0,   maxInf = 0;
        LinearExpr newLHS;

        for (std::size_t k = 0; k < con.lhs.varIds.size(); ++k) {
            const uint32_t origId = con.lhs.varIds[k];
            const uint32_t rid    = rec.varMap[origId];
            if (rid == UINT32_MAX) continue;

            const double c  = con.lhs.coeffs[k];
            newLHS.addTerm(newVars[rid], c);
            const double lb = hot.lb[origId];
            const double ub = hot.ub[origId];
            if (c > 0.0) {
                if (lb == -kInf) ++minInf; else minFin += c * lb;
                if (ub ==  kInf) ++maxInf; else maxFin += c * ub;
            } else if (c < 0.0) {
                if (ub ==  kInf) ++minInf; else minFin += c * ub;
                if (lb == -kInf) ++maxInf; else maxFin += c * lb;
            }
        }

        const bool leqRedundant =
            (con.sense == Sense::LessEq || con.sense == Sense::Equal) &&
            (maxInf == 0) && (maxFin <= adjustedRHS + tol);
        const bool geqRedundant =
            (con.sense == Sense::GreaterEq || con.sense == Sense::Equal) &&
            (minInf == 0) && (minFin >= adjustedRHS - tol);

        bool redundant = false;
        switch (con.sense) {
            case Sense::LessEq:    redundant = leqRedundant; break;
            case Sense::GreaterEq: redundant = geqRedundant; break;
            case Sense::Equal:     redundant = (leqRedundant && geqRedundant); break;
        }

        if (!redundant) {
            rec.conMap[i] = static_cast<uint32_t>(rec.reducedToOrigCon.size());
            rec.reducedToOrigCon.push_back(i);
            reduced.addLPConstraint(std::move(newLHS), con.sense, adjustedRHS);
        } else {
            ++rec.rowsEliminated;
        }
    }

    // ── Step 4: Set objective on the reduced model ────────────────────────────
    LinearExpr newObj;
    for (uint32_t k = 0; k < static_cast<uint32_t>(newVars.size()); ++k) {
        const double c = hot.obj[rec.reducedToOrig[k]];
        if (c != 0.0) newObj.addTerm(newVars[k], c);
    }
    newObj.constant = orig.getObjConstant();
    reduced.setObjective(std::move(newObj), orig.getObjSense());

    // ── Step 5: Add ghost vars for fixed variables (CP-only, lb == ub) ────────
    rec.lpVarCount = static_cast<uint32_t>(rec.reducedToOrig.size());
    for (const auto& [j, val] : rec.fixedVars) {
        rec.varMap[j] = static_cast<uint32_t>(reduced.numTotalVars());
        reduced.addGhostVar(val, cold.types[j], cold.labels[j]);
    }

    return reduced;
}

// ── CP-aware Elimination ───────────────────────────────────────────────────────

static std::optional<BuiltinConstraint> reduce(
    const AllDiffConstraint& con,
    const std::vector<uint32_t>& varMap)
{
    std::vector<Variable> newVars;
    newVars.reserve(con.vars.size());
    for (const Variable& v : con.vars)
        newVars.push_back(Variable{varMap[v.id]});
    if (newVars.size() < 2) return std::nullopt;
    return AllDiffConstraint{std::move(newVars)};
}

static std::optional<BuiltinConstraint> reduce(
    const CumulativeConstraint& con,
    const std::vector<uint32_t>& varMap)
{
    std::vector<Task> newTasks;
    newTasks.reserve(con.tasks.size());
    for (const Task& t : con.tasks)
        newTasks.push_back({Variable{varMap[t.varStart.id]}, t.duration, t.consumption});
    if (newTasks.empty()) return std::nullopt;
    return CumulativeConstraint{std::move(newTasks), con.capacity};
}

static std::optional<BuiltinConstraint> reduce(
    const BuiltinConstraint&     bc,
    const std::vector<uint32_t>& varMap)
{
    return std::visit([&](const auto& con) { return reduce(con, varMap); }, bc);
}

std::optional<BuiltinConstraint> reduce(const BuiltinConstraint& bc,
                                        const EliminationRecord& rec) {
    return reduce(bc, rec.varMap);
}

void presolveElimCP(const CPConstraints& cpOrig,
                    const EliminationRecord& rec,
                    Model& reduced) {
    if (cpOrig.empty()) return;
    for (const BuiltinConstraint& bc : cpOrig.builtins()) {
        auto r = reduce(bc, rec.varMap);
        if (r) reduced.addCPConstraint(std::move(*r));
    }
    for (const auto& c : cpOrig.customs())
        if (auto r = c->reduce(rec.varMap))
            reduced.addCPConstraint(std::move(r));
}

} // namespace baguette
