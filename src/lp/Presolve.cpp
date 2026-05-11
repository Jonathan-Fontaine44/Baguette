#include "baguette/lp/Presolve.hpp"

#include <chrono>
#include <cmath>
#include <limits>

#include "baguette/core/Config.hpp"
#include "baguette/core/Sense.hpp"
#include "baguette/core/Variable.hpp"
#include "baguette/model/Model.hpp"

namespace baguette {

namespace {

const double kInf = std::numeric_limits<double>::infinity();

// Runs one full pass of bound tightening over all constraints.
// Returns true if at least one bound was updated.
// Sets infeasible=true and returns false on empty-domain detection.
bool singlePass(Model& model, uint32_t& boundsTightened, bool& infeasible) {
    const double tol = lp_feasibility_tol;
    bool changed     = false;

    for (const auto& con : model.getLPConstraints()) {
        const auto&       varIds = con.lhs.varIds;
        const auto&       coeffs = con.lhs.coeffs;
        const std::size_t n      = varIds.size();
        if (n == 0) continue;

        // Re-read bounds at the start of each constraint so that tightenings
        // from previous constraints in this pass are immediately visible.
        const ModelHot& hot = model.getHot();

        // ── Activity bounds ─────────────────────────────────────────────────
        // minActFin / maxActFin: sum of finite contributions.
        // minActInf / maxActInf: count of -inf / +inf contributors.
        //
        // For c > 0: min contribution = c*lb  (−∞ if lb=−∞)
        //            max contribution = c*ub  (+∞ if ub=+∞)
        // For c < 0: min contribution = c*ub  (−∞ if ub=+∞)
        //            max contribution = c*lb  (+∞ if lb=−∞)
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

        // ── Infeasibility check ──────────────────────────────────────────────
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

        // ── Per-variable bound tightening ────────────────────────────────────
        for (std::size_t i = 0; i < n; ++i) {
            const uint32_t id = varIds[i];
            const double   c  = coeffs[i];
            // Snapshot bounds used in all excl formulas for this variable.
            // Reading from hot at this point captures any updates from earlier
            // constraints in this pass but preserves per-variable consistency
            // (each id appears at most once in varIds due to the LinearExpr
            // sorted-unique invariant, so bounds for id have not been updated
            // yet in this constraint's inner loop).
            const double lb = hot.lb[id];
            const double ub = hot.ub[id];

            if (c == 0.0) continue;

            // ── LEQ tightening (sum <= rhs) ──────────────────────────────────
            if (con.sense == Sense::LessEq || con.sense == Sense::Equal) {
                if (c > 0.0) {
                    // Tighten ub: x_k <= (rhs - excl) / c_k
                    // excl = sum of min contributions from all OTHER vars.
                    // k's min contribution: c*lb (finite) or -inf.
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
                    // c < 0: tighten lb: x_k >= (rhs - excl) / c_k  (flip)
                    // k's min contribution: c*ub (finite) or -inf if ub=+inf.
                    const int excl_inf = minInf - (ub == kInf ? 1 : 0);
                    if (excl_inf == 0) {
                        const double excl  = minFin - (ub == kInf ? 0.0 : c * ub);
                        const double newLb = (con.rhs - excl) / c; // c < 0 flips ineq
                        if (newLb > lb + tol) {
                            if (newLb > ub + tol) { infeasible = true; return false; }
                            model.setVarBounds(Variable{id}, newLb, ub);
                            ++boundsTightened;
                            changed = true;
                        }
                    }
                }
            }

            // ── GEQ tightening (sum >= rhs) ──────────────────────────────────
            if (con.sense == Sense::GreaterEq || con.sense == Sense::Equal) {
                // Re-read current bounds: LEQ may have updated them above.
                const double lb2 = model.getHot().lb[id];
                const double ub2 = model.getHot().ub[id];

                if (c > 0.0) {
                    // Tighten lb: x_k >= (rhs - excl) / c_k
                    // excl = sum of max contributions from all OTHER vars.
                    // k's max contribution: c*ub (snapshot, +inf if ub=+inf).
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
                    // c < 0: tighten ub: x_k <= (rhs - excl) / c_k (flip)
                    // k's max contribution: c*lb (snapshot, +inf if lb=-inf).
                    const int excl_inf = maxInf - (lb == -kInf ? 1 : 0);
                    if (excl_inf == 0) {
                        const double excl  = maxFin - (lb == -kInf ? 0.0 : c * lb);
                        const double newUb = (con.rhs - excl) / c; // c < 0 flips ineq
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

PresolveResult presolveInPlace(Model& model, uint32_t maxPasses,
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

    // Count fixed variables (lb == ub within tolerance).
    const auto& hot = model.getHot();
    for (std::size_t i = 0; i < hot.lb.size(); ++i) {
        if (hot.ub[i] - hot.lb[i] <= lp_feasibility_tol)
            ++result.fixedVars;
    }

    return result;
}

std::pair<Model, PresolveResult> presolve(const Model& model, uint32_t maxPasses,
                                          double timeLimitS,
                                          std::chrono::steady_clock::time_point startTime) {
    Model copy        = model;
    PresolveResult pr = presolveInPlace(copy, maxPasses, timeLimitS, startTime);
    return {std::move(copy), pr};
}

} // namespace baguette