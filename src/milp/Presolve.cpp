#include "baguette/milp/Presolve.hpp"

#include <chrono>
#include <cmath>
#include <limits>

#include "baguette/core/Sense.hpp"
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
    Model&   model,
    uint32_t maxCycles,
    double   intFeasTol,
    double   timeLimitS,
    std::chrono::steady_clock::time_point startTime)
{
    const double kIntTol = intFeasTol;
    constexpr double kInf    = std::numeric_limits<double>::infinity();

    MILPPresolveResult res;

    const auto&    types = model.getCold().types;
    const uint32_t n     = static_cast<uint32_t>(model.numVars());

    // Snap lb = ceil(lb), ub = floor(ub) for every Integer/Binary variable.
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

    // Initial integrality pass before LP propagation.
    if (!roundIntBounds()) { res.infeasible = true; return res; }

    // PR1 — Round RHS of all-integer constraints once before the outer loop.
    // For ∑ aᵢ xᵢ ≤ b where all xᵢ are Integer/Binary and all aᵢ are integer-
    // valued, the LHS is always integer at any feasible point, so ⌊b⌋ is a
    // valid tighter RHS (analogously ⌈b⌉ for ≥).  Reported in res.rhsRounded.
    // @note Complexity O(C × N) over all constraints × variables per constraint.
    [&]() {
        const auto& types = model.getCold().types;
        const auto& cons  = model.getLPConstraints();
        for (uint32_t ci = 0; ci < static_cast<uint32_t>(cons.size()); ++ci) {
            const Constraint& con = cons[ci];
            if (con.sense == Sense::Equal) continue;
            if (con.lhs.varIds.empty()) continue;

            bool eligible = true;
            for (std::size_t k = 0; k < con.lhs.varIds.size(); ++k) {
                const VarType t = types[con.lhs.varIds[k]];
                if (t != VarType::Integer && t != VarType::Binary) {
                    eligible = false; break;
                }
                if (std::abs(con.lhs.coeffs[k] - std::round(con.lhs.coeffs[k])) > kIntTol) {
                    eligible = false; break;
                }
            }
            if (!eligible) continue;

            const double rhs    = con.rhs;
            const double newRhs = (con.sense == Sense::LessEq)
                ? std::floor(rhs + kIntTol)
                : std::ceil(rhs  - kIntTol);
            if (std::abs(newRhs - rhs) > kIntTol) {
                model.setConstraintRHS(ci, newRhs);
                ++res.rhsRounded;
            }
        }
    }();

    for (uint32_t outer = 0; maxCycles == 0 || outer < maxCycles; ++outer) {
        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - startTime).count();
        if (elapsed >= timeLimitS) { res.timeLimitReached = true; break; }

        // One LP pass per outer cycle — interleaves LP propagation and integer
        // rounding more tightly, avoiding over-propagation before the next snap.
        PresolveResult pr = presolveTBInPlace(model, 1, timeLimitS, startTime);
        res.passesRun       += pr.passesRun;
        res.boundsTightened += pr.boundsTightened;
        if (pr.infeasible)       { res.infeasible = true; return res; }
        if (pr.timeLimitReached) { res.timeLimitReached = true; break; }

        // Re-apply integrality after each LP round.
        const uint32_t prevRounded = res.boundsRounded;
        if (!roundIntBounds()) { res.infeasible = true; return res; }

        // Fixed point: LP changed nothing AND rounding changed nothing.
        if (pr.boundsTightened == 0 && res.boundsRounded == prevRounded) break;
    }

    const auto& hot = model.getHot();
    for (uint32_t i = 0; i < n; ++i)
        if (hot.ub[i] - hot.lb[i] <= kIntTol)
            ++res.fixedVars;

    return res;
}

} // namespace baguette
