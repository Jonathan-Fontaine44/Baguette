#include "baguette/milp/Presolve.hpp"

#include <chrono>
#include <cmath>
#include <limits>

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

    for (uint32_t outer = 0; maxCycles == 0 || outer < maxCycles; ++outer) {
        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - startTime).count();
        if (elapsed >= timeLimitS) { res.timeLimitReached = true; break; }

        // LP bound-tightening to its own fixed point.
        PresolveResult pr = presolveTBInPlace(model, 0, timeLimitS, startTime);
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
