#include "baguette/milp/Presolve.hpp"

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

} // namespace baguette
