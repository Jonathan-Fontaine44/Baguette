#include "baguette/lp/Presolve.hpp"

#include <limits>

namespace baguette {

void postsolveElim(LPDetailedResult& r, const EliminationRecord& rec) {
    if (rec.varsEliminated == 0 && rec.rowsEliminated == 0) return;

    const bool hasPrimal = r.result.status == LPStatus::Optimal ||
                           r.result.status == LPStatus::MaxIter  ||
                           r.result.status == LPStatus::TimeLimit;

    if (hasPrimal) {
        r.result.objectiveValue += rec.objAdjustment;

        if (!r.result.primalValues.empty()) {
            std::vector<double> full(rec.origVarCount, 0.0);
            for (uint32_t k = 0; k < static_cast<uint32_t>(rec.reducedToOrig.size()); ++k)
                if (k < r.result.primalValues.size())
                    full[rec.reducedToOrig[k]] = r.result.primalValues[k];
            for (const auto& [origId, val] : rec.fixedVars)
                full[origId] = val;
            r.result.primalValues = std::move(full);
        }
    }

    if (r.result.status != LPStatus::Optimal) return;

    if (!r.dualValues.empty()) {
        std::vector<double> full(rec.origConstraintCount, 0.0);
        for (uint32_t k = 0; k < static_cast<uint32_t>(rec.reducedToOrigCon.size()); ++k)
            if (k < r.dualValues.size())
                full[rec.reducedToOrigCon[k]] = r.dualValues[k];
        r.dualValues = std::move(full);
    }

    if (!r.reducedCosts.empty()) {
        std::vector<double> full(rec.origVarCount, 0.0);
        for (uint32_t k = 0; k < static_cast<uint32_t>(rec.reducedToOrig.size()); ++k)
            if (k < r.reducedCosts.size())
                full[rec.reducedToOrig[k]] = r.reducedCosts[k];
        r.reducedCosts = std::move(full);
    }

    // Remap sensitivity to original model indices.
    // Eliminated constraints/variables get unbounded ranges as defaults.
    const double kInf = std::numeric_limits<double>::infinity();
    if (!r.sensitivity.rhsRange.empty()) {
        std::vector<std::array<double, 2>> full(rec.origConstraintCount, {-kInf, kInf});
        for (uint32_t k = 0; k < static_cast<uint32_t>(rec.reducedToOrigCon.size()); ++k)
            if (k < r.sensitivity.rhsRange.size())
                full[rec.reducedToOrigCon[k]] = r.sensitivity.rhsRange[k];
        r.sensitivity.rhsRange = std::move(full);
    }
    if (!r.sensitivity.objRange.empty()) {
        std::vector<std::array<double, 2>> full(rec.origVarCount, {-kInf, kInf});
        for (uint32_t k = 0; k < static_cast<uint32_t>(rec.reducedToOrig.size()); ++k)
            if (k < r.sensitivity.objRange.size())
                full[rec.reducedToOrig[k]] = r.sensitivity.objRange[k];
        r.sensitivity.objRange = std::move(full);
    }
}

} // namespace baguette
