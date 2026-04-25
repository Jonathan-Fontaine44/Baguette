#include "baguette/milp/CuttingPlanes.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "baguette/core/Sense.hpp"
#include "baguette/model/ModelEnums.hpp"

namespace baguette {

namespace {

// Return frac(x) ∈ [0, 1).
inline double frac(double x) { return x - std::floor(x); }

// GMI coefficient for a non-basic SF column with tableau entry a_bar,
// when the basic row has fractional part f_i.
// isInt: the SF column corresponds to an integer-typed original variable.
double gmiCoeff(double aBar, double fi, bool isInt) {
    if (isInt) {
        double aj = frac(aBar);
        return (aj <= fi) ? aj / fi : (1.0 - aj) / (1.0 - fi);
    }
    // Continuous
    return (aBar >= 0.0) ? aBar / fi : -aBar / (1.0 - fi);
}

} // namespace

std::vector<Cut> generateGMICuts(const std::vector<FractionalRow>& rows,
                                  const BasisRecord&                basis,
                                  const Model&                      model,
                                  uint32_t                          maxCuts,
                                  double                            intFeasTol) {
    const auto& hot         = model.getHot();
    const auto& types       = model.getCold().types;
    const auto& constraints = model.getConstraints();
    const double kInf        = std::numeric_limits<double>::infinity();

    std::vector<Cut> cuts;
    cuts.reserve(rows.size());

    for (const FractionalRow& fr : rows) {
        if (maxCuts > 0 && static_cast<uint32_t>(cuts.size()) >= maxCuts) break;

        const double fi = fr.fracVal;
        if (fi <= intFeasTol || fi >= 1.0 - intFeasTol) continue;

        Cut cut;
        cut.rhs = 1.0; // GMI cut in SF space: Σ gmi_j * x'_j ≥ 1 (before substitution)

        const std::size_t n = fr.tabRow.size();

        for (std::size_t j = 0; j < n; ++j) {
            const double aBar = fr.tabRow[j];
            if (std::abs(aBar) <= intFeasTol) continue;

            const ColumnKind kind   = basis.colKind[j];
            const uint32_t   origin = basis.colOrigin[j];

            double gmi = 0.0;

            // ── Compute GMI coefficient ──────────────────────────────────────
            switch (kind) {
                case ColumnKind::Original: {
                    bool isInt = (types[origin] == VarType::Integer ||
                                  types[origin] == VarType::Binary);
                    gmi = gmiCoeff(aBar, fi, isInt);
                    break;
                }
                case ColumnKind::Slack:
                case ColumnKind::UpperSlack:
                    gmi = gmiCoeff(aBar, fi, /*isInt=*/false);
                    break;
                case ColumnKind::FreeNeg:
                    continue; // skip — weakens the cut conservatively
            }

            if (std::abs(gmi) <= intFeasTol) continue;

            // ── Substitute back to model-variable space ──────────────────────
            switch (kind) {
                case ColumnKind::Original: {
                    // x'_j = varColSign * (x_j − varShiftVal); derive shift from current bounds.
                    double lb = hot.lb[origin];
                    double ub = hot.ub[origin];
                    int8_t s;
                    double v;
                    if (lb > -kInf) {
                        s = +1; v = lb; // lb-shift
                    } else if (ub < kInf) {
                        s = -1; v = ub; // ub-shift
                    } else {
                        continue; // free-split: excluded by computeCutData, safety guard
                    }
                    // gmi * x'_j = gmi*s*x_j − gmi*s*v; move constant to RHS
                    cut.expr.addTerm(Variable{origin}, gmi * static_cast<double>(s));
                    cut.rhs += gmi * static_cast<double>(s) * v;
                    break;
                }

                case ColumnKind::Slack: {
                    // Slack (LessEq) or surplus (GEQ) for model constraint k = origin.
                    //   LessEq: s_k = rhs_k − Σ a_kl*x_l  →  sign = −1
                    //   GEQ:    s_k = Σ a_kl*x_l − rhs_k  →  sign = +1
                    const Constraint& con  = constraints[origin];
                    double            sign = (con.sense == Sense::LessEq) ? -1.0 : +1.0;
                    for (std::size_t t = 0; t < con.lhs.size(); ++t) {
                        cut.expr.addTerm(Variable{con.lhs.varIds[t]},
                                         sign * gmi * con.lhs.coeffs[t]);
                    }
                    cut.rhs += sign * gmi * con.rhs; // LessEq: −= gmi*rhs; GEQ: += gmi*rhs
                    break;
                }

                case ColumnKind::UpperSlack: {
                    // s_ub = ub_v − x_v: gmi*s_ub = gmi*ub_v − gmi*x_v
                    cut.expr.addTerm(Variable{origin}, -gmi);
                    cut.rhs -= gmi * hot.ub[origin];
                    break;
                }

                case ColumnKind::FreeNeg:
                    break; // already continued above
            }
        }

        if (cut.expr.size() > 0)
            cuts.push_back(std::move(cut));
    }

    return cuts;
}

} // namespace baguette
