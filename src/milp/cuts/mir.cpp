#include "mir.hpp"

#include <cmath>
#include <limits>

#include "baguette/core/Sense.hpp"
#include "baguette/model/ModelEnums.hpp"

namespace baguette {

namespace {

inline double frac(double x) { return x - std::floor(x); }

// MIR coefficient for a non-negative variable with shifted constraint
// coefficient `a` and row fractional part `f`.
double mirCoeff(double a, double f, bool isInt) {
    if (a < 0.0) return 0.0; // negative coeffs dropped (conservative)
    if (isInt) {
        double fa = frac(a);
        return std::floor(a) + (fa <= f ? 0.0 : (fa - f) / (1.0 - f));
    }
    return a / f; // continuous: tighten
}

bool isIntType(VarType t) { return t == VarType::Integer || t == VarType::Binary; }

} // namespace

std::vector<Cut> generateMIRCuts(const LPDetailedResult& lp,
                                  const Model&            model,
                                  uint32_t                maxCuts,
                                  double                  intFeasTol) {
    const auto& hot         = model.getHot();
    const auto& types       = model.getCold().types;
    const auto& constraints = model.getLPConstraints();
    const auto& xLP         = lp.result.primalValues;
    const double kInf       = std::numeric_limits<double>::infinity();

    std::vector<Cut> cuts;

    for (const auto& con : constraints) {
        if (maxCuts > 0 && cuts.size() >= maxCuts) break;
        if (con.sense != Sense::LessEq) continue;

        const auto& lhs = con.lhs;
        double bShifted = con.rhsConst;
        bool skip = false;
        for (std::size_t k = 0; k < lhs.size(); ++k) {
            double lb = hot.lb[lhs.varIds[k]];
            if (lb == -kInf) { skip = true; break; }
            bShifted -= lhs.coeffs[k] * lb;
        }
        if (skip) continue;

        const double f = frac(bShifted);
        if (f <= intFeasTol || f >= 1.0 - intFeasTol) continue;

        Cut cut;
        cut.sense = Sense::LessEq;
        cut.rhs   = std::floor(bShifted);

        for (std::size_t k = 0; k < lhs.size(); ++k) {
            const uint32_t id = lhs.varIds[k];
            const double   a  = lhs.coeffs[k];
            const double   lb = hot.lb[id];
            double alpha = mirCoeff(a, f, isIntType(types[id]));
            if (std::abs(alpha) <= intFeasTol) continue;
            cut.expr.addTerm(Variable{id}, alpha);
            cut.rhs += alpha * lb; // unshift: α*(x-lb) ≤ ⌊b'⌋  →  αx ≤ ⌊b'⌋ + α*lb
        }

        if (cut.expr.empty()) continue;

        // Emit only if the current LP solution violates the cut.
        double lhsVal = 0.0;
        for (std::size_t k = 0; k < cut.expr.size(); ++k) {
            uint32_t id = cut.expr.varIds[k];
            if (id < xLP.size()) lhsVal += cut.expr.coeffs[k] * xLP[id];
        }
        if (lhsVal <= cut.rhs + intFeasTol) continue;

        cuts.push_back(std::move(cut));
    }

    return cuts;
}

std::vector<Cut> generateCMIRCuts(const LPDetailedResult& lp,
                                   const Model&            model,
                                   uint32_t                maxCuts,
                                   double                  intFeasTol) {
    const auto& hot         = model.getHot();
    const auto& types       = model.getCold().types;
    const auto& constraints = model.getLPConstraints();
    const auto& xLP         = lp.result.primalValues;
    const double kInf       = std::numeric_limits<double>::infinity();

    std::vector<Cut> cuts;

    for (const auto& con : constraints) {
        if (maxCuts > 0 && cuts.size() >= maxCuts) break;
        if (con.sense != Sense::GreaterEq) continue;

        // Complement all vars: x'ⱼ = ubⱼ − xⱼ.
        // b̄ = Σ aⱼ ubⱼ − b; constraint becomes Σ aⱼ x'ⱼ ≤ b̄ (x'ⱼ ≥ 0).
        const auto& lhs = con.lhs;
        double bBar = -con.rhsConst;
        bool skip = false;
        for (std::size_t k = 0; k < lhs.size(); ++k) {
            double ub = hot.ub[lhs.varIds[k]];
            double a  = lhs.coeffs[k];
            if (ub == kInf || a <= 0.0) { skip = true; break; }
            bBar += a * ub;
        }
        if (skip || bBar < 0.0) continue;

        const double f = frac(bBar);
        if (f <= intFeasTol || f >= 1.0 - intFeasTol) continue;

        // Apply MIR in complemented space (all coefficients are aⱼ > 0, lbⱼ=0).
        Cut cut;
        cut.sense = Sense::GreaterEq;
        // In complemented space: Σ α̃ⱼ x'ⱼ ≤ ⌊b̄⌋
        // x'ⱼ = ubⱼ − xⱼ  →  α̃ⱼ(ubⱼ − xⱼ) ≤ ⌊b̄⌋
        // →  Σ α̃ⱼ xⱼ ≥ Σ α̃ⱼ ubⱼ − ⌊b̄⌋
        cut.rhs = -std::floor(bBar);

        for (std::size_t k = 0; k < lhs.size(); ++k) {
            const uint32_t id = lhs.varIds[k];
            const double   a  = lhs.coeffs[k];
            const double   ub = hot.ub[id];
            double alpha = mirCoeff(a, f, isIntType(types[id]));
            if (std::abs(alpha) <= intFeasTol) continue;
            cut.expr.addTerm(Variable{id}, alpha);
            cut.rhs += alpha * ub;
        }

        if (cut.expr.empty()) continue;

        // Emit only if violated by current LP solution.
        double lhsVal = 0.0;
        for (std::size_t k = 0; k < cut.expr.size(); ++k) {
            uint32_t id = cut.expr.varIds[k];
            if (id < xLP.size()) lhsVal += cut.expr.coeffs[k] * xLP[id];
        }
        if (lhsVal >= cut.rhs - intFeasTol) continue;

        cuts.push_back(std::move(cut));
    }

    return cuts;
}

} // namespace baguette
