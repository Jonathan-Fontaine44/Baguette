#include "baguette/core/LinearExpr.hpp"

#include <algorithm>
#include <cassert>

namespace baguette {

void LinearExpr::addTerm(Variable var, double coeff) {
    auto it = std::lower_bound(varIds.begin(), varIds.end(), var.id);
    if (it != varIds.end() && *it == var.id) {
        std::size_t idx = static_cast<std::size_t>(it - varIds.begin());
        coeffs[idx] += coeff;
        if (coeffs[idx] == 0.0) {
            varIds.erase(it);
            coeffs.erase(coeffs.begin() + static_cast<std::ptrdiff_t>(idx));
        }
    } else {
        std::size_t idx = static_cast<std::size_t>(it - varIds.begin());
        varIds.insert(it, var.id);
        coeffs.insert(coeffs.begin() + static_cast<std::ptrdiff_t>(idx), coeff);
    }
}

void LinearExpr::scale(double factor) {
    for (auto& c : coeffs) c *= factor;
    constant *= factor;
}

LinearExpr operator*(double coeff, Variable var) {
    LinearExpr expr;
    expr.varIds.push_back(var.id);
    expr.coeffs.push_back(coeff);
    return expr;
}

LinearExpr operator*(Variable var, double coeff) {
    return coeff * var;
}

LinearExpr operator+(LinearExpr lhs, const LinearExpr& rhs) {
    // Linear merge of two sorted lists — O(n+m)
    LinearExpr result;
    result.constant = lhs.constant + rhs.constant;
    result.varIds.reserve(lhs.varIds.size() + rhs.varIds.size());
    result.coeffs.reserve(lhs.coeffs.size() + rhs.coeffs.size());

    std::size_t i = 0, j = 0;
    while (i < lhs.varIds.size() && j < rhs.varIds.size()) {
        if (lhs.varIds[i] < rhs.varIds[j]) {
            result.varIds.push_back(lhs.varIds[i]);
            result.coeffs.push_back(lhs.coeffs[i]);
            ++i;
        } else if (lhs.varIds[i] > rhs.varIds[j]) {
            result.varIds.push_back(rhs.varIds[j]);
            result.coeffs.push_back(rhs.coeffs[j]);
            ++j;
        } else {
            double sum = lhs.coeffs[i] + rhs.coeffs[j];
            if (sum != 0.0) {
                result.varIds.push_back(lhs.varIds[i]);
                result.coeffs.push_back(sum);
            }
            ++i; ++j;
        }
    }
    while (i < lhs.varIds.size()) {
        result.varIds.push_back(lhs.varIds[i]);
        result.coeffs.push_back(lhs.coeffs[i]);
        ++i;
    }
    while (j < rhs.varIds.size()) {
        result.varIds.push_back(rhs.varIds[j]);
        result.coeffs.push_back(rhs.coeffs[j]);
        ++j;
    }
    return result;
}

} // namespace baguette
