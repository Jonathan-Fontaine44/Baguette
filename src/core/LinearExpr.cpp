#include "baguette/core/LinearExpr.hpp"

#include <algorithm>
#include <cmath>

namespace baguette {

void LinearExpr::addTerm(Variable var, double coeff) {
    auto it = std::lower_bound(varIds.begin(), varIds.end(), var.id);
    if (it != varIds.end() && *it == var.id) {
        std::size_t idx = static_cast<std::size_t>(it - varIds.begin());
        coeffs[idx] += coeff;
        if (std::abs(coeffs[idx]) <= cancellation_tol) {
            varIds.erase(it);
            coeffs.erase(coeffs.begin() + static_cast<std::ptrdiff_t>(idx));
        }
    } else {
        if (std::abs(coeff) > cancellation_tol) {
            std::size_t idx = static_cast<std::size_t>(it - varIds.begin());
            varIds.insert(it, var.id);
            coeffs.insert(coeffs.begin() + static_cast<std::ptrdiff_t>(idx), coeff);
        }
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

LinearExpr& LinearExpr::operator*=(double factor) {
    scale(factor);
    return *this;
}

LinearExpr operator/(Variable var, double coeff) {
    return (1.0 / coeff) * var;
}

LinearExpr& LinearExpr::operator+=(const LinearExpr& rhs) {
    // Linear merge of two sorted lists — O(n+m)
    LinearExpr result;
    result.constant = constant + rhs.constant;
    result.varIds.reserve(varIds.size() + rhs.varIds.size());
    result.coeffs.reserve(coeffs.size() + rhs.coeffs.size());

    std::size_t i = 0, j = 0;
    while (i < varIds.size() && j < rhs.varIds.size()) {
        if (varIds[i] < rhs.varIds[j]) {
            result.varIds.push_back(varIds[i]);
            result.coeffs.push_back(coeffs[i]);
            ++i;
        } else if (varIds[i] > rhs.varIds[j]) {
            result.varIds.push_back(rhs.varIds[j]);
            result.coeffs.push_back(rhs.coeffs[j]);
            ++j;
        } else {
            double sum = coeffs[i] + rhs.coeffs[j];
            if (std::abs(sum) > cancellation_tol) {
                result.varIds.push_back(varIds[i]);
                result.coeffs.push_back(sum);
            }
            ++i; ++j;
        }
    }
    while (i < varIds.size()) {
        result.varIds.push_back(varIds[i]);
        result.coeffs.push_back(coeffs[i]);
        ++i;
    }
    while (j < rhs.varIds.size()) {
        result.varIds.push_back(rhs.varIds[j]);
        result.coeffs.push_back(rhs.coeffs[j]);
        ++j;
    }
    *this = std::move(result);
    return *this;
}

LinearExpr operator+(LinearExpr lhs, const LinearExpr& rhs) {
    lhs += rhs;
    return lhs;
}

LinearExpr& LinearExpr::operator-=(const LinearExpr& rhs) {
    LinearExpr neg = rhs;
    neg.scale(-1.0);
    return *this += neg;
}

LinearExpr operator-(LinearExpr lhs, const LinearExpr& rhs) {
    lhs -= rhs;
    return lhs;
}

LinearExpr& LinearExpr::operator/=(double factor) {
    scale(1.0 / factor);
    return *this;
}

LinearExpr operator/(LinearExpr lhs, double factor) {
    lhs /= factor;
    return lhs;
}

} // namespace baguette
