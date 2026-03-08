#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

#include "Variable.hpp"

namespace baguette {

/// Sparse linear expression: `sum(coeff_i * var_i) + constant`.
///
/// **Invariant:** `varIds` is always sorted in ascending order by variable ID.
/// This enables O(n+m) merge, binary search, and SIMD-friendly dot products.
/// `unordered_map` is intentionally avoided (random memory access, hash overhead).
struct LinearExpr {
    std::vector<std::uint32_t> varIds;  ///< Variable IDs, sorted ascending.
    std::vector<double>        coeffs;  ///< Coefficients, parallel to varIds.
    double                     constant = 0.0; ///< Constant offset.

    std::size_t size()  const { return varIds.size(); }
    bool        empty() const { return varIds.empty(); }

    /// Add the term `coeff * var` to the expression.
    ///
    /// If `var` is already present, its coefficient is incremented by `coeff`.
    /// If the resulting coefficient is exactly 0, the term is removed.
    /// The sorted order of `varIds` is preserved throughout.
    ///
    /// Complexity: O(n + log n), where n = `size()`.
    ///
    /// @param var   The variable to add.
    /// @param coeff The coefficient for that variable.
    void addTerm(Variable var, double coeff);

    /// Multiply all coefficients and the constant by @p factor.
    /// Complexity: O(n), where n = `size()`.
    void scale(double factor);
};

/// Create a single-term expression `coeff * var`.
LinearExpr operator*(double coeff, Variable var);
/// @copydoc operator*(double, Variable)
LinearExpr operator*(Variable var, double coeff);

/// Merge two expressions into a new one.
/// Terms present in both are summed; terms with a zero result are dropped.
/// Complexity: O(n+m), where n = `lhs.size()` and m = `rhs.size()`.
LinearExpr operator+(LinearExpr lhs, const LinearExpr& rhs);

} // namespace baguette
