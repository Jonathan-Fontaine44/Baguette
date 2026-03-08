#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

#include "Config.hpp"
#include "Variable.hpp"

namespace baguette {

/// Sparse linear expression: `sum(coeff_i * var_i) + constant`.
///
/// **Invariant:** `varIds` is always sorted in ascending order by variable ID.
/// This enables O(n+m) merge, binary search, and SIMD-friendly dot products.
/// `unordered_map` is intentionally avoided (random memory access, hash overhead).
///
/// Prefer building expressions via the `operator*` and `operator+` helpers,
/// which maintain the sorted invariant automatically.
///
/// @warning If you construct a LinearExpr manually, you must ensure:
/// - `varIds` is sorted in strictly ascending order (no duplicates).
/// - Every ID in `varIds` refers to a variable that exists in the target Model.
/// Violating either condition produces silent incorrect behaviour or undefined behaviour.
struct LinearExpr {
    std::vector<std::uint32_t> varIds;  ///< Variable IDs, sorted ascending.
    std::vector<double>        coeffs;  ///< Coefficients, parallel to varIds.
    double                     constant = 0.0; ///< Constant offset.

    std::size_t size()  const { return varIds.size(); }
    bool        empty() const { return varIds.empty(); }

    /// Add the term `coeff * var` to the expression.
    ///
    /// If `var` is already present, its coefficient is incremented by `coeff`.
    /// If `|result| <= baguette::cancellation_tol`, the term is removed.
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

    /// Multiply all coefficients and the constant by @p factor in-place.
    /// Equivalent to `scale(factor)`.
    /// Complexity: O(n), where n = `size()`.
    LinearExpr& operator*=(double factor);

    /// Merge @p rhs into this expression in-place.
    /// Equivalent to `*this = *this + rhs`.
    /// Complexity: O(n+m), where n = `size()` and m = `rhs.size()`.
    LinearExpr& operator+=(const LinearExpr& rhs);

    /// Subtract @p rhs from this expression in-place.
    /// Equivalent to `*this = *this - rhs`.
    /// Complexity: O(n+m), where n = `size()` and m = `rhs.size()`.
    LinearExpr& operator-=(const LinearExpr& rhs);

    /// Divide all coefficients and the constant by @p factor in-place.
    /// Equivalent to `scale(1.0 / factor)`.
    /// Complexity: O(n), where n = `size()`.
    LinearExpr& operator/=(double factor);
};

/// Create a single-term expression `coeff * var`.
LinearExpr operator*(double coeff, Variable var);
/// @copydoc operator*(double, Variable)
LinearExpr operator*(Variable var, double coeff);

/// Create a single-term expression `(1/coeff) * var`.
LinearExpr operator/(Variable var, double coeff);

/// Merge two expressions into a new one.
/// Terms present in both are summed; terms with a zero result are dropped.
/// Complexity: O(n+m), where n = `lhs.size()` and m = `rhs.size()`.
LinearExpr operator+(LinearExpr lhs, const LinearExpr& rhs);

/// Subtract two expressions.
/// Terms present in both are differenced; terms with a zero result are dropped.
/// Complexity: O(n+m), where n = `lhs.size()` and m = `rhs.size()`.
LinearExpr operator-(LinearExpr lhs, const LinearExpr& rhs);

/// Divide all coefficients and the constant of @p lhs by @p factor.
/// Complexity: O(n), where n = `lhs.size()`.
LinearExpr operator/(LinearExpr lhs, double factor);

} // namespace baguette
