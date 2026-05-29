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

    /// @return Number of variable terms (excluding the constant).
    std::size_t size()  const { return varIds.size(); }
    /// @return `true` if the expression has no variable terms (constant only).
    bool        empty() const { return varIds.empty(); }

    /// Add the term `coeff * var` to the expression.
    ///
    /// If `var` is already present, its coefficient is incremented by `coeff`.
    /// If `|result| <= baguette::cancellation_tol`, the term is removed.
    /// The sorted order of `varIds` is preserved throughout.
    ///
    /// @note Complexity: O(n + log n), where n = `size()`.
    ///
    /// @param var   The variable to add.
    /// @param coeff The coefficient for that variable.
    void addTerm(Variable var, double coeff);

    /// Multiply all coefficients and the constant by @p factor.
    /// @note Complexity: O(n), where n = `size()`.
    void scale(double factor);

    /// Multiply all coefficients and the constant by @p factor in-place.
    /// Equivalent to `scale(factor)`.
    /// @note Complexity: O(n), where n = `size()`.
    /// @return Reference to `*this`.
    LinearExpr& operator*=(double factor);

    /// Merge @p rhs into this expression in-place.
    /// Equivalent to `*this = *this + rhs`.
    /// @note Complexity: O(n+m), where n = `size()` and m = `rhs.size()`.
    /// @return Reference to `*this`.
    LinearExpr& operator+=(const LinearExpr& rhs);

    /// Subtract @p rhs from this expression in-place.
    /// Equivalent to `*this = *this - rhs`.
    /// @note Complexity: O(n+m), where n = `size()` and m = `rhs.size()`.
    /// @return Reference to `*this`.
    LinearExpr& operator-=(const LinearExpr& rhs);

    /// Divide all coefficients and the constant by @p factor in-place.
    /// Equivalent to `scale(1.0 / factor)`.
    /// @note Complexity: O(n), where n = `size()`.
    /// @return Reference to `*this`.
    LinearExpr& operator/=(double factor);

    /// Append @p var with coefficient +1 in-place.
    /// @note Complexity: O(n).
    LinearExpr& operator+=(Variable rhs) { addTerm(rhs,  1.0); return *this; }
    /// Subtract @p var (coefficient −1) in-place.
    /// @note Complexity: O(n).
    LinearExpr& operator-=(Variable rhs) { addTerm(rhs, -1.0); return *this; }
};

/// Create a single-term expression `coeff * var`.
LinearExpr operator*(double coeff, Variable var);
/// @copydoc baguette::operator*(double, Variable)
LinearExpr operator*(Variable var, double coeff);

/// Create a single-term expression `(1/coeff) * var`.
LinearExpr operator/(Variable var, double coeff);

// ── Variable arithmetic helpers ───────────────────────────────────────────────
// Inline: cold-loop helpers; declared after operator*(double,Variable).

/// Build expression `lhs + rhs` (both with coefficient 1).
/// @note Complexity: O(1).
inline LinearExpr operator+(Variable lhs, Variable rhs) {
    LinearExpr e; e.addTerm(lhs, 1.0); e.addTerm(rhs, 1.0); return e;
}
/// Build expression `lhs - rhs`.
/// @note Complexity: O(1).
inline LinearExpr operator-(Variable lhs, Variable rhs) {
    LinearExpr e; e.addTerm(lhs, 1.0); e.addTerm(rhs, -1.0); return e;
}
/// Append @p rhs with coefficient 1 to @p lhs.
/// @note Complexity: O(n).
inline LinearExpr operator+(LinearExpr lhs, Variable rhs) { lhs.addTerm(rhs,  1.0); return lhs; }
/// Subtract @p rhs (coefficient 1) from @p lhs.
/// @note Complexity: O(n).
inline LinearExpr operator-(LinearExpr lhs, Variable rhs) { lhs.addTerm(rhs, -1.0); return lhs; }
/// Prepend @p lhs (coefficient 1) to @p rhs expression.
/// @note Complexity: O(n).
inline LinearExpr operator+(Variable lhs, LinearExpr rhs) { rhs.addTerm(lhs, 1.0);  return rhs; }
/// Build `lhs − rhs_expr`.
/// @note Complexity: O(n).
inline LinearExpr operator-(Variable lhs, const LinearExpr& rhs) {
    LinearExpr e = rhs; e.scale(-1.0); e.addTerm(lhs, 1.0); return e;
}

/// Merge two expressions into a new one.
/// Terms present in both are summed; terms with a zero result are dropped.
/// @note Complexity: O(n+m), where n = `lhs.size()` and m = `rhs.size()`.
LinearExpr operator+(LinearExpr lhs, const LinearExpr& rhs);

/// Subtract two expressions.
/// Terms present in both are differenced; terms with a zero result are dropped.
/// @note Complexity: O(n+m), where n = `lhs.size()` and m = `rhs.size()`.
LinearExpr operator-(LinearExpr lhs, const LinearExpr& rhs);

/// Divide all coefficients and the constant of @p lhs by @p factor.
/// @note Complexity: O(n), where n = `lhs.size()`.
LinearExpr operator/(LinearExpr lhs, double factor);

} // namespace baguette
