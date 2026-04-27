#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "StandardForm.hpp"

namespace baguette::internal {

/// Revised-simplex basis using an explicit m×m basis inverse B⁻¹.
///
/// Unlike SimplexTableau (which stores the full m×(n+1) matrix B⁻¹A | B⁻¹b),
/// LUTable stores only B⁻¹ (m×m) and computes entering columns on demand.
///
/// Memory: O(m²) working space for B⁻¹ (vs O(m·n) for SimplexTableau).
/// Per-pivot: O(m²) to update B⁻¹ + O(m·n) full repricing — same asymptotic
/// as SimplexTableau but smaller working set when m ≪ n.
/// Periodic reinversion uses LU factorisation with partial pivoting.
struct LUTableau {
    std::size_t m = 0;  ///< Number of constraint rows.
    std::size_t n = 0;  ///< Number of standard-form columns.

    /// Explicit basis inverse B⁻¹, row-major, size m×m.
    /// Binv[i*m + k] = (B⁻¹)_{i,k}.
    std::vector<double> Binv;

    /// Current basic variable values: xB = B⁻¹ b, size m.
    std::vector<double> xB;

    /// Pricing vector: π = cB^T B⁻¹, size m.
    std::vector<double> pi;

    /// Reduced costs: rc[j] = c[j] − π^T a_j, size n+1.
    /// rc[n] = −(current standard-form objective value).
    std::vector<double> rc;

    /// basicCols[i] = column index of the basic variable for row i. Size m.
    std::vector<uint32_t> basicCols;

    /// Active column limit for entering selection.
    /// 0 = all n columns (phase I); nOrig in phase II to exclude artificials.
    std::size_t nActive = 0;

    // ── Stored for on-demand column computation ──────────────────────────────

    /// Shared reference to the standard-form constraint matrix A (m×n, row-major).
    std::shared_ptr<const std::vector<double>> A_ptr;
    std::vector<double> c;  ///< Standard-form objective, size n.
    std::vector<double> b;  ///< Standard-form RHS, size m.

    // ── Construction ────────────────────────────────────────────────────────

    /// Initialise from a standard form and an initial basis.
    /// Performs LU factorisation of B, computes B⁻¹, xB, π, and rc.
    /// @return false if the basis matrix is numerically singular.
    /// @note Complexity: O(m³) LU + O(m³) back-substitution + O(m·n) repricing.
    bool init(const LPStandardForm& sf, std::vector<uint32_t> initialBasis);

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// Entering column η = B⁻¹ a_j (length m).
    /// @note Complexity: O(m²).
    std::vector<double> enteringColumn(std::size_t j) const;

    /// Full tableau row r: t_r[j] = (B⁻¹ A)_{r,j} for j = 0..n-1.
    /// Needed for dual entering selection and cut-data extraction.
    /// @note Complexity: O(m·n).
    std::vector<double> tableauRow(std::size_t r) const;

    // ── Pivot selection ──────────────────────────────────────────────────────

    /// Bland's rule: smallest j with rc[j] < −lp_optimality_tol.
    /// @return j < n (entering column) or n (optimal).
    /// @note Complexity: O(nActive).
    std::size_t selectEntering() const;

    /// Minimum-ratio test for primal simplex (Bland's tie-breaking).
    /// Computes η = B⁻¹ a_j internally.
    /// @return r < m (leaving row) or m (unbounded).
    /// @note Complexity: O(m²+m).
    std::size_t selectLeaving(std::size_t enteringCol) const;

    /// Most-negative xB row for dual simplex (Bland's tie-breaking).
    /// @return r < m (leaving row) or m (primal feasible → optimal).
    /// @note Complexity: O(m).
    std::size_t selectLeavingDual() const;

    /// Dual entering: min ratio rc[j]/|t_r[j]| over j with t_r[j] < 0.
    /// Computes the full tableau row for leavingRow internally.
    /// @return j < n (entering column) or n (primal infeasible).
    /// @note Complexity: O(m·n).
    std::size_t selectEnteringDual(std::size_t leavingRow) const;

    // ── Pivot ────────────────────────────────────────────────────────────────

    /// Bring enteringCol into the basis at leavingRow (eta-file update).
    /// Updates Binv, xB, basicCols, π, and rc.
    /// @note Complexity: O(m²) Binv + O(m) xB + O(m²) π + O(m·n) rc.
    void pivot(std::size_t leavingRow, std::size_t enteringCol);

    // ── Reinversion ──────────────────────────────────────────────────────────

    /// Recompute B⁻¹ from scratch using LU factorisation with partial pivoting.
    /// Also refreshes A_ptr, c, b from sf (handles bounds-only updates).
    /// @return false if the basis is numerically singular.
    /// @note Complexity: O(m³) LU + O(m³) back-sub + O(m·n) repricing.
    [[nodiscard]] bool reinvert(const LPStandardForm& sf);

    /// Update the objective vector and recompute π and rc without reinversion.
    /// Used for the phase I → II transition.
    /// @note Complexity: O(m²) for π + O(m·n) for rc.
    void repriceObjective(const std::vector<double>& newC, std::size_t newNActive);

    // ── Query ────────────────────────────────────────────────────────────────

    /// @note Complexity: O(1).
    double objectiveValue() const { return -rc[n]; }

    /// Primal solution for all n columns (0 for non-basic).
    /// @note Complexity: O(m).
    std::vector<double> primalSolution() const;

private:
    void recomputePi();
    void recomputeReducedCosts();
};

} // namespace baguette::internal
