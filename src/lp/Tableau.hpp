#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "StandardForm.hpp"

namespace baguette::internal {

/// Full (non-revised) simplex tableau for m rows and n columns.
///
/// Stores the m × (n+1) augmented matrix [B⁻¹A | B⁻¹b] explicitly.
/// The reduced-cost row rc (length n+1) is kept separately:
///   rc[j]  = reduced cost of column j  (< 0 means improving for minimisation)
///   rc[n]  = −(current objective value)
///
/// Column indices match LPStandardForm::nCols exactly, so column IDs are
/// stable and can be recorded directly in BasisRecord.
///
/// The tableau supports periodic reinversion: calling reinvert(sf) rebuilds
/// B⁻¹ from scratch using the current basis to prevent floating-point drift.
struct Tableau {
    std::size_t m = 0; ///< Number of constraint rows.
    std::size_t n = 0; ///< Number of columns (NOT counting the rhs column).

    /// Constraint rows, row-major, size m × (n+1).
    /// tab[i*(n+1) + j] = B⁻¹A entry at row i, column j.
    /// tab[i*(n+1) + n] = B⁻¹b entry at row i (current rhs).
    std::vector<double> tab;

    /// Reduced-cost row, size n+1.
    std::vector<double> rc;

    /// basicCols[i] = index of the basic column in row i.  Size m.
    std::vector<uint32_t> basicCols;

    // ── Construction ────────────────────────────────────────────────────────

    /// Build the tableau from a standard form and an initial basis.
    ///
    /// @p initialBasis[i] is the column index of the basic variable for row i.
    /// Performs Gauss-Jordan elimination to express the tableau in terms of
    /// the given basis, and prices the objective row accordingly.
    ///
    /// @param sf           The standard-form LP.  For phase I, pass the augmented
    ///                     form whose sf.c already has 1 for artificial columns and
    ///                     0 for all others.  For phase II (and reinvert), pass the
    ///                     original standard form.
    /// @param initialBasis Column indices forming the initial basis (size == sf.nRows).
    /// @return true on success; false if the basis matrix is numerically singular
    ///         (all pivot candidates in some column are below pivot_tol).
    ///         On false the tableau is in an undefined state and must not be used.
    bool init(const LPStandardForm& sf,
              std::vector<uint32_t> initialBasis);

    // ── Simplex operations ───────────────────────────────────────────────────

    /// Select the entering column using Bland's rule:
    /// the smallest column index j with rc[j] < −lp_optimality_tol.
    /// Returns n if no improving column exists (current solution is optimal).
    std::size_t selectEntering() const;

    /// Select the leaving row using the minimum ratio test with full Bland's rule.
    /// Only rows with tab[i*(n+1) + enteringCol] > pivot_tol are considered.
    /// On ties (ratios within pivot_tol of each other), the row whose basic
    /// variable has the smallest column index is chosen, completing Bland's
    /// anti-cycling rule. Returns m if no such row exists (problem is unbounded).
    std::size_t selectLeaving(std::size_t enteringCol) const;

    // ── Dual-simplex pivot selection ─────────────────────────────────────────

    /// Dual-simplex leaving-row selection.
    ///
    /// Picks the row i with the most-negative rhs  tab[i*(n+1)+n] < −lp_feasibility_tol.
    /// Returns m if all rhs values are ≥ −lp_feasibility_tol (primal feasible → optimal
    /// since the dual simplex maintains dual feasibility throughout).
    std::size_t selectLeavingDual() const;

    /// Dual-simplex entering-column selection (minimum-ratio test on reduced costs).
    ///
    /// For the leaving row @p leavingRow, considers only columns j where
    /// tab[leavingRow*(n+1)+j] < −pivot_tol, and picks the one minimising
    ///   rc[j] / |tab[leavingRow*(n+1)+j]|
    /// with ties broken by smallest column index (Bland-style anti-cycling).
    ///
    /// Returns n if no column qualifies (all entries ≥ −pivot_tol), which signals
    /// that the primal LP is infeasible (dual LP is unbounded in that direction).
    ///
    /// @pre  All rc[j] ≥ 0  (dual feasibility is maintained by the caller).
    std::size_t selectEnteringDual(std::size_t leavingRow) const;

    /// Pivot: bring enteringCol into the basis at leavingRow.
    /// Updates tab, rc, and basicCols in-place.
    void pivot(std::size_t leavingRow, std::size_t enteringCol);

    /// Rebuild B⁻¹ from scratch using the current basicCols to reset
    /// accumulated floating-point errors.
    /// @param sf The standard-form LP (needed for the original A and c).
    /// @return true on success; false if the basis is numerically singular.
    ///         On false the tableau is in an undefined state and must not be used.
    [[nodiscard]] bool reinvert(const LPStandardForm& sf);

    // ── Solution extraction ──────────────────────────────────────────────────

    /// Current objective value (valid for both phase I and phase II).
    double objectiveValue() const { return -rc[n]; }

    /// Primal solution for all n columns.
    /// Basic variables take their rhs value; non-basic variables are 0.
    std::vector<double> primalSolution() const;

    /// Dual variables y = c_B B⁻¹.
    /// For the full tableau these are read from the reduced-cost entries of
    /// the slack columns, sign-corrected for negated rows and surplus sign.
    ///
    /// @param sf The standard form (needed for rowSlackCol and rowNegated).
    std::vector<double> dualSolution(const LPStandardForm& sf) const;
};

} // namespace baguette::internal
