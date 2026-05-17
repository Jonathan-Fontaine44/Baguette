#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "SimplexConfig.hpp"
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

    /// Per-solve numerical configuration. Set before init() so that all member
    /// functions use per-solve tolerances.
    SimplexConfig cfg;

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

    /// Bland's (default) or Dantzig's entering rule, per cfg.useDantzig.
    /// @return j < n (entering column) or n (optimal).
    /// @note Complexity: O(nActive).
    std::size_t selectEntering() const;

    /// Minimum-ratio test for primal simplex (Bland's tie-breaking).
    /// Computes η = B⁻¹ a_j internally.
    /// @return r < m (leaving row) or m (unbounded).
    /// @note Complexity: O(m²+m).
    std::size_t selectLeaving(std::size_t enteringCol) const;

    /// Same as selectLeaving but also returns the precomputed η = B⁻¹ a_j.
    /// Pass the returned eta directly to pivot() to avoid recomputing it.
    /// @return {leaving row (m = unbounded), η vector}.
    /// @note Complexity: O(m²+m).
    std::pair<std::size_t, std::vector<double>>
    selectLeavingWithEta(std::size_t enteringCol) const;

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

    /// Bring enteringCol into the basis at leavingRow.
    /// Computes η = B⁻¹ a_j internally; prefer the eta overload when η is
    /// already available (e.g. from selectLeavingWithEta) to avoid recomputing.
    /// @note Complexity: O(m²) η + O(m²) Binv + O(m) π + O(m·n) rc.
    void pivot(std::size_t leavingRow, std::size_t enteringCol);

    /// Bring enteringCol into the basis at leavingRow using a precomputed η.
    /// Updates Binv (row-major, cache-friendly), xB, basicCols, π, and rc
    /// using incremental formulas: π update O(m), rc update O(m·n) row-major.
    /// @note Complexity: O(m²) Binv + O(m) π + O(m·n) rc.
    void pivot(std::size_t leavingRow, std::size_t enteringCol,
               const std::vector<double>& eta);

    // ── BV extension fields ──────────────────────────────────────────────────────

    /// Per-column upper bound in lb-shifted space. Length n.
    /// colUB[j] = ub_j − lb_j (finite ub) or +∞ (unbounded). Populated by initBV.
    std::vector<double> colUB;

    /// Complement invariant: atUB[j] = true iff non-basic column j is complemented
    /// (uses −a_j direction). Basic variables always satisfy atUB[basicCols[i]] = false.
    /// Length n. Populated by initBV.
    std::vector<bool> atUB;

    // ── BV construction ──────────────────────────────────────────────────────────

    /// Initialise from a BV standard form and an initial basis.
    /// Same as init() but stores colUB, initialises atUB = false, and accepts
    /// LPStandardFormBV (no explicit UB rows, m = nOrigRows).
    /// @return false if the basis matrix is numerically singular.
    /// @note Complexity O(m³) LU + O(m³) back-sub + O(m·n) repricing.
    bool initBV(const LPStandardFormBV& sfbv, std::vector<uint32_t> initialBasis);

    // ── BV complement ────────────────────────────────────────────────────────────

    /// Toggle column j between AT_LB (x_j = 0) and AT_UB (x_j = colUB[j]).
    /// Updates xB, rc[j], and rc[n] using B⁻¹ a_j computed on demand.
    /// @note Complexity O(m²) due to enteringColumn().
    void complement(std::size_t j);

    // ── BV pivot selection ───────────────────────────────────────────────────────

    struct RatioResultBV {
        std::size_t leavingRow;  ///< m = no basis change (bound flip or unbounded).
        bool        boundFlip;   ///< Entering var hits its own UB before any basis change.
        bool        leavingAtUB; ///< Leaving basic variable exits to its upper bound.
    };

    /// BV ratio test: two-sided (basic vars may hit LB or UB), plus bound flip.
    /// Returns the result and η = B⁻¹ a_j (un-complemented entering column).
    /// Pass η to pivotBV() to avoid recomputing it.
    /// @note Complexity O(m²) for enteringColumn() + O(m) ratio loop.
    std::pair<RatioResultBV, std::vector<double>>
    selectLeavingBVWithEta(std::size_t enteringCol) const;

    // ── BV pivot ─────────────────────────────────────────────────────────────────

    /// BV pivot: un-complement entering if AT_UB (O(m)), then standard pivot
    /// (O(m²) B⁻¹ + O(m) π + O(m·n) rc), then complement leaving if leavingAtUB
    /// (O(m²) due to enteringColumn on updated B⁻¹).
    /// @param eta_orig  B⁻¹ a_j from selectLeavingBVWithEta (avoids recomputation).
    /// @note Complexity O(m·n) dominated by rc update.
    void pivotBV(std::size_t leavingRow, std::size_t enteringCol, bool leavingAtUB,
                 const std::vector<double>& eta_orig);

    // ── BV repricing ─────────────────────────────────────────────────────────────

    /// Recompute π and rc for a new objective, then apply AT_UB sign adjustments.
    /// Used for the phase I → II transition in the BV revised simplex.
    /// @note Complexity O(m²) for π + O(m·n) for rc + O(n) for AT_UB adjustment.
    void repriceBV(const std::vector<double>& newC, std::size_t newNActive);

    // ── Reinversion ──────────────────────────────────────────────────────────

    /// Recompute B⁻¹ from scratch using LU factorisation with partial pivoting.
    /// Also refreshes A_ptr, c, b from sf (handles bounds-only updates).
    /// @return false if the basis is numerically singular.
    /// @note Complexity: O(m³) LU + O(m³) back-sub + O(m·n) repricing.
    [[nodiscard]] bool reinvert(const LPStandardForm& sf);

    /// Recompute B⁻¹ from scratch for BV simplex, preserving the complement state.
    /// Saves atUB, refreshes A_ptr/c/b/colUB from sfbv, runs LU, then re-applies
    /// complement(j) for every j that was AT_UB.
    /// @return false if the basis is numerically singular.
    /// @note Complexity O(m³) LU + O(k·m²) complements, where k = |AT_UB|.
    [[nodiscard]] bool reinvertBV(const LPStandardFormBV& sfbv);

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

    /// Primal solution for BV simplex: basic vars from xB, AT_UB non-basics at colUB.
    /// @note Complexity O(m + n).
    std::vector<double> primalSolutionBV() const;

private:
    /// LU factorisation + B⁻¹ computation + xB/π/rc from current A_ptr, b, c.
    bool doReinvert();
    /// Apply AT_UB sign adjustments to rc[j] and rc[n] after full recomputation.
    void applyAtUBToRc();
    void recomputePi();
    void recomputeReducedCosts();
};

} // namespace baguette::internal
