#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "StandardForm.hpp"

namespace baguette::internal {

/// Bounded-variable simplex tableau.
///
/// Maintains the complement invariant: for each non-basic column j, if
/// atUB[j] = true the column has been negated in the tableau (complement
/// substitution xj' = colUB[j] − xj). Consequently the entering criterion
/// is always rc[j] < −tol (same as standard simplex), and the ratio test
/// additionally checks whether any basic variable would hit its upper bound.
///
/// The RHS column tab[i*(n+1)+n] stores the actual BFS value for basic
/// variable i, already accounting for non-basic AT_UB contributions.
///
/// @node Complexity
/// init(): O(m²n) Gauss-Jordan.
/// complement(): O(m).
/// selectEntering(): O(n).
/// selectLeavingBV() / selectLeavingDualBV(): O(m).
/// selectEnteringDualBV(): O(n).
/// pivotBV(): O(mn) + O(m) for optional complement.
struct SimplexTableauBV {
    std::size_t m = 0;
    std::size_t n = 0;

    /// Row-major augmented matrix [B⁻¹A | B⁻¹b_adj], size m × (n+1).
    std::vector<double> tab;

    /// Reduced-cost row, size n+1.  rc[n] = −z (current negative objective).
    std::vector<double> rc;

    /// basicCols[i] = column index of the basic variable in row i.
    std::vector<uint32_t> basicCols;

    bool        hasRedundantRow = false;
    std::size_t nActive         = 0; ///< 0 = all n columns active (phase I).

    /// Per-column upper bound in lb-shifted space. Length n.
    std::vector<double> colUB;

    /// Complement invariant: atUB[j] = true iff non-basic column j has been
    /// complemented (column negated, atUB[basicCols[i]] is always false).
    std::vector<bool> atUB;

    // ── Construction ─────────────────────────────────────────────────────────

    bool init(const LPStandardFormBV& sfbv, std::vector<uint32_t> initialBasis);

    /// Rebuild B⁻¹ from scratch, restoring the complement state afterward.
    [[nodiscard]] bool reinvert(const LPStandardFormBV& sf);

    // ── Complement operation ──────────────────────────────────────────────────

    /// Negate column j in all rows (including rc), update RHS += colUB[j]*new_col,
    /// and toggle atUB[j].  O(m).
    void complement(std::size_t j);

    // ── Pivot selection ───────────────────────────────────────────────────────

    /// Bland's rule: smallest j with rc[j] < −lp_optimality_tol.
    /// Works for both AT_LB and AT_UB columns (complement invariant).
    std::size_t selectEntering() const;

    struct RatioResult {
        std::size_t leavingRow;  ///< m = no basis change (bound flip or unbounded).
        bool        boundFlip;   ///< Entering var hits its own UB (no basis change).
        bool        leavingAtUB; ///< Leaving basic var exits to its UB.
    };

    /// BV ratio test: checks LB and UB of each basic variable.
    RatioResult selectLeavingBV(std::size_t enteringCol) const;

    // ── Dual pivot selection ──────────────────────────────────────────────────

    struct DualLeavingResult {
        std::size_t leavingRow; ///< m = primal feasible (stop).
        bool        exitsToUB;  ///< True when the leaving basic exceeds its UB.
    };

    /// Most-infeasible leaving rule: picks row with largest |violation| of LB or UB.
    /// Tiebreak: smallest basic column index (Bland anti-cycling).
    DualLeavingResult selectLeavingDualBV() const;

    /// Dual entering: minimum ratio rc[j]/|eta| to maintain dual feasibility.
    /// exitsToUB=false → need eta < 0 (standard dual); =true → need eta > 0.
    /// Returns n when infeasibility is certified (no valid entering column).
    std::size_t selectEnteringDualBV(std::size_t leavingRow, bool exitsToUB) const;

    // ── Pivot ─────────────────────────────────────────────────────────────────

    /// Standard Gauss-Jordan pivot, then complement(leaving) when leavingAtUB.
    void pivotBV(std::size_t leavingRow, std::size_t enteringCol, bool leavingAtUB);

    // ── Extraction ────────────────────────────────────────────────────────────

    double objectiveValue() const { return -rc[n]; }

    /// Returns actual shifted values: basic vars from RHS, AT_UB non-basics at colUB[j].
    std::vector<double> primalSolution() const;
};

/// Compute RHS and objective sensitivity ranges from an optimal BV tableau.
///
/// Analogous to extractSensitivity (Extractor.cpp) but for the bounded-variable
/// form: upper bounds enforced via colUB + atUB instead of explicit UB rows.
/// RHS ranging checks both LB=0 and UB constraints on basic variables.
/// Equal constraints are not supported (BV path falls back to primal for those).
SensitivityResult extractSensitivityBV(const SimplexTableauBV&      tab,
                                        const LPStandardFormBV&      sfbv,
                                        const Model&                 model,
                                        const std::vector<uint32_t>& equalArtCol = {});

} // namespace baguette::internal
