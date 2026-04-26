#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace baguette { namespace internal { struct LPStandardForm; } }

namespace baguette {

/// Status returned by the LP solver.
enum class LPStatus {
    Optimal,          ///< Solved to optimality.
    Infeasible,       ///< Problem has no feasible solution (phase-I objective > 0).
    Unbounded,        ///< Objective is unbounded below (Minimize) or above (Maximize).
    MaxIter,          ///< Iteration limit reached without conclusion.
    TimeLimit,        ///< Wall-clock time limit reached without conclusion.
    NumericalFailure  ///< Basis reinversion failed; tableau state is undefined.
                      ///  Feasibility of the problem is unknown. primalValues is empty.
};

/// Result returned by solve().
///
/// `primalValues` is populated whenever a feasible primal solution is available,
/// including when status is MaxIter or TimeLimit (if phase II had already started).
/// It is empty when status is Infeasible or Unbounded.
struct LPResult {
    LPStatus status         = LPStatus::Infeasible;

    /// Current objective value. Valid when Optimal; best-known value for
    /// MaxIter / TimeLimit if a feasible solution was reached.
    double objectiveValue   = 0.0;

    /// Primal solution indexed by Variable::id (size == Model::numVars()).
    /// Valid when Optimal; best-known when MaxIter or TimeLimit with feasible point.
    /// Empty when Infeasible or Unbounded.
    std::vector<double> primalValues;
};

// ── Advanced result ─────────────────────────────────────────────────────────

/// Identifies the origin of a column in the standard-form LP.
/// Stored in BasisRecord so that B&B can interpret each column without
/// looking inside the tableau.
enum class ColumnKind : uint8_t {
    Original,    ///< A shifted original variable  x' = x − lb.
    Slack,       ///< Slack / surplus for a model constraint row.
    UpperSlack,  ///< Slack for an upper-bound row  x' + s = ub − lb.
    FreeNeg      ///< Negative part x⁻ of a fully free variable split as x = x⁺ − x⁻.
};

/// Compact description of the current basis, suitable for warm-starting a
/// child B&B node.
///
/// Column indices refer to the standard-form column space, which is stable
/// across B&B nodes as long as variable bounds change but no new variables
/// are added.
///
/// The `sfCache` field carries the standard form built during the last
/// solveDualDetailed() call.  When passed back as @p warmBasis, the solver
/// reuses the cached constraint matrix A (O(1) copy via shared_ptr) and only
/// recomputes the RHS vector b for the new bounds, avoiding an O(m·n)
/// rebuild of the full standard form.
struct BasisRecord {
    /// basicCols[i] = index of the basic column in row i.
    /// Length == number of rows in the standard form.
    std::vector<uint32_t> basicCols;

    /// Kind of each column in the standard form (length == total columns).
    std::vector<ColumnKind> colKind;

    /// Origin of each column (length == total columns).
    /// ColumnKind::Original   → Variable::id of the corresponding model variable.
    /// ColumnKind::Slack      → index of the corresponding model constraint.
    /// ColumnKind::UpperSlack → index of the corresponding model variable.
    std::vector<uint32_t> colOrigin;

    /// Cached standard form for bounds-only warm restart in solveDualDetailed().
    /// Populated only when result.status == Optimal.  Consumers must not modify
    /// or interpret this field; it is internal to the solver.
    std::shared_ptr<internal::LPStandardForm> sfCache;
};

/// Farkas infeasibility certificate for the LP.
///
/// Provides a machine-verifiable proof of infeasibility in one of two forms:
///
/// **Tableau certificate** (`y` non-empty, `infeasVarId == -1`):
///   A vector `y` (size == Model::numConstraints()) such that
///     A_model^T y >= 0   (component-wise, original constraint coefficients)
///     b_model^T y < 0    (original RHS values)
///   Populated when infeasibility is detected by the simplex tableau
///   (dual-simplex blocking row or primal phase-I objective > 0).
///
/// **Bound certificate** (`y` empty, `infeasVarId >= 0`):
///   Variable `infeasVarId` has lb > ub (domain is empty after B&B branching).
///   Machine-verifiable: check model.getHot().lb[infeasVarId] > model.getHot().ub[infeasVarId].
///
/// Both fields are zero-initialised; at most one is set per Infeasible result.
struct FarkasRay {
    /// Farkas multipliers for model constraints (size == Model::numConstraints()).
    /// Empty for bound-violation infeasibility or when not available.
    std::vector<double> y;

    /// Variable id where lb > ub caused early infeasibility, or -1 if not applicable.
    int32_t infeasVarId = -1;
};

/// Sensitivity analysis: ranges for the current optimal basis to remain valid.
///
/// All ranges are expressed as actual parameter values (not deltas).
/// ±infinity entries indicate the bound is unlimited in that direction.
///
/// Interpretation:
///   - rhsRange[i] = {lo, hi}: b[i] can take any value in [lo, hi] while
///     the current basis remains primal feasible (and hence optimal, since
///     dual feasibility is preserved when only b changes).
///   - objRange[j] = {lo, hi}: c[j] can take any value in [lo, hi] while
///     the current basis remains dual feasible (and hence optimal, since
///     primal feasibility is preserved when only c changes).
///
/// Valid only when LPDetailedResult::result.status == Optimal.
struct SensitivityResult {
    /// RHS ranging, indexed by model constraint index.
    /// Size == Model::numConstraints().
    std::vector<std::array<double, 2>> rhsRange;

    /// Objective ranging, indexed by Variable::id.
    /// Size == Model::numVars().
    /// Fully free variables (lb = ub = ±∞) are reported as [−∞, +∞].
    std::vector<std::array<double, 2>> objRange;
};

/// Raw tableau row for a basic integer variable at a fractional optimal value.
///
/// Extracted by solveDualDetailed() when computeCutData == true.
/// Consumed by the MILP cutting-plane module (CuttingPlanes) to generate
/// Gomory Mixed-Integer (GMI) cuts; no MILP semantics are embedded here.
struct FractionalRow {
    /// Variable::id of the integer-typed model variable basic in this row.
    uint32_t origVarId = 0;

    /// Fractional part of the basic variable's optimal value: frac ∈ (0, 1).
    double fracVal = 0.0;

    /// B⁻¹A row for this tableau row, indexed 0 .. BasisRecord::colKind.size()-1.
    /// Entry j is (B⁻¹A)_{r,j}; basic columns have value 0 or 1 (identity) and
    /// can be skipped during cut generation.
    std::vector<double> tabRow;
};

/// Extended result returned by solveDetailed().
///
/// Contains an LPResult for the basic outcome; the additional fields are
/// valid only when noted:
///   - dualValues, reducedCosts, basis, sensitivity : valid when result.status == Optimal.
///   - farkas                                       : valid when result.status == Infeasible.
///   - fractionalRows                               : valid when result.status == Optimal
///                                                   and computeCutData == true.
///
/// Access the basic result via the public `result` member.
struct LPDetailedResult {
    /// Basic result: status, objective value, and primal solution.
    LPResult result;

    /// Dual variables (shadow prices), one per original model constraint.
    /// Size == Model::numConstraints(). Valid only when result.status == Optimal.
    std::vector<double> dualValues;

    /// Reduced costs for original model variables, indexed by Variable::id.
    /// Size == Model::numVars(). Valid only when result.status == Optimal.
    std::vector<double> reducedCosts;

    /// Basis record for B&B warm-start.
    /// Valid only when result.status == Optimal.
    BasisRecord basis;

    /// Farkas infeasibility certificate.
    /// Valid when result.status == Infeasible.
    /// farkas.y is non-empty for tableau-detected infeasibility.
    /// farkas.infeasVarId >= 0 for early lb > ub detection.
    FarkasRay farkas;

    /// Sensitivity analysis: RHS and objective coefficient ranging.
    /// Valid only when result.status == Optimal.
    SensitivityResult sensitivity;

    /// Raw fractional integer rows from the optimal tableau.
    /// Populated only when computeCutData == true in solveDualDetailed().
    /// Empty when status != Optimal or no basic integer variable is fractional.
    /// Consumed by CuttingPlanes::generateGMICuts() — do not interpret directly.
    std::vector<FractionalRow> fractionalRows;

    /// True when the caller supplied a non-empty warmBasis and solveDualDetailed()
    /// successfully seeded the dual simplex from it (no sfCache mismatch, no
    /// reinversion failure, no dual-feasibility fallback to cold primal).
    /// False when no warm basis was provided or a fallback to cold primal occurred.
    bool usedWarmStart = false;
};

} // namespace baguette
