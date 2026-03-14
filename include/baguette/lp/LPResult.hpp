#pragma once

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

/// Extended result returned by solveDetailed().
///
/// Contains an LPResult for the basic outcome; the additional fields
/// (dualValues, reducedCosts, basis) are valid only when result.status == Optimal.
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
};

} // namespace baguette
