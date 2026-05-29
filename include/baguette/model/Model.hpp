#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "baguette/core/LinearExpr.hpp"
#include "baguette/core/Sense.hpp"
#include "baguette/core/Variable.hpp"
#include "baguette/cp/CPConstraints.hpp"
#include "baguette/model/ModelData.hpp"

namespace baguette {

/// User-facing API for building an optimization model.
///
/// The modeling layer is called once before the solve — it may use
/// convenient data structures without impacting solver performance.
/// Internal hot/cold separation ensures the solver's memory layout
/// remains cache-friendly regardless of how the model was built.
class Model {
public:
    /// Add a continuous decision variable with an optional label.
    ///
    /// @param lb    Lower bound.
    /// @param ub    Upper bound.
    /// @param label Optional name for display and debugging.
    /// @return A Variable handle valid for the lifetime of this Model.
    /// @throws std::invalid_argument if lb > ub.
    Variable addVar(double lb, double ub, std::string label = "");

    /// Add a decision variable with an explicit type and optional label.
    ///
    /// @param lb    Lower bound.
    /// @param ub    Upper bound.
    /// @param type  Variable type (Continuous, Integer, Binary).
    /// @param label Optional name for display and debugging.
    /// @return A Variable handle valid for the lifetime of this Model.
    /// @throws std::invalid_argument if lb > ub.
    Variable addVar(double lb, double ub, VarType type, std::string label = "");

    /// Add a constraint and return its ConstraintId for later inspection.
    ///
    /// The original form (possibly two-sided) is stored and returned by
    /// getLPConstraint().  A normalized copy (rhs variable terms moved to lhs)
    /// is stored for the solver and returned by getLPConstraints().
    ///
    /// @throws std::out_of_range if any variable in @p c does not belong to this Model.
    ConstraintId addLPConstraint(LPConstraint c);

    /// Convenience overload — equivalent to addLPConstraint(lhs sense rhsConst).
    /// @throws std::out_of_range if any variable in @p lhs does not belong to this Model.
    ConstraintId addLPConstraint(LinearExpr lhs, Sense sense, double rhs) {
        return addLPConstraint(LPConstraint{std::move(lhs), sense, {}, rhs});
    }

    /// Set the objective function and optimization direction.
    ///
    /// Converts the sparse LinearExpr into the dense `hot.obj` vector.
    /// The constant term of @p expr is stored and added back to the reported
    /// objective value after solving (it does not affect the optimal solution).
    /// @throws std::out_of_range if any variable in @p expr does not belong to this Model.
    void setObjective(LinearExpr expr,
                      ObjSense sense = ObjSense::Minimize);

    /// Update the right-hand side of a constraint in place.
    ///
    /// Used by MILP presolve to tighten constraint RHS values when all
    /// variables are integral (PR1 — integer RHS rounding).
    /// @note No validation of @p conIdx is performed (intentional: same
    ///       philosophy as setVarBounds — fast path, caller is responsible).
    void setConstraintRHS(uint32_t conIdx, double newRhs);

    /// Update the bounds of @p var in place.
    ///
    /// This is the preferred method for Branch & Bound hot loops.  Modify
    /// bounds before solving a child node, then restore them for backtracking:
    ///
    /// @code
    ///   double savedLb = model.getHot().lb[x.id];
    ///   double savedUb = model.getHot().ub[x.id];
    ///   model.setVarBounds(x, newLb, newUb);
    ///   auto child = solveDualDetailed(model, 0, 0.0, startTime, parentBasis);
    ///   model.setVarBounds(x, savedLb, savedUb); // backtrack
    /// @endcode
    ///
    /// @warning The finiteness of bounds must not change relative to the solve
    ///          that produced the BasisRecord you plan to warm-start from.
    ///          See withVarBounds() for details.
    ///
    /// @note No validation of newLb ≤ newUb is performed here.
    ///       The solver detects empty domains via its early-infeasibility check.
    /// @note No validation of @p var.id is performed here (intentional: this
    ///       method sits on the critical path of the B&B hot loop and a bounds
    ///       check would add overhead to every node).  @p var must have been
    ///       obtained from addVar() on *this* model; passing a Variable from a
    ///       different Model instance is undefined behaviour.  A debug-mode
    ///       assertion catches this during development.
    void setVarBounds(Variable var, double newLb, double newUb);

    /// Return a copy of this model with updated bounds for @p var.
    ///
    /// Convenient for tests and single-shot exploration but incurs an O(model)
    /// copy (all lb/ub/obj/constraint vectors).  For B&B hot loops, prefer
    /// setVarBounds() with manual save/restore to avoid per-node allocations.
    ///
    /// @warning The finiteness of bounds must not change relative to the
    ///          model used to produce the BasisRecord you plan to warm-start
    ///          from.  Specifically, a variable that had an infinite bound in
    ///          the parent must keep an infinite bound in the child, and vice
    ///          versa.  Changing finiteness alters the number of rows/columns
    ///          in the standard form (upper-bound rows, free-split columns),
    ///          making the parent basis incompatible.  In that case
    ///          solveDualDetailed() falls back automatically to a cold start.
    ///
    /// @note No validation of newLb ≤ newUb is performed here.
    ///       The solver detects empty domains via its early-infeasibility check.
    /// @note @p var must have been obtained from addVar() on *this* model.
    ///       Passing a Variable from a different Model instance is undefined
    ///       behaviour.
    Model withVarBounds(Variable var, double newLb, double newUb) const;

    /// Add a ghost variable: fixed at @p fixedVal, visible to CP only.
    ///
    /// Ghost variables occupy the tail of the internal arrays ([numVars(), numTotalVars())).
    /// They must be added after all LP variables.  The LP solver sees only
    /// [0, numVars()) and is unaffected; CP constraints reference them by their
    /// full ID so propagation naturally enforces their fixed value.
    Variable addGhostVar(double fixedVal, VarType type, std::string label = "");

    /// @return Number of LP decision variables (ghost variables excluded).
    std::size_t numVars()        const { return hot.lb.size() - ghostVarCount; }
    /// @return Total variable count including ghost (CP-only) variables.
    std::size_t numTotalVars()   const { return hot.lb.size(); }
    /// @return Number of constraints added via addLPConstraint().
    std::size_t numConstraints() const { return constraints_.size(); }

    /// @return Hot data (bounds, objective coefficients) for solver access.
    const ModelHot&  getHot()  const { return hot; }
    /// @return Cold data (labels, types) for model inspection and output.
    const ModelCold& getCold() const { return cold; }

    /// @return All constraints in normalized form (rhs always empty, lhs.constant == 0).
    ///
    /// This is the solver-facing view: every element satisfies isNormalized().
    /// Do **not** use this for user-level debugging — the original two-sided form
    /// may have been simplified.  Use getLPConstraint(id) for the original.
    const std::vector<LPConstraint>& getLPConstraints() const { return constraints_; }

    /// @return The original constraint as submitted by the user (possibly two-sided).
    ///
    /// Preserves the form passed to addLPConstraint() for debugging: if the user
    /// wrote `expr1 >= expr2`, this returns that two-sided form.
    /// @throws std::out_of_range if @p id is invalid.
    LPConstraint getLPConstraint(ConstraintId id) const {
        if (id >= originals_.size())
            throw std::out_of_range("getLPConstraint: invalid ConstraintId");
        return originals_[id];
    }
    /// @return Optimization direction (Minimize or Maximize).
    ObjSense                      getObjSense()     const { return objSense; }
    /// @return Constant offset of the objective (from the constant term of setObjective()).
    double                        getObjConstant()  const { return objConstant; }

    /// Add a single CP constraint (built-in or user-defined).
    /// Updates cold.varToCP for built-in constraints.
    void addCPConstraint(BuiltinConstraint c);
    void addCPConstraint(std::shared_ptr<const CPConstraint> c) { cpConstraints.add(std::move(c)); }

    /// @return The CP constraints attached to this model.
    const CPConstraints& getCPConstraints() const { return cpConstraints; }

private:
    ModelHot      hot;
    ModelCold     cold;
    CPConstraints cpConstraints;
    std::vector<LPConstraint> constraints_; ///< Normalized forms — solver view.
    std::vector<LPConstraint> originals_;   ///< Original forms — user debug view.
    ObjSense objSense      = ObjSense::Minimize;
    double   objConstant   = 0.0;
    uint32_t ghostVarCount = 0;
};

} // namespace baguette
