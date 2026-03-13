#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "baguette/core/LinearExpr.hpp"
#include "baguette/core/Sense.hpp"
#include "baguette/core/Variable.hpp"
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

    /// Add a linear constraint: `lhs sense rhs`.
    /// @throws std::out_of_range if any variable in @p lhs does not belong to this Model.
    void addConstraint(LinearExpr lhs, Sense sense, double rhs);

    /// Set the objective function and optimization direction.
    ///
    /// Converts the sparse LinearExpr into the dense `hot.obj` vector.
    /// The constant term of @p expr is stored and added back to the reported
    /// objective value after solving (it does not affect the optimal solution).
    /// @throws std::out_of_range if any variable in @p expr does not belong to this Model.
    void setObjective(LinearExpr expr,
                      ObjSense sense = ObjSense::Minimize);

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

    /// @return Number of decision variables in the model.
    std::size_t numVars()        const { return hot.lb.size(); }
    /// @return Number of constraints added via addConstraint().
    std::size_t numConstraints() const { return constraints.size(); }

    /// @return Hot data (bounds, objective coefficients) for solver access.
    const ModelHot&               getHot()          const { return hot; }
    /// @return Cold data (labels, types) for model inspection and output.
    const ModelCold&              getCold()         const { return cold; }
    /// @return All constraints added via addConstraint().
    const std::vector<Constraint>& getConstraints() const { return constraints; }
    /// @return Optimization direction (Minimize or Maximize).
    ObjSense                      getObjSense()     const { return objSense; }
    /// @return Constant offset of the objective (from the constant term of setObjective()).
    double                        getObjConstant()  const { return objConstant; }

private:
    ModelHot  hot;
    ModelCold cold;
    std::vector<Constraint> constraints;
    ObjSense objSense    = ObjSense::Minimize;
    double   objConstant = 0.0;
};

} // namespace baguette
