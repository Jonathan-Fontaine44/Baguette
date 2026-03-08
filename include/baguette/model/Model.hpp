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
    /// Add a decision variable and return its handle.
    ///
    /// @param lb    Lower bound.
    /// @param ub    Upper bound.
    /// @param type  Variable type (Continuous, Integer, Binary).
    /// @param label Optional name for display and debugging.
    /// @return A Variable handle valid for the lifetime of this Model.
    Variable addVar(double lb, double ub,
                    VarType type  = VarType::Continuous,
                    std::string label = "");

    /// Add a linear constraint: `lhs sense rhs`.
    /// @warning All variables in @p lhs must have been created by this Model.
    void addConstraint(LinearExpr lhs, Sense sense, double rhs);

    /// Set the objective function and optimization direction.
    ///
    /// Converts the sparse LinearExpr into the dense `hot.obj` vector.
    /// The constant term of @p expr is ignored (offsets do not affect
    /// the optimal solution).
    /// @warning All variables in @p expr must have been created by this Model.
    /// Throws std::out_of_range if a variable ID exceeds numVars().
    void setObjective(LinearExpr expr,
                      ObjSense sense = ObjSense::Minimize);

    std::size_t numVars()        const { return hot.lb.size(); }
    std::size_t numConstraints() const { return constraints.size(); }

    const ModelHot&               getHot()         const { return hot; }
    const ModelCold&              getCold()        const { return cold; }
    const std::vector<Constraint>& getConstraints() const { return constraints; }
    ObjSense                      getObjSense()    const { return objSense; }

private:
    ModelHot  hot;
    ModelCold cold;
    std::vector<Constraint> constraints;
    ObjSense objSense = ObjSense::Minimize;
};

} // namespace baguette
