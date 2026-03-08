#include "baguette/model/Model.hpp"

#include <algorithm>
#include <stdexcept>

namespace baguette {

Variable Model::addVar(double lb, double ub, VarType type, std::string label) {
    auto id = static_cast<std::uint32_t>(hot.lb.size());

    hot.lb.push_back(lb);
    hot.ub.push_back(ub);
    hot.obj.push_back(0.0);

    cold.labels.push_back(std::move(label));
    cold.types.push_back(type);

    return Variable{id};
}

void Model::addConstraint(LinearExpr lhs, Sense sense, double rhs) {
    constraints.push_back({std::move(lhs), sense, rhs});
}

void Model::setObjective(LinearExpr expr, ObjSense sense) {
    objSense = sense;

    std::fill(hot.obj.begin(), hot.obj.end(), 0.0);

    for (std::size_t i = 0; i < expr.varIds.size(); ++i) {
        if (expr.varIds[i] >= hot.obj.size())
            throw std::out_of_range("setObjective: variable ID out of range (wrong model?)");
        hot.obj[expr.varIds[i]] = expr.coeffs[i];
    }
}

} // namespace baguette
