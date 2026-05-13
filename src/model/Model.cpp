#include "baguette/model/Model.hpp"

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <variant>

namespace baguette {

Variable Model::addVar(double lb, double ub, std::string label) {
    return addVar(lb, ub, VarType::Continuous, std::move(label));
}

Variable Model::addVar(double lb, double ub, VarType type, std::string label) {
    if (lb > ub)
        throw std::invalid_argument("Model::addVar: lb > ub for variable '" + label + "'");

    auto id = static_cast<std::uint32_t>(hot.lb.size());

    hot.lb.push_back(lb);
    hot.ub.push_back(ub);
    hot.obj.push_back(0.0);

    cold.labels.push_back(std::move(label));
    cold.types.push_back(type);
    cold.varToLP.emplace_back();
    cold.varToCP.emplace_back();

    return Variable{id};
}

Variable Model::addGhostVar(double fixedVal, VarType type, std::string label) {
    Variable v = addVar(fixedVal, fixedVal, type, std::move(label));
    ++ghostVarCount;
    return v;
}

void Model::addLPConstraint(LinearExpr lhs, Sense sense, double rhs) {
    for (uint32_t id : lhs.varIds)
        if (id >= hot.lb.size())
            throw std::out_of_range("addLPConstraint: variable ID " + std::to_string(id) + " out of range (wrong model?)");

    const auto conIdx = static_cast<uint32_t>(constraints.size());
    for (uint32_t k = 0; k < static_cast<uint32_t>(lhs.varIds.size()); ++k)
        cold.varToLP[lhs.varIds[k]].push_back({conIdx, k});

    constraints.push_back({std::move(lhs), sense, rhs});
}

void Model::setVarBounds(Variable var, double newLb, double newUb) {
    assert(var.id < hot.lb.size() && "setVarBounds: variable does not belong to this model");
    hot.lb[var.id] = newLb;
    hot.ub[var.id] = newUb;
}

Model Model::withVarBounds(Variable var, double newLb, double newUb) const {
    Model copy = *this;
    copy.setVarBounds(var, newLb, newUb);
    return copy;
}

void Model::setObjective(LinearExpr expr, ObjSense sense) {
    objSense    = sense;
    objConstant = expr.constant;

    std::fill(hot.obj.begin(), hot.obj.end(), 0.0);

    for (std::size_t i = 0; i < expr.varIds.size(); ++i) {
        if (expr.varIds[i] >= hot.obj.size())
            throw std::out_of_range("setObjective: variable ID out of range (wrong model?)");
        hot.obj[expr.varIds[i]] += expr.coeffs[i];
    }
}

void Model::addCPConstraint(BuiltinConstraint c) {
    const auto cpIdx = static_cast<uint32_t>(cpConstraints.builtins().size());
    std::visit([&](const auto& con) {
        using T = std::decay_t<decltype(con)>;
        if constexpr (std::is_same_v<T, AllDiffConstraint>) {
            for (const Variable& v : con.vars)
                cold.varToCP[v.id].push_back(cpIdx);
        } else if constexpr (std::is_same_v<T, CumulativeConstraint>) {
            for (const Task& t : con.tasks)
                cold.varToCP[t.varStart.id].push_back(cpIdx);
        }
    }, c);
    cpConstraints.add(std::move(c));
}

} // namespace baguette
