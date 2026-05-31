#pragma once
// Shared LP problem definitions - used by test_lp_methods.cpp and per-method
// test files (test_lp.cpp, test_dual_simplex.cpp, test_revised_simplex.cpp).

#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "baguette/core/Sense.hpp"
#include "baguette/lp/LPResult.hpp"
#include "baguette/model/Model.hpp"

// File-scope constexpr: accessible inside lambdas without capture.
inline constexpr double kLPInf = std::numeric_limits<double>::infinity();

/// One LP problem: a model builder, the expected solve status, and (when
/// Optimal) the expected objective value that every solving method must return.
struct LPTestCase {
    std::string name;
    baguette::LPStatus expectedStatus;
    double expectedObj;                    // meaningful only when Optimal
    std::function<baguette::Model()> build;
};

/// Returns all shared LP problems used by the cross-method parametrized suite.
inline std::vector<LPTestCase> makeLPTestSuite() {
    using namespace baguette;

    return {
        {"simple_max", LPStatus::Optimal, 4.0, []() {
            // max x+y  s.t. x+y<=4, x<=3, y<=3  →  obj=4
            Model m;
            auto x = m.addVar(0.0, kLPInf, "x");
            auto y = m.addVar(0.0, kLPInf, "y");
            m.addLPConstraint(1.0*x + 1.0*y, Sense::LessEq, 4.0);
            m.addLPConstraint(1.0*x,          Sense::LessEq, 3.0);
            m.addLPConstraint(1.0*y,          Sense::LessEq, 3.0);
            m.setObjective(1.0*x + 1.0*y, ObjSense::Maximize);
            return m;
        }},
        {"min_geq", LPStatus::Optimal, 8.0, []() {
            // min 2x+3y  s.t. x+y>=4, 2x+y>=6  →  x=4, y=0, obj=8
            Model m;
            auto x = m.addVar(0.0, kLPInf, "x");
            auto y = m.addVar(0.0, kLPInf, "y");
            m.addLPConstraint(1.0*x + 1.0*y, Sense::GreaterEq, 4.0);
            m.addLPConstraint(2.0*x + 1.0*y, Sense::GreaterEq, 6.0);
            m.setObjective(2.0*x + 3.0*y, ObjSense::Minimize);
            return m;
        }},
        {"equality", LPStatus::Optimal, 5.0, []() {
            // min x+y  s.t. x+y=5  →  obj=5
            Model m;
            auto x = m.addVar(0.0, kLPInf, "x");
            auto y = m.addVar(0.0, kLPInf, "y");
            m.addLPConstraint(1.0*x + 1.0*y, Sense::Equal, 5.0);
            m.setObjective(1.0*x + 1.0*y, ObjSense::Minimize);
            return m;
        }},
        {"infeasible", LPStatus::Infeasible, 0.0, []() {
            // x>=3 AND x<=2  →  infeasible
            Model m;
            auto x = m.addVar(0.0, kLPInf, "x");
            m.addLPConstraint(1.0*x, Sense::GreaterEq, 3.0);
            m.addLPConstraint(1.0*x, Sense::LessEq,    2.0);
            m.setObjective(1.0*x, ObjSense::Minimize);
            return m;
        }},
        {"unbounded", LPStatus::Unbounded, 0.0, []() {
            // min -x,  x>=0  →  unbounded
            Model m;
            auto x = m.addVar(0.0, kLPInf, "x");
            m.addLPConstraint(1.0*x, Sense::GreaterEq, 0.0);
            m.setObjective(-1.0*x, ObjSense::Minimize);
            return m;
        }},
        {"upper_bound", LPStatus::Optimal, 5.0, []() {
            // max x,  0<=x<=5  →  obj=5
            Model m;
            auto x = m.addVar(0.0, 5.0, "x");
            m.setObjective(1.0*x, ObjSense::Maximize);
            return m;
        }},
        {"lower_bound", LPStatus::Optimal, 3.0, []() {
            // min x,  3<=x<=10  →  obj=3
            Model m;
            auto x = m.addVar(3.0, 10.0, "x");
            m.setObjective(1.0*x, ObjSense::Minimize);
            return m;
        }},
        {"simple_min_leq", LPStatus::Optimal, 4.0, []() {
            // min x+y  s.t. x+y>=4, x<=5, y<=5  →  obj=4
            Model m;
            auto x = m.addVar(0.0, kLPInf, "x");
            auto y = m.addVar(0.0, kLPInf, "y");
            m.addLPConstraint(1.0*x + 1.0*y, Sense::GreaterEq, 4.0);
            m.addLPConstraint(1.0*x,          Sense::LessEq,    5.0);
            m.addLPConstraint(1.0*y,          Sense::LessEq,    5.0);
            m.setObjective(1.0*x + 1.0*y, ObjSense::Minimize);
            return m;
        }},
        {"three_var", LPStatus::Optimal, -26.0, []() {
            // min -x-2y-3z  s.t. x+y+z<=10, x,y,z in [0,6]  →  obj=-26
            Model m;
            auto x = m.addVar(0.0, 6.0, "x");
            auto y = m.addVar(0.0, 6.0, "y");
            auto z = m.addVar(0.0, 6.0, "z");
            m.addLPConstraint(1.0*x + 1.0*y + 1.0*z, Sense::LessEq, 10.0);
            m.setObjective(-1.0*x + -2.0*y + -3.0*z, ObjSense::Minimize);
            return m;
        }},
        {"max_5x4y", LPStatus::Optimal, 21.0, []() {
            // max 5x+4y  s.t. 6x+4y<=24, x+2y<=6  →  obj=21
            Model m;
            auto x = m.addVar(0.0, kLPInf, "x");
            auto y = m.addVar(0.0, kLPInf, "y");
            m.addLPConstraint(6.0*x + 4.0*y, Sense::LessEq, 24.0);
            m.addLPConstraint(1.0*x + 2.0*y, Sense::LessEq,  6.0);
            m.setObjective(5.0*x + 4.0*y, ObjSense::Maximize);
            return m;
        }},
    };
}

// ── Canonical builders reused by per-method test files ────────────────────────

inline baguette::Model makeSimpleMax() {
    using namespace baguette;
    Model m;
    auto x = m.addVar(0.0, kLPInf, "x");
    auto y = m.addVar(0.0, kLPInf, "y");
    m.addLPConstraint(1.0*x + 1.0*y, Sense::LessEq, 4.0);
    m.addLPConstraint(1.0*x,          Sense::LessEq, 3.0);
    m.addLPConstraint(1.0*y,          Sense::LessEq, 3.0);
    m.setObjective(1.0*x + 1.0*y, ObjSense::Maximize);
    return m;
}

inline baguette::Model makeMinWithGEQ() {
    using namespace baguette;
    Model m;
    auto x = m.addVar(0.0, kLPInf, "x");
    auto y = m.addVar(0.0, kLPInf, "y");
    m.addLPConstraint(1.0*x + 1.0*y, Sense::GreaterEq, 4.0);
    m.addLPConstraint(2.0*x + 1.0*y, Sense::GreaterEq, 6.0);
    m.setObjective(2.0*x + 3.0*y, ObjSense::Minimize);
    return m;
}

inline baguette::Model makeEqualityConstraint() {
    using namespace baguette;
    Model m;
    auto x = m.addVar(0.0, kLPInf, "x");
    auto y = m.addVar(0.0, kLPInf, "y");
    m.addLPConstraint(1.0*x + 1.0*y, Sense::Equal, 5.0);
    m.setObjective(1.0*x + 1.0*y, ObjSense::Minimize);
    return m;
}

inline baguette::Model makeInfeasible() {
    using namespace baguette;
    Model m;
    auto x = m.addVar(0.0, kLPInf, "x");
    m.addLPConstraint(1.0*x, Sense::GreaterEq, 3.0);
    m.addLPConstraint(1.0*x, Sense::LessEq,    2.0);
    m.setObjective(1.0*x, ObjSense::Minimize);
    return m;
}

inline baguette::Model makeUnbounded() {
    using namespace baguette;
    Model m;
    auto x = m.addVar(0.0, kLPInf, "x");
    m.addLPConstraint(1.0*x, Sense::GreaterEq, 0.0);
    m.setObjective(-1.0*x, ObjSense::Minimize);
    return m;
}

inline baguette::Model makeUpperBound() {
    using namespace baguette;
    Model m;
    auto x = m.addVar(0.0, 5.0, "x");
    m.setObjective(1.0*x, ObjSense::Maximize);
    return m;
}

inline baguette::Model makeLowerBound() {
    using namespace baguette;
    Model m;
    auto x = m.addVar(3.0, 10.0, "x");
    m.setObjective(1.0*x, ObjSense::Minimize);
    return m;
}

inline baguette::Model makeSimpleMinLEQ() {
    using namespace baguette;
    Model m;
    auto x = m.addVar(0.0, kLPInf, "x");
    auto y = m.addVar(0.0, kLPInf, "y");
    m.addLPConstraint(1.0*x + 1.0*y, Sense::GreaterEq, 4.0);
    m.addLPConstraint(1.0*x,          Sense::LessEq,    5.0);
    m.addLPConstraint(1.0*y,          Sense::LessEq,    5.0);
    m.setObjective(1.0*x + 1.0*y, ObjSense::Minimize);
    return m;
}

