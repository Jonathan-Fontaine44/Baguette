#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "baguette/lp/LPSolver.hpp"
#include "baguette/milp/BranchAndBound.hpp"
#include "baguette/milp/SecCuts.hpp"
#include "baguette/model/Model.hpp"
#include "baguette/model/ModelEnums.hpp"

using namespace baguette;
using Catch::Matchers::WithinAbs;

static constexpr double kTol = 1e-6;

// Helper: builds a 6-city symmetric TSP model with binary edge variables.
//
// Edge costs: 0 within each triangle ({0,1,2} and {3,4,5}), 100 across.
// LP relaxation (degree constraints only, no SEC) is integral:
//   x01=x02=x12=x34=x35=x45=1, all cross-edges=0  â†’  cost 0.
// This violates SEC for S={0,1,2}: x01+x02+x12=3 > |S|-1=2.
// Optimal Hamiltonian tour cost: 2 cross-edges Ã- 100 = 200.

static void buildTSP6(Model& m, std::vector<std::vector<Variable>>& xv) {
    const int n = 6;
    const double cost[6][6] = {
        {  0,  0,  0, 100, 100, 100},
        {  0,  0,  0, 100, 100, 100},
        {  0,  0,  0, 100, 100, 100},
        {100, 100, 100,  0,   0,   0},
        {100, 100, 100,  0,   0,   0},
        {100, 100, 100,  0,   0,   0},
    };

    xv.assign(n, std::vector<Variable>(n, Variable{0}));
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) {
            Variable v = m.addVar(0.0, 1.0, VarType::Binary,
                                  "x" + std::to_string(i) + std::to_string(j));
            xv[i][j] = xv[j][i] = v;
        }

    // Degree constraints: Î£â±¼ xáµ¢â±¼ = 2 for each city i.
    for (int i = 0; i < n; i++) {
        LinearExpr deg;
        for (int j = 0; j < n; j++) {
            if (j == i) continue;
            deg.addTerm(xv[i][j], 1.0);
        }
        m.addLPConstraint(deg, Sense::Equal, 2.0);
    }

    LinearExpr obj;
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            obj.addTerm(xv[i][j], cost[i][j]);
    m.setObjective(obj, ObjSense::Minimize);
}

// â”€â”€ SEC: Stoer-Wagner separation detects violated subtour â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//
// LP relaxation gives x01=x02=x12=x34=x35=x45=1 (two triangles, cost 0).
// SEC for S={0,1,2}: x01+x02+x12=3 > 2 â†’ violated.
// After adding the SEC, the LP optimal must rise above 0.

TEST_CASE("SEC: generator detects violated subtour in 6-city two-triangle LP", "[sec]") {
    const int n = 6;
    Model m;
    std::vector<std::vector<Variable>> xv;
    buildTSP6(m, xv);

    LPOptions lpOpts;
    lpOpts.enablePresolve = false;
    lpOpts.timeLimitS     = 2.0;
    LPDetailedResult lp = solveLPDetailed(m, lpOpts);

    REQUIRE(lp.result.status == LPStatus::Optimal);
    REQUIRE_THAT(lp.result.objectiveValue, WithinAbs(0.0, kTol));

    CutGenerator gen = makeSecGenerator(n, xv);
    std::vector<Cut> cuts = gen(lp, m);
    REQUIRE(cuts.size() >= 1);

    const Cut& cut = cuts[0];
    REQUIRE(cut.sense == Sense::LessEq);

    // LP solution must violate the cut.
    double lpLhs = 0.0;
    for (std::size_t k = 0; k < cut.expr.size(); ++k) {
        uint32_t id = cut.expr.varIds[k];
        if (id < lp.result.primalValues.size())
            lpLhs += cut.expr.coeffs[k] * lp.result.primalValues[id];
    }
    REQUIRE(lpLhs > cut.rhs + kTol);

    // A Hamiltonian tour (e.g. 0-1-2-3-4-5-0) satisfies every SEC:
    // each proper subset S has at most |S|-1 tour edges inside it.
    // Encode tour 0-1-2-3-4-5-0 in a value map.
    const uint32_t tourEdges[] = {
        xv[0][1].id, xv[1][2].id, xv[2][3].id,
        xv[3][4].id, xv[4][5].id, xv[0][5].id,
    };
    double tourLhs = 0.0;
    for (std::size_t k = 0; k < cut.expr.size(); ++k) {
        uint32_t id = cut.expr.varIds[k];
        double   val = 0.0;
        for (uint32_t eid : tourEdges) if (eid == id) { val = 1.0; break; }
        tourLhs += cut.expr.coeffs[k] * val;
    }
    REQUIRE(tourLhs <= cut.rhs + kTol);

    // Adding the SEC tightens the LP bound above 0.
    m.addLPConstraint(cut.expr, cut.sense, cut.rhs);
    LPDetailedResult lp2 = solveLPDetailed(m, lpOpts);
    REQUIRE(lp2.result.status == LPStatus::Optimal);
    REQUIRE(lp2.result.objectiveValue > kTol);
}

// â”€â”€ SEC via CutGenerator in B&B finds optimal Hamiltonian tour â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//
// Full B&B on the 6-city TSP.  Without SEC the LP bound is 0 (two triangles).
// With the CutGenerator the solver adds SECs at the root and converges to the
// optimal Hamiltonian tour of cost 200 (exactly 2 cross-edges).

TEST_CASE("SEC: B&B with makeSecGenerator finds optimal Hamiltonian tour", "[sec][bb]") {
    const int n = 6;
    Model m;
    std::vector<std::vector<Variable>> xv;
    buildTSP6(m, xv);

    BBOptions opts;
    opts.enableCuts      = false;
    opts.enableMIR       = false;
    opts.collectStats    = true;
    opts.presolveLevel  = 0;
    opts.enableElimination = false;
    opts.timeLimitS      = 10.0;
    opts.cutGenerators.push_back(makeSecGenerator(n, xv));

    MILPResult r = solveMILP(m, opts);

    REQUIRE(r.status == MILPStatus::Optimal);
    REQUIRE_THAT(r.objectiveValue, WithinAbs(200.0, kTol));
    REQUIRE(r.stats->cutsAdded >= 1);
}
