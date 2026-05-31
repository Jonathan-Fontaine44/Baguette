#include <catch2/catch_test_macros.hpp>
#include <sstream>
#include <string>

#include "baguette/core/Sense.hpp"
#include "baguette/milp/BranchAndBound.hpp"
#include "baguette/model/Model.hpp"
#include "baguette/model/ModelEnums.hpp"
#include "lp/MILP_problems.hpp"

using namespace baguette;

// ── Local builder ─────────────────────────────────────────────────────────────
//
// Cascade50: 10 groups × 5 integer vars ∈ [0,10].
// Each var is bounded above by 3.9 (so the tightest integer UB is 3).
// Each group must sum to at least 13.5 (tightest integer LB is 14).
// Minimum solution: each var = 3 in one group member and the rest balanced.
//
// Without presolve: LP relaxation is feasible but the integer tree has domain
// [0,10]^50 ≈ 10^50 nodes — exhaustion is impossible.
// With presolve: domains collapse to [2,3], tree becomes trivial.
//
// Used exclusively in the TimeLimit test.
static Model makeCascadeHard() {
    const int n = 50;
    Model m;
    std::vector<Variable> x;
    x.reserve(n);
    for (int i = 0; i < n; ++i)
        x.push_back(m.addVar(0.0, 10.0, VarType::Integer,
                              "c" + std::to_string(i)));
    for (int g = 0; g < n / 5; ++g) {
        LinearExpr sum;
        for (int k = 0; k < 5; ++k) {
            const int vi = g * 5 + k;
            m.addLPConstraint(1.0 * x[vi], Sense::LessEq, 3.9);
            sum += 1.0 * x[vi];
        }
        m.addLPConstraint(sum, Sense::GreaterEq, 13.5);
    }
    LinearExpr obj;
    for (auto& v : x) obj += 1.0 * v;
    m.setObjective(obj, ObjSense::Minimize);
    return m;
}

// ── Helper ────────────────────────────────────────────────────────────────────

static int countSubstr(const std::string& s, const std::string& needle) {
    int count = 0;
    std::size_t pos = 0;
    while ((pos = s.find(needle, pos)) != std::string::npos) {
        ++count;
        pos += needle.size();
    }
    return count;
}

// ── Test 1: feasible MILP — branching required ────────────────────────────────
//
// Knapsack10 (10 binary vars, capacity 50, max profit).
// LP relaxation optimal is fractional → B&B required to prove IP optimality.
//
// Expected proof structure:
//   BB-PROOF header, N 0 -1 (root), BRANCH 0 (LP was fractional),
//   DIFF events for children, INCUMBENT (solution found), RESULT Optimal.
//
// Also verifies that the proof result is consistent with the solver result.

TEST_CASE("Proof: feasible knapsack with branching", "[proof]") {
    Model m = baguette_test::makeKnapsack10();

    BBOptions opts;
    opts.presolveLevel = 1; // default; presolve tightens but does not close the gap

    std::ostringstream proof;
    opts.proofStream = &proof;

    const MILPResult r = solveMILP(m, opts);
    REQUIRE(r.status == MILPStatus::Optimal);

    const std::string s = proof.str();
    INFO("Proof:\n" << s);

    // Header present
    REQUIRE(s.find("BB-PROOF 0.7.0") != std::string::npos);
    // Root node declared (id=0, parent=-1)
    REQUIRE(s.find("N 0 -1") != std::string::npos);
    // Root LP was fractional → branching occurred
    REQUIRE(s.find("BRANCH 0 ") != std::string::npos);
    // Children were created → DIFF events present
    REQUIRE(s.find("DIFF") != std::string::npos);
    // At least one integer-feasible node was found
    REQUIRE(s.find("INCUMBENT") != std::string::npos);
    // Final result matches
    REQUIRE(s.find("RESULT Optimal") != std::string::npos);
    // No numerical issues — proof is fully verifiable
    REQUIRE(s.find("UNVERIFIED") == std::string::npos);
}

// ── Test 2: trivially infeasible — LP at root proves infeasibility ─────────────
//
// Knapsack with capacity=5 AND minLoad=6: 5 < 6, so no x ∈ [0,1]^10 satisfies
// both constraints simultaneously.  The LP at the root is infeasible and the
// solver produces a Farkas certificate without branching.
//
// Expected proof structure:
//   BB-PROOF header, N 0 -1, LP 0 - INFEASIBLE, FARKAS (certificate),
//   RESULT Infeasible.  No BRANCH event.

TEST_CASE("Proof: trivially infeasible - root LP Farkas certificate", "[proof]") {
    // capacity=5 < minLoad=6 is contradictory for any x in [0,1]^10.
    Model m = baguette_test::makeKnapsack10(5.0, 6.0);

    BBOptions opts;
    // Disable presolve so the infeasibility is detected by the root LP, not
    // by bound propagation.  With presolve=1, presolveMILPInPlace would catch
    // the contradiction before the B&B loop starts.
    opts.presolveLevel = 0;

    std::ostringstream proof;
    opts.proofStream = &proof;

    const MILPResult r = solveMILP(m, opts);
    REQUIRE(r.status == MILPStatus::Infeasible);

    const std::string s = proof.str();
    INFO("Proof:\n" << s);

    REQUIRE(s.find("BB-PROOF 0.7.0") != std::string::npos);
    // Root node declared
    REQUIRE(s.find("N 0 -1") != std::string::npos);
    // Root LP was infeasible
    REQUIRE(s.find("LP 0 - INFEASIBLE") != std::string::npos);
    // At least one Farkas certificate emitted (FARKAS_LP or FARKAS_BOUND)
    REQUIRE(s.find("FARKAS") != std::string::npos);
    REQUIRE(s.find("RESULT Infeasible") != std::string::npos);
    // No branching was needed — LP directly proved infeasibility
    REQUIRE(s.find("BRANCH") == std::string::npos);
}

// ── Test 3: non-trivially infeasible — tree exhausted ─────────────────────────
//
// Model: x1 + x2 = 1.5,  x1, x2 ∈ {0, 1}.
//
// LP relaxation is feasible: x1=x2=0.75 satisfies the equality.
// Integer programming is infeasible: all four binary combinations give
// x1+x2 ∈ {0, 1, 2}, none equals 1.5.
//
// With presolveLevel=0, B&B must explore the full tree before concluding
// infeasibility.  Expected exploration (MostFractional, HybridPlunge DFS):
//
//   Root LP (x1=x2=0.75) → branch on x1 (id=0, both tied at frac=0.25):
//     Left  (x1=0, id=1): LP infeasible → FARKAS
//     Right (x1=1, id=2): LP x2=0.5 → branch on x2:
//       RL (x2=0, id=3): LP infeasible → FARKAS
//       RR (x2=1, id=4): LP infeasible → FARKAS
//
// 3 Farkas certificates and 3 INFEASIBLE LP events expected.

TEST_CASE("Proof: non-trivially infeasible - branching required to prove", "[proof]") {
    Model m;
    Variable x1 = m.addVar(0.0, 1.0, VarType::Binary, "x1");
    Variable x2 = m.addVar(0.0, 1.0, VarType::Binary, "x2");
    m.addLPConstraint(1.0 * x1 + 1.0 * x2, Sense::Equal, 1.5);
    m.setObjective(1.0 * x1 + 1.0 * x2, ObjSense::Minimize);

    BBOptions opts;
    // Presolve would fix x1=1 (lb rounds up to 0.5 → 1), then x2=0.5 → empty
    // domain, catching infeasibility without any B&B.  Disable to force the proof
    // to contain the full branching tree.
    opts.presolveLevel = 0;

    std::ostringstream proof;
    opts.proofStream = &proof;

    const MILPResult r = solveMILP(m, opts);
    REQUIRE(r.status == MILPStatus::Infeasible);

    const std::string s = proof.str();
    INFO("Proof:\n" << s);

    REQUIRE(s.find("BB-PROOF 0.7.0") != std::string::npos);
    REQUIRE(s.find("N 0 -1") != std::string::npos);
    // Root LP was optimal (x1=x2=0.75) — branching was required to prove infeasibility
    REQUIRE(s.find("BRANCH 0 ") != std::string::npos);
    // 3 LP infeasible events (one per leaf of the 5-node tree)
    REQUIRE(countSubstr(s, "INFEASIBLE") >= 3);
    // 3 Farkas certificates
    REQUIRE(countSubstr(s, "FARKAS") >= 3);
    REQUIRE(s.find("RESULT Infeasible") != std::string::npos);
    // The tree was exhausted with no solution
    REQUIRE(s.find("INCUMBENT") == std::string::npos);
}

// ── Test 4: time-limited solve ────────────────────────────────────────────────
//
// Cascade50 without presolve.  The integer tree has ~10^50 nodes; solving to
// optimality is impossible in any reasonable time.  With timeLimitS=3.0, the
// solver explores as many nodes as it can and returns MILPStatus::TimeLimit.
//
// Expected: proof contains a header, multiple node events, and a TimeLimit result.
// The solver respects the 3-second budget, so the test finishes in ≤ ~4 s.

TEST_CASE("Proof: time-limited solve emits TimeLimit result", "[proof]") {
    Model m = makeCascadeHard();

    BBOptions opts;
    opts.presolveLevel = 0;   // disable presolve so the tree stays intractable
    opts.timeLimitS    = 3.0; // hard wall-clock limit

    std::ostringstream proof;
    opts.proofStream = &proof;

    const MILPResult r = solveMILP(m, opts);
    REQUIRE(r.status == MILPStatus::TimeLimit);

    const std::string s = proof.str();
    INFO("Proof size: " << s.size() << " bytes");

    REQUIRE(s.find("BB-PROOF 0.7.0") != std::string::npos);
    REQUIRE(s.find("RESULT TimeLimit") != std::string::npos);
    // Multiple nodes were explored before the time limit
    REQUIRE(countSubstr(s, "\nN ") > 5);
}
