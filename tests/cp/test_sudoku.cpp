#include <cmath>
#include <functional>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "baguette/cp/constraints/AllDiff.hpp"
#include "baguette/milp/BranchAndBound.hpp"
#include "baguette/milp/MILPResult.hpp"
#include "baguette/model/Model.hpp"
#include "baguette/model/ModelEnums.hpp"

using namespace baguette;

// ── Test case ─────────────────────────────────────────────────────────────────

struct SudokuTestCase {
    std::string                    name;
    int                            B;        ///< Block size (N = B*B).
    bool                           feasible; ///< Expected result.
    std::vector<std::vector<int>>  clues;    ///< 0 = unknown, nonzero = fixed cell.
};

// ── Suite ─────────────────────────────────────────────────────────────────────

static std::vector<SudokuTestCase> makeSudokuSuite() {
    return {
        // 4×4 Sudoku (B=2). Unique solution:
        //   1 2 3 4 / 3 4 1 2 / 2 1 4 3 / 4 3 2 1
        {"4x4_easy", 2, true, {
            {1, 2, 0, 0},
            {0, 0, 1, 2},
            {0, 0, 4, 3},
            {4, 3, 0, 0},
        }},

        // 9×9 Sudoku (B=3). Classic easy puzzle with unique solution.
        {"9x9_easy", 3, true, {
            {5, 3, 0, 0, 7, 0, 0, 0, 0},
            {6, 0, 0, 1, 9, 5, 0, 0, 0},
            {0, 9, 8, 0, 0, 0, 0, 6, 0},
            {8, 0, 0, 0, 6, 0, 0, 0, 3},
            {4, 0, 0, 8, 0, 3, 0, 0, 1},
            {7, 0, 0, 0, 2, 0, 0, 0, 6},
            {0, 6, 0, 0, 0, 0, 2, 8, 0},
            {0, 0, 0, 4, 1, 9, 0, 0, 5},
            {0, 0, 0, 0, 8, 0, 0, 7, 9},
        }},

        // 9×9 Sudoku (B=3). Same puzzle but with an extra 9 forced at row 8 col 0,
        // conflicting with the existing clue 9 at row 8 col 8 - row AllDiff violated.
        {"9x9_infeasible", 3, false, {
            {5, 3, 0, 0, 7, 0, 0, 0, 0},
            {6, 0, 0, 1, 9, 5, 0, 0, 0},
            {0, 9, 8, 0, 0, 0, 0, 6, 0},
            {8, 0, 0, 0, 6, 0, 0, 0, 3},
            {4, 0, 0, 8, 0, 3, 0, 0, 1},
            {7, 0, 0, 0, 2, 0, 0, 0, 6},
            {0, 6, 0, 0, 0, 0, 2, 8, 0},
            {0, 0, 0, 4, 1, 9, 0, 0, 5},
            {9, 0, 0, 0, 8, 0, 0, 7, 9},
        }},
    };
}

// ── Model builder ─────────────────────────────────────────────────────────────

struct SudokuModel {
    Model                              m;
    std::vector<std::vector<Variable>> x; ///< x[row][col]
};

static SudokuModel buildSudokuModel(int B, const std::vector<std::vector<int>>& clues) {
    const int N = B * B;
    SudokuModel sm;
    sm.x.reserve(N);
    for (int i = 0; i < N; ++i) {
        sm.x.push_back({});
        sm.x.back().reserve(N);
        for (int j = 0; j < N; ++j) {
            const double lb = clues[i][j] ? double(clues[i][j]) : 1.0;
            const double ub = clues[i][j] ? double(clues[i][j]) : double(N);
            sm.x.back().push_back(sm.m.addVar(lb, ub, VarType::Integer));
        }
    }

    sm.m.setObjective({}, ObjSense::Minimize);

    for (int i = 0; i < N; ++i)
        sm.m.addCPConstraint(
            AllDiffConstraint(std::vector<Variable>(sm.x[i].begin(), sm.x[i].end())));

    for (int j = 0; j < N; ++j) {
        std::vector<Variable> col;
        col.reserve(N);
        for (int i = 0; i < N; ++i) col.push_back(sm.x[i][j]);
        sm.m.addCPConstraint(AllDiffConstraint(std::move(col)));
    }

    for (int bi = 0; bi < B; ++bi)
        for (int bj = 0; bj < B; ++bj) {
            std::vector<Variable> box;
            box.reserve(N);
            for (int i = bi * B; i < (bi + 1) * B; ++i)
                for (int j = bj * B; j < (bj + 1) * B; ++j)
                    box.push_back(sm.x[i][j]);
            sm.m.addCPConstraint(AllDiffConstraint(std::move(box)));
        }

    return sm;
}

// ── Solution checker ──────────────────────────────────────────────────────────

static void requireValidSudoku(int B,
                                const std::vector<std::vector<int>>&      clues,
                                const std::vector<double>&                sol,
                                const std::vector<std::vector<Variable>>& x) {
    const int N = B * B;

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            if (clues[i][j])
                REQUIRE(int(std::round(sol[x[i][j].id])) == clues[i][j]);

    auto requireDistinct = [&](const std::vector<Variable>& vars) {
        for (int a = 0; a < int(vars.size()); ++a) {
            const int va = int(std::round(sol[vars[a].id]));
            REQUIRE(va >= 1);
            REQUIRE(va <= N);
            for (int b = a + 1; b < int(vars.size()); ++b)
                REQUIRE(va != int(std::round(sol[vars[b].id])));
        }
    };

    for (int i = 0; i < N; ++i) requireDistinct(x[i]);

    for (int j = 0; j < N; ++j) {
        std::vector<Variable> col;
        for (int i = 0; i < N; ++i) col.push_back(x[i][j]);
        requireDistinct(col);
    }

    for (int bi = 0; bi < B; ++bi)
        for (int bj = 0; bj < B; ++bj) {
            std::vector<Variable> box;
            for (int i = bi * B; i < (bi + 1) * B; ++i)
                for (int j = bj * B; j < (bj + 1) * B; ++j)
                    box.push_back(x[i][j]);
            requireDistinct(box);
        }
}

// ── Test ──────────────────────────────────────────────────────────────────────

TEST_CASE("Sudoku suite - AllDiff-only MIP", "[cp][sudoku]") {
    static const auto suite = makeSudokuSuite();
    auto i = GENERATE(range(std::size_t{0}, suite.size()));
    const auto& tc = suite[i];

    DYNAMIC_SECTION(tc.name) {
        auto [m, x] = buildSudokuModel(tc.B, tc.clues);

        BBOptions opts;
        opts.timeLimitS            = 60.0;
        opts.cpPropagateToFixpoint = true;
        const MILPResult res = solveMILP(m, opts);

        if (tc.feasible) {
            REQUIRE(res.status == MILPStatus::Optimal);
            requireValidSudoku(tc.B, tc.clues, res.primalValues, x);
        } else {
            REQUIRE(res.status == MILPStatus::Infeasible);
        }
    }
}
