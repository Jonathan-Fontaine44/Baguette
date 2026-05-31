#include <benchmark/benchmark.h>

#include <filesystem>
#include <fstream>
#include <functional>
#include <vector>

#include "baguette/cp/constraints/AllDiff.hpp"
#include "baguette/milp/BranchAndBound.hpp"
#include "baguette/model/Model.hpp"
#include "baguette/model/ModelEnums.hpp"
#include "lp/MILP_problems.hpp"

using namespace baguette;

// ── Log directory ─────────────────────────────────────────────────────────────
//
// Proof files are written to log/ relative to the working directory at
// benchmark runtime.  Run from the project root so they land in log/.

static const char* kLogDir = "log";

// ── Generic Sudoku builder ────────────────────────────────────────────────────
//
// Pure AllDiff model: N*N integer variables, AllDiff on N rows + N columns +
// B*B blocks.  No LP constraints.  clues[i][j] = 0 (free) or a fixed digit.

static Model buildSudoku(int B, const std::vector<std::vector<int>>& clues) {
    const int N = B * B;
    Model m;
    std::vector<std::vector<Variable>> x(N, std::vector<Variable>(N));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            const double lb = clues[i][j] ? double(clues[i][j]) : 1.0;
            const double ub = clues[i][j] ? double(clues[i][j]) : double(N);
            x[i][j] = m.addVar(lb, ub, VarType::Integer);
        }
    m.setObjective({}, ObjSense::Minimize);
    for (int i = 0; i < N; ++i)
        m.addCPConstraint(AllDiffConstraint(std::vector<Variable>(x[i].begin(), x[i].end())));
    for (int j = 0; j < N; ++j) {
        std::vector<Variable> col;
        for (int i = 0; i < N; ++i) col.push_back(x[i][j]);
        m.addCPConstraint(AllDiffConstraint(std::move(col)));
    }
    for (int bi = 0; bi < B; ++bi)
        for (int bj = 0; bj < B; ++bj) {
            std::vector<Variable> box;
            for (int i = bi*B; i < (bi+1)*B; ++i)
                for (int j = bj*B; j < (bj+1)*B; ++j)
                    box.push_back(x[i][j]);
            m.addCPConstraint(AllDiffConstraint(std::move(box)));
        }
    return m;
}

static Model makeSudoku4x4() {
    return buildSudoku(2, {
        {1, 2, 0, 0},
        {0, 0, 1, 2},
        {0, 0, 4, 3},
        {4, 3, 0, 0},
    });
}

// Classic easy 9x9 puzzle — unique solution, solved purely by CP propagation.
static Model makeSudoku9x9() {
    return buildSudoku(3, {
        {5, 3, 0, 0, 7, 0, 0, 0, 0},
        {6, 0, 0, 1, 9, 5, 0, 0, 0},
        {0, 9, 8, 0, 0, 0, 0, 6, 0},
        {8, 0, 0, 0, 6, 0, 0, 0, 3},
        {4, 0, 0, 8, 0, 3, 0, 0, 1},
        {7, 0, 0, 0, 2, 0, 0, 0, 6},
        {0, 6, 0, 0, 0, 0, 2, 8, 0},
        {0, 0, 0, 4, 1, 9, 0, 0, 5},
        {0, 0, 0, 0, 8, 0, 0, 7, 9},
    });
}

// Same puzzle with digit 9 forced at both col 0 and col 8 of row 8.
// CP detects AllDiff DuplicateFixed at the root — no LP or branching needed.
static Model makeSudoku9x9Infeasible() {
    return buildSudoku(3, {
        {5, 3, 0, 0, 7, 0, 0, 0, 0},
        {6, 0, 0, 1, 9, 5, 0, 0, 0},
        {0, 9, 8, 0, 0, 0, 0, 6, 0},
        {8, 0, 0, 0, 6, 0, 0, 0, 3},
        {4, 0, 0, 8, 0, 3, 0, 0, 1},
        {7, 0, 0, 0, 2, 0, 0, 0, 6},
        {0, 6, 0, 0, 0, 0, 2, 8, 0},
        {0, 0, 0, 4, 1, 9, 0, 0, 5},
        {9, 0, 0, 0, 8, 0, 0, 7, 9}, // conflict: col 0 = col 8 = 9
    });
}

// ── Core runner ───────────────────────────────────────────────────────────────
//
// Three proof modes controlled by (withProof, logFile):
//   withProof=false, logFile=any   - NoProof: proofStream nullptr, zero overhead.
//   withProof=true,  logFile=null  - StringStream: proof to std::ostringstream,
//                                    measures serialisation cost without disk I/O.
//   withProof=true,  logFile="..."  - File: proof written to logFile each iteration
//                                    (truncated), measures full I/O overhead.
//
// Comparing (File - StringStream) isolates disk I/O cost.
// Comparing (StringStream - NoProof) isolates ProofWriter serialisation cost.
//
// Counters:
//   nodes       - B&B nodes explored
//   obj         - objective value
//   proof_bytes - proof size in bytes (StringStream and File only)

static void runProof(benchmark::State&      state,
                     bool                   withProof,
                     const char*            logFile,
                     std::function<Model()> build) {
    std::filesystem::create_directories(kLogDir);

    for (auto _ : state) {
        BBOptions opts;
        opts.collectStats = true;

        std::ostringstream ss;
        std::ofstream      f;

        if (withProof) {
            if (logFile) {
                f.open(logFile, std::ios::trunc);
                opts.proofStream = &f;
            } else {
                opts.proofStream = &ss;
            }
        }

        MILPResult r = solveMILP(build(), opts);
        benchmark::DoNotOptimize(r.objectiveValue);

        if (r.stats) {
            state.counters["nodes"] = double(r.stats->nodesExplored);
            state.counters["obj"]   = r.objectiveValue;
        }
        if (withProof) {
            const std::size_t bytes = logFile
                ? static_cast<std::size_t>(f.tellp())
                : ss.str().size();
            state.counters["proof_bytes"] = double(bytes);
        }
    }
}

// ── FacilityLocation 5x10 ────────────────────────────────────────────────────
//
// 5 facilities x 10 clients: 55 binary variables, 60 constraints.
// LP optimal = 67, IP optimal = 69.  Real B&B gap: branching required.
// Proof contains N/DIFF/LP/BRANCH/INCUMBENT/PRUNE_BOUND/FARKAS events.

BENCHMARK_CAPTURE(runProof, FacilityLocation5x10/NoProof,
    false, nullptr,
    []() { return baguette_test::makeFacilityLocation5x10(); })
    ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(runProof, FacilityLocation5x10/StringStream,
    true, nullptr,
    []() { return baguette_test::makeFacilityLocation5x10(); })
    ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(runProof, FacilityLocation5x10/File,
    true, "log/proof_facility5x10.log",
    []() { return baguette_test::makeFacilityLocation5x10(); })
    ->Unit(benchmark::kMillisecond);

// ── SetPartitioning small ─────────────────────────────────────────────────────
//
// 10 elements, 30 columns (10 singletons + 20 compound, seed 0xC0FFEE42).
// 90 binary variables, 10 equality constraints.
// LP optimal = 16, significant integrality gap - generates many B&B nodes.
// Proof volume is larger than FacilityLocation, stresses the buffer path.

BENCHMARK_CAPTURE(runProof, SetPartitioningSmall/NoProof,
    false, nullptr,
    []() { return baguette_test::makeSetPartitioningSmall(); })
    ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(runProof, SetPartitioningSmall/StringStream,
    true, nullptr,
    []() { return baguette_test::makeSetPartitioningSmall(); })
    ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(runProof, SetPartitioningSmall/File,
    true, "log/proof_setpart_small.log",
    []() { return baguette_test::makeSetPartitioningSmall(); })
    ->Unit(benchmark::kMillisecond);

// ── Sudoku 4x4 (CP-heavy) ─────────────────────────────────────────────────────
//
// Pure AllDiff model: no LP constraints.  CP propagation alone solves
// the puzzle at the root (cpPropagateToFixpoint=true, default).
// Proof consists entirely of CP_INFEASIBLE events (AllDiff witnesses)
// plus a single INCUMBENT at the leaf.

BENCHMARK_CAPTURE(runProof, Sudoku4x4/NoProof,
    false, nullptr,
    makeSudoku4x4)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_CAPTURE(runProof, Sudoku4x4/StringStream,
    true, nullptr,
    makeSudoku4x4)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_CAPTURE(runProof, Sudoku4x4/File,
    true, "log/proof_sudoku4x4.log",
    makeSudoku4x4)
    ->Unit(benchmark::kMicrosecond);

// ── Sudoku 9x9 feasible (CP-heavy) ───────────────────────────────────────────
//
// 81 integer variables, 27 AllDiff constraints (9 rows + 9 cols + 9 blocks).
// CP propagation alone solves the easy puzzle (no LP branching needed).
// Proof contains AllDiff witnesses for every CP_INFEASIBLE pruning event
// during the search, plus the INCUMBENT at the solution node.

BENCHMARK_CAPTURE(runProof, Sudoku9x9/NoProof,
    false, nullptr,
    makeSudoku9x9)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(runProof, Sudoku9x9/StringStream,
    true, nullptr,
    makeSudoku9x9)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(runProof, Sudoku9x9/File,
    true, "log/proof_sudoku9x9.log",
    makeSudoku9x9)
    ->Unit(benchmark::kMillisecond);

// ── Sudoku 9x9 infeasible (root CP conflict) ─────────────────────────────────
//
// Same 9x9 puzzle but row 8 has digit 9 at both col 0 and col 8.
// AllDiff(row 8) fires DuplicateFixed at the root CP pass - instant.
// The proof is minimal (header + N 0 + CP_INFEASIBLE + RESULT): lowest
// possible overhead, one CP witness written, no B&B at all.

BENCHMARK_CAPTURE(runProof, Sudoku9x9Infeasible/NoProof,
    false, nullptr,
    makeSudoku9x9Infeasible)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_CAPTURE(runProof, Sudoku9x9Infeasible/StringStream,
    true, nullptr,
    makeSudoku9x9Infeasible)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_CAPTURE(runProof, Sudoku9x9Infeasible/File,
    true, "log/proof_sudoku9x9_infeasible.log",
    makeSudoku9x9Infeasible)
    ->Unit(benchmark::kMicrosecond);
