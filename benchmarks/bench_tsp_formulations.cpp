// TSP Formulation Comparison Benchmark
//
// Compares six TSP formulations on random Euclidean instances (n = 5, 10, 15):
//
//   Directed formulations (MTZ position vars / flow vars / explicit SECs):
//     MTZ      — Miller-Tucker-Zemlin (weak LP bound, O(n²) size)
//     LMTZ     — Lifted MTZ / Desrochers-Laporte (tighter LP, same size)
//     SCF      — Single-Commodity Flow (LP bound = DFJ, O(n²) size)
//     MCF      — Multi-Commodity Flow (LP bound = DFJ, O(n³) size; n ≤ 10)
//     DFJ      — Dantzig-Fulkerson-Johnson explicit SEC (LP = DFJ; n ≤ 10)
//
//   Undirected formulation with dynamic SEC cuts:
//     SEC      — Degree-2 relaxation + Stoer-Wagner SEC cut generator
//
//   Each directed formulation is also run with GMI cuts (+GMI variant).
//
// Metrics per run:
//   rootLP    — LP relaxation at the root (integrality ignored)
//   optimal   — best integer objective found
//   gap%      — (optimal − rootLP) / optimal × 100  (root LP gap)
//   nodes     — B&B nodes explored
//   solves    — total LP solves across the tree
//   cuts      — total cuts added (GMI or SEC)
//   time ms   — B&B wall-clock time (model build and root LP excluded)
//   status    — Optimal / TimeLimit / ...
//
// B&B settings (shared across all formulations for a fair comparison):
//   LP method      : DualSimplexBV
//   presolve       : disabled (isolates formulation LP-bound quality)
//   branch         : MostFractional
//   node selection : HybridPlunge
//   time limit     : 120 s (configurable via kTimeLimitS below)
//
// Build (requires -DBAGUETTE_BENCHMARKS=ON):
//   cmake -S . -B build/Release -DCMAKE_BUILD_TYPE=Release -DBAGUETTE_BENCHMARKS=ON
//   cmake --build build/Release --config Release
//   ./build/Release/benchmarks/BaguetteTSPBench

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "baguette/core/Sense.hpp"
#include "baguette/lp/LPSolver.hpp"
#include "baguette/milp/BranchAndBound.hpp"
#include "baguette/milp/SecCuts.hpp"
#include "baguette/model/Model.hpp"
#include "baguette/model/ModelEnums.hpp"
#include "lp/MILP_problems.hpp"

using namespace baguette;
using namespace baguette_test;
using Clock = std::chrono::high_resolution_clock;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

static constexpr double   kTimeLimitS    = 120.0;
static constexpr int      kMaxGMICuts    = 300;
static constexpr int      kMaxGMIPerNode = 10;

static const std::vector<int>      kSizes = {5, 10, 15};
static const std::vector<uint64_t> kSeeds = {42, 1234, 9999};

// ─────────────────────────────────────────────────────────────────────────────
// Random Euclidean instance
// ─────────────────────────────────────────────────────────────────────────────

struct City { double x, y; };

static double euclidDist(const City& a, const City& b) {
    double dx = a.x - b.x, dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

static std::vector<City> genCities(int n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(0.0, 100.0);
    std::vector<City> cities(n);
    for (auto& c : cities) { c.x = u(rng); c.y = u(rng); }
    return cities;
}

// Full directed arc list (both directions).
static std::vector<TspArc> toArcs(const std::vector<City>& cities) {
    int n = static_cast<int>(cities.size());
    std::vector<TspArc> arcs;
    arcs.reserve(n * (n - 1));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (i != j) arcs.push_back({i, j, euclidDist(cities[i], cities[j])});
    return arcs;
}

// ─────────────────────────────────────────────────────────────────────────────
// Run result
// ─────────────────────────────────────────────────────────────────────────────

struct RunResult {
    std::string formulation;
    int         n       = 0;
    uint64_t    seed    = 0;
    double      rootLP  = 0.0;   // LP relaxation value at root
    double      optimal = 0.0;   // best IP objective found
    double      gapPct  = std::numeric_limits<double>::quiet_NaN(); // root LP gap
    uint32_t    nodes   = 0;
    uint32_t    solves  = 0;
    uint32_t    cuts    = 0;
    double      timeMs  = 0.0;
    MILPStatus  status  = MILPStatus::Infeasible;
};

// ─────────────────────────────────────────────────────────────────────────────
// B&B options factory
// ─────────────────────────────────────────────────────────────────────────────

static BBOptions makeOpts(bool gmi) {
    BBOptions opts;
    opts.enableCuts        = gmi;
    opts.enableMIR         = false;
    opts.collectStats      = true;
    opts.enablePresolve    = false;
    opts.enableElimination = false;
    opts.timeLimitS        = kTimeLimitS;
    opts.branchStrat       = BranchStrategy::MostFractional;
    opts.nodeSelect        = NodeSelection::HybridPlunge;
    opts.lpOpts.method     = LPMethod::DualSimplexBV;
    if (gmi) {
        opts.maxCutsPerNode = kMaxGMIPerNode;
        opts.maxTotalCuts   = kMaxGMICuts;
    }
    return opts;
}

// ─────────────────────────────────────────────────────────────────────────────
// Root LP measurement (LP relaxation, ignores integrality)
// ─────────────────────────────────────────────────────────────────────────────

static double rootLP(const Model& m) {
    LPOptions lo;
    lo.method         = LPMethod::DualSimplexBV;
    lo.enablePresolve = false;
    lo.timeLimitS     = 30.0;
    LPResult r = solveLP(m, lo);
    return r.status == LPStatus::Optimal ? r.objectiveValue
                                         : std::numeric_limits<double>::quiet_NaN();
}

// ─────────────────────────────────────────────────────────────────────────────
// Directed formulation run
// ─────────────────────────────────────────────────────────────────────────────

static Model buildDirected(const std::string& tag, int n,
                            const std::vector<TspArc>& arcs) {
    if      (tag == "MTZ")  return makeTSP(n, arcs);
    else if (tag == "LMTZ") return makeTSPLifted(n, arcs);
    else if (tag == "SCF")  return makeTSPFlow(n, arcs);
    else if (tag == "MCF")  return makeTSPMCF(n, arcs);
    else                    return makeTSPDFJ(n, arcs);
}

static RunResult runDirected(const std::string& tag, int n, uint64_t seed,
                              const std::vector<City>& cities, bool gmi) {
    const auto  arcs = toArcs(cities);
    const Model m    = buildDirected(tag, n, arcs);

    RunResult res;
    res.formulation = gmi ? (tag + "+GMI") : tag;
    res.n           = n;
    res.seed        = seed;
    res.rootLP      = rootLP(m);

    auto opts = makeOpts(gmi);
    auto t0   = Clock::now();
    auto r    = solveMILP(m, opts);
    res.timeMs = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

    res.optimal = r.objectiveValue;
    res.status  = r.status;
    if (r.stats) {
        res.nodes  = r.stats->nodesExplored;
        res.solves = r.stats->lpSolvesTotal;
        res.cuts   = r.stats->cutsAdded;
    }
    if (r.status == MILPStatus::Optimal && res.optimal > 1e-9)
        res.gapPct = (res.optimal - res.rootLP) / res.optimal * 100.0;
    return res;
}

// ─────────────────────────────────────────────────────────────────────────────
// Undirected + SEC cut generator
// ─────────────────────────────────────────────────────────────────────────────

static RunResult runSEC(int n, uint64_t seed, const std::vector<City>& cities) {
    // Build undirected model: binary edge vars + degree-2 constraints.
    Model m;
    std::vector<std::vector<Variable>> ev(n, std::vector<Variable>(n, Variable{0}));
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            Variable v = m.addVar(0.0, 1.0, VarType::Binary);
            ev[i][j] = ev[j][i] = v;
        }
    for (int i = 0; i < n; ++i) {
        LinearExpr deg;
        for (int j = 0; j < n; ++j)
            if (j != i) deg.addTerm(ev[i][j], 1.0);
        m.addLPConstraint(deg, Sense::Equal, 2.0);
    }
    LinearExpr obj;
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            obj.addTerm(ev[i][j], euclidDist(cities[i], cities[j]));
    m.setObjective(obj, ObjSense::Minimize);

    RunResult res;
    res.formulation = "SEC";
    res.n           = n;
    res.seed        = seed;
    res.rootLP      = rootLP(m);   // 2-factor LP (no subtour constraints yet)

    auto opts = makeOpts(false);
    opts.cutGenerators.push_back(makeSecGenerator(n, ev));

    auto t0 = Clock::now();
    auto r  = solveMILP(m, opts);
    res.timeMs = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

    res.optimal = r.objectiveValue;
    res.status  = r.status;
    if (r.stats) {
        res.nodes  = r.stats->nodesExplored;
        res.solves = r.stats->lpSolvesTotal;
        res.cuts   = r.stats->cutsAdded;
    }
    if (r.status == MILPStatus::Optimal && res.optimal > 1e-9)
        res.gapPct = (res.optimal - res.rootLP) / res.optimal * 100.0;
    return res;
}

// ─────────────────────────────────────────────────────────────────────────────
// Output
// ─────────────────────────────────────────────────────────────────────────────

static void printSeparator(char c = '-', int w = 105) {
    std::cout << std::string(w, c) << '\n';
}

static void printHeader() {
    printSeparator('=');
    std::cout
        << std::left  << std::setw(14) << "Formulation"
        << std::right
        << std::setw(8)  << "rootLP"
        << std::setw(10) << "optimal"
        << std::setw(8)  << "gap%"
        << std::setw(8)  << "nodes"
        << std::setw(8)  << "solves"
        << std::setw(6)  << "cuts"
        << std::setw(11) << "time(ms)"
        << std::setw(12) << "status"
        << '\n';
    printSeparator();
}

static void printRow(const RunResult& r) {
    std::cout
        << std::left  << std::setw(14) << r.formulation
        << std::right << std::fixed
        << std::setw(8)  << std::setprecision(2) << r.rootLP
        << std::setw(10) << std::setprecision(2) << r.optimal;
    if (std::isnan(r.gapPct))
        std::cout << std::setw(8) << "—";
    else
        std::cout << std::setw(7) << std::setprecision(1) << r.gapPct << "%";
    std::cout
        << std::setw(8)  << r.nodes
        << std::setw(8)  << r.solves
        << std::setw(6)  << r.cuts
        << std::setw(11) << std::setprecision(1) << r.timeMs
        << std::setw(12) << to_string(r.status)
        << '\n';
}

static void printGroupHeader(int n, uint64_t seed) {
    std::cout << "\nn = " << n << "  |  seed = " << seed << '\n';
    printHeader();
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "=== TSP Formulation Comparison Benchmark ===\n";
    std::cout << "Time limit: " << kTimeLimitS << " s  |  LP: DualSimplexBV  "
              << "|  Presolve: OFF  |  Branch: MostFractional  "
              << "|  Node: HybridPlunge\n";
    std::cout << "Directed (+GMI: up to " << kMaxGMICuts << " GMI cuts, "
              << kMaxGMIPerNode << "/node)\n";
    std::cout << "MCF and DFJ only run for n ≤ 10 (O(n³) / O(2^n) formulations)\n";
    std::cout << "SEC: undirected degree-2 + Stoer-Wagner cut generator (no GMI)\n";

    for (int n : kSizes) {
        for (uint64_t seed : kSeeds) {
            const auto cities = genCities(n, seed);

            printGroupHeader(n, seed);

            // ── Directed, pure B&B ──────────────────────────────────────────
            for (const char* tag : {"MTZ", "LMTZ", "SCF"})
                printRow(runDirected(tag, n, seed, cities, false));
            if (n <= 10) {
                printRow(runDirected("MCF", n, seed, cities, false));
                printRow(runDirected("DFJ", n, seed, cities, false));
            }

            // ── Directed, B&B + GMI ─────────────────────────────────────────
            printSeparator(' ');  // visual gap before GMI group
            for (const char* tag : {"MTZ", "LMTZ", "SCF"})
                printRow(runDirected(tag, n, seed, cities, true));

            // ── Undirected + SEC cuts ───────────────────────────────────────
            printSeparator(' ');
            printRow(runSEC(n, seed, cities));
        }
    }

    std::cout << '\n';
    printSeparator('=');
    std::cout << "Done.\n";
    return 0;
}
