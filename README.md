# Baguette

[![GitHub](https://img.shields.io/badge/GitHub-Baguette-blue?logo=github)](https://github.com/Jonathan-Fontaine44/Baguette)

**Baguette** is a research-oriented C++20 solver for hybrid **MILP + Constraint Programming** models.

It implements a Branch & Bound engine enriched with CP global constraints (starting with AllDiff), where a **CP propagator** is enforced at every B&B node to:
1. Reduce variable domains (CP filtering) before/after LP solves
2. Generate valid cutting planes from CP reductions

The CP propagation level can be **Bounds Consistency (BC)** or **Arc Consistency (AC)** — the architecture supports both, though the initial implementation targets BC (see [Roadmap](#roadmap)).

> **No third-party solver is used.** The simplex, B&B, and propagation algorithms are implemented from scratch for research purposes.

---

## Scientific objective

The central thesis is that CP propagation and LP relaxation are complementary: CP reduces domains cheaply (O(n log n) for AllDiff with Bounds Consistency), which tightens LP bounds and enables cut generation that would otherwise require expensive enumeration. The hybrid loop runs as:

```
Node entry → CP propagation → LP solve → CP cut generation → branch
```

See [Hooker, *Integrated Methods for Optimization*, 2007] for the theoretical foundation.

---

## Architecture

```
baguette/
├── include/baguette/
│   ├── core/          ← Variable (pure handle), Domain, LinearExpr, Sense
│   ├── model/         ← Model (user API: addVar, addConstraint, setObjective)
│   ├── lp/            ← StandardForm, full simplex tableau (B⁻¹A explicit), LPSolver interface
│   ├── milp/          ← BranchAndBound, Node, BranchingStrategy, CuttingPlanes
│   └── cp/            ← DomainStore (trail backtracking), AllDiffPropagator
├── src/               ← implementations
├── tests/             ← Catch2 v3
└── CMakeLists.txt
```

### Key design decisions (Data-Oriented Design)

Performance-critical paths follow strict DOD rules to maximize cache efficiency:

| Rule | Forbidden | Correct |
|---|---|---|
| No virtual in hot loops | `virtual propagate()` + vtable | `PropagatorStore` with typed vectors |
| No hash maps in hot paths | `unordered_map<VarID, double>` | sorted pairs `(varIds[], coeffs[])` |
| Hot/cold separation | `struct { string label; double lb; }` | `ModelHot` + `ModelCold` separate |
| No `unique_ptr` in collections | `vector<unique_ptr<Propagator>>` | `vector<AllDiffPropagator>` (inline) |
| No per-node `new` | `new Node()` per node | pool allocator per depth level |
| Contiguous VarIDs per constraint | arbitrary IDs in AllDiff | renumbering at model compilation |

The modeling layer (called once before solve) is exempt — `addVar()` / `addConstraint()` may use convenient structures.

---

## Build

**Requirements:** CMake 3.20+, a C++20-capable compiler (GCC 11+, Clang 14+, MSVC 19.29+), Catch2 v3.

```bash
cmake -S . -B build                      # configure
cmake --build build --config Debug       # compile
ctest --test-dir build -C Debug          # run tests
doxygen Doxyfile                         # generate documentation → docs/html/index.html
```

---

## Roadmap

| Version | Status | Scope |
| ------- | ------ | ----- |
| `v0.1.0` | ✅ Done | Core layer: `Variable`, `Domain`, `LinearExpr` (sorted SoA, `operator+/+=/-=`), `Model` API (`addVar`, `addConstraint`, `setObjective`), `ModelEnums`, global `Config` tolerances. 38 Catch2 tests. |
| `v0.2.0` | ✅ Done | LP solver: standard form conversion, full simplex tableau (phase I + II), `LPSolver` public interface. |
| `v0.2.1` | ✅ Done | Handle infinite bounds (`lb = -∞`, `ub = +∞`) in the simplex: fully free variables split as `x = x⁺ − x⁻`. |
| `v0.2.2` | ✅ Done | LP dual: `dualStandardForm` (mathematical dual as a solvable SF), dual simplex algorithm (`selectLeavingDual` / `selectEnteringDual`), `solveDual` / `solveDualDetailed` public API with automatic fallback to primal. 32 new Catch2 tests (strong duality, GEQ handling, infeasibility, double-dual). |
| `v0.2.3` | ✅ Done | B&B warm-start via dual simplex: `Model::withVarBounds` (child-node model from bound tightening), `solveWarm` / `solveDetailedWarm` — reinvert the child tableau from the parent's `BasisRecord` (skipping Phase I), then run the dual simplex to restore primal feasibility. Automatic fallback to cold primal start when the bound-finiteness invariant is violated or the warm basis is not dual-feasible. 7 new Catch2 tests (identity, left/right branch, infeasible domain/constraints, incompatible basis, two-level B&B tree). |
| `v0.2.4` | ✅ Done | Farkas certificate: when `solveDetailed` / `solveDualDetailed` returns `Infeasible`, populate a `FarkasRay` in `LPDetailedResult` — a vector `y` extracted from the blocking tableau row such that `A^T y ≥ 0` and `b^T y < 0`, providing a machine-verifiable proof of infeasibility. Early `lb > ub` detection (B&B branching) sets `FarkasRay::infeasVarId` instead. 7 new Catch2 tests. |
| `v0.2.5` | ✅ Done | Sensitivity analysis: `SensitivityResult` populated at solve time (opt-in via `computeSensitivity = true`). Per-constraint **RHS ranging** — interval `[lo, hi]` over which the current basis stays primal feasible, read from `B⁻¹ e_i` via the slack/surplus/artificial column. Per-variable **objective ranging** — interval over which the basis stays dual feasible, via ratio test on the non-basic reduced cost (non-basic variable) or dual ratio test over all non-basic columns (basic variable). Works on both primal and dual simplex paths. The flag defaults to `false` to avoid O(m·n) overhead in B&B hot loops. 16 new Catch2 tests. |
| `v0.3.0` | ✅ Done | Branch & Bound: `solveMILP()` with `BBOptions` (BestBound / DepthFirst node selection, FirstFractional / MostFractional branching, node and time limits). LP warm-started via `solveDualDetailed()` with shared `startTime` propagated across all nodes. Incumbent tracking, pruning by bound and infeasibility. `MILPStatus`: Optimal, Infeasible, Unbounded, TimeLimit, MaxNodes. 8 new Catch2 tests. |
| `v0.3.1` | ✅ Done | Branch & Cut: GMI cuts from the final simplex tableau, pseudo-cost branching, `BBOptions::enableCuts/maxCutsPerNode`, `MILPResult::cutsAdded`. New: `CuttingPlanes.hpp/.cpp`. 10 new Catch2 tests. |
| `v0.3.2` | ✅ Done | Node memory optimisation: replaced full bounds snapshots (`O(n)` per node) with per-branch **delta records** (`BoundChange{varId, newLb, newUb}`) accumulated as a trail from the root. Dirty-variable tracking limits `restoreBounds` to O(prev\_depth + curr\_depth) work. 1 new Catch2 test (BestBound vs DepthFirst deep tree). |
| `v0.3.3` | ✅ Done | Hybrid node-selection strategy: `NodeSelection::HybridPlunge` — plunges DFS until the first integer-feasible incumbent is found, then heapifies the queue and switches to BestBound to prove optimality. 1 new Catch2 test. |
| `v0.4.0` | ✅ Done | CP propagation: `propagateCP()` dispatches via `std::variant<AllDiffConstraint, CumulativeConstraint>` — adding a new constraint type requires only a typed struct, a `propagate()` overload, and one line in `AnyConstraint`. **AllDiff BC**: fixed-value elimination to fixpoint + range feasibility check (O(K² × I)). **Cumulative BC**: compulsory-region overload detection + earliest-start tightening (O(N² × D × I)). Integrated in `solveMILP()` as a new `const CPConstraints& cp = {}` parameter: propagation runs after `restoreBounds()`, before the LP, with CP-tightened bounds tracked in `dirtyVars` for automatic backtracking. 12 new Catch2 tests. |
| `v0.4.1` | ✅ Done | Hybrid closed/open CP architecture: two-tier `CPConstraints` — built-ins (`AllDiff`, `Cumulative`) dispatch via `std::visit` on `BuiltinConstraint = std::variant<...>` (zero virtual overhead); user-defined constraints inherit abstract `CPConstraint` base and are stored as `shared_ptr<const CPConstraint>` (virtual dispatch, one call per node). `Model::addCPConstraints(const CPConstraints&)` integrates CP semantically into the model — `solveMILP()` no longer takes a `cp` parameter, it reads `model.getCPConstraints()` internally. `shared_ptr<const>` keeps Model copyable without `clone()`. |
| `v0.4.2` | ✅ Done | B&C profiling: opt-in `BBStats` struct in `MILPResult`, controlled by `BBOptions::collectStats` (default false, zero overhead when disabled). Fields: `nodesExplored`, `cutsAdded`, `nodesWithCuts`, `nodesPrunedByBound`, `nodesPrunedByInfeasibility`, `warmStartFallbacks`, `lpSolvesTotal`, `cutsPerDepth`. `LPDetailedResult::usedWarmStart` surfaces warm-start fallbacks from `solveDualDetailed()`. |
| `v0.5.0` | Planned | **Revised simplex with LU decomposition**: replaces the current dense m×n tableau (full row update at each pivot, O(mn)) with LU factorization of the basis B ∈ ℝᵐˣᵐ with partial pivoting. Each pivot only computes the entering column B⁻¹aⱼ and the leaving row eᵣᵀB⁻¹ in O(m²) instead of O(mn). Rank-1 update of the factorization (eta-file or Forrest-Tomlin) avoids a full refactorization at each step. Required infrastructure for interior-point methods (v0.5.1–0.5.2); decisive improvement for large sparse LPs where m ≪ n. |
| `v0.5.1` | Planned | **Primal-dual interior-point method** (path-following, Mehrotra predictor-corrector): at each iteration, solves an augmented KKT system of size (m+n)×(m+n) via the LU factorization from v0.5.0 to compute the affine and centering directions. The barrier parameter μ = xᵀs/n decreases toward 0 along the central path; convergence in O(√n) iterations. Returns the same `LPResult` as the simplex — both solvers are interchangeable via an `LPMethod` parameter. |
| `v0.5.2` | Planned | **Log-barrier method** (classical, Fiacco-McCormick): transforms inequalities into min cᵀx − μ Σ ln(bᵢ − aᵢᵀx) and solves the sequence of unconstrained Newton subproblems as μ → 0. Simpler than the primal-dual (no explicit dual variables), but only superlinear convergence on the central path — slower iterations in practice. Validates v0.5.1 results and serves as a pedagogical reference. |
| `v0.6.0` | Planned | **B&B log-proof**: machine-verifiable certificate of optimality. Every node records its LP bound, branching decision, and pruning reason (bound dominated, Farkas infeasibility, or integer-feasible leaf). The complete trace constitutes a proof that no unexplored node could improve the incumbent, independently verifiable without re-running the solver. |
| `v1.0.0` | Planned | Complete solver, stable public API, full documentation. |

### CP propagation strategy

The CP layer starts with **Bounds Consistency (BC)**: operates on `[lb, ub]` intervals only, no value enumeration.
**Arc Consistency (AC)** is planned as a future phase for stronger domain reductions.

---

## References

- Quimper et al., *An Efficient Bounds Consistency Algorithm for the Global Cardinality Constraint*, CP 2003
- Régin, *A Filtering Algorithm for Constraints of Difference in CSPs*, AAAI 1994
- Hooker, *Integrated Methods for Optimization*, Springer 2007
- Chvátal, *Linear Programming*, W.H. Freeman 1983

### Revised simplex with LU decomposition (v0.5.0)

- Dantzig, G.B., & Orchard-Hays, W., *The Product Form for the Inverse in the Simplex Method*, Mathematical Tables and Other Aids to Computation 8(46), 1954 — original revised simplex; maintains B⁻¹ as a product of eta-matrices rather than the full tableau
- Bartels, R.H., & Golub, G.H., *The Simplex Method of Linear Programming Using LU Decomposition*, Communications of the ACM 12(5), 1969 — replaces the eta-file product with a numerically stable LU factorisation of the basis
- Forrest, J.J.H., & Tomlin, J.A., *Updated Triangular Factors of the Basis to Maintain Sparsity in the Product Form Simplex Method*, Mathematical Programming 2(1), 1972 — rank-1 update of the LU factors (Forrest-Tomlin update) to preserve sparsity between pivots

### Primal-dual interior-point method (v0.5.1)

- Karmarkar, N., *A New Polynomial-Time Algorithm for Linear Programming*, Combinatorica 4(4), 1984 — first polynomial-time interior-point algorithm for LP (projective method)
- Mehrotra, S., *On the Implementation of a Primal-Dual Interior Point Method*, SIAM Journal on Optimization 2(4), 1992 — predictor-corrector variant of the primal-dual path-following method; standard reference for modern IPM implementations

### Log-barrier method (v0.5.2)

- Fiacco, A.V., & McCormick, G.P., *The Sequential Unconstrained Minimization Technique for Nonlinear Programing, a Primal-Dual Method*, Management Science 10(2), 1964 — original SUMT barrier method paper
- Fiacco, A.V., & McCormick, G.P., *Nonlinear Programming: Sequential Unconstrained Minimization Techniques*, Wiley, 1968 (republished SIAM 1990) — reference monograph with complete convergence theory

---

## Acknowledgements

Baguette is developed with AI assistance (Claude, Anthropic), under the supervision of **Dr. Jonathan Fontaine**.

---

## License

Baguette is distributed under the [GNU Lesser General Public License v3.0](LICENSE).
