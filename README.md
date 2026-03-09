# Baguette

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
│   ├── lp/            ← LPMatrix (CSR), RevisedSimplex, LPSolver interface
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
| `v0.2.0` | ✅ Done | LP solver: standard form conversion, revised simplex (phase I + II), `LPSolver` public interface. |
| `v0.2.1` | ✅ Done | Handle infinite bounds (`lb = -∞`, `ub = +∞`) in the simplex: fully free variables split as `x = x⁺ − x⁻`. |
| `v0.2.2` | Planned | LP dual generation. |
| `v0.3.0` | Planned | Branch & Bound: node queue, branching strategy, incumbent tracking. |
| `v0.4.0` | Planned | CP propagation — Bounds Consistency for AllDiff and Cumulative. |
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

---

## License

Baguette is distributed under the [GNU Lesser General Public License v3.0](LICENSE).
