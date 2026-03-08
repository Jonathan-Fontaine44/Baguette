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
```

---

## Roadmap

### Phase 1 — Bounds Consistency (current)

The first implementation uses **Bounds Consistency (BC)** for all CP propagators:

- O(n log n) per AllDiff node via sort + sweep (Quimper et al., 2003)
- Operates on `[lb, ub]` intervals only — no value enumeration

### Phase 2 — Arc Consistency (future)

**Arc Consistency (AC)** is a strictly stronger form of propagation that reasons over individual values in variable domains, not just bounds. It is planned as a future extension:

- AC for AllDiff via bipartite matching (Régin, 1994)
- Stronger domain reductions and tighter cuts
- Higher per-node cost — trade-off to be evaluated experimentally

---

## References

- Quimper et al., *An Efficient Bounds Consistency Algorithm for the Global Cardinality Constraint*, CP 2003
- Régin, *A Filtering Algorithm for Constraints of Difference in CSPs*, AAAI 1994
- Hooker, *Integrated Methods for Optimization*, Springer 2007
- Chvátal, *Linear Programming*, W.H. Freeman 1983

---

## License

Baguette is distributed under the [GNU Lesser General Public License v3.0](LICENSE).
