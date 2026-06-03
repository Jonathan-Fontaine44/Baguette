# LP Solver

The `lp/` module solves the LP relaxation of a model. It is called once per B&B node and produces the bound, the dual certificate, and the warm-start basis for the next node.

## Entry points

```cpp
LPResult         solveLP(const Model&, const LPOptions& = {});
LPDetailedResult solveLPDetailed(const Model&, const LPOptions& = {});
```

`solveLPDetailed` is the B&B-facing API. It orchestrates:
1. Presolve (bound-tightening then elimination, if enabled)
2. Standard-form construction
3. Algorithm dispatch
4. Postsolve (reconstruct dual/primal on the original variable space)

Integer and Binary variables are treated as continuous (LP relaxation).

## Algorithms

| Method | Description | Warm-start | Notes |
|--------|-------------|-----------|-------|
| `Auto` | `DualSimplexBV` with fallback to `PrimalSimplexBV` | yes | default |
| `PrimalSimplex` | Two-phase primal simplex (phase I + II) | no | adds explicit UB rows |
| `DualSimplex` | Dual simplex with primal fallback | yes | |
| `RevisedSimplex` | Two-phase primal, maintains B^-1 via LU | no | O(m^2) memory, m << n |
| `PrimalSimplexBV` | Primal simplex with bounded-variable complement | no | no UB row inflation |
| `DualSimplexBV` | Dual simplex with bounded-variable complement | yes | default for B&B nodes |
| `RevisedSimplexBV` | Revised primal BV, periodic LU reinversion | yes (v0.6.3+) | |
| `ShortStepIPM` | Short-step feasible path-following IPM, O(sqrt(n) log(1/eps)) | no | |
| `MehrotraIPM` | Primal-dual infeasible-start, predictor-corrector, 15-50 iters | no | good root solve |
| `NetworkSimplex` | Detects node-arc incidence; O(n) pivots vs O(m^2) | no | 100-1000x on network LPs |

The `BV` (bounded-variable) methods enforce upper bounds via the complement invariant (`x' = ub - x`) rather than adding explicit UB rows. This keeps m = nOrigRows throughout, avoiding O(n) row inflation on bounded problems. All B&B node solves use `DualSimplexBV` by default.

## LPOptions

Key fields relevant to B&B:

| Field | Default | Purpose |
|-------|---------|---------|
| `method` | `Auto` | Overridden per-node by `BBOptions::nodeMethod` |
| `warmBasis` | empty | Parent node's `BasisRecord`; enables dual warm-start |
| `enablePresolve` | true | Bound-tightening pass before the LP |
| `enableElimination` | true | Fixed-variable removal before the LP |
| `computeCutData` | false | Populate `fractionalRows` for GMI cut generation |
| `reinversionPeriod` | 50 | Rebuild B^-1 every N pivots to cap floating-point drift |
| `feasibilityTol` | 1e-9 | Primal feasibility threshold |
| `optimalityTol` | 1e-9 | Reduced-cost threshold for optimality |

## Results

### LPResult (basic)

```cpp
struct LPResult {
    LPStatus status;           // Optimal, Infeasible, Unbounded, MaxIter, TimeLimit, NumericalFailure
    double   objectiveValue;
    vector<double> primalValues;  // indexed by Variable::id
};
```

`primalValues` is populated for `Optimal`, and for `MaxIter`/`TimeLimit` if a feasible point was reached. It is empty for `Infeasible` and `Unbounded`.

### LPDetailedResult (B&B-facing)

Extends `LPResult` with:

| Field | Valid when | Description |
|-------|------------|-------------|
| `dualValues` | Optimal | Shadow prices, one per constraint |
| `reducedCosts` | Optimal | Reduced costs, indexed by `Variable::id` |
| `basis` | Optimal | `BasisRecord` for warm-starting child nodes |
| `farkas` | Infeasible | Infeasibility certificate (see below) |
| `sensitivity` | Optimal + flag | RHS and objective ranging |
| `fractionalRows` | Optimal + flag | Raw tableau rows for GMI cut generation |
| `usedWarmStart` | always | Whether the warm basis was successfully used |
| `iterationsUsed` | always | Total simplex pivots or IPM Newton steps |

## Warm-start mechanism

The `BasisRecord` carries:
- `basicCols[]` - which standard-form column is basic in each row
- `colKind[]` / `colOrigin[]` - maps columns back to model variables/constraints
- `sfCache` - `shared_ptr` to the full standard form built during the parent solve

When `solveDualDetailed` receives a non-empty `warmBasis`, it reuses `sfCache` via an O(1) `shared_ptr` copy and only recomputes the RHS vector `b` for the new bounds. This avoids the O(m*n) standard-form rebuild at every node - the dominant setup cost for large models.

If `sfCache` dimensions do not match the current model (e.g., after a cut was added), the warm-start is skipped and a cold primal solve is performed instead.

## Farkas certificate

When the LP is infeasible, `LPDetailedResult::farkas` holds one of two forms:

**Bound certificate** (`infeasVarId >= 0`): variable `infeasVarId` has `lb > ub` after B&B branching. Verifiable by reading `model.getHot()`.

**Tableau certificate** (`y` non-empty): dual multipliers such that:
```
A^T * y >= 0   (component-wise, original constraint coefficients)
b^T * y < 0    (original RHS)
```
This is the classic Farkas lemma certificate: it proves no primal feasible point exists. Derived from the blocking row of the dual simplex or the phase-I objective of the primal simplex.

Both forms are written into the proof log. See [proof_system.md](proof_system.md) for the exact syntax.

## Design assumptions

- The LP solver never modifies the `Model`. It builds a working copy for standard-form construction.
- Sensitivity analysis is O(m*n) and explicitly disabled in B&B hot loops (`computeSensitivity = false` by default).
- `NumericalFailure` (basis reinversion failure) is a recoverable error: the B&B node is logged as `UNVERIFIED` in the proof and the search continues.
