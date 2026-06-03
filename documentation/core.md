# Core

The `core/` module defines the primitive types shared by every other module. It has no dependencies within Baguette.

## Variable

```cpp
struct Variable { std::uint32_t id; };
```

A `Variable` is a typed integer handle - 4 bytes, trivially copyable, with no pointer back to the `Model`. All variable data (bounds, label, type) lives in the model, indexed by `id`.

This design is intentional: during Branch & Bound, child nodes copy or modify bounds without copying the variable itself. A `Variable` can be stored in constraints, linear expressions, and B&B queues freely.

**Invariant**: a `Variable` is only valid for the `Model` that created it via `Model::addVar()`. No runtime check enforces this; the contract is caller-side.

## LinearExpr

```cpp
struct LinearExpr {
    std::vector<std::uint32_t> varIds;  // sorted ascending
    std::vector<double>        coeffs;  // parallel to varIds
    double                     constant = 0.0;
};
```

A sparse linear expression `sum(coeff_i * var_i) + constant`.

**Invariant**: `varIds` is always sorted in strictly ascending order, with no duplicates. This enables:
- O(n+m) merge via `operator+=` (single linear scan)
- O(log n) binary search in `addTerm`
- cache-friendly dot products (contiguous memory, no hash overhead)

`addTerm` handles duplicate variables by incrementing the existing coefficient. If the result falls below `cancellation_tol` (default 1e-9), the term is removed to avoid accumulating near-zero noise.

**Building expressions**: use the operator overloads (`3.0 * x + 2.0 * y - z`), which maintain the sorted invariant automatically. Manual construction must ensure sorted, deduplicated `varIds`.

## Config

Three global tolerances, each settable at runtime:

| Name | Default | Usage |
|------|---------|-------|
| `cancellation_tol` | 1e-9 | Zero threshold for coefficient cancellation in `LinearExpr::addTerm` |
| `lp_feasibility_tol` | 1e-9 | Primal feasibility check in the simplex; also used by `Domain::isFixed()` |
| `lp_optimality_tol` | 1e-9 | Dual optimality check - reduced cost threshold |

These are `inline` globals (not constants) to allow problem-specific tuning without recompilation. The LP solver's per-solve `LPOptions` can override the last two locally.

## Sense

```cpp
enum class Sense { LessEq, Equal, GreaterEq };
```

Direction of an LP or CP constraint. Used in `LPConstraint` and printed verbatim in the proof log.

## Design assumptions

- Variable IDs are dense integers starting at 0. Ghost variables (CP-only) occupy the tail of the array and are never seen by the LP solver.
- `double` precision throughout. No interval arithmetic, no exact rational arithmetic.
- Coefficient cancellation is purely local to `LinearExpr`; the LP solver does not re-cancel during standard-form construction.
