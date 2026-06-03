# Model

The `model/` module is the user-facing API for problem construction. It is called once before any solve and is never accessed on the hot path of the LP or B&B loop.

## Hot / Cold separation

Model data is split into two structs:

```
ModelHot   { lb[], ub[], obj[] }       -- accessed every simplex iteration
ModelCold  { labels[], types[], ... }  -- accessed only during construction and output
```

The LP solver receives `ModelHot` directly and iterates over contiguous arrays. `ModelCold` (variable labels, types, reverse indices) is never touched during solving. This layout ensures that simplex inner loops are cache-friendly regardless of problem size.

## Variables

```cpp
Variable addVar(double lb, double ub, VarType type, std::string label = "");
```

`VarType` is `Continuous`, `Integer`, or `Binary`. The returned `Variable` is a handle valid for the lifetime of the model.

**Ghost variables** are CP-only variables fixed at a constant value, invisible to the LP solver:

```cpp
Variable addGhostVar(double fixedVal, VarType type, std::string label = "");
```

Ghost variables occupy indices `[numVars(), numTotalVars())`. The LP solver sees only `[0, numVars())`. CP constraints reference them by their full ID; propagation enforces their fixed value via bound equality.

**Invariant**: ghost variables must be added after all LP variables.

## LP Constraints

```cpp
ConstraintId addLPConstraint(LPConstraint c);
```

Two copies of each constraint are stored:
- `originals_` - the user-supplied form, possibly two-sided (`lhs sense rhs_expr`). Accessible via `getLPConstraint(id)` for debugging.
- `constraints_` - the normalized form: all variable terms on `lhs`, `rhs` is a scalar, `lhs.constant == 0`. This is the solver-facing view, returned by `getLPConstraints()`.

Normalization moves RHS variable terms to the LHS and adjusts the constant.

## Objective

```cpp
void setObjective(LinearExpr expr, ObjSense sense = ObjSense::Minimize);
```

Converts the sparse `LinearExpr` into the dense `hot.obj` vector (size == `numVars()`). The constant term is stored separately in `objConstant` and added back to the reported objective value after solving without affecting optimality.

## Reverse indices

`ModelCold` maintains two reverse indices built during construction:

- `varToLP[v]` - list of constraint indices where variable `v` appears. Used by bound-tightening presolve to propagate a bound change to the right constraints.
- `varToCP[v]` - list of CP constraint indices where variable `v` participates. Used by CP propagation dirty-set tracking.

## Bound manipulation

```cpp
void setVarBounds(Variable var, double newLb, double newUb);  // in-place, O(1)
Model withVarBounds(Variable var, double newLb, double newUb) const;  // copy, O(model)
```

`setVarBounds` is the B&B hot-loop API: modify bounds before solving a child node, solve, then restore. No allocation. `withVarBounds` copies the full model - convenient for tests, too slow for B&B.

**Warm-start constraint**: the finiteness of bounds must not change relative to the parent solve that produced the `BasisRecord`. Changing a bound from finite to infinite (or vice versa) alters the number of rows/columns in the standard form, making the parent basis structurally incompatible. The solver detects this via `sfCache` dimension mismatch and falls back to a cold start.

## Presolve

Two presolve stages operate on a copy of the model (the original is never mutated):

**Bound-tightening (`presolveTB`)**: for each constraint `sum(a_i x_i) <= b`, computes the implied upper bound on each `x_i` from the bounds of the other variables. Repeats to fixpoint. Terminates because each pass can only narrow bounds. Convergence is guaranteed; `presolveMaxPasses` guards against slow convergence.

**Elimination (`presolveElim`)**: removes fixed variables (`lb == ub`) and always-satisfied constraints, reducing problem size before the LP solve. `postsolveElim()` reconstructs the full dual/primal solution afterwards.

## Design assumptions

- The model is built sequentially; no thread safety on construction.
- All variable IDs are stable after creation - adding more variables does not renumber existing ones.
- The model is not modified during a solve. B&B uses `setVarBounds` which only mutates `hot.lb`/`hot.ub`, never the constraint matrix.
