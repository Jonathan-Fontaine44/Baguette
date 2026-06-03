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

## Example: TSP n=5 with MTZ + AllDiff

A Travelling Salesman Problem on 5 cities illustrates how LP constraints (flow + MTZ subtour elimination) and a CP constraint (AllDiff on position variables) coexist in the same model.

### Problem

Minimize the total travel cost of a tour visiting all 5 cities exactly once.

Distance matrix (asymmetric):

```
     0   1   2   3   4
0  [ -   3   6  10   7 ]
1  [ 3   -   2   8   4 ]
2  [ 6   2   -   5   9 ]
3  [ 10  8   5   -   1 ]
4  [ 7   4   9   1   - ]
```

Optimal tour: 0 -> 1 -> 2 -> 3 -> 4 -> 0, cost = 18.

### Decision variables

- `x[i][j]` in {0,1}: 1 if the tour goes directly from city i to city j (20 binary variables).
- `u[i]` in {0..4} integer: position of city i in the tour, for MTZ subtour elimination.
  - `u[0]` is fixed to 0 (root city, no subtour can loop back without passing through 0).
  - `u[1..4]` range over {1..4} and must all be distinct.

### Code

```cpp
#include "baguette/model/Model.hpp"
#include "baguette/milp/BranchAndBound.hpp"
#include "baguette/cp/constraints/AllDiff.hpp"

using namespace baguette;

const int n = 5;
const double dist[n][n] = {
    { 0,  3,  6, 10,  7},
    { 3,  0,  2,  8,  4},
    { 6,  2,  0,  5,  9},
    {10,  8,  5,  0,  1},
    { 7,  4,  9,  1,  0},
};

Model model;

// Binary routing variables x[i][j] (i != j)
Variable x[n][n];
for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
        if (i != j)
            x[i][j] = model.addVar(0, 1, VarType::Binary,
                                   "x" + std::to_string(i) + std::to_string(j));

// Integer position variables for MTZ: u[0] fixed, u[1..4] in [1, n-1]
Variable u[n];
u[0] = model.addVar(0, 0, VarType::Integer, "u0");
for (int i = 1; i < n; i++)
    u[i] = model.addVar(1, n - 1, VarType::Integer, "u" + std::to_string(i));

// Objective: minimize total tour distance
LinearExpr obj;
for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
        if (i != j)
            obj += dist[i][j] * x[i][j];
model.setObjective(obj, ObjSense::Minimize);

// Each city is left exactly once: sum_j x[i][j] = 1
for (int i = 0; i < n; i++) {
    LinearExpr row;
    for (int j = 0; j < n; j++)
        if (i != j) row += x[i][j];
    model.addLPConstraint(row, Sense::Equal, 1.0);
}

// Each city is entered exactly once: sum_i x[i][j] = 1
for (int j = 0; j < n; j++) {
    LinearExpr col;
    for (int i = 0; i < n; i++)
        if (i != j) col += x[i][j];
    model.addLPConstraint(col, Sense::Equal, 1.0);
}

// MTZ subtour elimination: u[i] - u[j] + n*x[i][j] <= n-1  (i,j in 1..4, i != j)
// If x[i][j]=1 (tour goes i->j), forces u[j] >= u[i]+1, so positions are strictly ordered.
for (int i = 1; i < n; i++)
    for (int j = 1; j < n; j++)
        if (i != j)
            model.addLPConstraint(u[i] - u[j] + (double)n * x[i][j],
                                  Sense::LessEq, (double)(n - 1));

// AllDiff on u[1..4]: positions must be distinct integers in [1,4]
// Redundant with MTZ here but enables CP propagation to fix positions early
// and prune subtree branches before any LP solve.
std::vector<Variable> uVars(u + 1, u + n);
model.addCPConstraint(AllDiffConstraint(uVars));

// Solve
BBOptions opts;
opts.presolveLevel = 2;  // includes CP fixpoint propagation at root
MILPResult result = solveMILP(model, opts);
// result.objectiveValue == 18.0, result.status == MILPStatus::Optimal
```

### What happens at each B&B node

1. **CP propagation** runs AllDiff on `u[1..4]`. Once a position variable is fixed by branching, its value is eliminated from the domains of all other position variables. This can close multiple siblings without any LP solve.
2. **LP relaxation** is solved. The MTZ constraints ensure that fractional `u` values cannot form subtours in the relaxation.
3. **Branching** targets the fractional `x[i][j]` or `u[i]` with the highest priority (MostFractional by default).

### Variable count summary

| Variable group | Count  | Type             |
| -------------- | ------ | ---------------- |
| `x[i][j]`      | 20     | Binary           |
| `u[0]`         | 1      | Integer (fixed)  |
| `u[1..4]`      | 4      | Integer [1, 4]   |
| **Total**      | **25** |                  |

| Constraint group | Count  |
| ---------------- | ------ |
| Degree out       | 5      |
| Degree in        | 5      |
| MTZ              | 12     |
| AllDiff (CP)     | 1      |
| **Total**        | **23** |

## Design assumptions

- The model is built sequentially; no thread safety on construction.
- All variable IDs are stable after creation - adding more variables does not renumber existing ones.
- The model is not modified during a solve. B&B uses `setVarBounds` which only mutates `hot.lb`/`hot.ub`, never the constraint matrix.
