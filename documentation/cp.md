# CP Module

The `cp/` module implements constraint propagation. It is called at each B&B node before the LP solve to tighten variable domains and prune nodes that are provably infeasible without solving any LP.

## Container: CPConstraints

Constraints are stored in two separate tiers to control dispatch overhead:

```
builtins:  std::variant<AllDiffConstraint, CumulativeConstraint, ...>
           -> zero-overhead std::visit dispatch (no virtual call)

customs_:  std::vector<shared_ptr<const CPConstraint>>
           -> virtual dispatch (once per constraint per node)
```

Built-ins use `std::variant` because their types are known at compile time and they are called millions of times across a B&B tree. Custom user constraints use virtual dispatch because their types are open-ended and their propagation cost (O(K^2) or more) dwarfs the dispatch overhead.

## Built-in: AllDiff

```
AllDiffConstraint: all variables take distinct integer values
```

**Algorithm**: Bounds Consistency with Hall interval detection.

1. For each fixed variable (lb == ub), remove its value from every other variable's domain.
2. Detect Hall intervals: a set S of variables whose combined domain covers exactly |S| values cannot share values with variables outside S. Any variable outside S whose domain intersects this interval is tightened.
3. Repeat to fixpoint.

**Complexity**: O(K^2 * I) where K = number of variables, I = iterations to fixpoint (I <= K).

**Infeasibility**: detected when the domain of a variable becomes empty, or when a set of K variables must share fewer than K distinct values.

## Built-in: Cumulative

```
CumulativeConstraint: sum of resource consumption over overlapping tasks <= capacity
```

**Algorithm**: Compulsory part reasoning (Time-Tabling).

For each task i, the compulsory part is the time interval `[lct_i, est_i + p_i)` where the task is guaranteed to execute regardless of scheduling (`lct` = latest completion time, `est` = earliest start time, `p` = duration). For each task i, advance its earliest start past any window where the sum of compulsory loads from all other tasks exceeds `capacity - consumption_i`.

**Complexity**: O(N * D * I) where N = number of tasks, D = domain width, I = iterations.

**Infeasibility**: detected when the earliest start of a task exceeds its latest start.

## User-defined constraints

```cpp
class CPConstraint {
public:
    virtual PropagationResult propagate(Model& model) const = 0;
    virtual bool     cpFeasible(const vector<double>& sol, double tol) const = 0;
    virtual uint32_t cpViolatedVar(const vector<double>& sol, double tol) const = 0;
    virtual shared_ptr<const CPConstraint> reduce(const vector<uint32_t>& varMap) const;
};
```

A user-defined constraint subclasses `CPConstraint` and implements:
- `propagate` - tighten bounds, return changed variables and optional `CPFailureWitness`
- `cpFeasible` - check feasibility of a complete solution (used at integer-feasible nodes)
- `cpViolatedVar` - return the variable most violating the constraint (for branching heuristics)
- `reduce` - remap variable IDs after elimination presolve; return `nullptr` to drop the constraint

Constraints must be **immutable after construction** and are stored as `shared_ptr<const CPConstraint>` so the `Model` remains copyable without a deep clone.

## Propagation loop

```cpp
PropagationResult propagateCP(Model& model, bool toFixpoint);
```

Calls all constraints in sequence. If `toFixpoint` is true (default in B&B), repeats until no bounds change. The loop terminates in at most O(V) iterations (each iteration must narrow at least one bound).

The result carries:
- `status` - `Feasible` or `Infeasible`
- `changedVarIds` - sorted list of variables whose bounds were tightened (logged as `CP_TIGHTEN` in the proof)
- `witness` - the `CPFailureWitness` if infeasible (logged as `CP_INFEASIBLE` in the proof)

## CPFailureWitness

```cpp
struct CPFailureWitness {
    std::string constraintDesc;   // "AllDiff", "Cumulative(cap=N)", "Custom(idx=N)"
    uint32_t    constraintIdx;
    vector<uint32_t> varIds;      // variables that witness the infeasibility
    vector<double>   varLb;       // their lower bounds at failure time
    vector<double>   varUb;       // their upper bounds at failure time
};
```

A verifier re-runs the constraint's feasibility check on these `(varId, lb, ub)` triples without re-solving the full LP. This makes CP infeasibility nodes machine-verifiable at O(K^2) cost instead of O(m*n) for an LP.

## Integration with B&B

At each B&B node, after bound restoration and before the LP solve:

1. `propagateCP` is called (to fixpoint if `cpPropagateToFixpoint = true`).
2. All tightened bounds are recorded in `dirtyVars` and logged as `CP_TIGHTEN`.
3. If `Infeasible`, the node is pruned immediately (no LP solve), logged as `CP_INFEASIBLE` with the witness.
4. On backtrack, `restoreBounds` resets all dirty bounds to the parent node's values.

## Design assumptions

- CP propagation modifies `Model::hot.lb` / `Model::hot.ub` directly, then B&B restores them on backtrack. The constraint matrix is never modified.
- Ghost variables are propagated by CP constraints but fixed by construction; propagation cannot widen their domain.
- `cpFeasible` is only called when all integer variables are within `intFeasTol` of an integer value. It validates the CP constraints on the rounded solution.
