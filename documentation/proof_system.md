# Proof System

The proof system produces a machine-verifiable certificate of the entire B&B search. A verifier can confirm correctness without re-running the solver by checking each node's certificate independently.

## Activation

```cpp
std::ofstream proofFile("proof.log");
BBOptions opts;
opts.proofStream = &proofFile;
solveMILP(model, opts);
```

When `proofStream` is null (default), no proof is generated and no overhead is incurred.

## Format v0.7.0

The proof is a text stream, one event per line, written in the order events occur during the B&B search.

### Header

```
BB-PROOF 0.7.0
VARS n  CONS m
```

`VARS n` is the number of LP variables (excludes ghost variables). `CONS m` is the number of LP constraints. Ghost variable bounds are also listed in the header for the verifier.

### Node lifecycle

Each B&B node goes through this sequence of events:

```
N id parentId|-1
DIFF id varname [lb, ub]
CP_TIGHTEN id v1=[lb1,ub1] v2=[lb2,ub2] ...
INT_ROUND id v1=[lb1,ub1] ...
LP id obj OPTIMAL
  -- or --
LP id - INFEASIBLE
FARKAS_BOUND id v lb=x ub=y
  -- or --
FARKAS_LP id n
  w1 * (expr1 sense1 rhs1)
  ...
BRANCH id varname frac L=l R=r
  -- or --
INCUMBENT id obj
SOL id [v1=val1, v2=val2, ...]
  -- or --
LEAF id obj
SOL id [v1=val1, ...]
  -- or --
PRUNE_BOUND id incumbent=x by=n
  -- or --
CP_INFEASIBLE id
UNVERIFIED id {MAX_ITER|NUMERICAL_FAILURE}
```

### Token reference

| Token | Arguments | Meaning |
|-------|-----------|---------|
| `N` | `id parentId\|-1` | Node declaration. `parentId = -1` for the root. |
| `DIFF` | `id varname [lb, ub]` | The single bound change that created this child node. Absent for the root. |
| `CP_TIGHTEN` | `id v=[lb,ub] ...` | CP propagation tightened at least one variable domain at this node. |
| `INT_ROUND` | `id v=[lb,ub] ...` | Integer-bound rounding (ceil lb / floor ub) changed at least one domain. |
| `LP` | `id obj OPTIMAL` | LP solved to optimality with this objective value. |
| `LP` | `id - INFEASIBLE` | LP infeasible. Always followed by a `FARKAS_*` line. |
| `FARKAS_BOUND` | `id v lb=x ub=y` | Bound certificate: variable `v` has empty domain `[x, y]` with `x > y`. |
| `FARKAS_LP` | `id n` + n lines | Tableau certificate: `n` weighted constraints whose combination proves infeasibility. |
| `BRANCH` | `id varname frac L=l R=r` | Branching decision on `varname` at fractional value `frac`. `l`, `r` are child proof IDs; `-` if the child was not created (e.g., budget exhausted). |
| `INCUMBENT` | `id obj` | This node found a new best integer solution. Always followed by `SOL`. |
| `LEAF` | `id obj` | Integer-feasible but does not improve the current best. Always followed by `SOL`. |
| `SOL` | `id [v1=val1, ...]` | Primal solution at an integer-feasible node. All LP variables listed. |
| `PRUNE_BOUND` | `id incumbent=x by=n` | Node pruned: LP bound cannot improve incumbent `x`. `by=n` is the proof ID of the node that established the incumbent. |
| `CP_INFEASIBLE` | `id` | Node pruned by CP propagation. The `CPFailureWitness` is embedded in the preceding `CP_TIGHTEN` context. |
| `UNVERIFIED` | `id MAX_ITER\|NUMERICAL_FAILURE` | LP hit its iteration limit or had a numerical failure. This node's result cannot be certified. |
| `CUT` | `nodeId (expr sense rhs)` | A globally valid cut was added during processing of node `nodeId`. |
| `RESULT` | `status obj` | Final solver outcome. Always the last line. Forces a full flush. |

### Example: two-node trace

```
BB-PROOF 0.7.0
VARS 2  CONS 1
N 0 -1
LP 0 3.5 OPTIMAL
BRANCH 0 x1 0.5 L=1 R=2
N 1 0
DIFF 1 x1 [0, 0]
LP 1 - INFEASIBLE
FARKAS_BOUND 1 x1 lb=1 ub=0
N 2 0
DIFF 2 x1 [1, 1]
LP 2 4.0 OPTIMAL
INCUMBENT 2 4.0
SOL 2 [x1=1.0, x2=2.0]
RESULT Optimal 4.0
```

## Farkas certificates

### Bound certificate

```
FARKAS_BOUND id v lb=x ub=y
```

Variable `v` has `lb = x > y = ub` after B&B branching. Verification: read `model.lb[v]` and `model.ub[v]` at this node's bound state and check `lb > ub`.

### Tableau certificate

```
FARKAS_LP id n
  w1 * (a1_1*x1 + ... LessEq b1)
  w2 * (a2_1*x1 + ... GreaterEq b2)
  ...
```

`n` non-zero weighted constraints from the model. The verifier checks:
1. Each `(expr sense rhs)` is a valid model constraint.
2. `sum(w_i * a_i_j) = 0` for each variable j (the combination cancels all variables).
3. `sum(w_i * b_i) < 0` (the combination yields a contradictory RHS).

This is Farkas' lemma: the existence of such `w` proves that no primal feasible point exists.

## CP infeasibility witness

When CP propagation reports infeasibility, the proof records the witness embedded in the `CP_TIGHTEN` / `CP_INFEASIBLE` sequence:

```
CP_TIGHTEN id x1=[2,2] x2=[2,2] x3=[2,2]
CP_INFEASIBLE id
```

The `CPFailureWitness` carries:
- `constraintDesc`: `"AllDiff"`, `"Cumulative(cap=N)"`, or `"Custom(idx=N)"`
- `varIds`, `varLb`, `varUb`: the minimal set of variables and their domains at failure

Verification: re-run the named constraint's feasibility check on the listed `(id, lb, ub)` triples. For AllDiff: three variables all fixed to value 2 is immediately infeasible. For Cumulative: check that the compulsory load sum exceeds capacity. Complexity: O(K^2) for AllDiff, O(N) for Cumulative.

## Integer-feasible solutions

```
INCUMBENT id 4.0
SOL id [x1=1.0, x2=2.0]
```

Verification:
1. For each integer/binary variable, check `|val - round(val)| <= intFeasTol`.
2. Re-evaluate all LP constraints: substitute solution values and verify each `expr sense rhs`.
3. Re-evaluate all CP constraints: call `cpFeasible(sol, tol)`.
4. Compute objective and confirm it matches the logged value.
5. For `INCUMBENT`: confirm this is strictly better than all preceding `INCUMBENT` nodes.

## Cutting planes in proof

```
CUT 5 (2*x1 + x2 LessEq 3)
```

Logged when a globally valid cut is added. The verifier checks that the cut is valid for the problem's integer hull (e.g., derived from the Gomory procedure on the tableau row of node 5). All nodes after this `CUT` event implicitly include this constraint.

## Unverified nodes

```
UNVERIFIED 7 MAX_ITER
```

Node 7's LP hit its iteration limit. The solver continued by discarding this node (conservative: assumes it might be feasible and optimal). The proof is incomplete for this subtree. A verifier must flag this node as not certified and report the gap.

## Performance

Proof generation adds:
- One mutex acquire per node event
- One string append per event (~100 bytes average)
- One 64 KB buffer flush every ~640 nodes

This is negligible compared to LP solve time on any non-trivial instance. The buffer is flushed automatically at 64 KB and forced on `RESULT`.

## Thread safety

`allocId()` is lock-free (atomic `fetch_add`). All `write*()` methods acquire an internal mutex. The format is designed for future multi-threaded B&B where multiple workers write concurrently to the same proof stream.

## Roadmap

Format v0.7.0 is an intermediate format. v0.7.1 will add automatic transcription to [veriPB](https://gitlab.com/MIAOresearch/software/VeriPB) format for integration with the veriPB verifier.
