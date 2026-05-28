#pragma once

#include <vector>

#include "baguette/milp/BranchAndBound.hpp"
#include "baguette/model/Model.hpp"

namespace baguette {

/// Build a CutGenerator for Subtour Elimination Constraints (SEC)
/// for a symmetric TSP instance formulated with binary edge variables.
///
/// For any proper subset S ⊂ {0..n-1} (|S| ≥ 2), the SEC reads:
///
///     Σᵢ∈S Σⱼ∈S, j>i  xᵢⱼ  ≤  |S| − 1
///
/// A SEC is violated when the LP solution contains a sub-tour restricted to S
/// (i.e. the out-cut capacity Σᵢ∈S Σⱼ∉S xᵢⱼ < 2).
///
/// At each B&B node the generator:
///  1. Builds an undirected weighted graph from the LP solution.
///  2. Computes the global minimum cut via Stoer-Wagner (O(n²)).
///  3. If the cut capacity < 2 − intFeasTol, returns the violated SEC for the
///     lighter side S.
///
/// Only one cut is returned per call (the globally tightest one).  The B&B
/// framework re-invokes the generator after each LP re-solve, so additional
/// violated SECs are found in subsequent iterations.
///
/// @param n         Number of cities (0..n-1).
/// @param edgeVar   n×n matrix; edgeVar[i][j] (i < j) is the Variable for
///                  edge (i, j).  The lower triangle is not accessed.
/// @param intFeasTol Cut is emitted only when cutCapacity < 2 − intFeasTol.
///
/// Usage:
///   opts.cutGenerators.push_back(makeSecGenerator(n, edgeVar));
///
/// @note Complexity O(n²) per B&B node (Stoer-Wagner adjacency-matrix).
CutGenerator makeSecGenerator(int                                   n,
                               std::vector<std::vector<Variable>>   edgeVar,
                               double                               intFeasTol = 1e-6);

} // namespace baguette
