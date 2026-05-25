#include "baguette/milp/SecCuts.hpp"

#include <algorithm>
#include <limits>
#include <vector>

namespace baguette {

namespace {

// Stoer-Wagner global minimum cut — O(n²) adjacency-matrix implementation.
//
// Mutates `w` in-place (contractions).  Requires n ≥ 2 and w[i][j] ≥ 0.
//
// @par Complexity O(n²)
struct SWResult { double value; std::vector<int> side; };

SWResult stoerWagner(int n, std::vector<std::vector<double>> w) {
    // nodes[i] = original vertices merged into super-node i
    std::vector<std::vector<int>> nodes(n);
    for (int i = 0; i < n; i++) nodes[i] = {i};
    std::vector<bool> active(n, true);

    double           bestVal  = std::numeric_limits<double>::infinity();
    std::vector<int> bestSide;

    for (int phase = 0; phase < n - 1; phase++) {
        // Maximum adjacency ordering: greedily add the most-connected vertex.
        std::vector<double> key(n, 0.0);
        std::vector<bool>   inA(n, false);
        int prev = -1, last = -1;

        for (int step = 0, alive = n - phase; step < alive; step++) {
            int u = -1;
            for (int i = 0; i < n; i++)
                if (active[i] && !inA[i] && (u == -1 || key[i] > key[u]))
                    u = i;

            inA[u] = true;
            prev   = last;
            last   = u;

            for (int i = 0; i < n; i++)
                if (active[i] && !inA[i])
                    key[i] += w[u][i];
        }

        // Cut of phase: {last} vs everything in A \ {last}.
        // key[last] = total weight of edges from last to the rest of A.
        if (key[last] < bestVal) {
            bestVal  = key[last];
            bestSide = nodes[last];
        }

        // Contract: merge last into prev.
        for (int i = 0; i < n; i++) {
            w[prev][i] += w[last][i];
            w[i][prev] += w[i][last];
        }
        nodes[prev].insert(nodes[prev].end(),
                           nodes[last].begin(), nodes[last].end());
        active[last] = false;
    }

    return {bestVal, std::move(bestSide)};
}

} // namespace

CutGenerator makeSecGenerator(int                                  n,
                               std::vector<std::vector<Variable>>  edgeVar,
                               double                              intFeasTol) {
    return [n, ev = std::move(edgeVar), intFeasTol]
           (const LPDetailedResult& lp, const Model& /*model*/) -> std::vector<Cut> {
        if (n < 3) return {};

        const auto& x = lp.result.primalValues;

        // Build support graph from LP solution.
        std::vector<std::vector<double>> w(n, std::vector<double>(n, 0.0));
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++) {
                const uint32_t id  = ev[i][j].id;
                const double   val = (id < static_cast<uint32_t>(x.size())) ? x[id] : 0.0;
                w[i][j] = w[j][i] = std::max(0.0, val);
            }

        auto [cutVal, sideS] = stoerWagner(n, std::move(w));

        if (cutVal >= 2.0 - intFeasTol) return {};

        // Use the smaller side to minimise the number of terms in the cut.
        if (static_cast<int>(sideS.size()) > n - static_cast<int>(sideS.size())) {
            std::vector<bool> inS(n, false);
            for (int v : sideS) inS[v] = true;
            sideS.clear();
            for (int v = 0; v < n; v++)
                if (!inS[v]) sideS.push_back(v);
        }

        const int sSize = static_cast<int>(sideS.size());
        if (sSize < 2 || sSize >= n) return {};

        // Build SEC: Σᵢ∈S Σⱼ∈S, j>i  xᵢⱼ  ≤  |S| - 1
        std::vector<bool> inS(n, false);
        for (int v : sideS) inS[v] = true;

        Cut cut;
        cut.sense = Sense::LessEq;
        cut.rhs   = static_cast<double>(sSize - 1);
        for (int i = 0; i < n; i++) {
            if (!inS[i]) continue;
            for (int j = i + 1; j < n; j++) {
                if (!inS[j]) continue;
                cut.expr.addTerm(ev[i][j], 1.0);
            }
        }

        if (cut.expr.empty()) return {};
        return {std::move(cut)};
    };
}

} // namespace baguette
