#pragma once

#include <chrono>
#include <cstdint>
#include <limits>

#include "baguette/lp/LPResult.hpp"
#include "baguette/model/Model.hpp"

namespace baguette {

/// Shared clock type for all solver time limits.
using SolverClock = std::chrono::steady_clock;

/// LP solving algorithm.
enum class LPMethod {
    Auto,           ///< Try dual simplex first; fall back to primal when preconditions fail.
    PrimalSimplex,  ///< Two-phase primal simplex (phase I + II).
    DualSimplex,    ///< Dual simplex with automatic primal fallback.
    RevisedSimplex, ///< Two-phase primal revised simplex: maintains B⁻¹ explicitly (m×m)
                    ///< via LU factorisation instead of the full m×n tableau.
                    ///< Same algorithmic path as PrimalSimplex, smaller memory footprint
                    ///< when m ≪ n. Warm-start not supported on this path.
    ShortStepIPM,   ///< Short-step feasible path-following interior-point method.
                    ///< Fixed step α = 1/(1+√n) keeps the iterate in the N₂(θ)
                    ///< neighbourhood; convergence in O(√n log(1/ε)) iterations.
                    ///< Warm-start and sensitivity analysis not supported.
};

/// Options for LP solves — analogous to BBOptions for MILP.
///
/// Default-constructed LPOptions{} produces a cold-start Auto solve with no
/// iteration or time limit, and no warm basis.
struct LPOptions {
    /// Solving algorithm. Auto tries dual simplex and falls back to primal.
    LPMethod method = LPMethod::Auto;

    /// Maximum number of simplex pivots. 0 = unlimited.
    uint32_t maxIter = 0;

    /// Wall-clock time limit in seconds. Default = unlimited.
    double timeLimitS = std::numeric_limits<double>::infinity();

    /// Reference clock for timeLimitS. Set once at the B&B root and reuse across
    /// nodes to enforce a shared time budget. Defaults to construction time.
    SolverClock::time_point startTime = SolverClock::now();

    /// Warm-start basis from a previous solve (e.g. parent B&B node).
    /// Empty = cold start. Only honoured by DualSimplex / Auto.
    BasisRecord warmBasis;

    /// If true, populate LPDetailedResult::sensitivity (RHS and objective ranging).
    /// O(m·n) overhead — avoid in B&B hot loops. Default false.
    bool computeSensitivity = false;

    /// If true, populate LPDetailedResult::fractionalRows for GMI cut generation.
    /// Default false.
    bool computeCutData = false;
};

/// Solve the LP relaxation of @p model.
///
/// Integer and Binary variables are treated as continuous (LP relaxation).
///
/// @note Complexity: O(m·n) for standard-form setup, then O(K·m·n) for the
///   simplex, where K = total pivot count (problem-dependent).
LPResult solveLP(const Model& model, const LPOptions& opts = {});

/// Solve the LP relaxation of @p model and return the full detailed result.
///
/// In addition to the basic LPResult, provides dual variables, reduced costs,
/// a BasisRecord for B&B warm-starting, Farkas certificates, and optionally
/// sensitivity analysis and cut data.
///
/// @note Dual variables, reduced costs, and basis are valid only when
///       result.status == Optimal. Farkas is valid when status == Infeasible.
LPDetailedResult solveLPDetailed(const Model& model, const LPOptions& opts = {});

} // namespace baguette
