#pragma once

#include <chrono>
#include <cstdint>
#include <limits>
#include <ostream>
#include <string_view>

#include "baguette/lp/LPResult.hpp"
#include "baguette/model/Model.hpp"

namespace baguette {

/// Shared clock type for all solver time limits.
using SolverClock = std::chrono::steady_clock;

/// LP solving algorithm.
enum class LPMethod {
    Auto,           ///< DualSimplexBV with automatic fallback to PrimalSimplexBV.
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
    MehrotraIPM,    ///< Primal-dual infeasible-start IPM with Mehrotra predictor-corrector.
                    ///< Affine predictor (μ=0) + centering corrector with σ=(μ_aff/μ)³.
                    ///< Detects infeasibility and unboundedness. Typically 15–50 iterations.
                    ///< Warm-start and sensitivity analysis not supported.
    PrimalSimplexBV,///< Two-phase primal simplex with bounded-variable (BV) ratio test.
                    ///< Variable upper bounds enforced via complement invariant — no
                    ///< explicit UB rows added, keeping m = nOrigRows. This eliminates
                    ///< the O(n) row inflation of PrimalSimplex on bounded problems.
                    ///< Warm-start and sensitivity analysis not supported.
    DualSimplexBV,  ///< Dual simplex with bounded-variable complement invariant.
                    ///< m = nOrigRows (no UB rows). Leaving variable exits to LB or UB;
                    ///< entering selection maintains dual feasibility in both cases.
                    ///< Warm-start via BasisRecord::atUBCache (BV→BV). Falls back to
                    ///< PrimalSimplexBV when dual feasibility cannot be established.
                    ///< Sensitivity analysis not supported.
    RevisedSimplexBV, ///< Two-phase primal revised simplex with bounded-variable invariant.
                    ///< Combines the O(m²) memory footprint of RevisedSimplex with the
                    ///< UB-row elimination of BV methods: m = nOrigRows, no explicit UB
                    ///< rows. Periodic LU reinversion every reinversion_period pivots.
                    ///< Sensitivity analysis and warm-start not supported.
    NetworkSimplex, ///< Primal network simplex for min-cost flow LPs.
                    ///< Detects node-arc incidence structure (equality constraints,
                    ///< ±1 coefficients, each variable in exactly 2 rows). If detected,
                    ///< basis = rooted spanning tree; pivots in O(n) vs O(m²) for the
                    ///< general simplex, giving 100–1000× speed-ups on network LPs.
                    ///< Falls back to DualSimplexBV when the model is not a pure network.
                    ///< Sensitivity analysis and warm-start not supported.
};

inline std::string_view to_string(LPMethod m) {
    switch (m) {
        case LPMethod::Auto:             return "Auto";
        case LPMethod::PrimalSimplex:    return "PrimalSimplex";
        case LPMethod::DualSimplex:      return "DualSimplex";
        case LPMethod::RevisedSimplex:   return "RevisedSimplex";
        case LPMethod::ShortStepIPM:     return "ShortStepIPM";
        case LPMethod::MehrotraIPM:      return "MehrotraIPM";
        case LPMethod::PrimalSimplexBV:   return "PrimalSimplexBV";
        case LPMethod::DualSimplexBV:     return "DualSimplexBV";
        case LPMethod::RevisedSimplexBV:  return "RevisedSimplexBV";
        case LPMethod::NetworkSimplex:    return "NetworkSimplex";
    }
    return "Unknown";
}

inline std::ostream& operator<<(std::ostream& os, LPMethod m) {
    return os << to_string(m);
}

/// Options for LP solves — analogous to BBOptions for MILP.
///
/// Default-constructed LPOptions{} produces a cold-start Auto solve with no
/// iteration or time limit, and no warm basis.
struct LPOptions {
    /// Solving algorithm. Auto uses DualSimplexBV with fallback to PrimalSimplexBV.
    LPMethod method = LPMethod::Auto;

    /// Maximum number of simplex pivots. 0 = unlimited.
    uint32_t maxIter = 0;

    /// Wall-clock time limit in seconds. Default = unlimited.
    double timeLimitS = std::numeric_limits<double>::infinity();

    /// Reference clock for timeLimitS. Set once at the B&B root and reuse across
    /// nodes to enforce a shared time budget. Defaults to construction time.
    SolverClock::time_point startTime = SolverClock::now();

    /// Warm-start basis from a previous solve (e.g. parent B&B node).
    /// Empty = cold start. Honoured by DualSimplex, DualSimplexBV, and Auto.
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
