#pragma once

namespace baguette {

/// Global zero threshold for coefficient cancellation.
///
/// A coefficient whose absolute value is ≤ `precision` is treated as zero
/// and removed from a LinearExpr (in addTerm and operator+).
///
/// Default: 1e-15 (appropriate for coefficients of order 1).
/// Tighten for ill-conditioned models; relax for large-magnitude coefficients.
inline double precision = 1e-15;

/// Set the global zero threshold used when cancelling coefficients.
/// @param eps New threshold value (must be ≥ 0).
inline void set_precision(double eps) { precision = eps; }

} // namespace baguette
