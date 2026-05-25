#pragma once

#include "baguette/core/LinearExpr.hpp"
#include "baguette/core/Sense.hpp"

namespace baguette {

/// A linear cut expressed in original model-variable space.
///
/// Used by the B&C loop (built-in GMI) and by user CutGenerator callbacks.
/// The sense field allows any direction; GMI always produces GreaterEq (default).
struct Cut {
    LinearExpr expr;
    Sense      sense = Sense::GreaterEq;
    double     rhs   = 0.0;
};

} // namespace baguette
