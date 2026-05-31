#pragma once

#include <atomic>
#include <cstdint>
#include <limits>
#include <mutex>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

#include "baguette/cp/CPTypes.hpp"
#include "baguette/lp/LPResult.hpp"
#include "baguette/milp/MILPResult.hpp"
#include "baguette/model/Model.hpp"

namespace baguette::internal {

/// Buffered, thread-safe proof writer for the Baguette B&B log-proof format.
///
/// Emits a machine-verifiable certificate of the B&B search to an std::ostream.
/// Output is accumulated in a 64 KB string buffer; it is flushed automatically
/// when the buffer is full, on writeResult(), and in the destructor.
///
/// @par Format (v0.7.0 intermediate - will be transcribed to veriPB in v0.7.1)
///
///   BB-PROOF 0.7.0
///   VARS n  CONS m
///
///   N  id parentId|-1          node declaration
///   DIFF id varname [lb, ub]   branching bound (absent for the root)
///   LP id obj OPTIMAL          LP solved optimally
///   LP id - INFEASIBLE         LP infeasible (followed by FARKAS_*)
///   FARKAS_BOUND id v lb=x ub=y  bound certificate (lb > ub)
///   FARKAS_LP id n             tableau certificate: n explicit terms follow
///     yi * (expr sense rhs)    ...one line per non-zero yi
///   BRANCH id v frac L=l R=r  branching decision; l/r are child proof IDs, - = absent
///   INCUMBENT id obj           this node establishes a new (better) incumbent
///   LEAF id obj                integer-feasible but does not improve incumbent
///   PRUNE_BOUND id incumbent=x by=n  pruned: LP bound dominated by incumbent
///   CP_INFEASIBLE id           pruned: CP propagation proved infeasibility
///   UNVERIFIED id {MAX_ITER|NUMERICAL_FAILURE}  cannot certify this node
///   CUT nodeId (expr sense rhs)  cut added globally during processing of nodeId
///
///   RESULT {status} obj        final result (always last, triggers flush)
///
/// @par Thread safety
///   allocId() is lock-free (atomic fetch_add). All write*() methods acquire
///   an internal mutex - safe for future multi-threaded B&B where multiple
///   workers explore sub-trees concurrently. In the current single-threaded
///   solver the mutex is uncontested and adds negligible overhead.
class ProofWriter {
    static constexpr uint32_t    kNoNode         = std::numeric_limits<uint32_t>::max();
    static constexpr std::size_t kFlushThreshold = 64 * 1024; // 64 KB

    std::ostream*         os_;
    std::mutex            mu_;
    std::string           buf_;
    std::atomic<uint32_t> nextId_{0};

    void       flushLocked();
    void       appendLocked(std::string_view sv);

    static std::string fmtDouble(double v);
    static std::string fmtVar(uint32_t id, const ModelCold& cold);
    static std::string fmtSense(Sense s);
    static std::string fmtExpr(const LinearExpr& expr, const ModelCold& cold);
    static std::string fmtConstraint(const LPConstraint& c, const ModelCold& cold);
    static std::string fmtTightenLine(std::string_view tag, uint32_t id,
                                       const std::vector<uint32_t>& varIds,
                                       const ModelHot& hot, const ModelCold& cold);

public:
    explicit ProofWriter(std::ostream* os) : os_(os) {}
    ~ProofWriter();

    /// Sentinel value for "no node" (root's parent, absent child in BRANCH).
    static constexpr uint32_t noNode() noexcept { return kNoNode; }

    /// Allocate a fresh node ID. Lock-free; safe to call concurrently.
    /// @par Complexity O(1) - single atomic fetch_add.
    uint32_t allocId() noexcept {
        return nextId_.fetch_add(1u, std::memory_order_relaxed);
    }

    /// Write the proof header (VARS / CONS counts) from the post-presolve model.
    void writeHeader(const Model& model);

    /// Declare node @p id. Pass noNode() as @p parentId for the root.
    void writeNode(uint32_t id, uint32_t parentId);

    /// Write the single bound-change that created this child node.
    /// Not written for the root (no branching decision created it).
    void writeDiff(uint32_t id, uint32_t varId,
                   double lb, double ub, const ModelCold& cold);

    /// LP solved to optimality.
    void writeLPOptimal(uint32_t id, double obj);

    /// LP infeasible - always followed by writeFarkas().
    void writeLPInfeasible(uint32_t id);

    /// Explicit Farkas certificate. @p farkas must match the model state at solve time.
    void writeFarkas(uint32_t id, const FarkasRay& farkas, const Model& model);

    /// Record the branching decision at @p id. Use noNode() for absent children.
    void writeBranch(uint32_t id, uint32_t varId, double fracVal,
                     uint32_t leftId, uint32_t rightId, const ModelCold& cold);

    /// This node established a new (strictly better) incumbent.
    /// Always followed by writeSolution().
    void writeIncumbent(uint32_t id, double obj);

    /// Integer-feasible leaf that does not improve the current incumbent.
    /// Always followed by writeSolution().
    void writeLeaf(uint32_t id, double obj);

    /// Write the primal solution at an integer-feasible node (INCUMBENT or LEAF).
    /// @p sol contains the LP variable values (size == Model::numVars(), indexed
    ///   by Variable::id). The verifier can re-check all LP and CP constraints
    ///   using these values together with the ghost variable bounds in the header.
    void writeSolution(uint32_t id,
                       const std::vector<double>& sol,
                       const ModelCold&           cold);

    /// Node pruned because its LP bound cannot improve the incumbent.
    /// @p incNodeId is the proof ID of the node that established @p incVal.
    void writePruneByBound(uint32_t id, double incVal, uint32_t incNodeId);

    /// One CP propagation pass tightened at least one domain.
    /// @p varIds  variables whose bounds changed (from PropagationResult::changedVarIds).
    /// @p hot     model hot data read immediately after the propagation call.
    void writeCpTighten(uint32_t id,
                        const std::vector<uint32_t>& varIds,
                        const ModelHot&  hot,
                        const ModelCold& cold);

    /// The integer-bound rounding step (ceil lb / floor ub on Integer/Binary vars)
    /// changed at least one domain.  Logged before the LP solve, after CP propagation.
    void writeIntRound(uint32_t id,
                       const std::vector<uint32_t>& varIds,
                       const ModelHot&  hot,
                       const ModelCold& cold);

    /// Node pruned by CP propagation (no LP solve was performed).
    /// @p witness describes the failing constraint and the variables with their domains.
    void writeCpInfeasible(uint32_t id,
                            const std::optional<CPFailureWitness>& witness,
                            const ModelCold& cold);

    /// Node result is unverifiable - LP returned MaxIter or NumericalFailure.
    void writeUnverified(uint32_t id, LPStatus status);

    /// A globally-valid cut was added to the model during processing of @p nodeId.
    void writeCutAdded(uint32_t nodeId, const LinearExpr& expr,
                       Sense sense, double rhs, const ModelCold& cold);

    /// Write the global result. Forces a full buffer flush.
    void writeResult(MILPStatus status, double obj);

    /// Force-flush the internal buffer to the stream.
    void flush();
};

} // namespace baguette::internal
