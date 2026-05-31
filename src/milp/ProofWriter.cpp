#include "milp/ProofWriter.hpp"

#include <cmath>
#include <cstdio>
#include <string>

#include "baguette/core/Sense.hpp"
#include "baguette/model/ModelData.hpp"

namespace baguette::internal {

// ── Formatting helpers ─────────────────────────────────────────────────────────

std::string ProofWriter::fmtDouble(double v) {
    if (std::isinf(v)) return v > 0.0 ? "+inf" : "-inf";
    if (std::isnan(v)) return "nan";
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.17g", v);
    return buf;
}

std::string ProofWriter::fmtVar(uint32_t id, const ModelCold& cold) {
    if (id < cold.labels.size() && !cold.labels[id].empty())
        return cold.labels[id];
    char buf[16];
    std::snprintf(buf, sizeof(buf), "x%u", id);
    return buf;
}

std::string ProofWriter::fmtSense(Sense s) {
    switch (s) {
        case Sense::LessEq:    return "<=";
        case Sense::GreaterEq: return ">=";
        case Sense::Equal:     return "==";
    }
    return "?";
}

// Write a LinearExpr as "a1*v1 + a2*v2 - a3*v3 + ...".
// Handles coefficients of ±1 without the "1*" prefix.
// Constant term appended if non-zero.
std::string ProofWriter::fmtExpr(const LinearExpr& expr, const ModelCold& cold) {
    if (expr.empty())
        return expr.constant == 0.0 ? "0" : fmtDouble(expr.constant);

    std::string s;
    bool first = true;
    for (std::size_t i = 0; i < expr.size(); ++i) {
        double c          = expr.coeffs[i];
        const std::string vn = fmtVar(expr.varIds[i], cold);
        if (first) {
            if      (c ==  1.0) s += vn;
            else if (c == -1.0) s += "-" + vn;
            else                s += fmtDouble(c) + "*" + vn;
            first = false;
        } else {
            if (c >= 0.0) { s += " + "; }
            else          { s += " - "; c = -c; }
            if (c == 1.0) s += vn;
            else          s += fmtDouble(c) + "*" + vn;
        }
    }
    if (expr.constant != 0.0) {
        if (expr.constant > 0.0) s += " + " + fmtDouble(expr.constant);
        else                     s += " - " + fmtDouble(-expr.constant);
    }
    return s;
}

// Format a normalized LPConstraint as "expr sense rhs".
std::string ProofWriter::fmtConstraint(const LPConstraint& c, const ModelCold& cold) {
    return fmtExpr(c.lhs, cold) + " " + fmtSense(c.sense) + " " + fmtDouble(c.rhsConst);
}

// ── Buffer management ──────────────────────────────────────────────────────────

ProofWriter::~ProofWriter() {
    flush();
}

void ProofWriter::flushLocked() {
    if (!buf_.empty() && os_) {
        os_->write(buf_.data(), static_cast<std::streamsize>(buf_.size()));
        buf_.clear();
    }
}

void ProofWriter::appendLocked(std::string_view sv) {
    buf_ += sv;
    if (buf_.size() >= kFlushThreshold)
        flushLocked();
}

void ProofWriter::flush() {
    std::lock_guard lock(mu_);
    flushLocked();
}

// ── Public write methods ───────────────────────────────────────────────────────

void ProofWriter::writeHeader(const Model& model) {
    std::string s;
    s += "BB-PROOF 0.7.0\n";
    s += "VARS " + std::to_string(model.numVars())        + "\n";
    s += "CONS " + std::to_string(model.numConstraints()) + "\n";
    std::lock_guard lock(mu_);
    appendLocked(s);
}

void ProofWriter::writeNode(uint32_t id, uint32_t parentId) {
    std::string s = "N " + std::to_string(id) + " ";
    s += (parentId == kNoNode) ? "-1" : std::to_string(parentId);
    s += "\n";
    std::lock_guard lock(mu_);
    appendLocked(s);
}

void ProofWriter::writeDiff(uint32_t id, uint32_t varId,
                             double lb, double ub, const ModelCold& cold) {
    std::string s = "DIFF " + std::to_string(id) + " "
                  + fmtVar(varId, cold)
                  + " [" + fmtDouble(lb) + ", " + fmtDouble(ub) + "]\n";
    std::lock_guard lock(mu_);
    appendLocked(s);
}

void ProofWriter::writeLPOptimal(uint32_t id, double obj) {
    std::string s = "LP " + std::to_string(id) + " " + fmtDouble(obj) + " OPTIMAL\n";
    std::lock_guard lock(mu_);
    appendLocked(s);
}

void ProofWriter::writeLPInfeasible(uint32_t id) {
    std::string s = "LP " + std::to_string(id) + " - INFEASIBLE\n";
    std::lock_guard lock(mu_);
    appendLocked(s);
}

void ProofWriter::writeFarkas(uint32_t id, const FarkasRay& farkas, const Model& model) {
    std::string s;
    if (farkas.infeasVarId >= 0) {
        // Bound certificate: lb > ub after branching
        const uint32_t vid = static_cast<uint32_t>(farkas.infeasVarId);
        s += "FARKAS_BOUND " + std::to_string(id) + " "
           + fmtVar(vid, model.getCold())
           + " lb=" + fmtDouble(model.getHot().lb[vid])
           + " ub=" + fmtDouble(model.getHot().ub[vid]) + "\n";
    } else if (!farkas.y.empty()) {
        // Tableau certificate: explicit weighted combination of model constraints.
        // y has size == model.numConstraints() (including any cuts added so far).
        // A^T y >= 0 and b^T y < 0 => the current system is infeasible.
        const auto& cons = model.getLPConstraints();
        const auto& cold = model.getCold();
        uint32_t nz = 0;
        for (std::size_t i = 0; i < farkas.y.size() && i < cons.size(); ++i)
            if (farkas.y[i] != 0.0) ++nz;
        s += "FARKAS_LP " + std::to_string(id) + " " + std::to_string(nz) + "\n";
        for (std::size_t i = 0; i < farkas.y.size() && i < cons.size(); ++i) {
            if (farkas.y[i] == 0.0) continue;
            s += "  " + fmtDouble(farkas.y[i]) + " * (" + fmtConstraint(cons[i], cold) + ")\n";
        }
    }
    // No certificate available (e.g. early bound infeasibility before LP, already covered
    // by FARKAS_BOUND; or NumericalFailure which produces an UNVERIFIED event instead).
    std::lock_guard lock(mu_);
    appendLocked(s);
}

void ProofWriter::writeBranch(uint32_t id, uint32_t varId, double fracVal,
                               uint32_t leftId, uint32_t rightId, const ModelCold& cold) {
    std::string s = "BRANCH " + std::to_string(id) + " "
                  + fmtVar(varId, cold) + " " + fmtDouble(fracVal) + " L=";
    s += (leftId  == kNoNode) ? "-" : std::to_string(leftId);
    s += " R=";
    s += (rightId == kNoNode) ? "-" : std::to_string(rightId);
    s += "\n";
    std::lock_guard lock(mu_);
    appendLocked(s);
}

void ProofWriter::writeIncumbent(uint32_t id, double obj) {
    std::string s = "INCUMBENT " + std::to_string(id) + " " + fmtDouble(obj) + "\n";
    std::lock_guard lock(mu_);
    appendLocked(s);
}

void ProofWriter::writeLeaf(uint32_t id, double obj) {
    std::string s = "LEAF " + std::to_string(id) + " " + fmtDouble(obj) + "\n";
    std::lock_guard lock(mu_);
    appendLocked(s);
}

void ProofWriter::writePruneByBound(uint32_t id, double incVal, uint32_t incNodeId) {
    std::string s = "PRUNE_BOUND " + std::to_string(id)
                  + " incumbent=" + fmtDouble(incVal) + " by=";
    s += (incNodeId == kNoNode) ? "?" : std::to_string(incNodeId);
    s += "\n";
    std::lock_guard lock(mu_);
    appendLocked(s);
}

void ProofWriter::writeCpInfeasible(uint32_t id,
                                     const std::optional<CPFailureWitness>& witness,
                                     const ModelCold& cold) {
    std::string s = "CP_INFEASIBLE " + std::to_string(id);
    if (!witness) {
        s += "\n";
    } else {
        s += " ";
        s += witness->constraintDesc;
        s += " [";
        for (std::size_t i = 0; i < witness->varIds.size(); ++i) {
            if (i > 0) s += ", ";
            s += fmtVar(witness->varIds[i], cold);
            s += "=[";
            s += fmtDouble(witness->varLb[i]);
            s += ", ";
            s += fmtDouble(witness->varUb[i]);
            s += "]";
        }
        s += "]\n";
    }
    std::lock_guard lock(mu_);
    appendLocked(s);
}

void ProofWriter::writeUnverified(uint32_t id, LPStatus status) {
    std::string s = "UNVERIFIED " + std::to_string(id) + " ";
    s += (status == LPStatus::MaxIter) ? "MAX_ITER" : "NUMERICAL_FAILURE";
    s += "\n";
    std::lock_guard lock(mu_);
    appendLocked(s);
}

void ProofWriter::writeCutAdded(uint32_t nodeId, const LinearExpr& expr,
                                 Sense sense, double rhs, const ModelCold& cold) {
    std::string s = "CUT " + std::to_string(nodeId) + " ("
                  + fmtExpr(expr, cold) + " " + fmtSense(sense) + " " + fmtDouble(rhs) + ")\n";
    std::lock_guard lock(mu_);
    appendLocked(s);
}

void ProofWriter::writeResult(MILPStatus status, double obj) {
    std::string s = "RESULT ";
    s += to_string(status);
    s += " ";
    s += fmtDouble(obj);
    s += "\n";
    std::lock_guard lock(mu_);
    appendLocked(s);
    flushLocked(); // force full flush - this is always the last line
}

} // namespace baguette::internal
