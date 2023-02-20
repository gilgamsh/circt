//===- InlineArcs.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Support/BackedgeBuilder.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arc-inline"

using namespace circt;
using namespace arc;
using llvm::SetVector;
using llvm::SmallDenseSet;
using mlir::SymbolUserMap;

namespace {
struct InlineArcsPass : public InlineArcsBase<InlineArcsPass> {
  InlineArcsPass() = default;
  InlineArcsPass(const InlineArcsPass &pass) : InlineArcsPass() {}

  void runOnOperation() override;
  bool shouldInline(DefineOp defOp, ArrayRef<Operation *> users);
  void inlineUse(DefineOp defOp, StateOp useOp, bool removeArc);

  Statistic numInlinedArcs{this, "inlined-arcs", "Arcs inlined at a use site"};
  Statistic numRemovedArcs{this, "removed-arcs",
                           "Arcs removed after full inlining"};
  Statistic numTrivialArcs{this, "trivial-arcs", "Arcs with very few ops"};
  Statistic numSingleUseArcs{this, "single-use-arcs", "Arcs with a single use"};
};
} // namespace

void InlineArcsPass::runOnOperation() {
  auto module = getOperation();
  SymbolTableCollection symbolTable;
  SymbolUserMap symbolUserMap(symbolTable, module);

  for (auto defOp : llvm::make_early_inc_range(module.getOps<DefineOp>())) {
    // Check if we should inline the arc.
    auto users = symbolUserMap.getUsers(defOp);
    if (!shouldInline(defOp, users))
      continue;
    LLVM_DEBUG(llvm::dbgs() << "Inlining " << defOp.getSymName() << "\n");

    // Inline all uses of the arc. Currently we inline all of them but in the
    // future we may decide per use site whether to inline or not.
    unsigned numUsersLeft = users.size();
    for (auto *user : users) {
      auto useOp = dyn_cast<StateOp>(user);
      if (!useOp)
        continue;
      if (useOp.getLatency() > 0)
        continue;
      inlineUse(defOp, useOp, --numUsersLeft == 0);
      ++numInlinedArcs;
    }

    // Track if we have completely inlined the arc.
    if (numUsersLeft == 0)
      ++numRemovedArcs;
  }
}

bool InlineArcsPass::shouldInline(DefineOp defOp, ArrayRef<Operation *> users) {
  // Count the number of non-trivial ops in the arc. If there are only a few,
  // inline the arc.
  unsigned numNonTrivialOps = 0;
  defOp.getBodyBlock().walk([&](Operation *op) {
    if (!op->hasTrait<OpTrait::ConstantLike>() && !isa<OutputOp>(op))
      ++numNonTrivialOps;
  });
  if (numNonTrivialOps <= 3) {
    ++numTrivialArcs;
    return true;
  }
  LLVM_DEBUG(llvm::dbgs() << "Arc " << defOp.getSymName() << " has "
                          << numNonTrivialOps << " non-trivial ops\n");

  // Check if the arc is only ever used once.
  if (users.size() == 1) {
    ++numSingleUseArcs;
    return true;
  }

  return false;
}

void InlineArcsPass::inlineUse(DefineOp defOp, StateOp useOp, bool removeArc) {
  OpBuilder builder(useOp);
  if (removeArc) {
    // Simple implementation where we can just move the operations since the
    // arc will be removed after inlining anyway.
    for (auto &op :
         llvm::make_early_inc_range(defOp.getBodyBlock().getOperations())) {
      op.remove();
      // Recursively replace all block arguments used by the current operation.
      op.walk([&](Operation *op) {
        for (auto &operand : op->getOpOperands()) {
          if (auto arg = operand.get().dyn_cast<BlockArgument>();
              arg && arg.getOwner() == &defOp.getBodyBlock())
            operand.set(useOp.getOperands()[arg.getArgNumber()]);
        }
      });
      if (auto outputOp = dyn_cast<OutputOp>(&op)) {
        useOp.replaceAllUsesWith(outputOp.getOperands());
        op.erase();
        continue;
      }
      builder.insert(&op);
    }
  } else {
    // General implementation where we leave the original arc intact and build
    // up a clone of each operation.
    DenseMap<Value, Value> mapping; // mapping from outer to inner values

    for (auto [oldValue, newValue] :
         llvm::zip(defOp.getArguments(), useOp.getOperands()))
      mapping.insert({oldValue, newValue});

    BackedgeBuilder backedgeBuilder(builder, defOp.getLoc());
    SmallDenseMap<Value, Backedge, 8> backedges;

    for (auto &oldOp : defOp.getBodyBlock().getOperations()) {
      if (auto outputOp = dyn_cast<OutputOp>(&oldOp)) {
        for (auto [result, output] :
             llvm::zip(useOp.getResults(), outputOp.getOperands())) {
          auto mapped = mapping.lookup(output);
          assert(mapped);
          result.replaceAllUsesWith(mapped);
        }
        continue;
      }
      auto *newOp = oldOp.clone();
      builder.insert(newOp);
      for (auto &operand : newOp->getOpOperands()) {
        auto &mapped = mapping[operand.get()];
        if (!mapped) {
          auto backedge = backedgeBuilder.get(operand.get().getType());
          backedges.insert({operand.get(), backedge});
          mapped = backedge;
        }
        operand.set(mapped);
      }
      for (auto [newResult, oldResult] :
           llvm::zip(newOp->getResults(), oldOp.getResults())) {
        mapping[oldResult] = newResult;
        auto it = backedges.find(oldResult);
        if (it != backedges.end()) {
          it->second.setValue(newResult);
          backedges.erase(it);
        }
      }
    }
  }

  // Remove the use and optionally the arc.
  useOp->erase();
  if (removeArc)
    defOp->erase();
}

std::unique_ptr<Pass> arc::createInlineArcsPass() {
  return std::make_unique<InlineArcsPass>();
}
