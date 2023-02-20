//===- AddTaps.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace circt;
using namespace arc;
using namespace hw;

namespace {
struct AddTapsPass : public AddTapsBase<AddTapsPass> {
  void runOnOperation() override;

  using AddTapsBase::tapPorts;
  using AddTapsBase::tapWires;
};
} // namespace

void AddTapsPass::runOnOperation() {
  // Add taps for all module ports.
  if (tapPorts) {
    getOperation().walk([](HWModuleOp moduleOp) {
      auto *outputOp = moduleOp.getBodyBlock()->getTerminator();
      ModulePortInfo ports = moduleOp.getPorts();

      // Add taps to inputs.
      auto builder = OpBuilder::atBlockBegin(moduleOp.getBodyBlock());
      for (auto [port, arg] : llvm::zip(ports.inputs, moduleOp.getArguments()))
        builder.create<arc::TapOp>(arg.getLoc(), arg, port.getName());

      // Add taps to outputs.
      builder.setInsertionPoint(outputOp);
      for (auto [port, result] :
           llvm::zip(ports.outputs, outputOp->getOperands()))
        builder.create<arc::TapOp>(result.getLoc(), result, port.getName());
    });
  }

  // Add taps for wires.
  if (tapWires) {
    getOperation().walk([](sv::WireOp wireOp) {
      sv::ReadInOutOp readOp;
      for (auto *user : wireOp->getUsers())
        if (auto op = dyn_cast<sv::ReadInOutOp>(user))
          readOp = op;

      OpBuilder builder(wireOp);
      if (!readOp) {
        builder.setInsertionPointAfter(wireOp);
        readOp = builder.create<sv::ReadInOutOp>(wireOp.getLoc(), wireOp);
      }

      builder.setInsertionPointAfter(readOp);
      builder.create<arc::TapOp>(readOp.getLoc(), readOp, wireOp.getName());
    });
  }
}

std::unique_ptr<Pass> arc::createAddTapsPass() {
  return std::make_unique<AddTapsPass>();
}
