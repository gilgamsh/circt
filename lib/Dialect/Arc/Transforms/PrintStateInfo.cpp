//===- PrintStateInfo.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Conversion/LLHDToLLVM.h"
#include "circt/Support/Namespace.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/ControlFlowSinkUtils.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Transforms/TopologicalSortUtils.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/ToolOutputFile.h"

#define DEBUG_TYPE "arc-print-state-info"

using namespace mlir;
using namespace circt;
using namespace arc;
using namespace hw;

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

namespace {
struct StateInfo {
  enum Type { Input, Output, Register, Memory, Wire } type;
  StringAttr name;
  unsigned offset;
  unsigned numBits;
  unsigned memoryStride = 0; // byte separation between memory words
  unsigned memoryDepth = 0;  // number of words in a memory
};

struct ModelInfo {
  size_t numStateBytes;
  std::vector<StateInfo> states;
};

struct PrintStateInfoPass : public PrintStateInfoBase<PrintStateInfoPass> {
  void runOnOperation() override;
  void runOnOperation(llvm::raw_ostream &outputStream);
  void collectStates(Value storage, unsigned offset,
                     std::vector<StateInfo> &stateInfos);

  ModelOp modelOp;
  DenseMap<ModelOp, ModelInfo> modelInfos;

  using PrintStateInfoBase::stateFile;
};
} // namespace

void PrintStateInfoPass::runOnOperation() {
  // Print to the output file if one was given, or stdout otherwise.
  if (!stateFile.empty()) {
    std::error_code ec;
    llvm::ToolOutputFile outputFile(stateFile, ec,
                                    llvm::sys::fs::OpenFlags::OF_None);
    if (ec) {
      mlir::emitError(getOperation().getLoc(), "unable to open state file: ")
          << ec.message();
      return signalPassFailure();
    }
    runOnOperation(outputFile.os());
    outputFile.keep();
  } else {
    runOnOperation(llvm::outs());
    llvm::outs() << "\n";
  }
}

void PrintStateInfoPass::runOnOperation(llvm::raw_ostream &outputStream) {
  llvm::json::OStream json(outputStream, 2);
  json.array([&] {
    std::vector<StateInfo> states;
    for (auto modelOp : getOperation().getOps<ModelOp>()) {
      auto storageArg = modelOp.getBody().getArgument(0);
      auto storageType = storageArg.getType().cast<StorageType>();
      states.clear();
      collectStates(storageArg, 0, states);
      llvm::sort(states, [](auto &a, auto &b) { return a.offset < b.offset; });

      json.object([&] {
        json.attribute("name", modelOp.getName());
        json.attribute("numStateBytes", storageType.getSize());
        json.attributeArray("states", [&] {
          for (const auto &state : states) {
            json.object([&] {
              if (state.name && !state.name.getValue().empty())
                json.attribute("name", state.name.getValue());
              json.attribute("offset", state.offset);
              json.attribute("numBits", state.numBits);
              auto typeStr = [](StateInfo::Type type) {
                switch (type) {
                case StateInfo::Input:
                  return "input";
                case StateInfo::Output:
                  return "output";
                case StateInfo::Register:
                  return "register";
                case StateInfo::Memory:
                  return "memory";
                case StateInfo::Wire:
                  return "wire";
                }
                return "";
              };
              json.attribute("type", typeStr(state.type));
              if (state.type == StateInfo::Memory) {
                json.attribute("stride", state.memoryStride);
                json.attribute("depth", state.memoryDepth);
              }
            });
          }
        });
      });
    }
  });
}

void PrintStateInfoPass::collectStates(Value storage, unsigned offset,
                                       std::vector<StateInfo> &stateInfos) {
  for (auto *op : storage.getUsers()) {
    if (auto substorage = dyn_cast<AllocStorageOp>(op)) {
      assert(substorage.getOffset().has_value());
      collectStates(substorage.getOutput(), *substorage.getOffset() + offset,
                    stateInfos);
      continue;
    }
    if (!op->hasAttr("name"))
      continue;
    if (isa<AllocStateOp, RootInputOp, RootOutputOp>(op)) {
      auto result = op->getResult(0);
      auto &stateInfo = stateInfos.emplace_back();
      stateInfo.type = StateInfo::Register;
      if (isa<RootInputOp>(op))
        stateInfo.type = StateInfo::Input;
      else if (isa<RootOutputOp>(op))
        stateInfo.type = StateInfo::Output;
      else if (auto alloc = dyn_cast<AllocStateOp>(op)) {
        if (alloc.getTap())
          stateInfo.type = StateInfo::Wire;
      }
      stateInfo.name = op->getAttrOfType<StringAttr>("name");
      stateInfo.offset =
          op->getAttrOfType<IntegerAttr>("offset").getValue().getZExtValue() +
          offset;
      stateInfo.numBits =
          result.getType().cast<StateType>().getType().getWidth();
      continue;
    }
    if (auto memOp = dyn_cast<AllocMemoryOp>(op)) {
      auto memType = memOp.getType();
      auto intType = memType.getWordType();
      auto &stateInfo = stateInfos.emplace_back();
      stateInfo.type = StateInfo::Memory;
      stateInfo.name = op->getAttrOfType<StringAttr>("name");
      stateInfo.offset =
          op->getAttrOfType<IntegerAttr>("offset").getValue().getZExtValue() +
          offset;
      stateInfo.numBits = intType.getWidth();
      stateInfo.memoryStride =
          op->getAttrOfType<IntegerAttr>("stride").getValue().getZExtValue();
      stateInfo.memoryDepth = memType.getNumWords();
      continue;
    }
  }
}

std::unique_ptr<Pass> arc::createPrintStateInfoPass(StringRef stateFile) {
  auto pass = std::make_unique<PrintStateInfoPass>();
  if (!stateFile.empty())
    pass->stateFile.assign(stateFile);
  return pass;
}
