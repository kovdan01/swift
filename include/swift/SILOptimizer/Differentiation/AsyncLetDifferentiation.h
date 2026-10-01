//===--- AsyncLetDifferentiation.h - Async-let VJP transport ----*- C++ -*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2026 the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//
#ifndef SWIFT_SILOPTIMIZER_DIFFERENTIATION_ASYNCLETDIFFERENTIATION_H
#define SWIFT_SILOPTIMIZER_DIFFERENTIATION_ASYNCLETDIFFERENTIATION_H

#include "swift/SILOptimizer/Analysis/AsyncLetActivity.h"
#include "swift/SILOptimizer/Differentiation/DifferentiationInvoker.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace swift {
class DifferentiableActivityInfo;
class SILBuilder;
namespace autodiff {
class ADContext;
class LinearMapInfo;

FunctionType *getAsyncLetPullbackASTType(const AsyncLetAccess &access);
CanSILFunctionType getAsyncLetDirectPullbackType(const AsyncLetAccess &access);

/// Adapt an owned synchronous scalar PB to a different seed/result ABI.
/// Unlike the general AD thunk helper, this supports flattening/packing result
/// tuples when the number of formal results changes. Only the validated
/// nongeneric Float/Double async-let domain is supported. The returned closure
/// owns the original PB and borrows it on every invocation.
SILValue reabstractAsyncLetPullback(ADContext &context, SILBuilder &builder,
                                    SILLocation loc, SILValue function,
                                    CanSILFunctionType target);

/// Returns true after issuing a diagnostic for an unsupported async-let region.
bool diagnoseUnsupportedAsyncLets(ADContext &context,
                                  const DifferentiableActivityInfo &activity,
                                  DifferentiationInvoker invoker);

class AsyncLetVJPEmitter {
  struct State {
    const AsyncLetAccess *access = nullptr;
    Type payloadAST;
    SILType payloadType;
    SILFunction *entry = nullptr;
    SILValue buffer, valueAddress, pullbackAddress;
    SILValue escapingClosure, noescapeClosure;
  };
  ADContext &context;
  SILFunction &vjp;
  const DifferentiableActivityInfo &activity;
  const AutoDiffConfig &config;
  LinearMapInfo &linearMaps;
  SmallVector<State, 4> states;

  SILFunction *createTaskEntry(const AsyncLetAccess &access,
                               SILType payloadType, unsigned ordinal);

public:
  AsyncLetVJPEmitter(ADContext &context, SILFunction &vjp,
                     const DifferentiableActivityInfo &activity,
                     const AutoDiffConfig &config, LinearMapInfo &linearMaps);

  /// Clone an intercepted instruction and record its normal SSA mapping.
  /// Returns false for an instruction that should use ordinary VJP cloning.
  /// savePullback appends a checkpoint at the current original instruction.
  bool clone(SILInstruction *inst, SILBuilder &builder,
             llvm::function_ref<SILValue(SILValue)> remap,
             llvm::function_ref<void(SILValue, SILValue)> map,
             llvm::function_ref<void(SILValue)> savePullback);
};

} // namespace autodiff
} // namespace swift
#endif
