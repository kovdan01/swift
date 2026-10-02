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

/// Indices are in the lifted initializer's explicit capture parameter list.
/// Nondifferentiable and inactive captures remain primal inputs, not gradients.
IndexSubset *
getAsyncLetParameterIndices(const AsyncLetAccess &access,
                            const DifferentiableActivityInfo &activity,
                            const AutoDiffConfig &config);
FunctionType *
getAsyncLetPullbackASTType(const AsyncLetAccess &access,
                           const DifferentiableActivityInfo &activity,
                           const AutoDiffConfig &config);

/// A private invocation convention with an indirect tangent seed and one
/// indirect gradient result. The result may be a tuple of capture tangents.
/// This is not a claim that tangent values are loadable or trivial.
CanSILFunctionType
getAsyncLetPullbackBufferType(const AsyncLetAccess &access,
                              const DifferentiableActivityInfo &activity,
                              const AutoDiffConfig &config);

/// Use the shared AD thunk builder, including its result-regrouping support.
SILValue reabstractAsyncLetPullback(ADContext &context, SILBuilder &builder,
                                    SILLocation loc, SILValue function,
                                    CanSILFunctionType target);

bool diagnoseUnsupportedAsyncLets(ADContext &context,
                                  const DifferentiableActivityInfo &activity,
                                  const AutoDiffConfig &config,
                                  DifferentiationInvoker invoker);

/// Run after the normal derivative worklists have resolved callees. Unknown
/// residual provenance is diagnosed, not treated as a successful proof.
bool diagnoseAsyncLetResidualTransfers(ADContext &context);

class AsyncLetVJPEmitter {
  struct State {
    const AsyncLetAccess *access = nullptr;
    Type payloadAST;
    SILType payloadType;
    SILFunction *entry = nullptr;
    SILValue buffer, valueAddress, pullbackAddress;
  };
  ADContext &context;
  SILFunction &vjp;
  const DifferentiableActivityInfo &activity;
  const AutoDiffConfig &config;
  LinearMapInfo &linearMaps;
  DifferentiationInvoker invoker;
  SmallVector<State, 4> states;

  SILFunction *createTaskEntry(const AsyncLetAccess &access,
                               SILType payloadType, unsigned ordinal);

public:
  AsyncLetVJPEmitter(ADContext &context, SILFunction &vjp,
                     const DifferentiableActivityInfo &activity,
                     const AutoDiffConfig &config, LinearMapInfo &linearMaps,
                     DifferentiationInvoker invoker);

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
