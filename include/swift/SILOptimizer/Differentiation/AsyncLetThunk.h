//===--- AsyncLetThunk.h - Async-let autodiff thunk emission ----*- C++ -*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2026 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#ifndef SWIFT_SILOPTIMIZER_DIFFERENTIATION_ASYNCLETTHUNK_H
#define SWIFT_SILOPTIMIZER_DIFFERENTIATION_ASYNCLETTHUNK_H

#include "swift/AST/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include <utility>

namespace swift {

class SILFunction;
class SILModule;

namespace autodiff {

/// Inputs for emitting a previously declared task-entry adapter. These types
/// are already in the thunk's generic environment.
struct AsyncLetThunkBody {
  CanSILFunctionType expectedType;
  CanType formalType;
  CanType dynamicSelfType;
};

/// Emit a completed AD pass's task-entry adapters using ordinary SILGen.
/// All derivative bodies must be complete before this runs: the existing
/// SILGenModule destructor finalizes and verifies the module.
void emitAsyncLetThunkBodies(
    SILModule &module,
    llvm::ArrayRef<std::pair<SILFunction *, AsyncLetThunkBody>> thunks);

} // namespace autodiff
} // namespace swift

#endif // SWIFT_SILOPTIMIZER_DIFFERENTIATION_ASYNCLETTHUNK_H
