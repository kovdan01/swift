//===--- AsyncLetActivity.h - Logical async-let dataflow --------*- C++ -*-===//
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
#ifndef SWIFT_SILOPTIMIZER_ANALYSIS_ASYNCLETACTIVITY_H
#define SWIFT_SILOPTIMIZER_ANALYSIS_ASYNCLETACTIVITY_H

#include "swift/SIL/SILInstruction.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <memory>
#include <string>

namespace swift {

bool isExperimentalAsyncAutoDiffEnabled();

/// A semantic edge from the captured environment to each read of a child
/// result. Task handles and raw pointers are transport, NOT differentiable
/// values.
///
/// This first implementation deliberately accepts only the single-block,
/// local-buffer lowering. An unrecognized start is retained with a diagnostic
/// reason, rather than disappearing from activity analysis as an inactive
/// value.
struct AsyncLetAccess {
  BuiltinInst *start = nullptr;
  PartialApplyInst *closure = nullptr;
  SILFunction *initializer = nullptr;
  AllocStackInst *buffer = nullptr;
  AddressToPointerInst *pointer = nullptr;
  SILInstruction *finish = nullptr;
  BuiltinInst *end = nullptr;
  DeallocStackInst *deallocation = nullptr;
  SmallVector<SILValue, 4> captures;
  SmallVector<LoadInst *, 4> reads;
  SmallVector<PointerToAddressInst *, 4> addresses;
  SmallVector<ApplyInst *, 4> gets;
  SmallVector<SILInstruction *, 16> transport;
  std::string unsupportedReason;

  bool isSupported() const { return unsupportedReason.empty(); }
  SILType getValueType() const { return buffer->getType().getObjectType(); }
};

class AsyncLetActivity {
  SmallVector<std::unique_ptr<AsyncLetAccess>, 4> accesses;
  llvm::DenseMap<LoadInst *, const AsyncLetAccess *> reads;
  llvm::SmallPtrSet<SILInstruction *, 32> transport;

public:
  explicit AsyncLetActivity(SILFunction &function);
  ArrayRef<std::unique_ptr<AsyncLetAccess>> getAccesses() const {
    return accesses;
  }
  bool empty() const { return accesses.empty(); }
  const AsyncLetAccess *getRead(SILInstruction *inst) const {
    auto *load = dyn_cast_or_null<LoadInst>(inst);
    return load ? reads.lookup(load) : nullptr;
  }
  bool isTransport(SILInstruction *inst) const { return transport.count(inst); }
};

/// Deliberately narrow initial transport domain. It does not assert that every
/// Differentiable/Sendable primal has a Sendable tape.
bool isAsyncAutoDiffScalar(SILType type);

} // namespace swift
#endif
