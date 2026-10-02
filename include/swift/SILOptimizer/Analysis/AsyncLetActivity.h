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

/// The numerical value materialized by an async-let result read. Copying an
/// address-only result initializes a destination rather than defining an SSA
/// object. Task handles and the runtime's raw pointers are never cotangents.
SILValue getAsyncLetReadValue(SILInstruction *read);

/// A statically associated structured child and its typed result accesses.
/// Cleanup sites can be mutually exclusive. The recognizer proves the lifetime
/// protocol on the CFG; it does not impose an order on physical block layout.
struct AsyncLetAccess {
  BuiltinInst *start = nullptr;
  PartialApplyInst *closure = nullptr;
  FunctionRefInst *initializerRef = nullptr;
  SILFunction *initializer = nullptr;
  AllocStackInst *buffer = nullptr;
  AddressToPointerInst *pointer = nullptr;
  SmallVector<SILInstruction *, 4> finishes;
  SmallVector<BuiltinInst *, 4> ends;
  SmallVector<DeallocStackInst *, 4> deallocations;
  SmallVector<SILValue, 4> captures;
  SmallVector<SILInstruction *, 4> reads;
  SmallVector<PointerToAddressInst *, 4> addresses;
  SmallVector<ApplyInst *, 4> gets;
  SmallVector<SILInstruction *, 8> closureConversions;
  SmallVector<SILInstruction *, 16> transport;
  std::string unsupportedReason;

  bool isSupported() const { return unsupportedReason.empty(); }
  SILType getValueType() const { return buffer->getType().getObjectType(); }
};

class AsyncLetActivity {
  SmallVector<std::unique_ptr<AsyncLetAccess>, 4> accesses;
  llvm::DenseMap<SILInstruction *, const AsyncLetAccess *> reads;
  llvm::SmallPtrSet<SILInstruction *, 32> transport;

public:
  explicit AsyncLetActivity(SILFunction &function);
  ArrayRef<std::unique_ptr<AsyncLetAccess>> getAccesses() const {
    return accesses;
  }
  bool empty() const { return accesses.empty(); }
  const AsyncLetAccess *getRead(SILInstruction *inst) const {
    return reads.lookup(inst);
  }
  bool isTransport(SILInstruction *inst) const { return transport.count(inst); }
};

} // namespace swift
#endif
