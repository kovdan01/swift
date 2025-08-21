//===--- AutoDiffClosureSpecializationBridging.h --------------------------===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2025 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#ifndef SWIFT_SILOPTIMIZER_ADCSBRIDGING_H
#define SWIFT_SILOPTIMIZER_ADCSBRIDGING_H

#include "swift/SIL/SILBridging.h"

SWIFT_BEGIN_NULLABILITY_ANNOTATIONS

struct BridgedTypeHasher {
  unsigned operator()(const BridgedType &value) const {
    return llvm::DenseMapInfo<void *>::getHashValue(value.opaqueValue);
  }
};

using BranchTracingEnumDict =
    std::unordered_map<BridgedType, BridgedType, BridgedTypeHasher>;

struct ClosureAndIdxInPayload {
  ClosureAndIdxInPayload(BridgedInstruction closure, SwiftInt idxInPayload)
      : closure(closure), idxInPayload(idxInPayload) {}
  BridgedInstruction closure;
  SwiftInt idxInPayload;
};

using VectorOfClosureAndIdxInPayload = std::vector<ClosureAndIdxInPayload>;

struct BridgedClosureInfoCFG {
  BridgedType enumType;
  SwiftInt enumCaseIdx;
  BridgedInstruction closure;
  SwiftInt idxInPayload;
};

using VectorOfBridgedClosureInfoCFG = std::vector<BridgedClosureInfoCFG>;

SWIFT_IMPORT_UNSAFE BranchTracingEnumDict autodiffSpecializeBranchTracingEnums(
    BridgedFunction topVjp, BridgedType topEnum,
    const VectorOfBridgedClosureInfoCFG &vectorOfClosureInfoCFG);

SWIFT_IMPORT_UNSAFE BridgedArgument recreateEnumBlockArgument(
    BridgedArgument arg, const BranchTracingEnumDict &dict);
SWIFT_IMPORT_UNSAFE BridgedArgument recreateTupleBlockArgument(
    BridgedArgument arg, const BranchTracingEnumDict &dict,
    const VectorOfClosureAndIdxInPayload &closuresBuffersForPb);
SWIFT_IMPORT_UNSAFE BridgedArgument recreateOptionalBlockArgument(
    BridgedBasicBlock bbBridged, BridgedType optionalType);

SWIFT_END_NULLABILITY_ANNOTATIONS

#endif
