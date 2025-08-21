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

struct BridgedClosureInfoCFG {
  BridgedType enumType;
  SwiftInt enumCaseIdx;
  BridgedInstruction closure;
  SwiftInt idxInPayload;
};

using VectorOfBridgedClosureInfoCFG = std::vector<BridgedClosureInfoCFG>;

struct BridgedAutoDiffClosureSpecializationHelper {
  SWIFT_IMPORT_UNSAFE BranchTracingEnumDict rewriteAllEnums(
      BridgedFunction topVjp, BridgedType topEnum,
      const VectorOfBridgedClosureInfoCFG &vectorOfClosureInfoCFG) const;
};

SWIFT_END_NULLABILITY_ANNOTATIONS

#endif
