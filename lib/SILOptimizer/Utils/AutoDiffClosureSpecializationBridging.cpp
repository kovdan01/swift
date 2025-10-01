//===--- AutoDiffClosureSpecializationBridging.cpp ------------------------===//
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

#define DEBUG_TYPE "autodiff-closure-specialization-bridging"

#include "swift/SILOptimizer/AutoDiffClosureSpecializationBridging.h"
#include "swift/AST/ParameterList.h"
#include "swift/SIL/SILBasicBlock.h"
#include "swift/SIL/SILInstruction.h"
#include "swift/SILOptimizer/Differentiation/ADContext.h"
#include "swift/SILOptimizer/Differentiation/LinearMapInfo.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace swift;

ClosureAndIdxInPayload::ClosureAndIdxInPayload(BridgedInstruction closure,
                                               SwiftInt idxInPayload)
    : closure(closure), idxInPayload(idxInPayload) {}

static SILType getBranchingTraceEnumLoweredTypeImpl(EnumDecl *ed,
                                                    SILFunction &vjp) {
  return autodiff::getLoweredTypeImpl(
      ed->getDeclaredInterfaceType()->getCanonicalType(), &vjp,
      vjp.getModule().Types);
}

BridgedType getBranchingTraceEnumLoweredType2(BridgedEnumDecl ed,
                                                    BridgedFunction vjp) {
  return getBranchingTraceEnumLoweredTypeImpl(ed.unbridged(), *vjp.getFunction());
}

BridgedType getBranchingTraceEnumLoweredType(BridgedDeclObj ed,
                                             BridgedFunction vjp) {
  return getBranchingTraceEnumLoweredTypeImpl(ed.getAs<EnumDecl>(), *vjp.getFunction());
}

BridgedNullableGenericParamList cloneGenericParameters(BridgedASTContext ctx, BridgedDeclContext dc,
                                                                           BridgedCanGenericSignature sig) {
  return autodiff::cloneGenericParameters(ctx.unbridged(), dc.unbridged(), sig.unbridged());
}

// NOTE: this is adopted from
// lib/SILOptimizer/Differentiation/PullbackCloner.cpp.
/// Remap any archetypes into the current function's context.
static SILType remapType(SILType ty, SILFunction &f) {
  if (ty.hasArchetype())
    ty = ty.mapTypeOutOfContext();
  auto remappedType = ty.getASTType()->getReducedType(
      f.getLoweredFunctionType()->getSubstGenericSignature());
  auto remappedSILType =
      SILType::getPrimitiveType(remappedType, ty.getCategory());
  if (f.getGenericEnvironment())
    return f.mapTypeIntoContext(remappedSILType);
  return remappedSILType;
}

BridgedSourceFile autodiffGetSourceFile(BridgedFunction f) {
  return {&autodiff::getSourceFile(f.getFunction())};
}

BridgedArgument specializePayloadTupleBBArgInPullback(BridgedArgument arg,
                                                      BridgedType enumType,
                                                      SwiftInt caseIdx) {
  SILArgument *oldArg = arg.getArgument();
  unsigned argIdx = oldArg->getIndex();
  SILBasicBlock *bb = oldArg->getParentBlock();
  assert(!bb->isEntry());
  SILModule &silModule = bb->getModule();

  SILType newEnumType = enumType.unbridged();
  EnumDecl *newED = newEnumType.getEnumOrBoundGenericEnum();
  assert(newED != nullptr);

  CanType newPayloadTupleTy;
  for (EnumElementDecl *newEED : newED->getAllElements()) {
    unsigned currentCaseIdx = silModule.getCaseIndex(newEED);
    if (currentCaseIdx != caseIdx)
      continue;

    newPayloadTupleTy = newEED->getPayloadInterfaceType()->getCanonicalType();
  }

  assert(newPayloadTupleTy != CanType());

  SILType newPayloadTupleSILTy =
      remapType(autodiff::getLoweredTypeImpl(
                    newPayloadTupleTy, bb->getFunction(), silModule.Types),
                *bb->getFunction());

  ValueOwnershipKind oldOwnership = bb->getArgument(argIdx)->getOwnershipKind();

  SILPhiArgument *newArg =
      bb->insertPhiArgument(argIdx, newPayloadTupleSILTy, oldOwnership);

  return {newArg};
}
