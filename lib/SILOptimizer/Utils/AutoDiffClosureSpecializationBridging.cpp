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
#include "swift/SILOptimizer/Differentiation/ADContext.h"
#include "swift/SILOptimizer/Differentiation/LinearMapInfo.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/STLExtras.h"

using namespace swift;

unsigned BridgedTypeHasher::operator()(const BridgedType &value) const {
  return llvm::DenseMapInfo<void *>::getHashValue(value.opaqueValue);
}

static SILType getBranchingTraceEnumLoweredType(EnumDecl *ed,
                                                SILFunction &vjp) {
  return autodiff::getBranchingTraceEnumLoweredTypeImpl(ed, &vjp,
                                                        vjp.getModule().Types);
}

static Type getCapturedArgTypesTupleForClosure(const SILInstruction *closure,
                                               ASTContext &ctx) {
  llvm::SmallVector<TupleTypeElt, 8> capturedArgTypes;

  if (const auto *pai = dyn_cast<PartialApplyInst>(closure)) {
    capturedArgTypes.reserve(pai->getArguments().size());
    for (const SILValue &arg : pai->getArguments())
      capturedArgTypes.emplace_back(arg->getType().getASTType(), Identifier{});
  } else {
    assert(isa<ThinToThickFunctionInst>(closure));
  }

  return TupleType::get(capturedArgTypes, ctx);
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

llvm::SmallVector<Type, 8> getBranchTracingEnumPreds(Type bteType) {
  llvm::SmallVector<Type, 8> btePreds;
  EnumDecl *ed = bteType->getEnumOrBoundGenericEnum();
  for (EnumCaseDecl *ecd : ed->getAllCases()) {
    assert(ecd->getElements().size() == 1);
    EnumElementDecl *oldEED = ecd->getElements().front();

    assert(oldEED->getParameterList()->size() == 1);
    ParamDecl &oldParamDecl = *oldEED->getParameterList()->front();

    auto *tt = cast<TupleType>(oldParamDecl.getInterfaceType().getPointer());

    if (tt->getNumElements() > 0 && !tt->getElement(0).getName().empty()) {
      assert(tt->getElement(0).getName().is("predecessor"));
      btePreds.emplace_back(tt->getElement(0).getType());
    }
  }
  return btePreds;
}

void iterateOverBranchTracingEnumPreds(
    llvm::DenseMap<Type, llvm::SmallVector<Type, 8>> &bteToPredsDict,
    const Type &currentBTEType) {
  assert(currentBTEType->isCanonical());
  llvm::SmallVector<Type, 8> currentBTEPreds =
      getBranchTracingEnumPreds(currentBTEType);
  bteToPredsDict[currentBTEType] = currentBTEPreds;
  for (const Type &currentBTEPred : currentBTEPreds) {
    if (!bteToPredsDict.contains(currentBTEPred)) {
      iterateOverBranchTracingEnumPreds(bteToPredsDict, currentBTEPred);
    }
  }
}

llvm::SmallVector<Type, 8> getBranchTracingEnumQueue(BridgedType topBTEType) {
  llvm::DenseMap<Type, llvm::SmallVector<Type, 8>> bteToPredsDict;
  iterateOverBranchTracingEnumPreds(bteToPredsDict,
                                    topBTEType.unbridged().getASTType());

  llvm::SmallVector<Type, 8> bteQueue;
  std::size_t totalEnums = bteToPredsDict.size();
  bteQueue.reserve(totalEnums);
  for (std::size_t i = 0; i < totalEnums; ++i) {
    for (const auto &[bteType, btePreds] : bteToPredsDict) {
      if (!btePreds.empty())
        continue;
      TypeBase *bteTypePointer = bteType.getPointer();
      assert(llvm::find_if(bteQueue, [bteTypePointer](const Type &type) {
               return bteTypePointer == type.getPointer();
             }) == bteQueue.end());
      bteQueue.emplace_back(bteType);
      break;
    }
    assert(bteQueue.size() == i + 1);
    bteToPredsDict.erase(bteQueue.back());
    for (auto &[bteType, btePreds] : bteToPredsDict) {
      llvm::erase_if(btePreds, [&bteQueue](const Type &type) {
        return bteQueue.back().getPointer() == type.getPointer();
      });
    }
  }
  assert(bteQueue.size() == totalEnums);

  return bteQueue;
}

bool operator==(const BridgedType &lhs, const BridgedType &rhs) {
  return lhs.opaqueValue == rhs.opaqueValue;
}

struct EnumTypeAndCaseIdx {
  BridgedType enumType;
  SwiftInt caseIdx;
};

static bool operator==(const EnumTypeAndCaseIdx &lhs,
                       const EnumTypeAndCaseIdx &rhs) {
  return lhs.enumType == rhs.enumType && lhs.caseIdx == rhs.caseIdx;
}

struct EnumTypeAndCaseIdxHasher {
  unsigned operator()(const EnumTypeAndCaseIdx &value) const {
    return llvm::DenseMapInfo<std::pair<void *, SwiftInt>>::getHashValue(
        std::pair<void *, SwiftInt>(value.enumType.opaqueValue, value.caseIdx));
  }
};

using BranchTracingEnumToClosuresDict =
    std::unordered_map<EnumTypeAndCaseIdx,
                       llvm::SmallVector<ClosureAndIdxInPayload, 8>,
                       EnumTypeAndCaseIdxHasher>;

// NOTE: Branch tracing enum creation logic was adopted from
// LinearMapInfo::createBranchingTraceDecl.
static BridgedType autodiffSpecializeBranchTracingEnum(
    BridgedType bteType, SILFunction &topVjp,
    const BranchTracingEnumToClosuresDict &bteToClosuresDict,
    const SpecializedBranchTracingEnumDict &specBTEDict) {
  EnumDecl *oldED = bteType.unbridged().getEnumOrBoundGenericEnum();
  assert(oldED && "Expected valid enum type");
  // TODO: switch to contains() after transition to C++20
  assert(specBTEDict.find(bteType) == specBTEDict.end());

  SILModule &silModule = topVjp.getModule();
  ASTContext &astContext = oldED->getASTContext();

  std::string newEDNameStr = oldED->getNameStr().str() + "_spec";

  llvm::SmallVector<ParameterList *, 8> newPLs;

  for (EnumCaseDecl *oldECD : oldED->getAllCases()) {
    assert(oldECD->getElements().size() == 1);
    EnumElementDecl *oldEED = oldECD->getElements().front();

    unsigned caseIdx = silModule.getCaseIndex(oldEED);

    static llvm::SmallVector<ClosureAndIdxInPayload, 8> emptyClosureAndIdxList;
    auto closureAndIdxListIt =
        bteToClosuresDict.find(EnumTypeAndCaseIdx{bteType, caseIdx});
    const llvm::SmallVector<ClosureAndIdxInPayload, 8> &closureAndIdxList =
        (closureAndIdxListIt == bteToClosuresDict.end()
             ? emptyClosureAndIdxList
             : closureAndIdxListIt->second);

    assert(oldEED->getParameterList()->size() == 1);
    ParamDecl &oldParamDecl = *oldEED->getParameterList()->front();

    const auto *oldPayloadTupleType =
        cast<TupleType>(oldParamDecl.getInterfaceType().getPointer());
    llvm::SmallVector<TupleTypeElt, 8> newPayloadTupleElementTypes;
    newPayloadTupleElementTypes.reserve(oldPayloadTupleType->getNumElements());

    std::string newECDNameSuffix;

    for (unsigned idxInPayloadTuple = 0;
         idxInPayloadTuple < oldPayloadTupleType->getNumElements();
         ++idxInPayloadTuple) {
      auto closureAndIdxInPayloadIt = llvm::find_if(
          closureAndIdxList,
          [idxInPayloadTuple](
              const ClosureAndIdxInPayload &closureAndIdxInPayload) {
            return closureAndIdxInPayload.idxInPayload == idxInPayloadTuple;
          });
      Type newPayloadTupleEltType;
      if (closureAndIdxInPayloadIt != closureAndIdxList.end()) {
        newECDNameSuffix += '_' + std::to_string(idxInPayloadTuple);
        newPayloadTupleEltType = getCapturedArgTypesTupleForClosure(
            closureAndIdxInPayloadIt->closure.unbridged(), astContext);
        // TODO: delete throwing function support
        if (oldPayloadTupleType->getElementType(idxInPayloadTuple)
                ->isOptional()) {
          assert(idxInPayloadTuple + 1 ==
                 oldPayloadTupleType->getNumElements());
          newPayloadTupleEltType =
              OptionalType::get(newPayloadTupleEltType)->getCanonicalType();
        }
      } else {
        newPayloadTupleEltType =
            oldPayloadTupleType->getElementType(idxInPayloadTuple);
        // TODO
        for (const auto &[enumTypeOld, enumTypeNew] : specBTEDict) {
          if (enumTypeOld.unbridged()
                  .getASTType()
                  ->mapTypeOutOfContext()
                  .getPointer() == newPayloadTupleEltType.getPointer()) {
            assert(idxInPayloadTuple == 0);
            newPayloadTupleEltType = enumTypeNew.unbridged().getASTType();
          }
        }
      }
      Identifier label =
          oldPayloadTupleType->getElement(idxInPayloadTuple).getName();
      newPayloadTupleElementTypes.emplace_back(newPayloadTupleEltType, label);
    }

    Type newTupleType = TupleType::get(newPayloadTupleElementTypes, astContext)
                            ->mapTypeOutOfContext();

    auto *newParamDecl = ParamDecl::cloneWithoutType(astContext, &oldParamDecl);
    newParamDecl->setInterfaceType(newTupleType);

    newPLs.emplace_back(ParameterList::create(astContext, {newParamDecl}));

    if (!newECDNameSuffix.empty())
      newEDNameStr += '_' + oldEED->getNameStr().str() + newECDNameSuffix;
  }

  CanGenericSignature genericSig = nullptr;
  if (auto *derivativeFnGenEnv = topVjp.getGenericEnvironment())
    genericSig =
        derivativeFnGenEnv->getGenericSignature().getCanonicalSignature();
  GenericParamList *genericParams = nullptr;
  if (genericSig)
    genericParams = autodiff::cloneGenericParameters(
        astContext, oldED->getDeclContext(), genericSig);

  Identifier newEDName = astContext.getIdentifier(newEDNameStr);

  auto *newED = new (astContext) EnumDecl(
      /*EnumLoc*/ SourceLoc(), /*Name*/ newEDName, /*NameLoc*/ SourceLoc(),
      /*Inherited*/ {}, /*GenericParams*/ genericParams,
      /*DC*/
      oldED->getDeclContext());
  newED->setImplicit();
  if (genericSig)
    newED->setGenericSignature(genericSig);

  for (auto [idx, oldECD] : llvm::enumerate(oldED->getAllCases())) {
    EnumElementDecl *oldEED = oldECD->getElements().front();
    auto *newPL = newPLs[idx];
    auto *newEED = new (astContext) EnumElementDecl(
        /*IdentifierLoc*/ SourceLoc(),
        DeclName(astContext.getIdentifier(oldEED->getNameStr())), newPL,
        SourceLoc(), /*RawValueExpr*/ nullptr, newED);
    newEED->setImplicit();
    auto *newECD = EnumCaseDecl::create(
        /*CaseLoc*/ SourceLoc(), {newEED}, newED);
    newECD->setImplicit();
    newED->addMember(newEED);
    newED->addMember(newECD);
  }

  // MYTODO: is this correct?
  newED->setAccess(AccessLevel::Public);
  auto &file = autodiff::getSourceFile(&topVjp).getOrCreateSynthesizedFile();
  file.addTopLevelDecl(newED);
  file.getParentModule()->clearLookupCache();

  SILType newEnumType =
      remapType(getBranchingTraceEnumLoweredType(newED, topVjp), topVjp);

  return newEnumType;
}

SpecializedBranchTracingEnumDict autodiffSpecializeBranchTracingEnums(
    BridgedFunction bridgedTopVjp, BridgedType topEnum,
    const VectorOfBranchTracingEnumAndClosureInfo
        &vectorOfBranchTracingEnumAndClosureInfo) {
  SILFunction &topVjp = *bridgedTopVjp.getFunction();

  BranchTracingEnumToClosuresDict closuresBuffers;

  for (const BranchTracingEnumAndClosureInfo &elem :
       vectorOfBranchTracingEnumAndClosureInfo) {
    closuresBuffers[EnumTypeAndCaseIdx{elem.enumType, elem.enumCaseIdx}]
        .emplace_back(elem.closure, elem.idxInPayload);
  }

  llvm::SmallVector<Type, 8> enumQueue = getBranchTracingEnumQueue(topEnum);
  SpecializedBranchTracingEnumDict dict;

  for (const Type &t : enumQueue) {
    EnumDecl *ed = t->getEnumOrBoundGenericEnum();

    SILType silType =
        remapType(getBranchingTraceEnumLoweredType(ed, topVjp), topVjp);

    dict[BridgedType(silType)] = autodiffSpecializeBranchTracingEnum(
        BridgedType(silType), topVjp, closuresBuffers, dict);
  }

  return dict;
}

BridgedArgument specializeBranchTracingEnumBBArgInVJP(
    BridgedArgument arg, const SpecializedBranchTracingEnumDict &dict) {
  swift::ValueOwnershipKind oldOwnership =
      arg.getArgument()->getOwnershipKind();

  swift::SILArgument *oldArg = arg.getArgument();
  swift::SILBasicBlock *bb = oldArg->getParentBlock();
  assert(!bb->isEntry());
  unsigned index = oldArg->getIndex();
  // TODO: switch to contains() after transition to C++20
  assert(dict.find(oldArg->getType()) != dict.end());
  SILType type = dict.at(BridgedType(oldArg->getType())).unbridged();
  swift::SILPhiArgument *newArg =
      bb->insertPhiArgument(index, type, oldOwnership);
  oldArg->replaceAllUsesWith(newArg);
  bb->eraseArgument(index + 1);
  return {newArg};
}

BridgedArgument specializeOptionalBBArgInPullback(BridgedBasicBlock bbBridged,
                                                  BridgedType optionalType) {
  swift::SILBasicBlock *bb = bbBridged.unbridged();
  assert(!bb->isEntry());
  SILArgument *oldArg = bb->getArgument(0);

  SILModule &module = bb->getFunction()->getModule();

  SILType silType = optionalType.unbridged();
  assert(silType.getASTType()->isOptional());

  swift::ValueOwnershipKind oldOwnership =
      bb->getArgument(0)->getOwnershipKind();

  CanType type =
      silType.getASTType()->getOptionalObjectType()->getCanonicalType();
  Lowering::AbstractionPattern pattern(
      bb->getFunction()->getLoweredFunctionType()->getSubstGenericSignature(),
      type);
  SILType loweredType = module.Types.getLoweredType(
      pattern, type, TypeExpansionContext::minimal());

  swift::SILPhiArgument *newArg =
      bb->insertPhiArgument(0, loweredType, oldOwnership);

  oldArg->replaceAllUsesWith(newArg);
  bb->eraseArgument(1);

  return {newArg};
}

BridgedArgument specializePayloadTupleBBArgInPullback(
    BridgedArgument arg,
    const SpecializedBranchTracingEnumDict &branchTracingEnumDict,
    const VectorOfClosureAndIdxInPayload &closuresBuffersForPb) {
  swift::SILArgument *oldArg = arg.getArgument();
  unsigned argIdx = oldArg->getIndex();
  swift::SILBasicBlock *bb = oldArg->getParentBlock();
  assert(!bb->isEntry());
  ASTContext &astContext = bb->getModule().getASTContext();
  auto *oldTupleTy =
      llvm::cast<swift::TupleType>(oldArg->getType().getASTType().getPointer());
  llvm::SmallVector<swift::TupleTypeElt, 8> newTupleElTypes;
  for (unsigned idxInPayload = 0; idxInPayload < oldTupleTy->getNumElements();
       ++idxInPayload) {
    auto it = llvm::find_if(
        closuresBuffersForPb,
        [idxInPayload](const ClosureAndIdxInPayload &closureAndIdxInPayload) {
          return closureAndIdxInPayload.idxInPayload == idxInPayload;
        });

    if (it == closuresBuffersForPb.end()) {
      Type type = oldTupleTy->getElementType(idxInPayload);
      for (const auto &[enumTypeOld, enumTypeNew] : branchTracingEnumDict) {
        if (enumTypeOld.unbridged().getASTType().getPointer() ==
            type.getPointer()) {
          assert(idxInPayload == 0);
          type = enumTypeNew.unbridged().getASTType();
        }
      }
      newTupleElTypes.emplace_back(
          type, oldTupleTy->getElement(idxInPayload).getName());
      continue;
    }

    CanType canType =
        getCapturedArgTypesTupleForClosure(it->closure.unbridged(), astContext)
            ->getCanonicalType();

    // TODO: delete throwing function support
    if (oldTupleTy->getElementType(idxInPayload)->isOptional()) {
      assert(idxInPayload + 1 == oldTupleTy->getNumElements());
      canType = OptionalType::get(canType)->getCanonicalType();
    }
    newTupleElTypes.emplace_back(canType);
  }
  auto newTupleTy = swift::SILType::getFromOpaqueValue(
      swift::TupleType::get(newTupleElTypes, astContext));

  swift::ValueOwnershipKind oldOwnership =
      bb->getArgument(argIdx)->getOwnershipKind();

  swift::SILPhiArgument *newArg =
      bb->insertPhiArgument(argIdx, newTupleTy, oldOwnership);
  oldArg->replaceAllUsesWith(newArg);
  bb->eraseArgument(argIdx + 1);

  return {newArg};
}
