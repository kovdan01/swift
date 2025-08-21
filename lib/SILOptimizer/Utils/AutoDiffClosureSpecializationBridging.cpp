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

#include "llvm/ADT/STLExtras.h"

using namespace swift;

static SILType getBranchingTraceEnumLoweredType(EnumDecl *ed,
                                                SILFunction *vjp) {
  return autodiff::getBranchingTraceEnumLoweredTypeImpl(ed, vjp,
                                                        vjp->getModule().Types);
}

static Type getCapturedArgTypesForClosure(const SILInstruction *closure,
                                          ASTContext &ctx) {
  SmallVector<TupleTypeElt, 4> paramTuple;

  if (const auto *pai = dyn_cast<PartialApplyInst>(closure)) {
    paramTuple.reserve(pai->getArguments().size());
    for (const SILValue &arg : pai->getArguments())
      paramTuple.emplace_back(arg->getType().getASTType(), Identifier{});
  } else {
    assert(isa<ThinToThickFunctionInst>(closure));
  }

  return TupleType::get(paramTuple, ctx);
}

// TODO: can we use LinearMapInfo::remapTypeInDerivative?

/// Remap any archetypes into the current function's context.
static SILType remapType(SILType ty, SILFunction *foo) {
  if (ty.hasArchetype())
    ty = ty.mapTypeOutOfContext();
  auto remappedType = ty.getASTType()->getReducedType(
      foo->getLoweredFunctionType()->getSubstGenericSignature());
  auto remappedSILType =
      SILType::getPrimitiveType(remappedType, ty.getCategory());
  // FIXME: Sometimes getPullback() doesn't have a generic environment, in which
  // case callers are apparently happy to receive an interface type.
  if (foo->getGenericEnvironment())
    return foo->mapTypeIntoContext(remappedSILType);
  return remappedSILType;
}

std::vector<Type> getPredTypes(Type enumType) {
  std::vector<Type> ret;
  EnumDecl *ed = enumType->getEnumOrBoundGenericEnum();
  for (EnumCaseDecl *ecd : ed->getAllCases()) {
    assert(ecd->getElements().size() == 1);
    EnumElementDecl *oldEED = ecd->getElements().front();

    assert(oldEED->getParameterList()->size() == 1);
    ParamDecl &oldParamDecl = *oldEED->getParameterList()->front();

    auto *tt = cast<TupleType>(oldParamDecl.getInterfaceType().getPointer());

    if (tt->getNumElements() > 0 && !tt->getElement(0).getName().empty()) {
      assert(tt->getElement(0).getName().is("predecessor"));
      ret.emplace_back(tt->getElement(0).getType());
    }
  }
  return ret;
}

void helper(llvm::DenseMap<Type, std::vector<Type>> &predTypes,
            const Type &currentEnumType) {
  assert(currentEnumType->isCanonical());
  std::vector<Type> currentPredTypes = getPredTypes(currentEnumType);
  predTypes[currentEnumType] = currentPredTypes;
  for (const Type &t : currentPredTypes) {
    if (!predTypes.contains(t)) {
      helper(predTypes, t);
    }
  }
}

std::vector<Type> getEnumQueue(BridgedType topEnum) {
  llvm::DenseMap<Type, std::vector<Type>> predTypes;
  helper(predTypes, topEnum.unbridged().getASTType());

  std::vector<Type> enumQueue;
  std::size_t totalEnums = predTypes.size();
  for (std::size_t i = 0; i < totalEnums; ++i) {
    for (const auto &[enumType, currentPreds] : predTypes) {
      if (!currentPreds.empty())
        continue;
      TypeBase *enumTypePointer = enumType.getPointer();
      assert(std::find_if(enumQueue.begin(), enumQueue.end(),
                          [enumTypePointer](const Type &val) {
                            return enumTypePointer == val.getPointer();
                          }) == enumQueue.end());
      enumQueue.emplace_back(enumType);
      break;
    }
    assert(enumQueue.size() == i + 1);
    predTypes.erase(enumQueue.back());
    for (auto &[enumType, _] : predTypes) {
      std::vector<Type> &currentPredTypes = predTypes.find(enumType)->second;
      auto it = std::find_if(currentPredTypes.begin(), currentPredTypes.end(),
                             [&enumQueue](const Type &val) {
                               return enumQueue.back().getPointer() ==
                                      val.getPointer();
                             });
      if (it != currentPredTypes.end())
        currentPredTypes.erase(it);
    }
  }

  return enumQueue;
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
    BridgedType enumType, BridgedFunction topVjp,
    const BranchTracingEnumToClosuresDict &branchTracingEnumToClosuresDict,
    const BranchTracingEnumDict &branchTracingEnumDict) {
  EnumDecl *oldED = enumType.unbridged().getEnumOrBoundGenericEnum();
  assert(oldED && "Expected valid enum type");
  // TODO: switch to contains() after transition to C++20
  assert(branchTracingEnumDict.find(enumType.unbridged()) ==
         branchTracingEnumDict.end());

  SILModule &module = topVjp.getFunction()->getModule();
  ASTContext &astContext = oldED->getASTContext();

  CanGenericSignature genericSig = nullptr;
  if (auto *derivativeFnGenEnv = topVjp.getFunction()->getGenericEnvironment())
    genericSig =
        derivativeFnGenEnv->getGenericSignature().getCanonicalSignature();
  GenericParamList *genericParams = nullptr;
  if (genericSig)
    genericParams = autodiff::cloneGenericParameters(
        astContext, oldED->getDeclContext(), genericSig);

  // TODO: use better naming
  Twine edNameStr = oldED->getNameStr() + "_specialized";
  Identifier edName = astContext.getIdentifier(edNameStr.str());

  auto *ed = new (astContext) EnumDecl(
      /*EnumLoc*/ SourceLoc(), /*Name*/ edName, /*NameLoc*/ SourceLoc(),
      /*Inherited*/ {}, /*GenericParams*/ genericParams,
      /*DC*/
      oldED->getDeclContext());
  ed->setImplicit();
  if (genericSig)
    ed->setGenericSignature(genericSig);

  for (EnumCaseDecl *oldECD : oldED->getAllCases()) {
    assert(oldECD->getElements().size() == 1);
    EnumElementDecl *oldEED = oldECD->getElements().front();

    unsigned enumIdx = module.getCaseIndex(oldEED);

    static llvm::SmallVector<ClosureAndIdxInPayload, 8> emptyClosuresBuffer =
        {};

    auto it = branchTracingEnumToClosuresDict.find(
        EnumTypeAndCaseIdx{enumType, enumIdx});
    const llvm::SmallVector<ClosureAndIdxInPayload, 8> &closuresBuffer =
        (it == branchTracingEnumToClosuresDict.end() ? emptyClosuresBuffer
                                                     : it->second);

    assert(oldEED->getParameterList()->size() == 1);
    ParamDecl &oldParamDecl = *oldEED->getParameterList()->front();

    auto *tt = cast<TupleType>(oldParamDecl.getInterfaceType().getPointer());
    SmallVector<TupleTypeElt, 4> newElements;
    newElements.reserve(tt->getNumElements());

    for (unsigned idxInPayload = 0; idxInPayload < tt->getNumElements();
         ++idxInPayload) {
      auto it = llvm::find_if(
          closuresBuffer,
          [idxInPayload](const ClosureAndIdxInPayload &closureAndIdxInPayload) {
            return closureAndIdxInPayload.idxInPayload == idxInPayload;
          });
      Type type;
      if (it != closuresBuffer.end()) {
        type =
            getCapturedArgTypesForClosure(it->closure.unbridged(), astContext);
        // TODO: delete throwing function support
        if (tt->getElementType(idxInPayload)->isOptional()) {
          assert(idxInPayload + 1 == tt->getNumElements());
          type = OptionalType::get(type)->getCanonicalType();
        }
      } else {
        type = tt->getElementType(idxInPayload);
        for (const auto &[enumTypeOld, enumTypeNew] : branchTracingEnumDict) {
          if (enumTypeOld.unbridged()
                  .getASTType()
                  ->mapTypeOutOfContext()
                  .getPointer() == type.getPointer()) {
            assert(idxInPayload == 0);
            type = enumTypeNew.unbridged().getASTType();
          }
        }
      }
      Identifier label = tt->getElement(idxInPayload).getName();
      newElements.emplace_back(type, label);
    }

    Type newTupleType =
        TupleType::get(newElements, astContext)->mapTypeOutOfContext();

    auto *newParamDecl = ParamDecl::cloneWithoutType(astContext, &oldParamDecl);
    newParamDecl->setInterfaceType(newTupleType);

    auto *newPL = ParameterList::create(astContext, {newParamDecl});

    auto *newEED = new (astContext) EnumElementDecl(
        /*IdentifierLoc*/ SourceLoc(),
        DeclName(astContext.getIdentifier(oldEED->getNameStr())), newPL,
        SourceLoc(), /*RawValueExpr*/ nullptr, ed);
    newEED->setImplicit();
    auto *newECD = EnumCaseDecl::create(
        /*CaseLoc*/ SourceLoc(), {newEED}, ed);
    newECD->setImplicit();
    ed->addMember(newEED);
    ed->addMember(newECD);
  }

  ed->setAccess(AccessLevel::Public);
  auto &file = autodiff::getSourceFile(topVjp.getFunction())
                   .getOrCreateSynthesizedFile();
  file.addTopLevelDecl(ed);
  file.getParentModule()->clearLookupCache();

  SILType newEnumType =
      remapType(getBranchingTraceEnumLoweredType(ed, topVjp.getFunction()),
                topVjp.getFunction());

  return newEnumType;
}

BranchTracingEnumDict autodiffSpecializeBranchTracingEnums(
    BridgedFunction topVjp, BridgedType topEnum,
    const VectorOfBridgedClosureInfoCFG &vectorOfClosureInfoCFG) {

  BranchTracingEnumToClosuresDict closuresBuffers;

  for (const BridgedClosureInfoCFG &elem : vectorOfClosureInfoCFG) {
    closuresBuffers[EnumTypeAndCaseIdx{elem.enumType, elem.enumCaseIdx}]
        .emplace_back(elem.closure, elem.idxInPayload);
  }

  std::vector<Type> enumQueue = getEnumQueue(topEnum);
  BranchTracingEnumDict dict;

  for (const Type &t : enumQueue) {
    EnumDecl *ed = t->getEnumOrBoundGenericEnum();

    SILType silType =
        remapType(getBranchingTraceEnumLoweredType(ed, topVjp.getFunction()),
                  topVjp.getFunction());

    dict[BridgedType(silType)] = autodiffSpecializeBranchTracingEnum(
        BridgedType(silType), topVjp, closuresBuffers, dict);
  }

  return dict;
}

BridgedArgument recreateEnumBlockArgument(BridgedArgument arg,
                                          const BranchTracingEnumDict &dict) {
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

BridgedArgument recreateOptionalBlockArgument(BridgedBasicBlock bbBridged,
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

BridgedArgument recreateTupleBlockArgument(
    BridgedArgument arg, const BranchTracingEnumDict &branchTracingEnumDict,
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
        getCapturedArgTypesForClosure(it->closure.unbridged(), astContext)
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
