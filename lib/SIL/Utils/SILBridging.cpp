//===--- SILBridging.cpp --------------------------------------------------===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2023 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#include "swift/SIL/SILBridging.h"

#ifdef PURE_BRIDGING_MODE
// In PURE_BRIDGING_MODE, briding functions are not inlined and therefore inluded in the cpp file.
#include "swift/SIL/SILBridgingImpl.h"
#endif

#include "swift/AST/Attr.h"
#include "swift/AST/ParameterList.h"
#include "swift/AST/SemanticAttrs.h"
#include "swift/Basic/Assertions.h"
#include "swift/SIL/MemAccessUtils.h"
#include "swift/SIL/OwnershipUtils.h"
#include "swift/SIL/ParseTestSpecification.h"
#include "swift/SIL/SILBuilder.h"
#include "swift/SIL/SILCloner.h"
#include "swift/SIL/SILContext.h"
#include "swift/SIL/SILGlobalVariable.h"
#include "swift/SIL/SILNode.h"
#include "swift/SIL/Test.h"
#include <cstring>
#include <stdio.h>
#include <string>

using namespace swift;

namespace {

bool nodeMetatypesInitialized = false;

// Filled in by class registration in initializeSwiftModules().
SwiftMetatype nodeMetatypes[(unsigned)SILNodeKind::Last_SILNode + 1];

}

bool swiftModulesInitialized() {
  return nodeMetatypesInitialized;
}

// Does return null if initializeSwiftModules() is never called.
SwiftMetatype SILNode::getSILNodeMetatype(SILNodeKind kind) {
  SwiftMetatype metatype = nodeMetatypes[(unsigned)kind];
  if (nodeMetatypesInitialized && !metatype) {
    ABORT([&](auto &out) {
      out << "Instruction " << getSILInstructionName((SILInstructionKind)kind)
          << " not registered";
    });
  }
  return metatype;
}

static std::unordered_map<
    BridgedType,
    llvm::DenseMap<SwiftInt, llvm::SmallVector<
                                 std::pair<BridgedInstruction, SwiftInt>, 8>>,
    BridgedTypeHasher>
    closuresBuffers;

static llvm::SmallVector<std::pair<BridgedInstruction, SwiftInt>, 8>
    closuresBuffersForPb;

struct SILTypeHasher {
  unsigned operator()(const SILType &value) const {
    return llvm::DenseMapInfo<SILType>::getHashValue(value);
  }
};

static std::unordered_map<SILType, SILType, SILTypeHasher> enumDict;

//===----------------------------------------------------------------------===//
//                          Class registration
//===----------------------------------------------------------------------===//

static llvm::StringMap<SILNodeKind> valueNamesToKind;

/// Registers the metatype of a swift SIL class.
/// Called by initializeSwiftModules().
void registerBridgedClass(BridgedStringRef bridgedClassName, SwiftMetatype metatype) {
  StringRef className = bridgedClassName.unbridged();
  nodeMetatypesInitialized = true;

  // Handle the important non Node classes.
  if (className == "BasicBlock")
    return SILBasicBlock::registerBridgedMetatype(metatype);
  if (className == "GlobalVariable")
    return SILGlobalVariable::registerBridgedMetatype(metatype);
  if (className == "Argument") {
    nodeMetatypes[(unsigned)SILNodeKind::SILPhiArgument] = metatype;
    return;
  }
  if (className == "FunctionArgument") {
    nodeMetatypes[(unsigned)SILNodeKind::SILFunctionArgument] = metatype;
    return;
  }

  if (valueNamesToKind.empty()) {
#define VALUE(ID, PARENT) \
    valueNamesToKind[#ID] = SILNodeKind::ID;
#define NON_VALUE_INST(ID, NAME, PARENT, MEMBEHAVIOR, MAYRELEASE) \
    VALUE(ID, NAME)
#define ARGUMENT(ID, PARENT) \
    VALUE(ID, NAME)
#define SINGLE_VALUE_INST(ID, NAME, PARENT, MEMBEHAVIOR, MAYRELEASE) \
    VALUE(ID, NAME)
#define MULTIPLE_VALUE_INST(ID, NAME, PARENT, MEMBEHAVIOR, MAYRELEASE) \
    VALUE(ID, NAME)
#include "swift/SIL/SILNodes.def"
  }

  std::string prefixedName;
  auto iter = valueNamesToKind.find(className);
  if (iter == valueNamesToKind.end()) {
    // Try again with a "SIL" prefix. For example Argument -> SILArgument.
    prefixedName = std::string("SIL") + std::string(className);
    iter = valueNamesToKind.find(prefixedName);
    if (iter == valueNamesToKind.end()) {
      ABORT([&](auto &out) {
        out << "Unknown bridged node class " << className;
      });
    }
    className = prefixedName;
  }
  SILNodeKind kind = iter->second;
  SwiftMetatype existingTy = nodeMetatypes[(unsigned)kind];
  if (existingTy) {
    ABORT([&](auto &out) {
      out << "Double registration of class " << className;
    });
  }
  nodeMetatypes[(unsigned)kind] = metatype;
}

//===----------------------------------------------------------------------===//
//                                Test
//===----------------------------------------------------------------------===//

void registerTest(BridgedStringRef name, void *nativeSwiftContext) {
  swift::test::FunctionTest::createNativeSwiftFunctionTest(
      name.unbridged(), nativeSwiftContext, /*isSILTest=*/ true);
}

bool BridgedTestArguments::hasUntaken() const {
  return arguments->hasUntaken();
}

BridgedStringRef BridgedTestArguments::takeString() const {
  return arguments->takeString();
}

bool BridgedTestArguments::takeBool() const { return arguments->takeBool(); }

SwiftInt BridgedTestArguments::takeInt() const { return arguments->takeUInt(); }

BridgedOperand BridgedTestArguments::takeOperand() const {
  return {arguments->takeOperand()};
}

BridgedValue BridgedTestArguments::takeValue() const {
  return {arguments->takeValue()};
}

BridgedInstruction BridgedTestArguments::takeInstruction() const {
  return {arguments->takeInstruction()->asSILNode()};
}

BridgedArgument BridgedTestArguments::takeArgument() const {
  return {arguments->takeBlockArgument()};
}

BridgedBasicBlock BridgedTestArguments::takeBlock() const {
  return {arguments->takeBlock()};
}

BridgedFunction BridgedTestArguments::takeFunction() const {
  return {arguments->takeFunction()};
}

/// Returns the lowered SIL type of the branching trace enum associated with
/// the given original block.
static SILType getBranchingTraceEnumLoweredType(EnumDecl *ed,
                                                SILFunction *vjp) {
  auto traceDeclType = ed->getDeclaredInterfaceType()->getCanonicalType();
  Lowering::AbstractionPattern pattern(
      vjp->getLoweredFunctionType()->getSubstGenericSignature(), traceDeclType);
  Lowering::TypeConverter typeConverter(*ed->getParentModule());
  return typeConverter.getLoweredType(pattern, traceDeclType,
                                      TypeExpansionContext::minimal());
}

/// Remap any archetypes into the current function's context.
SILType remapType(SILType ty, SILFunction *foo) {
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

static SourceFile &getSourceFile(SILFunction *f) {
  if (f->hasLocation())
    if (auto *declContext = f->getLocation().getAsDeclContext())
      if (auto *parentSourceFile = declContext->getParentSourceFile())
        return *parentSourceFile;
  for (auto *file : f->getModule().getSwiftModule()->getFiles())
    if (auto *sourceFile = dyn_cast<SourceFile>(file))
      return *sourceFile;
  llvm_unreachable("Could not resolve SourceFile from SILFunction");
}

static Type getPAICapturedArgTypes(const PartialApplyInst *pai,
                                   ASTContext &ctx) {
  SmallVector<TupleTypeElt, 4> paramTuple;
  paramTuple.reserve(pai->getArguments().size());
  for (const SILValue &arg : pai->getArguments())
    paramTuple.emplace_back(arg->getType().getASTType(), Identifier{});
  return TupleType::get(paramTuple, ctx);
}

BridgedArgument
BridgedBasicBlock::recreateEnumBlockArgument(BridgedArgument arg) const {
  assert(!unbridged()->isEntry());
  swift::ValueOwnershipKind oldOwnership =
      arg.getArgument()->getOwnershipKind();

  swift::SILArgument *oldArg = arg.getArgument();
  unsigned index = oldArg->getIndex();
  // TODO: switch to contains() after transition to C++20
  assert(enumDict.find(oldArg->getType()) != enumDict.end());
  SILType type = enumDict.at(oldArg->getType());
  swift::SILPhiArgument *newArg =
      unbridged()->insertPhiArgument(index, type, oldOwnership);
  oldArg->replaceAllUsesWith(newArg);
  eraseArgument(index + 1);
  return {newArg};
}

BridgedArgument BridgedBasicBlock::recreateOptionalBlockArgument(
    BridgedType optionalType) const {
  swift::SILBasicBlock *bb = unbridged();
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
  eraseArgument(1);

  return {newArg};
}

BridgedArgument
BridgedBasicBlock::recreateTupleBlockArgument(BridgedArgument arg) const {
  swift::SILBasicBlock *bb = unbridged();
  assert(!bb->isEntry());
  swift::SILArgument *oldArg = arg.getArgument();
  unsigned argIdx = oldArg->getIndex();
  auto *oldTupleTy =
      llvm::cast<swift::TupleType>(oldArg->getType().getASTType().getPointer());
  llvm::SmallVector<swift::TupleTypeElt, 8> newTupleElTypes;
  for (unsigned i = 0; i < oldTupleTy->getNumElements(); ++i) {
    unsigned idxInClosuresBuffer = -1;
    for (unsigned j = 0; j < closuresBuffersForPb.size(); ++j) {
      if (closuresBuffersForPb[j].second == i) {
        if (idxInClosuresBuffer != unsigned(-1)) {
          assert(closuresBuffersForPb[j].first.unbridged() ==
                 closuresBuffersForPb[idxInClosuresBuffer].first.unbridged());
        }
        idxInClosuresBuffer = j;
      }
    }

    if (idxInClosuresBuffer == unsigned(-1)) {
      Type type = oldTupleTy->getElementType(i);
      for (const auto &[enumTypeOld, enumTypeNew] : enumDict) {
        if (enumTypeOld.getDebugDescription() == "$" + type.getString()) {
          assert(i == 0);
          type = enumTypeNew.getASTType();
        }
      }
      newTupleElTypes.emplace_back(type, oldTupleTy->getElement(i).getName());
      continue;
    }

    CanType canType;
    if (auto *pai = dyn_cast<PartialApplyInst>(
            closuresBuffersForPb[idxInClosuresBuffer].first.unbridged())) {
      canType = getPAICapturedArgTypes(pai, bb->getModule().getASTContext())
                    ->getCanonicalType();
    } else {
      assert(isa<ThinToThickFunctionInst>(
          closuresBuffersForPb[idxInClosuresBuffer].first.unbridged()));
      canType = TupleType::get({}, bb->getModule().getASTContext())
                    ->getCanonicalType();
    }
    if (oldTupleTy->getElementType(i)->isOptional()) {
      assert(i + 1 == oldTupleTy->getNumElements());
      canType = OptionalType::get(canType)->getCanonicalType();
    }
    newTupleElTypes.emplace_back(canType);
  }
  auto newTupleTy = swift::SILType::getFromOpaqueValue(
      swift::TupleType::get(newTupleElTypes, bb->getModule().getASTContext()));

  swift::ValueOwnershipKind oldOwnership =
      bb->getArgument(argIdx)->getOwnershipKind();

  swift::SILPhiArgument *newArg =
      bb->insertPhiArgument(argIdx, newTupleTy, oldOwnership);
  oldArg->replaceAllUsesWith(newArg);
  eraseArgument(argIdx + 1);

  return {newArg};
}

namespace {
struct SpecializeCandidateInfo {
  unsigned closureIdxInPayloadTuple;
  llvm::SmallVector<SILValue, 8> capturedArgs;
};

using SpecializeCandidate =
    llvm::DenseMap<SILInstruction *, SpecializeCandidateInfo>;
using BranchTracingEnumCases = llvm::DenseMap<unsigned, SpecializeCandidate>;

} // namespace

//===----------------------------------------------------------------------===//
//                                SILFunction
//===----------------------------------------------------------------------===//

static_assert((int)BridgedFunction::EffectsKind::ReadNone == (int)swift::EffectsKind::ReadNone);
static_assert((int)BridgedFunction::EffectsKind::ReadOnly == (int)swift::EffectsKind::ReadOnly);
static_assert((int)BridgedFunction::EffectsKind::ReleaseNone == (int)swift::EffectsKind::ReleaseNone);
static_assert((int)BridgedFunction::EffectsKind::ReadWrite == (int)swift::EffectsKind::ReadWrite);
static_assert((int)BridgedFunction::EffectsKind::Unspecified == (int)swift::EffectsKind::Unspecified);
static_assert((int)BridgedFunction::EffectsKind::Custom == (int)swift::EffectsKind::Custom);

static_assert((int)BridgedFunction::PerformanceConstraints::None == (int)swift::PerformanceConstraints::None);
static_assert((int)BridgedFunction::PerformanceConstraints::NoAllocation == (int)swift::PerformanceConstraints::NoAllocation);
static_assert((int)BridgedFunction::PerformanceConstraints::NoLocks == (int)swift::PerformanceConstraints::NoLocks);
static_assert((int)BridgedFunction::PerformanceConstraints::NoRuntime == (int)swift::PerformanceConstraints::NoRuntime);
static_assert((int)BridgedFunction::PerformanceConstraints::NoExistentials == (int)swift::PerformanceConstraints::NoExistentials);
static_assert((int)BridgedFunction::PerformanceConstraints::NoObjCBridging == (int)swift::PerformanceConstraints::NoObjCBridging);

static_assert((int)BridgedFunction::InlineStrategy::InlineDefault == (int)swift::InlineDefault);
static_assert((int)BridgedFunction::InlineStrategy::NoInline == (int)swift::NoInline);
static_assert((int)BridgedFunction::InlineStrategy::AlwaysInline == (int)swift::AlwaysInline);

static_assert((int)BridgedFunction::ThunkKind::IsNotThunk == (int)swift::IsNotThunk);
static_assert((int)BridgedFunction::ThunkKind::IsThunk == (int)swift::IsThunk);
static_assert((int)BridgedFunction::ThunkKind::IsReabstractionThunk == (int)swift::IsReabstractionThunk);
static_assert((int)BridgedFunction::ThunkKind::IsSignatureOptimizedThunk == (int)swift::IsSignatureOptimizedThunk);

BridgedOwnedString BridgedFunction::getDebugDescription() const {
  std::string str;
  llvm::raw_string_ostream os(str);
  getFunction()->print(os);
  str.pop_back(); // Remove trailing newline.
  return BridgedOwnedString(str);
}

BridgedSubstitutionMap BridgedFunction::getMethodSubstitutions(BridgedSubstitutionMap contextSubstitutions,
                                                               BridgedCanType selfType) const {
  swift::SILFunction *f = getFunction();
  swift::GenericSignature genericSig = f->getLoweredFunctionType()->getInvocationGenericSignature();

  if (!genericSig || genericSig->areAllParamsConcrete())
    return swift::SubstitutionMap();

  SubstitutionMap contextSubs = contextSubstitutions.unbridged();
  if (selfType.unbridged() &&
      contextSubs.getGenericSignature().getGenericParams().size() + 1 == genericSig.getGenericParams().size()) {

    // If this is a default witness methods (`selfType` != nil) it has generic self type. In this case
    // the generic self parameter is at depth 0 and the actual generic parameters of the substitution map
    // are at depth + 1, e.g:
    // ```
    //     @convention(witness_method: P) <τ_0_0><τ_1_0 where τ_0_0 : GenClass<τ_1_0>.T>
    //                                       ^      ^
    //                                    self      params of substitution map at depth + 1
    // ```
    return swift::SubstitutionMap::get(genericSig,
      [&](SubstitutableType *type) -> Type {
        GenericTypeParamType *genericParam = cast<GenericTypeParamType>(type);
        // The self type is τ_0_0
        if (genericParam->getDepth() == 0 && genericParam->getIndex() == 0)
          return selfType.unbridged();

        // Lookup the substitution map types at depth - 1.
        auto *depthMinus1Param = GenericTypeParamType::getType(genericParam->getDepth() - 1,
                                                               genericParam->getIndex(),
                                                               genericParam->getASTContext());
        return swift::QuerySubstitutionMap{contextSubs}(depthMinus1Param);
      },
      swift::LookUpConformanceInModule());

  }
  return swift::SubstitutionMap::get(genericSig,
                                     swift::QuerySubstitutionMap{contextSubs},
                                     swift::LookUpConformanceInModule());
}

//===----------------------------------------------------------------------===//
//                               SILBasicBlock
//===----------------------------------------------------------------------===//

BridgedOwnedString BridgedBasicBlock::getDebugDescription() const {
  std::string str;
  llvm::raw_string_ostream os(str);
  unbridged()->print(os);
  str.pop_back(); // Remove trailing newline.
  return BridgedOwnedString(str);
}

//===----------------------------------------------------------------------===//
//                                SILValue
//===----------------------------------------------------------------------===//

BridgedOwnedString BridgedValue::getDebugDescription() const {
  std::string str;
  llvm::raw_string_ostream os(str);
  getSILValue()->print(os);
  str.pop_back(); // Remove trailing newline.
  return BridgedOwnedString(str);
}

BridgedValue::Kind BridgedValue::getKind() const {
  SILValue v = getSILValue();
  if (isa<SingleValueInstruction>(v)) {
    return BridgedValue::Kind::SingleValueInstruction;
  } else if (isa<SILArgument>(v)) {
    return BridgedValue::Kind::Argument;
  } else if (isa<MultipleValueInstructionResult>(v)) {
    return BridgedValue::Kind::MultipleValueInstructionResult;
  } else if (isa<SILUndef>(v)) {
    return BridgedValue::Kind::Undef;
  }
  llvm_unreachable("unknown SILValue");
}

ArrayRef<SILValue> BridgedValueArray::getValues(SmallVectorImpl<SILValue> &storage) {
  for (unsigned idx = 0; idx < count; ++idx) {
    storage.push_back(base[idx].value.getSILValue());
  }
  return storage;
}

bool BridgedValue::findPointerEscape() const {
  return swift::findPointerEscape(getSILValue());
}

//===----------------------------------------------------------------------===//
//                                SILArgument
//===----------------------------------------------------------------------===//

static_assert((int)BridgedArgumentConvention::Indirect_In == (int)swift::SILArgumentConvention::Indirect_In);
static_assert((int)BridgedArgumentConvention::Indirect_In_Guaranteed == (int)swift::SILArgumentConvention::Indirect_In_Guaranteed);
static_assert((int)BridgedArgumentConvention::Indirect_Inout == (int)swift::SILArgumentConvention::Indirect_Inout);
static_assert((int)BridgedArgumentConvention::Indirect_InoutAliasable == (int)swift::SILArgumentConvention::Indirect_InoutAliasable);
static_assert((int)BridgedArgumentConvention::Indirect_Out == (int)swift::SILArgumentConvention::Indirect_Out);
static_assert((int)BridgedArgumentConvention::Direct_Owned == (int)swift::SILArgumentConvention::Direct_Owned);
static_assert((int)BridgedArgumentConvention::Direct_Unowned == (int)swift::SILArgumentConvention::Direct_Unowned);
static_assert((int)BridgedArgumentConvention::Direct_Guaranteed == (int)swift::SILArgumentConvention::Direct_Guaranteed);
static_assert((int)BridgedArgumentConvention::Pack_Owned == (int)swift::SILArgumentConvention::Pack_Owned);
static_assert((int)BridgedArgumentConvention::Pack_Inout == (int)swift::SILArgumentConvention::Pack_Inout);
static_assert((int)BridgedArgumentConvention::Pack_Guaranteed == (int)swift::SILArgumentConvention::Pack_Guaranteed);
static_assert((int)BridgedArgumentConvention::Pack_Out == (int)swift::SILArgumentConvention::Pack_Out);

//===----------------------------------------------------------------------===//
//                                Linkage
//===----------------------------------------------------------------------===//

static_assert((int)BridgedLinkage::Public == (int)swift::SILLinkage::Public);
static_assert((int)BridgedLinkage::PublicNonABI == (int)swift::SILLinkage::PublicNonABI);
static_assert((int)BridgedLinkage::Package == (int)swift::SILLinkage::Package);
static_assert((int)BridgedLinkage::PackageNonABI == (int)swift::SILLinkage::PackageNonABI);
static_assert((int)BridgedLinkage::Hidden == (int)swift::SILLinkage::Hidden);
static_assert((int)BridgedLinkage::Shared == (int)swift::SILLinkage::Shared);
static_assert((int)BridgedLinkage::Private == (int)swift::SILLinkage::Private);
static_assert((int)BridgedLinkage::PublicExternal == (int)swift::SILLinkage::PublicExternal);
static_assert((int)BridgedLinkage::PackageExternal == (int)swift::SILLinkage::PackageExternal);
static_assert((int)BridgedLinkage::HiddenExternal == (int)swift::SILLinkage::HiddenExternal);

//===----------------------------------------------------------------------===//
//                                Operand
//===----------------------------------------------------------------------===//

void BridgedOperand::changeOwnership(BridgedValue::Ownership from, BridgedValue::Ownership to) const {
  swift::ForwardingOperand forwardingOp(op);
  assert(forwardingOp);
  forwardingOp.replaceOwnershipKind(BridgedValue::unbridge(from), BridgedValue::unbridge(to));
}

//===----------------------------------------------------------------------===//
//                            SILGlobalVariable
//===----------------------------------------------------------------------===//

BridgedOwnedString BridgedGlobalVar::getDebugDescription() const {
  std::string str;
  llvm::raw_string_ostream os(str);
  getGlobal()->print(os);
  str.pop_back(); // Remove trailing newline.
  return BridgedOwnedString(str);
}

bool BridgedGlobalVar::canBeInitializedStatically() const {
  SILGlobalVariable *global = getGlobal();
  auto expansion = ResilienceExpansion::Maximal;
  if (hasPublicVisibility(global->getLinkage()))
    expansion = ResilienceExpansion::Minimal;

  auto props = global->getModule().Types.getTypeProperties(
      global->getLoweredType(),
      TypeExpansionContext::noOpaqueTypeArchetypesSubstitution(expansion));
  return props.isFixedABI();
}

bool BridgedGlobalVar::mustBeInitializedStatically() const {
  SILGlobalVariable *global = getGlobal();
  return global->mustBeInitializedStatically();
}

bool BridgedGlobalVar::isConstValue() const {
  SILGlobalVariable *global = getGlobal();
  if (const auto &decl = global->getDecl())
    return decl->isConstValue();
  return false;
}

//===----------------------------------------------------------------------===//
//                            SILDeclRef
//===----------------------------------------------------------------------===//

BridgedOwnedString BridgedDeclRef::getDebugDescription() const {
  std::string str;
  llvm::raw_string_ostream os(str);
  unbridged().print(os);
  return BridgedOwnedString(str);
}

//===----------------------------------------------------------------------===//
//                            SILVTable
//===----------------------------------------------------------------------===//

static_assert(sizeof(BridgedVTableEntry) >= sizeof(swift::SILVTableEntry),
              "BridgedVTableEntry has wrong size");

static_assert((int)BridgedVTableEntry::Kind::Normal == (int)swift::SILVTableEntry::Normal);
static_assert((int)BridgedVTableEntry::Kind::Inherited == (int)swift::SILVTableEntry::Inherited);
static_assert((int)BridgedVTableEntry::Kind::Override == (int)swift::SILVTableEntry::Override);

BridgedOwnedString BridgedVTable::getDebugDescription() const {
  std::string str;
  llvm::raw_string_ostream os(str);
  vTable->print(os);
  str.pop_back(); // Remove trailing newline.
  return BridgedOwnedString(str);
}

BridgedOwnedString BridgedVTableEntry::getDebugDescription() const {
  std::string str;
  llvm::raw_string_ostream os(str);
  unbridged().print(os);
  str.pop_back(); // Remove trailing newline.
  return BridgedOwnedString(str);
}

//===----------------------------------------------------------------------===//
//                    SILVWitnessTable, SILDefaultWitnessTable
//===----------------------------------------------------------------------===//

static_assert(sizeof(BridgedWitnessTableEntry) >= sizeof(swift::SILWitnessTable::Entry),
              "BridgedWitnessTableEntry has wrong size");

static_assert((int)BridgedWitnessTableEntry::Kind::invalid == (int)swift::SILWitnessTable::WitnessKind::Invalid);
static_assert((int)BridgedWitnessTableEntry::Kind::method == (int)swift::SILWitnessTable::WitnessKind::Method);
static_assert((int)BridgedWitnessTableEntry::Kind::associatedType == (int)swift::SILWitnessTable::WitnessKind::AssociatedType);
static_assert((int)BridgedWitnessTableEntry::Kind::associatedConformance == (int)swift::SILWitnessTable::WitnessKind::AssociatedConformance);
static_assert((int)BridgedWitnessTableEntry::Kind::baseProtocol == (int)swift::SILWitnessTable::WitnessKind::BaseProtocol);

BridgedOwnedString BridgedWitnessTableEntry::getDebugDescription() const {
  std::string str;
  llvm::raw_string_ostream os(str);
  unbridged().print(os, /*verbose=*/ false, PrintOptions::printSIL());
  str.pop_back(); // Remove trailing newline.
  return BridgedOwnedString(str);
}

BridgedOwnedString BridgedWitnessTable::getDebugDescription() const {
  std::string str;
  llvm::raw_string_ostream os(str);
  table->print(os);
  str.pop_back(); // Remove trailing newline.
  return BridgedOwnedString(str);
}

BridgedOwnedString BridgedDefaultWitnessTable::getDebugDescription() const {
  std::string str;
  llvm::raw_string_ostream os(str);
  table->print(os);
  str.pop_back(); // Remove trailing newline.
  return BridgedOwnedString(str);
}

//===----------------------------------------------------------------------===//
//                               SILDebugLocation
//===----------------------------------------------------------------------===//

static_assert(sizeof(BridgedLocation) >= sizeof(swift::SILDebugLocation),
              "BridgedLocation has wrong size");

BridgedOwnedString BridgedLocation::getDebugDescription() const {
  std::string str;
  llvm::raw_string_ostream os(str);
  SILLocation loc = getLoc().getLocation();
  loc.print(os);
#ifndef NDEBUG
  if (const SILDebugScope *scope = getLoc().getScope()) {
    if (DeclContext *dc = loc.getAsDeclContext()) {
      os << ", scope=";
      scope->print(dc->getASTContext().SourceMgr, os, /*indent*/ 2);
    } else {
      os << ", scope=?";
    }
  }
#endif
  return BridgedOwnedString(str);
}

//===----------------------------------------------------------------------===//
//                               SILInstruction
//===----------------------------------------------------------------------===//

static_assert((int)BridgedMemoryBehavior::None == (int)swift::MemoryBehavior::None);
static_assert((int)BridgedMemoryBehavior::MayRead == (int)swift::MemoryBehavior::MayRead);
static_assert((int)BridgedMemoryBehavior::MayWrite == (int)swift::MemoryBehavior::MayWrite);
static_assert((int)BridgedMemoryBehavior::MayReadWrite == (int)swift::MemoryBehavior::MayReadWrite);
static_assert((int)BridgedMemoryBehavior::MayHaveSideEffects == (int)swift::MemoryBehavior::MayHaveSideEffects);

BridgedOwnedString BridgedInstruction::getDebugDescription() const {
  std::string str;
  llvm::raw_string_ostream os(str);
  unbridged()->print(os);
  str.pop_back(); // Remove trailing newline.
  return BridgedOwnedString(str);
}

bool BridgedInstruction::mayAccessPointer() const {
  return ::mayAccessPointer(unbridged());
}

bool BridgedInstruction::mayLoadWeakOrUnowned() const {
  return ::mayLoadWeakOrUnowned(unbridged());
}

bool BridgedInstruction::maySynchronize() const {
  return ::maySynchronize(unbridged());
}

bool BridgedInstruction::mayBeDeinitBarrierNotConsideringSideEffects() const {
  return ::mayBeDeinitBarrierNotConsideringSideEffects(unbridged());
}

//===----------------------------------------------------------------------===//
//                               BridgedBuilder
//===----------------------------------------------------------------------===//

static llvm::SmallVector<std::pair<swift::EnumElementDecl *, swift::SILBasicBlock *>, 16>
convertCases(SILType enumTy, const void * _Nullable enumCases, SwiftInt numEnumCases) {
  using BridgedCase = const std::pair<SwiftInt, BridgedBasicBlock>;
  llvm::ArrayRef<BridgedCase> cases(static_cast<BridgedCase *>(enumCases),
                                    (unsigned)numEnumCases);
  llvm::SmallDenseMap<SwiftInt, swift::EnumElementDecl *> mappedElements;
  swift::EnumDecl *enumDecl = enumTy.getEnumOrBoundGenericEnum();
  for (auto elemWithIndex : llvm::enumerate(enumDecl->getAllElements())) {
    mappedElements[elemWithIndex.index()] = elemWithIndex.value();
  }
  llvm::SmallVector<std::pair<swift::EnumElementDecl *, swift::SILBasicBlock *>, 16> convertedCases;
  for (auto c : cases) {
    assert(mappedElements.count(c.first) && "wrong enum element index");
    convertedCases.push_back({mappedElements[c.first], c.second.unbridged()});
  }
  return convertedCases;
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

BranchTracingEnumDict
BridgedAutoDiffClosureSpecializationHelper::rewriteAllEnums(
    BridgedFunction topVjp, BridgedType topEnum,
    const VectorOfBridgedClosureInfoCFG &vectorOfClosureInfoCFG) const {
  std::unordered_map<
      BridgedType,
      llvm::DenseMap<SwiftInt, llvm::SmallVector<
                                   std::pair<BridgedInstruction, SwiftInt>, 8>>,
      BridgedTypeHasher>
      closuresBuffers;

  for (const BridgedClosureInfoCFG &elem : vectorOfClosureInfoCFG) {
    closuresBuffers[elem.enumType][elem.enumCaseIdx].emplace_back(
        elem.closure, elem.idxInPayload);
  }

  std::vector<Type> enumQueue = getEnumQueue(topEnum);
  BranchTracingEnumDict dict;

  for (const Type &t : enumQueue) {
    EnumDecl *ed = t->getEnumOrBoundGenericEnum();

    SILType silType =
        remapType(getBranchingTraceEnumLoweredType(ed, topVjp.getFunction()),
                  topVjp.getFunction());

    dict[BridgedType(silType)] = rewriteBranchTracingEnum(
        BridgedType(silType), topVjp, closuresBuffers, dict);
  }

  return dict;
}

BridgedOwnedString BridgedType::getEnumTypeCaseName(SwiftInt caseIdx) const {
  EnumDecl *ed = unbridged().getEnumOrBoundGenericEnum();
  SwiftInt idx = 0;
  for (EnumElementDecl *elem : ed->getAllElements()) {
    if (idx == caseIdx)
      return elem->getNameStr();
    ++idx;
  }
  assert(false);
}

BridgedInstruction
BridgedBuilder::createOptionalSome(BridgedValue value) const {
  EnumElementDecl *someEltDecl =
      unbridged().getASTContext().getOptionalSomeDecl();
  EnumInst *optionalSome = unbridged().createEnum(
      loc.getLoc().getLocation(), value.getSILValue(), someEltDecl,
      SILType::getOptionalType(value.getType().unbridged()),
      value.getSILValue()->getOwnershipKind());
  return optionalSome;
}

BridgedInstruction
BridgedBuilder::createOptionalNone(BridgedValueArray tupleElements) const {
  EnumElementDecl *noneEltDecl =
      unbridged().getASTContext().getOptionalNoneDecl();

  llvm::SmallVector<swift::SILValue, 16> elementValues;
  llvm::ArrayRef<swift::SILValue> values =
      tupleElements.getValues(elementValues);
  llvm::SmallVector<swift::TupleTypeElt, 16> tupleTyElts;
  tupleTyElts.reserve(values.size());
  for (const swift::SILValue &value : values) {
    tupleTyElts.emplace_back(value->getType().getASTType());
  }
  swift::Type tupleTy =
      swift::TupleType::get(tupleTyElts, unbridged().getASTContext());
  swift::SILType silTupleTy =
      swift::SILType::getPrimitiveObjectType(tupleTy->getCanonicalType());

  EnumInst *optionalNone =
      unbridged().createEnum(loc.getLoc().getLocation(), SILValue(),
                             noneEltDecl, SILType::getOptionalType(silTupleTy));

  return optionalNone;
}

// MYTODO: copied from LinearMapInfo.cpp. Is this needed?
/// Clone the generic parameters of the given generic signature and return a new
/// `GenericParamList`.
static GenericParamList *cloneGenericParameters(ASTContext &ctx,
                                                DeclContext *dc,
                                                CanGenericSignature sig) {
  SmallVector<GenericTypeParamDecl *, 2> clonedParams;
  for (auto paramType : sig.getGenericParams()) {
    auto *clonedParam = GenericTypeParamDecl::createImplicit(
        dc, paramType->getName(), paramType->getDepth(), paramType->getIndex(),
        paramType->getParamKind());
    clonedParam->setDeclContext(dc);
    clonedParams.push_back(clonedParam);
  }
  return GenericParamList::create(ctx, SourceLoc(), clonedParams, SourceLoc());
}

BridgedType BridgedType::mapTypeOutOfContext() const {
  return {unbridged().mapTypeOutOfContext()};
}

BridgedType
BridgedAutoDiffClosureSpecializationHelper::rewriteBranchTracingEnum(
    BridgedType enumType, BridgedFunction topVjp,
    std::unordered_map<
        BridgedType,
        llvm::DenseMap<
            SwiftInt,
            llvm::SmallVector<std::pair<BridgedInstruction, SwiftInt>, 8>>,
        BridgedTypeHasher> &closuresBuffers,
    const BranchTracingEnumDict &dict) const {
  EnumDecl *oldED = enumType.unbridged().getEnumOrBoundGenericEnum();
  assert(oldED && "Expected valid enum type");
  // TODO: switch to contains() after transition to C++20
  assert(dict.find(enumType.unbridged()) == dict.end());

  SILModule &module = topVjp.getFunction()->getModule();
  ASTContext &astContext = oldED->getASTContext();

  CanGenericSignature genericSig = nullptr;
  if (auto *derivativeFnGenEnv = topVjp.getFunction()->getGenericEnvironment())
    genericSig =
        derivativeFnGenEnv->getGenericSignature().getCanonicalSignature();
  GenericParamList *genericParams = nullptr;
  if (genericSig)
    genericParams =
        cloneGenericParameters(astContext, oldED->getDeclContext(), genericSig);

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

    llvm::SmallVector<std::pair<BridgedInstruction, SwiftInt>, 8>
        *closuresBuffer = &closuresBuffers[enumType.unbridged()][enumIdx];

    assert(oldEED->getParameterList()->size() == 1);
    ParamDecl &oldParamDecl = *oldEED->getParameterList()->front();

    auto *tt = cast<TupleType>(oldParamDecl.getInterfaceType().getPointer());
    SmallVector<TupleTypeElt, 4> newElements;
    newElements.reserve(tt->getNumElements());

    for (unsigned i = 0; i < tt->getNumElements(); ++i) {
      Type type;
      unsigned idxInClosuresBuffer = -1;
      for (unsigned j = 0; j < closuresBuffer->size(); ++j) {
        if ((*closuresBuffer)[j].second == i) {
          assert(idxInClosuresBuffer == unsigned(-1) ||
                 (*closuresBuffer)[j].first.unbridged() ==
                     (*closuresBuffer)[idxInClosuresBuffer].first.unbridged());
          idxInClosuresBuffer = j;
        }
      }
      if (idxInClosuresBuffer != unsigned(-1)) {
        if (const auto *PAI = dyn_cast<PartialApplyInst>(
                (*closuresBuffer)[idxInClosuresBuffer].first.unbridged())) {
          type = getPAICapturedArgTypes(PAI, astContext);
        } else {
          assert(isa<ThinToThickFunctionInst>(
              (*closuresBuffer)[idxInClosuresBuffer].first.unbridged()));
          type = TupleType::get({}, astContext);
        }
        if (tt->getElementType(i)->isOptional()) {
          assert(i + 1 == tt->getNumElements());
          type = OptionalType::get(type)->getCanonicalType();
        }
      } else {
        type = tt->getElementType(i);
        // TODO: make this less fragile
        for (const auto &[enumTypeOld, enumTypeNew] : enumDict) {
          if (enumTypeOld.unbridged().getDebugDescription() ==
              "$" + type.getString()) {
            assert(i == 0);
            type = enumTypeNew.unbridged().getASTType();
          }
        }
      }
      Identifier label = tt->getElement(i).getName();
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
  auto &file = getSourceFile(topVjp.getFunction()).getOrCreateSynthesizedFile();
  file.addTopLevelDecl(ed);
  file.getParentModule()->clearLookupCache();

  SILType newEnumType =
      remapType(getBranchingTraceEnumLoweredType(ed, topVjp.getFunction()),
                topVjp.getFunction());

  return newEnumType;
}

BridgedInstruction BridgedBuilder::createSwitchEnumInst(BridgedValue enumVal, OptionalBridgedBasicBlock defaultBlock,
                                        const void * _Nullable enumCases, SwiftInt numEnumCases) const {
  return {unbridged().createSwitchEnum(regularLoc(),
                                       enumVal.getSILValue(),
                                       defaultBlock.unbridged(),
                                       convertCases(enumVal.getSILValue()->getType(), enumCases, numEnumCases))};
}

BridgedInstruction BridgedBuilder::createSwitchEnumAddrInst(BridgedValue enumAddr,
                                                            OptionalBridgedBasicBlock defaultBlock,
                                                            const void * _Nullable enumCases,
                                                            SwiftInt numEnumCases) const {
  return {unbridged().createSwitchEnumAddr(regularLoc(),
                                           enumAddr.getSILValue(),
                                           defaultBlock.unbridged(),
                                           convertCases(enumAddr.getSILValue()->getType(), enumCases, numEnumCases))};
}

//===----------------------------------------------------------------------===//
//                             BridgedCloner
//===----------------------------------------------------------------------===//

// Need to put ClonerWithFixedLocation into namespace swift to forward reference
// it in SILBridging.h.
namespace swift {

class ClonerWithFixedLocation : public SILCloner<ClonerWithFixedLocation> {
  friend class SILInstructionVisitor<ClonerWithFixedLocation>;
  friend class SILCloner<ClonerWithFixedLocation>;

  SILDebugLocation insertLoc;

public:
  ClonerWithFixedLocation(SILGlobalVariable *gVar)
  : SILCloner<ClonerWithFixedLocation>(gVar),
  insertLoc(ArtificialUnreachableLocation(), nullptr) {}

  ClonerWithFixedLocation(SILInstruction *insertionPoint)
  : SILCloner<ClonerWithFixedLocation>(*insertionPoint->getFunction()),
  insertLoc(insertionPoint->getDebugLocation()) {
    Builder.setInsertionPoint(insertionPoint);
  }

  SILValue getClonedValue(SILValue v) {
    return getMappedValue(v);
  }

  void cloneInst(SILInstruction *inst) {
    visit(inst);
  }

protected:

  SILLocation remapLocation(SILLocation loc) {
    return insertLoc.getLocation();
  }

  const SILDebugScope *remapScope(const SILDebugScope *DS) {
    return insertLoc.getScope();
  }
};

} // namespace swift

BridgedCloner::BridgedCloner(BridgedGlobalVar var, BridgedContext context)
  : cloner(new ClonerWithFixedLocation(var.getGlobal())) {
  context.context->notifyNewCloner();
}

BridgedCloner::BridgedCloner(BridgedInstruction inst,
                             BridgedContext context)
    : cloner(new ClonerWithFixedLocation(inst.unbridged())) {
  context.context->notifyNewCloner();
}

void BridgedCloner::destroy(BridgedContext context) {
  delete cloner;
  cloner = nullptr;
  context.context->notifyClonerDestroyed();
}

BridgedValue BridgedCloner::getClonedValue(BridgedValue v) {
  return {cloner->getClonedValue(v.getSILValue())};
}

bool BridgedCloner::isValueCloned(BridgedValue v) const {
  return cloner->isValueCloned(v.getSILValue());
}

void BridgedCloner::clone(BridgedInstruction inst) {
  cloner->cloneInst(inst.unbridged());
}

void BridgedCloner::recordFoldedValue(BridgedValue origValue, BridgedValue mappedValue) {
  cloner->recordFoldedValue(origValue.getSILValue(), mappedValue.getSILValue());
}

//===----------------------------------------------------------------------===//
//                               BridgedContext
//===----------------------------------------------------------------------===//

BridgedOwnedString BridgedContext::getModuleDescription() const {
  std::string str;
  llvm::raw_string_ostream os(str);
  context->getModule()->print(os);
  str.pop_back(); // Remove trailing newline.
  return BridgedOwnedString(str);
}

OptionalBridgedFunction BridgedContext::lookUpNominalDeinitFunction(BridgedDeclObj nominal)  const {
  return {context->getModule()->lookUpMoveOnlyDeinitFunction(nominal.getAs<swift::NominalTypeDecl>())};
}

BridgedFunction BridgedContext::
createEmptyFunction(BridgedStringRef name,
                    const BridgedParameterInfo * _Nullable bridgedParams,
                    SwiftInt paramCount,
                    bool hasSelfParam,
                    BridgedFunction fromFunc) const {
  llvm::SmallVector<SILParameterInfo> params;
  for (unsigned idx = 0; idx < paramCount; ++idx) {
    params.push_back(bridgedParams[idx].unbridged());
  }
  return {context->createEmptyFunction(name.unbridged(), params, hasSelfParam, fromFunc.getFunction())};
}

BridgedGlobalVar BridgedContext::createGlobalVariable(BridgedStringRef name, BridgedType type,
                                                      BridgedLinkage linkage,
                                                      bool isLet,
                                                      bool markedAsUsed) const {
  auto *global = SILGlobalVariable::create(
      *context->getModule(),
      (swift::SILLinkage)linkage, IsNotSerialized,
      name.unbridged(), type.unbridged());
  if (isLet)
    global->setLet(true);
  global->setMarkedAsUsed(markedAsUsed);
  return {global};
}

void BridgedContext::moveFunctionBody(BridgedFunction sourceFunc, BridgedFunction destFunc) const {
  context->moveFunctionBody(sourceFunc.getFunction(), destFunc.getFunction());
}

//===----------------------------------------------------------------------===//
//                           SILContext
//===----------------------------------------------------------------------===//

SILContext::~SILContext() {}

void SILContext::verifyEverythingIsCleared() {
  ASSERT(allocatedSlabs.empty() && "StackList is leaking slabs");
  ASSERT(numBlockSetsAllocated == 0 && "Not all BasicBlockSets deallocated");
  ASSERT(numNodeSetsAllocated == 0 && "Not all NodeSets deallocated");
  ASSERT(numOperandSetsAllocated == 0 && "Not all OperandSets deallocated");
  ASSERT(numClonersAllocated == 0 && "Not all cloners deallocated");
}

FixedSizeSlab *SILContext::allocSlab(FixedSizeSlab *afterSlab) {
  FixedSizeSlab *slab = getModule()->allocSlab();
  if (afterSlab) {
    allocatedSlabs.insert(std::next(afterSlab->getIterator()), *slab);
  } else {
    allocatedSlabs.push_back(*slab);
  }
  return slab;
}

FixedSizeSlab *SILContext::freeSlab(FixedSizeSlab *slab) {
  FixedSizeSlab *prev = nullptr;
  assert(!allocatedSlabs.empty());
  if (&allocatedSlabs.front() != slab)
    prev = &*std::prev(slab->getIterator());

  allocatedSlabs.remove(*slab);
  getModule()->freeSlab(slab);
  return prev;
}

BasicBlockSet *SILContext::allocBlockSet() {
  ASSERT(numBlockSetsAllocated < BlockSetCapacity &&
         "too many BasicBlockSets allocated");

  auto *storage = (BasicBlockSet *)blockSetStorage + numBlockSetsAllocated;
  BasicBlockSet *set = new (storage) BasicBlockSet(function);
  aliveBlockSets[numBlockSetsAllocated] = true;
  ++numBlockSetsAllocated;
  return set;
}

void SILContext::freeBlockSet(BasicBlockSet *set) {
  int idx = set - (BasicBlockSet *)blockSetStorage;
  assert(idx >= 0 && idx < numBlockSetsAllocated);
  assert(aliveBlockSets[idx] && "double free of BasicBlockSet");
  aliveBlockSets[idx] = false;

  while (numBlockSetsAllocated > 0 && !aliveBlockSets[numBlockSetsAllocated - 1]) {
    auto *set = (BasicBlockSet *)blockSetStorage + numBlockSetsAllocated - 1;
    set->~BasicBlockSet();
    --numBlockSetsAllocated;
  }
}

NodeSet *SILContext::allocNodeSet() {
  ASSERT(numNodeSetsAllocated < NodeSetCapacity &&
         "too many NodeSets allocated");

  auto *storage = (NodeSet *)nodeSetStorage + numNodeSetsAllocated;
  NodeSet *set = new (storage) NodeSet(function);
  aliveNodeSets[numNodeSetsAllocated] = true;
  ++numNodeSetsAllocated;
  return set;
}

void SILContext::freeNodeSet(NodeSet *set) {
  int idx = set - (NodeSet *)nodeSetStorage;
  assert(idx >= 0 && idx < numNodeSetsAllocated);
  assert(aliveNodeSets[idx] && "double free of NodeSet");
  aliveNodeSets[idx] = false;

  while (numNodeSetsAllocated > 0 && !aliveNodeSets[numNodeSetsAllocated - 1]) {
    auto *set = (NodeSet *)nodeSetStorage + numNodeSetsAllocated - 1;
    set->~NodeSet();
    --numNodeSetsAllocated;
  }
}

OperandSet *SILContext::allocOperandSet() {
  ASSERT(numOperandSetsAllocated < OperandSetCapacity &&
         "too many OperandSets allocated");

  auto *storage = (OperandSet *)operandSetStorage + numOperandSetsAllocated;
  OperandSet *set = new (storage) OperandSet(function);
  aliveOperandSets[numOperandSetsAllocated] = true;
  ++numOperandSetsAllocated;
  return set;
}

void SILContext::freeOperandSet(OperandSet *set) {
  int idx = set - (OperandSet *)operandSetStorage;
  assert(idx >= 0 && idx < numOperandSetsAllocated);
  assert(aliveOperandSets[idx] && "double free of OperandSet");
  aliveOperandSets[idx] = false;

  while (numOperandSetsAllocated > 0 && !aliveOperandSets[numOperandSetsAllocated - 1]) {
    auto *set = (OperandSet *)operandSetStorage + numOperandSetsAllocated - 1;
    set->~OperandSet();
    --numOperandSetsAllocated;
  }
}

//===----------------------------------------------------------------------===//
//                           BridgedVerifier
//===----------------------------------------------------------------------===//

static BridgedVerifier::VerifyFunctionFn verifyFunctionFunction = nullptr;

void BridgedVerifier::registerVerifier(VerifyFunctionFn verifyFunctionFn) {
  verifyFunctionFunction = verifyFunctionFn;
}

void BridgedVerifier::runSwiftFunctionVerification(SILFunction * _Nonnull f, SILContext * _Nonnull context) {
  if (!verifyFunctionFunction)
    return;

  verifyFunctionFunction({context}, {f});
}

void BridgedVerifier::verifierError(BridgedStringRef message,
                                    OptionalBridgedInstruction atInstruction,
                                    OptionalBridgedArgument atArgument) {
  Twine msg(message.unbridged());
  verificationFailure(msg, atInstruction.unbridged(), atArgument.unbridged(), /*extraContext=*/nullptr);
}
