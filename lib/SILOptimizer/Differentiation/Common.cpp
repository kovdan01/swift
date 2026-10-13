//===--- Common.cpp - Automatic differentiation common utils --*- C++ -*---===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2019 - 2020 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//
//
// Automatic differentiation common utilities.
//
//===----------------------------------------------------------------------===//

#include "swift/SILOptimizer/Differentiation/Common.h"
#include "swift/AST/AutoDiff.h"
#include "swift/AST/Builtins.h"
#include "swift/AST/SubstitutionMap.h"
#include "swift/SILOptimizer/Analysis/DominanceAnalysis.h"
#include "llvm/Support/CommandLine.h"
#include "swift/SIL/SILDifferentiabilityWitness.h"

#define DEBUG_TYPE "differentiation"

#include "swift/AST/SemanticAttrs.h"
#include "swift/AST/TypeCheckRequests.h"
#include "swift/Basic/Assertions.h"
#include "swift/Basic/STLExtras.h"
#include "swift/SIL/ApplySite.h"
#include "swift/SILOptimizer/Analysis/ArraySemantic.h"
#include "swift/SILOptimizer/Analysis/DifferentiableActivityAnalysis.h"
#include "swift/SILOptimizer/Differentiation/ADContext.h"

namespace swift {
namespace autodiff {

raw_ostream &getADDebugStream() { return llvm::dbgs() << "[AD] "; }

// Given a pointer_to_address instruction, return corresponding
// startAsyncLetWithLocalBuffer provided that input instruction
// is the address of the async let result buffer (so, the last operand)
BuiltinInst *getAsyncLetStart(const PointerToAddressInst *pta) {
  for (auto use : pta->getOperand()->getUses()) {
    auto *bi = dyn_cast<BuiltinInst>(use->getUser());
    if (!bi)
      continue;

    if (bi->getBuiltinKind() ==
            BuiltinValueKind::StartAsyncLetWithLocalBuffer &&
        use->getOperandNumber() == bi->getNumOperands() - 1)
      return bi;
  }

  return nullptr;
}

bool shouldDifferentiateAsyncLet(
    BuiltinInst *start, const DifferentiableActivityInfo &activity,
    const AutoDiffConfig &config) {
  return start->getBuiltinKind() ==
             BuiltinValueKind::StartAsyncLetWithLocalBuffer &&
         activity.isActive(start->getArguments().back(), config);
}

static llvm::cl::opt<bool> EnableAsyncAutoDiff(
    "enable-experimental-async-autodiff", llvm::cl::Hidden,
    llvm::cl::init(false));

bool isExperimentalAsyncAutoDiffEnabled() {
  return EnableAsyncAutoDiff;
}

AllocStackInst *AsyncLetInfo::getBuffer() const {
  auto *pointer = cast<AddressToPointerInst>(start->getArguments()[2]);
  return cast<AllocStackInst>(pointer->getOperand());
}

IndexSubset *AsyncLetInfo::getParameterIndices() const {
  auto &function = *start->getFunction();
  auto captures = initializer->getArguments();
  SmallVector<unsigned, 4> indices;
  for (unsigned i = 0; i != captures.size(); ++i)
    if (captures[i]->getType().isDifferentiable(function.getModule()))
      indices.push_back(i);
  // Preserve capture positions, including gaps for nondifferentiable values.
  return IndexSubset::get(function.getASTContext(), captures.size(), indices);
}

Type getLinearMapASTType(CanSILFunctionType silFnTy) {
  SmallVector<AnyFunctionType::Param, 8> params;
  for (auto &param : silFnTy->getParameters()) {
    ParameterTypeFlags flags;
    if (param.isAutoDiffSemanticResult())
      flags = flags.withInOut(true);
    params.push_back(
        AnyFunctionType::Param(param.getInterfaceType(), Identifier(), flags));
  }
  AnyFunctionType *astFnTy;
  if (auto genSig = silFnTy->getSubstGenericSignature()) {
    // FIXME: Verify ExtInfo state is correct, not working by accident.
    GenericFunctionType::ExtInfo info;
    astFnTy = GenericFunctionType::get(
        genSig, params, /* yields */ {},
        silFnTy->getAllResultsInterfaceType().getASTType(), info);
  } else {
    FunctionType::ExtInfo info;
    astFnTy = FunctionType::get(
        params, /* yields */ {},
        silFnTy->getAllResultsInterfaceType().getASTType(), info);
  }
  Type resultType =
      astFnTy->hasArchetype() ? astFnTy->mapTypeOutOfEnvironment() : astFnTy;
  return resultType;
}

Type AsyncLetInfo::getPullbackType() const {
  auto &function = *start->getFunction();
  auto &module = function.getModule();
  auto *child = cast<FunctionRefInst>(initializer->getCallee())
                    ->getReferencedFunction();
  auto *results = IndexSubset::get(function.getASTContext(), 1, {0});
  auto *witness = getExactDifferentiabilityWitness(
      module, child, getParameterIndices(), results);
  assert(witness && witness->getVJP() && "prepare the child VJP first");
  // Ordinary AD is the authority for the seed and gradient types. Do not
  // reconstruct a parallel signature by looking up tangent types here.
  auto derivativeType = witness->getVJP()->getLoweredFunctionType()
                            ->getUnsubstitutedType(module);
  auto pullbackType = derivativeType->getResults().back()
                          .getSILStorageInterfaceType()
                          .castTo<SILFunctionType>()
                          ->getUnsubstitutedType(module);
  assert(!pullbackType->isAsync() && !pullbackType->hasErrorResult() &&
         !pullbackType->getSubstGenericSignature());
  // The closure itself is owned. Only its numerical interface is restricted
  // to trivial values by this prototype.
  for (auto parameter : pullbackType->getParameters())
    assert(parameter.getSILStorageInterfaceType().getObjectType()
               .isTrivial(function) && "nontrivial async-let seed");
  for (auto result : pullbackType->getResults())
    assert(result.getSILStorageInterfaceType().getObjectType()
               .isTrivial(function) && "nontrivial async-let gradient");
  return getLinearMapASTType(pullbackType);
}

bool AsyncLetInfo::isInitializerInstruction(SILInstruction *inst) const {
  SILValue value = start->getArguments()[1];
  while (true) {
    if (value->getDefiningInstruction() == inst)
      return true;
    if (value == initializer)
      return false;
    // findAsyncLet has checked the conversion chain.
    value = value->getDefiningInstruction()->getOperand(0);
  }
}

bool AsyncLetInfo::isGet(SILInstruction *inst) const {
  return llvm::any_of(gets, [&](ApplyInst *get) { return inst == get; });
}

bool AsyncLetInfo::isResultAddress(SILInstruction *inst) const {
  auto *address = dyn_cast_or_null<PointerToAddressInst>(inst);
  return address && getAsyncLetStart(address) == start;
}

// Do not classify the CFG as a particular diamond or loop shape. SILGen has
// already established the structured task lifetime, which the forward cloner
// preserves instruction for instruction. Only validate the use/def associations
// needed to change the result type. Ordinary AD owns control-flow history.
static AsyncLetInfo findAsyncLet(BuiltinInst *start,
                                 DominanceInfo &dominance) {
  auto &function = *start->getFunction();
  AsyncLetInfo a;
  a.start = start;

  assert(function.isAsync());
  assert(!function.isSerialized() &&
         !function.getActorIsolation().isActorIsolated());
  auto type = function.getLoweredFunctionType();
  assert(!type->getSubstGenericSignature() && !type->hasErrorResult());
  // Parent parameter/result conventions are handled by ordinary AD.
  assert(a.start->getArguments().size() == 3);
  assert(a.start->getSubstitutions().getReplacementTypes().size() == 1 &&
         a.start->getSubstitutions().getConformances().empty());
  auto *buffer = a.getBuffer();
  assert(buffer->getType().getObjectType().isTrivial(function) &&
         "only concrete trivial async-let results are implemented");

  SILValue closure = a.start->getArguments()[1];
  SILInstruction *consumer = a.start;
  while (true) {
    // The converted closure is private to this start, apart from cleanup.
    for (auto *use : closure->getUses())
      assert((use->getUser() == consumer ||
              isa<DestroyValueInst, DebugValueInst>(use->getUser()) ||
              isBuiltinInst(use->getUser(),
                            BuiltinValueKind::EndAsyncLetLifetime)) &&
             "shared initializer closure");
    if ((a.initializer = dyn_cast<PartialApplyInst>(closure)))
      break;
    consumer = closure->getDefiningInstruction();
    assert((isa<ConvertFunctionInst, ConvertEscapeToNoEscapeInst>(consumer)));
    closure = consumer->getOperand(0);
  }
  assert(a.initializer->getSubstitutionMap().empty());
  auto captures = a.initializer->getArguments();
  // Trivial control values are forwarded too; only values with a tangent
  // space are differentiation parameters. There is no capture-count limit.
  for (auto capture : captures)
    assert(capture->getType().isObject() &&
           capture->getType().isTrivial(function) &&
           "only by-value trivial async-let captures are implemented");
  assert(isa<FunctionRefInst>(a.initializer->getCallee()));
  assert(dominance.dominates(buffer, a.initializer) &&
         dominance.dominates(a.initializer, start));

  SmallVector<SILInstruction *, 2> finishes, ends, deallocs;
  SmallVector<PointerToAddressInst *, 2> resultAddresses;
  auto pointer = a.start->getArguments()[2];
  for (auto *use : pointer->getUses()) {
    auto *inst = use->getUser();
    if (inst == a.start)
      continue;
    if (auto *addr = dyn_cast<PointerToAddressInst>(inst)) {
      assert(getAsyncLetStart(addr) == start &&
             "inconsistent async-let result association");
      assert(addr->getType() == buffer->getType());
      // An aggregate can be read whole or through typed field projections.
      // The existing memory/projection visitors differentiate these accesses.
      SmallVector<SILValue, 4> addresses{addr};
      while (!addresses.empty()) {
        auto address = addresses.pop_back_val();
        for (auto *addressUse : address->getUses()) {
          auto *user = addressUse->getUser();
          if (isa<DebugValueInst>(user))
            continue;
          if (auto *field = dyn_cast<StructElementAddrInst>(user)) {
            addresses.push_back(field);
          } else if (auto *element = dyn_cast<TupleElementAddrInst>(user)) {
            addresses.push_back(element);
          } else if (auto *copy = dyn_cast<CopyAddrInst>(user)) {
            assert(copy->getSrc() == address && !copy->isTakeOfSrc() &&
                   "expected a nonconsuming async-let result copy");
          } else {
            auto *load = cast<LoadInst>(user);
            assert(load->getOwnershipQualifier() ==
                   LoadOwnershipQualifier::Trivial);
          }
        }
      }
      resultAddresses.push_back(addr);
    } else if (auto *apply = dyn_cast<ApplyInst>(inst)) {
      auto *callee = apply->getCalleeFunction();
      assert(callee && apply->getArguments().size() == 2 &&
             apply->getArguments()[0] == a.start &&
             apply->getArguments()[1] == pointer);
      if (callee->getName() == "swift_asyncLet_get")
        a.gets.push_back(apply);
      else {
        assert(callee->getName() == "swift_asyncLet_finish");
        finishes.push_back(apply);
      }
    } else {
      assert(isBuiltinInst(inst, BuiltinValueKind::FinishAsyncLet));
      auto args = cast<BuiltinInst>(inst)->getArguments();
      assert(args.size() == 2 && args[0] == a.start && args[1] == pointer);
      finishes.push_back(inst);
    }
  }
  for (auto *use : a.start->getUses()) {
    auto *inst = use->getUser();
    if (a.isGet(inst) || llvm::is_contained(finishes, inst))
      continue;
    assert(isBuiltinInst(inst, BuiltinValueKind::EndAsyncLetLifetime));
    ends.push_back(inst);
  }
  for (auto *use : buffer->getUses()) {
    auto *inst = use->getUser();
    if (inst == pointer->getDefiningInstruction() || isa<DebugValueInst>(inst))
      continue;
    deallocs.push_back(cast<DeallocStackInst>(inst));
  }
  assert(!finishes.empty() && !ends.empty() && !deallocs.empty());
  assert(resultAddresses.empty() || !a.gets.empty());
  // These are SSA dominance checks, not a second task-lifetime or AD-loop
  // analysis. In particular, waits in alternative arms can precede a joined
  // result read, and cleanup-only paths need not contain a numerical read.
  for (auto *get : a.gets)
    assert(dominance.dominates(start, get));
  for (auto *address : resultAddresses)
    assert(dominance.dominates(start, address));
  for (auto *finish : finishes)
    assert(dominance.dominates(start, finish));
  for (auto *end : ends)
    assert(dominance.dominates(start, end));
  for (auto *dealloc : deallocs)
    assert(dominance.dominates(buffer, dealloc));
  LLVM_DEBUG({
    getADDebugStream() << "Matched trivial async let with " << a.gets.size()
                       << " wait(s), " << resultAddresses.size()
                       << " result address(es), "
                       << a.getParameterIndices()->getNumIndices()
                       << " differentiable capture(s):\n"
                       << *a.start << *a.initializer;
  });
  return a;
}

SmallVector<AsyncLetInfo, 2> findAsyncLets(
    SILFunction &function, const DifferentiableActivityInfo &activity,
    const AutoDiffConfig &config) {
  SmallVector<AsyncLetInfo, 2> result;
  if (!EnableAsyncAutoDiff)
    return result;
  SmallVector<BuiltinInst *, 4> starts;
  for (auto &bb : function)
    for (auto &inst : bb)
      if (auto *start = dyn_cast<BuiltinInst>(&inst);
          start && shouldDifferentiateAsyncLet(start, activity, config))
        starts.push_back(start);
  if (starts.empty())
    return result;
  for (auto &bb : function)
    for (auto &inst : bb)
      if (auto *hop = dyn_cast<HopToExecutorInst>(&inst))
        assert(isa<EnumInst>(hop->getTargetExecutor()) &&
               !cast<EnumInst>(hop->getTargetExecutor())->hasOperand());
  DominanceInfo dominance(&function);
  for (auto *start : starts)
    result.push_back(findAsyncLet(start, dominance));
  return result;
}

const AsyncLetInfo *getAsyncLet(
    ArrayRef<AsyncLetInfo> asyncLets, SILInstruction *inst) {
  if (!inst)
    return nullptr;
  // Reuse the PR's association rather than duplicating its pointer-use walk.
  if (auto *address = dyn_cast<PointerToAddressInst>(inst))
    inst = getAsyncLetStart(address);
  if (!inst)
    return nullptr;
  for (const auto &a : asyncLets)
    if (inst == a.start || a.isGet(inst) || inst == a.getBuffer() ||
        inst == a.start->getArguments()[2]->getDefiningInstruction() ||
        a.isInitializerInstruction(inst))
      return &a;
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static SILValue getArrayValueOfElementAddress(SILValue v) {
  while (true) {
    switch (v->getKind()) {
    case ValueKind::IndexAddrInst:
    case ValueKind::RefTailAddrInst:
    case ValueKind::UncheckedRefCastInst:
    case ValueKind::StructExtractInst:
    case ValueKind::BeginBorrowInst:
      v = cast<SingleValueInstruction>(v)->getOperand(0);
      break;
    default:
      return v;
    }
  }
}

ApplyInst *getAllocateUninitializedArrayIntrinsicElementAddress(SILValue v) {
  SILValue arr = getArrayValueOfElementAddress(v);

  auto *mvir = dyn_cast<MultipleValueInstructionResult>(arr);
  if (!mvir)
    return nullptr;

  // Return the `array.uninitialized_intrinsic` application, if it exists.
  if (auto *dti = dyn_cast<DestructureTupleInst>(mvir->getParent()))
    return ArraySemanticsCall(dti->getOperand(),
                              semantics::ARRAY_UNINITIALIZED_INTRINSIC);
  return nullptr;
}

bool isSemanticMemberAccessor(SILFunction *original) {
  auto *dc = original->getDeclContext();
  if (!dc)
    return false;
  auto *decl = dc->getAsDecl();
  if (!decl)
    return false;
  auto *accessor = dyn_cast<AccessorDecl>(decl);
  if (!accessor)
    return false;
  // Currently, only getters, setters and _modify accessors are supported.
  if (accessor->getAccessorKind() != AccessorKind::Get &&
      accessor->getAccessorKind() != AccessorKind::Set &&
      accessor->getAccessorKind() != AccessorKind::Modify)
    return false;
  // Accessor must come from a `var` declaration.
  auto *varDecl = dyn_cast<VarDecl>(accessor->getStorage());
  if (!varDecl)
    return false;
  // Return true for stored property accessors.
  if (varDecl->hasStorage() && varDecl->isInstanceMember())
    return true;
  // Return true for properties that have attached property wrappers.
  if (varDecl->hasAttachedPropertyWrapper())
    return true;
  // Otherwise, return false.
  // User-defined accessors can never be supported because they may use custom
  // logic that does not semantically perform a member access.
  return false;
}

bool hasSemanticMemberAccessorCallee(ApplySite applySite) {
  if (auto *FRI = dyn_cast<FunctionRefBaseInst>(applySite.getCallee()))
    if (auto *F = FRI->getReferencedFunctionOrNull())
      return isSemanticMemberAccessor(F);
  return false;
}

void forEachApplyDirectResult(
    FullApplySite applySite,
    llvm::function_ref<void(SILValue)> resultCallback) {
  switch (applySite.getKind()) {
  case FullApplySiteKind::ApplyInst: {
    auto *ai = cast<ApplyInst>(applySite.getInstruction());
    if (!ai->getType().is<TupleType>()) {
      resultCallback(ai);
      return;
    }
    if (auto *dti = ai->getSingleUserOfType<DestructureTupleInst>())
      for (auto directResult : dti->getResults())
        resultCallback(directResult);
    break;
  }
  case FullApplySiteKind::BeginApplyInst: {
    auto *bai = cast<BeginApplyInst>(applySite.getInstruction());
    for (auto directResult : bai->getResults())
      resultCallback(directResult);
    break;
  }
  case FullApplySiteKind::TryApplyInst: {
    auto *tai = cast<TryApplyInst>(applySite.getInstruction());
    for (auto *succBB : tai->getSuccessorBlocks())
      for (auto *arg : succBB->getArguments())
        resultCallback(arg);
    break;
  }
  }
}

void collectAllFormalResultsInTypeOrder(SILFunction &function,
                                        SmallVectorImpl<SILValue> &results) {
  SILFunctionConventions convs = function.getConventions();
  auto indResults = function.getIndirectResults();
  auto *retInst = cast<ReturnInst>(function.findReturnBB()->getTerminator());
  auto retVal = retInst->getOperand();
  SmallVector<SILValue, 8> dirResults;
  if (auto *tupleInst =
          dyn_cast_or_null<TupleInst>(retVal->getDefiningInstruction()))
    dirResults.append(tupleInst->getElements().begin(),
                      tupleInst->getElements().end());
  else
    dirResults.push_back(retVal);
  unsigned indResIdx = 0, dirResIdx = 0;
  for (auto &resInfo : convs.getResults())
    results.push_back(resInfo.isFormalDirect() ? dirResults[dirResIdx++]
                                               : indResults[indResIdx++]);
  // Treat semantic result parameters as semantic results.
  // Append them` parameters after formal results.
  for (auto i : range(convs.getNumParameters())) {
    auto paramInfo = convs.getParameters()[i];
    if (!paramInfo.isAutoDiffSemanticResult())
      continue;
    auto *argument = function.getArgumentsWithoutIndirectResults()[i];
    results.push_back(argument);
  }
  // Treat yields as semantic results. Note that we can only differentiate
  // @yield_once with simple control flow, so we can assume that the function
  // contains only a single `yield` instruction
  auto yieldIt =
    std::find_if(function.begin(), function.end(),
      [](const SILBasicBlock &BB) -> bool {
        const TermInst *TI = BB.getTerminator();
        return isa<YieldInst>(TI);
    });
  if (yieldIt != function.end()) {
    auto *yieldInst = cast<YieldInst>(yieldIt->getTerminator());
    for (auto yield : yieldInst->getOperandValues())
      results.push_back(yield);
  }
}

void collectAllDirectResultsInTypeOrder(SILFunction &function,
                                        SmallVectorImpl<SILValue> &results) {
  auto *retInst = cast<ReturnInst>(function.findReturnBB()->getTerminator());
  auto retVal = retInst->getOperand();
  if (auto *tupleInst = dyn_cast<TupleInst>(retVal))
    results.append(tupleInst->getElements().begin(),
                   tupleInst->getElements().end());
  else
    results.push_back(retVal);
}

void collectAllActualResultsInTypeOrder(
    FullApplySite fai, ArrayRef<SILValue> extractedDirectResults,
    SmallVectorImpl<SILValue> &results) {
  auto calleeConvs = fai.getSubstCalleeConv();
  unsigned indResIdx = 0, dirResIdx = 0;
  for (auto &resInfo : calleeConvs.getResults()) {
    results.push_back(resInfo.isFormalDirect()
                          ? extractedDirectResults[dirResIdx++]
                          : fai.getIndirectSILResults()[indResIdx++]);
  }
}

void collectMinimalIndicesForFunctionCall(
    FullApplySite ai, const AutoDiffConfig &parentConfig,
    const DifferentiableActivityInfo &activityInfo,
    SmallVectorImpl<SILValue> &results, SmallVectorImpl<unsigned> &paramIndices,
    SmallVectorImpl<unsigned> &resultIndices) {
  auto calleeFnTy = ai.getSubstCalleeType();
  auto calleeConvs = ai.getSubstCalleeConv();

  // Parameter indices are indices (in the callee type signature) of parameter
  // arguments that are varied or are arguments.
  // Record all parameter indices in type order.
  unsigned currentParamIdx = 0;
  for (auto applyArg : ai.getArgumentsWithoutIndirectResults()) {
    if (activityInfo.isActive(applyArg, parentConfig))
      paramIndices.push_back(currentParamIdx);
    ++currentParamIdx;
  }

  // Result indices are indices (in the callee type signature) of results that
  // are useful.
  SmallVector<SILValue, 8> directResults;
  forEachApplyDirectResult(ai, [&](SILValue directResult) {
    directResults.push_back(directResult);
  });
  auto indirectResults = ai.getIndirectSILResults();
  // Record all results and result indices in type order.
  results.reserve(calleeFnTy->getNumResults());
  unsigned dirResIdx = 0;
  unsigned indResIdx = calleeConvs.getSILArgIndexOfFirstIndirectResult();
  for (const auto &resAndIdx : enumerate(calleeConvs.getResults())) {
    const auto &res = resAndIdx.value();
    unsigned idx = resAndIdx.index();
    if (res.isFormalDirect()) {
      results.push_back(directResults[dirResIdx]);
      if (auto dirRes = directResults[dirResIdx])
        if (dirRes && activityInfo.isActive(dirRes, parentConfig))
          resultIndices.push_back(idx);
      ++dirResIdx;
    } else {
      results.push_back(indirectResults[indResIdx]);
      if (activityInfo.isActive(indirectResults[indResIdx], parentConfig))
        resultIndices.push_back(idx);
      ++indResIdx;
    }
  }
  
  // Record all semantic result parameters as results.
  auto semanticResultParamResultIndex = calleeFnTy->getNumResults();
  for (const auto &paramAndIdx : enumerate(calleeConvs.getParameters())) {
    const auto &param = paramAndIdx.value();
    if (!param.isAutoDiffSemanticResult())
      continue;
    unsigned idx = paramAndIdx.index() + calleeFnTy->getNumIndirectFormalResults();
    results.push_back(ai.getArgument(idx));
    resultIndices.push_back(semanticResultParamResultIndex++);
  }

  // Record all yields. While we do not have a way to represent direct yields
  // (_read accessors) we run activity analysis for them. These will be
  // diagnosed later.
  if (BeginApplyInst *bai = dyn_cast<BeginApplyInst>(*ai)) {
    for (const auto &yieldAndIdx : enumerate(calleeConvs.getYields())) {
      results.push_back(bai->getYieldedValues()[yieldAndIdx.index()]);
      resultIndices.push_back(semanticResultParamResultIndex++);
    }
  }

  // Make sure the function call has active results.
#ifndef NDEBUG
  assert(results.size() == calleeFnTy->getNumAutoDiffSemanticResults());
  assert(llvm::any_of(results, [&](SILValue result) {
    return activityInfo.isActive(result, parentConfig);
  }));
#endif
}

std::optional<std::pair<SILDebugLocation, SILDebugVariable>>
findDebugLocationAndVariable(SILValue originalValue) {
  if (auto *asi = dyn_cast<AllocStackInst>(originalValue))
    return swift::transform(asi->getVarInfo(false),  [&](SILDebugVariable var) {
      return std::make_pair(asi->getDebugLocation(), var);
    });
  for (auto *use : originalValue->getUses()) {
    if (auto *dvi = dyn_cast<DebugValueInst>(use->getUser()))
      return swift::transform(dvi->getVarInfo(false), [&](SILDebugVariable var) {
        // We need to drop `op_deref` here as we're transferring debug info
        // location from debug_value instruction (which describes how to get value)
        // into alloc_stack (which describes the location)
        if (var.DIExpr.startsWithDeref())
          var.DIExpr.eraseElement(var.DIExpr.element_begin());
        return std::make_pair(dvi->getDebugLocation(), var);
      });
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Diagnostic utilities
//===----------------------------------------------------------------------===//

SILLocation getValidLocation(SILValue v) {
  auto loc = v.getLoc();
  if (loc.isNull() || loc.getSourceLoc().isInvalid())
    loc = v->getFunction()->getLocation();
  return loc;
}

SILLocation getValidLocation(SILInstruction *inst) {
  auto loc = inst->getLoc();
  if (loc.isNull() || loc.getSourceLoc().isInvalid())
    loc = inst->getFunction()->getLocation();
  return loc;
}

//===----------------------------------------------------------------------===//
// Tangent property lookup utilities
//===----------------------------------------------------------------------===//

VarDecl *getTangentStoredProperty(ADContext &context, VarDecl *originalField,
                                  CanType baseType, SILLocation loc,
                                  DifferentiationInvoker invoker) {
  auto &astCtx = context.getASTContext();
  auto tanFieldInfo = evaluateOrDefault(
      astCtx.evaluator, TangentStoredPropertyRequest{originalField, baseType},
      TangentPropertyInfo(nullptr));
  // If no error, return the tangent property.
  if (tanFieldInfo)
    return tanFieldInfo.tangentProperty;
  // Otherwise, diagnose error and return nullptr.
  assert(tanFieldInfo.error);
  auto *parentDC = originalField->getDeclContext();
  assert(parentDC->isTypeContext());
  auto parentDeclName = parentDC->getSelfNominalTypeDecl()->getNameStr();
  auto fieldName = originalField->getNameStr();
  auto sourceLoc = loc.getSourceLoc();
  switch (tanFieldInfo.error->kind) {
  case TangentPropertyInfo::Error::Kind::NoDerivativeOriginalProperty:
    llvm_unreachable(
        "`@noDerivative` stored property accesses should not be "
        "differentiated; activity analysis should not mark as varied");
  case TangentPropertyInfo::Error::Kind::NominalParentNotDifferentiable:
    context.emitNondifferentiabilityError(
        sourceLoc, invoker,
        diag::autodiff_stored_property_parent_not_differentiable,
        parentDeclName, fieldName);
    break;
  case TangentPropertyInfo::Error::Kind::OriginalPropertyNotDifferentiable:
    context.emitNondifferentiabilityError(
        sourceLoc, invoker, diag::autodiff_stored_property_not_differentiable,
        parentDeclName, fieldName, originalField->getInterfaceType());
    break;
  case TangentPropertyInfo::Error::Kind::ParentTangentVectorNotStruct:
    context.emitNondifferentiabilityError(
        sourceLoc, invoker, diag::autodiff_stored_property_tangent_not_struct,
        parentDeclName, fieldName);
    break;
  case TangentPropertyInfo::Error::Kind::TangentPropertyNotFound:
    context.emitNondifferentiabilityError(
        sourceLoc, invoker,
        diag::autodiff_stored_property_no_corresponding_tangent, parentDeclName,
        fieldName);
    break;
  case TangentPropertyInfo::Error::Kind::TangentPropertyWrongType:
    context.emitNondifferentiabilityError(
        sourceLoc, invoker, diag::autodiff_tangent_property_wrong_type,
        parentDeclName, fieldName, tanFieldInfo.error->getType());
    break;
  case TangentPropertyInfo::Error::Kind::TangentPropertyNotStored:
    context.emitNondifferentiabilityError(
        sourceLoc, invoker, diag::autodiff_tangent_property_not_stored,
        parentDeclName, fieldName);
    break;
  }
  return nullptr;
}

VarDecl *getTangentStoredProperty(ADContext &context,
                                  SingleValueInstruction *projectionInst,
                                  CanType baseType,
                                  DifferentiationInvoker invoker) {
  assert(isa<StructExtractInst>(projectionInst) ||
         isa<StructElementAddrInst>(projectionInst) ||
         isa<RefElementAddrInst>(projectionInst));
  Projection proj(projectionInst);
  auto loc = getValidLocation(projectionInst);
  auto *field = proj.getVarDecl(projectionInst->getOperand(0)->getType());
  return getTangentStoredProperty(context, field, baseType,
                                  loc, invoker);
}

//===----------------------------------------------------------------------===//
// Code emission utilities
//===----------------------------------------------------------------------===//

SILValue joinElements(ArrayRef<SILValue> elements, SILBuilder &builder,
                      SILLocation loc) {
  if (elements.size() == 1)
    return elements.front();
  return builder.createTuple(loc, elements);
}

void extractAllElements(SILValue value, SILBuilder &builder,
                        SmallVectorImpl<SILValue> &results) {
  auto tupleType = value->getType().getAs<TupleType>();
  if (!tupleType) {
    results.push_back(value);
    return;
  }
  if (builder.hasOwnership()) {
    auto *dti = builder.createDestructureTuple(value.getLoc(), value);
    results.append(dti->getResults().begin(), dti->getResults().end());
    return;
  }
  for (auto i : range(tupleType->getNumElements()))
    results.push_back(builder.createTupleExtract(value.getLoc(), value, i));
}

SILValue emitMemoryLayoutSize(
    SILBuilder &builder, SILLocation loc, CanType type) {
  auto &ctx = builder.getASTContext();
  auto id = ctx.getIdentifier(getBuiltinName(BuiltinValueKind::Sizeof));
  auto *builtin = cast<FuncDecl>(getBuiltinValueDecl(ctx, id));
  auto metatypeTy = SILType::getPrimitiveObjectType(
      CanMetatypeType::get(type, MetatypeRepresentation::Thin));
  auto metatypeVal = builder.createMetatype(loc, metatypeTy);
  return builder.createBuiltin(
      loc, id, SILType::getBuiltinWordType(ctx),
      SubstitutionMap::get(
          builtin->getGenericSignature(), ArrayRef<Type>{type},
          LookUpConformanceInModule()),
      {metatypeVal});
}

SILValue emitProjectTopLevelSubcontext(
    SILBuilder &builder, SILLocation loc, SILValue context,
    SILType subcontextType) {
  assert(context->getOwnershipKind() == OwnershipKind::Guaranteed);
  auto &ctx = builder.getASTContext();
  auto id = ctx.getIdentifier(
      getBuiltinName(BuiltinValueKind::AutoDiffProjectTopLevelSubcontext));
  assert(context->getType() == SILType::getNativeObjectType(ctx));
  auto *subcontextAddr = builder.createBuiltin(
      loc, id, SILType::getRawPointerType(ctx), SubstitutionMap(), {context});
  return builder.createPointerToAddress(
      loc, subcontextAddr, subcontextType.getAddressType(), /*isStrict*/ true);
}

//===----------------------------------------------------------------------===//
// Utilities for looking up derivatives of functions
//===----------------------------------------------------------------------===//

/// Returns the AbstractFunctionDecl corresponding to `F`. If there isn't one,
/// returns `nullptr`.
AbstractFunctionDecl *findAbstractFunctionDecl(SILFunction *F) {
  auto *DC = F->getDeclContext();
  if (!DC)
    return nullptr;
  auto *D = DC->getAsDecl();
  if (!D)
    return nullptr;
  return dyn_cast<AbstractFunctionDecl>(D);
}

SILDifferentiabilityWitness *
getExactDifferentiabilityWitness(SILModule &module, SILFunction *original,
                                 IndexSubset *parameterIndices,
                                 IndexSubset *resultIndices) {
  for (auto *w : module.lookUpDifferentiabilityWitnessesForFunction(
           original->getName())) {
    if (w->getParameterIndices() == parameterIndices &&
        w->getResultIndices() == resultIndices)
      return w;
  }
  return nullptr;
}

std::optional<AutoDiffConfig>
findMinimalDerivativeConfiguration(AbstractFunctionDecl *original,
                                   IndexSubset *parameterIndices,
                                   IndexSubset *&minimalASTParameterIndices) {
  std::optional<AutoDiffConfig> minimalConfig = std::nullopt;
  auto configs = original->getDerivativeFunctionConfigurations();
  for (auto &config : configs) {
    auto *silParameterIndices = autodiff::getLoweredParameterIndices(
        config.parameterIndices,
        original->getInterfaceType()->castTo<AnyFunctionType>());

    if (silParameterIndices->getCapacity() < parameterIndices->getCapacity()) {
      silParameterIndices =
        silParameterIndices->extendingCapacity(original->getASTContext(),
                                               parameterIndices->getCapacity());
    }

    // If all indices in `parameterIndices` are in `daParameterIndices`, and
    // it has fewer indices than our current candidate and a primitive VJP,
    // then `attr` is our new candidate.
    //
    // NOTE(TF-642): `attr` may come from a un-partial-applied function and
    // have larger capacity than the desired indices. We expect this logic to
    // go away when `partial_apply` supports `@differentiable` callees.
    if (silParameterIndices->isSupersetOf(parameterIndices->extendingCapacity(
            original->getASTContext(), silParameterIndices->getCapacity())) &&
        // fewer parameters than before
        (!minimalConfig ||
         silParameterIndices->getNumIndices() <
             minimalConfig->parameterIndices->getNumIndices())) {
      minimalASTParameterIndices = config.parameterIndices;
      minimalConfig =
          AutoDiffConfig(silParameterIndices, config.resultIndices,
                         autodiff::getDifferentiabilityWitnessGenericSignature(
                             original->getGenericSignature(),
                             config.derivativeGenericSignature));
    }
  }
  return minimalConfig;
}

SILDifferentiabilityWitness *getOrCreateMinimalASTDifferentiabilityWitness(
    SILModule &module, SILFunction *original, DifferentiabilityKind kind,
    IndexSubset *parameterIndices, IndexSubset *resultIndices) {
  // Explicit differentiability witnesses only exist on SIL functions that come
  // from AST functions.
  auto *originalAFD = findAbstractFunctionDecl(original);
  if (!originalAFD)
    return nullptr;

  IndexSubset *minimalASTParameterIndices = nullptr;
  auto minimalConfig = findMinimalDerivativeConfiguration(
      originalAFD, parameterIndices, minimalASTParameterIndices);
  if (!minimalConfig)
    return nullptr;

  std::string originalName = original->getName().str();
  // If original function requires a foreign entry point, use the foreign SIL
  // function to get or create the minimal differentiability witness.
  if (requiresForeignEntryPoint(originalAFD)) {
    originalName = SILDeclRef(originalAFD).asForeign().mangle();
    original = module.lookUpFunction(SILDeclRef(originalAFD).asForeign());
  }

  auto *existingWitness = module.lookUpDifferentiabilityWitness(
      {originalName, kind, *minimalConfig});
  if (existingWitness)
    return existingWitness;

  assert(original->isExternalDeclaration() &&
         "SILGen should create differentiability witnesses for all function "
         "definitions with explicit differentiable attributes");

  bool isDefaultDerivative =
      isa<ProtocolDecl>(originalAFD->getDeclContext()) &&
      !originalAFD->getAttrs().hasAttribute<DifferentiableAttr>();

  return SILDifferentiabilityWitness::createDeclaration(
      module,
      // Witness for @_alwaysEmitIntoClient original function must be emitted,
      // otherwise a linker error would occur due to undefined reference to the
      // witness symbol.
      original->isAlwaysEmitIntoClient() ? SILLinkage::PublicNonABI
                                         : SILLinkage::PublicExternal,
      original, kind, minimalConfig->parameterIndices,
      minimalConfig->resultIndices, minimalConfig->derivativeGenericSignature,
      isDefaultDerivative);
}

} // end namespace autodiff
} // end namespace swift
