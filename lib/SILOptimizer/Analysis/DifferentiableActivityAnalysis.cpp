//===--- DifferentiableActivityAnalysis.h ----------------------*- C++ -*-===//
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

#define DEBUG_TYPE "differentiation"

#include "swift/AST/AutoDiff.h"
#include "swift/AST/Builtins.h"
#include "swift/AST/SubstitutionMap.h"
#include "swift/SIL/AbstractionPattern.h"
#include "swift/SILOptimizer/Analysis/DominanceAnalysis.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "swift/SILOptimizer/Analysis/DifferentiableActivityAnalysis.h"
#include "swift/SILOptimizer/Differentiation/Common.h"

#include "swift/AST/SemanticAttrs.h"
#include "swift/Basic/Assertions.h"
#include "swift/SIL/NodeDatastructures.h"
#include "swift/SIL/Projection.h"
#include "swift/SIL/SILArgument.h"
#include "swift/SILOptimizer/Analysis/ArraySemantic.h"
#include "swift/SILOptimizer/Analysis/DominanceAnalysis.h"
#include "swift/SILOptimizer/PassManager/PassManager.h"

using namespace swift;
using namespace swift::autodiff;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static bool isWithoutDerivative(SILValue v) {
  if (auto *fnRef = dyn_cast<FunctionRefInst>(v))
    return fnRef->getReferencedFunction()->hasSemanticsAttr(
        "autodiff.nonvarying");
  return false;
}

std::unique_ptr<DifferentiableActivityCollection>
DifferentiableActivityAnalysis::newFunctionAnalysis(SILFunction *f) {
  assert(dominanceAnalysis && "Expect a valid dominance analysis");
  assert(postDominanceAnalysis && "Expect a valid post-dominance analysis");
  return std::make_unique<DifferentiableActivityCollection>(
      *f, dominanceAnalysis->get(f), postDominanceAnalysis->get(f));
}

void DifferentiableActivityAnalysis::initialize(SILPassManager *pm) {
  dominanceAnalysis = pm->getAnalysis<DominanceAnalysis>();
  postDominanceAnalysis = pm->getAnalysis<PostDominanceAnalysis>();
}

SILAnalysis *swift::createDifferentiableActivityAnalysis(SILModule *m) {
  return new DifferentiableActivityAnalysis();
}

DifferentiableActivityCollection::DifferentiableActivityCollection(
    SILFunction &f, DominanceInfo *di, PostDominanceInfo *pdi)
    : function(f), domInfo(di), postDomInfo(pdi) {}

static llvm::cl::opt<bool> EnableAsyncAutoDiff(
    "enable-experimental-async-autodiff", llvm::cl::Hidden,
    llvm::cl::init(false));

bool swift::isExperimentalAsyncAutoDiffEnabled() {
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

FunctionType *AsyncLetInfo::getPullbackType() const {
  auto &function = *start->getFunction();
  auto &ctx = function.getASTContext();
  auto tangentType = [&](SILType primal) -> CanType {
    auto space = primal.getASTType()->getAutoDiffTangentSpace(
        LookUpConformanceInModule());
    assert(space && "an active async-let value must have a tangent space");
    auto tangent = space->getCanonicalType();
    auto lowered = function.getLoweredType(
        Lowering::AbstractionPattern::getOpaque(), tangent).getObjectType();
    assert(lowered.isTrivial(function) &&
           "nontrivial or address-only async-let tangents are not implemented");
    return tangent;
  };
  AnyFunctionType::Param seed(tangentType(getBuffer()->getType()), Identifier(),
                             ParameterTypeFlags());
  auto captures = initializer->getArguments();
  SmallVector<TupleTypeElt, 4> fields;
  for (auto index : getParameterIndices()->getIndices())
    fields.emplace_back(tangentType(captures[index]->getType()));
  assert(!fields.empty() && "an active child must have differentiable captures");
  // A single capture's tangent is one result, even if it is itself a tuple.
  Type gradients = fields.size() == 1 ? fields.front().getType()
                                     : TupleType::get(fields, ctx);
  return FunctionType::get({seed}, /*yields*/ {}, gradients,
                           FunctionType::ExtInfo());
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
  return llvm::any_of(resultAddresses, [&](PointerToAddressInst *address) {
    return inst == address;
  });
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
  auto pointer = a.start->getArguments()[2];
  for (auto *use : pointer->getUses()) {
    auto *inst = use->getUser();
    if (inst == a.start)
      continue;
    if (auto *addr = dyn_cast<PointerToAddressInst>(inst)) {
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
      a.resultAddresses.push_back(addr);
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
  assert(a.resultAddresses.empty() || !a.gets.empty());
  // These are SSA dominance checks, not a second task-lifetime or AD-loop
  // analysis. In particular, waits in alternative arms can precede a joined
  // result read, and cleanup-only paths need not contain a numerical read.
  for (auto *get : a.gets)
    assert(dominance.dominates(start, get));
  for (auto *address : a.resultAddresses)
    assert(dominance.dominates(start, address));
  for (auto *finish : finishes)
    assert(dominance.dominates(start, finish));
  for (auto *end : ends)
    assert(dominance.dominates(start, end));
  for (auto *dealloc : deallocs)
    assert(dominance.dominates(buffer, dealloc));
  LLVM_DEBUG({
    getADDebugStream() << "Matched trivial async let with " << a.gets.size()
                       << " wait(s), " << a.resultAddresses.size()
                       << " result address(es), "
                       << a.getParameterIndices()->getNumIndices()
                       << " differentiable capture(s):\n"
                       << *a.start << *a.initializer;
  });
  return a;
}

SmallVector<AsyncLetInfo, 2> swift::findAsyncLets(SILFunction &function) {
  SmallVector<AsyncLetInfo, 2> result;
  if (!EnableAsyncAutoDiff)
    return result;
  SmallVector<BuiltinInst *, 4> starts;
  for (auto &bb : function)
    for (auto &inst : bb)
      if (auto *start = isBuiltinInst(
              &inst, BuiltinValueKind::StartAsyncLetWithLocalBuffer))
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

const AsyncLetInfo *
DifferentiableActivityInfo::getAsyncLet(SILInstruction *inst) const {
  if (!inst)
    return nullptr;
  for (const auto &a : asyncLets)
    if (inst == a.start || a.isGet(inst) || a.isResultAddress(inst) ||
        inst == a.getBuffer() ||
        inst == a.start->getArguments()[2]->getDefiningInstruction() ||
        a.isInitializerInstruction(inst))
      return &a;
  return nullptr;
}

DifferentiableActivityInfo::DifferentiableActivityInfo(
    DifferentiableActivityCollection &parent, GenericSignature derivGenSig)
    : parent(parent), derivativeGenericSignature(derivGenSig) {
  asyncLets = findAsyncLets(parent.function);
  analyze(parent.domInfo, parent.postDomInfo);
}

SILFunction &DifferentiableActivityInfo::getFunction() const {
  return parent.function;
}

void DifferentiableActivityInfo::analyze(DominanceInfo *di,
                                         PostDominanceInfo *pdi) {
  auto &function = getFunction();
  LLVM_DEBUG(getADDebugStream()
             << "Running activity analysis on @" << function.getName() << '\n');
  // Inputs are just function's arguments, count `n`.
  auto paramArgs = function.getArgumentsWithoutIndirectResults();
  for (auto value : paramArgs)
    inputValues.push_back(value);
  LLVM_DEBUG({
    auto &s = getADDebugStream();
    s << "Inputs in @" << function.getName() << ":\n";
    for (auto val : inputValues)
      s << val << '\n';
  });
  // Outputs are indirect result buffers and return values, count `m`.
  // For the purposes of differentiation, we consider yields to be results as
  // well
  collectAllFormalResultsInTypeOrder(function, outputValues);
  LLVM_DEBUG({
    auto &s = getADDebugStream();
    s << "Outputs in @" << function.getName() << ":\n";
    for (auto val : outputValues)
      s << val << '\n';
  });

  // Propagate variedness starting from the inputs.
  assert(variedValueSets.empty());
  for (auto inputAndIdx : enumerate(inputValues)) {
    auto input = inputAndIdx.value();
    unsigned i = inputAndIdx.index();
    variedValueSets.push_back({});
    setVariedAndPropagateToUsers(input, i);
  }

  // Mark differentiable outputs as useful.
  assert(usefulValueSets.empty());
  for (auto outputAndIdx : enumerate(outputValues)) {
    auto output = outputAndIdx.value();
    unsigned i = outputAndIdx.index();
    usefulValueSets.push_back({});
    setUsefulAndPropagateToOperands(output, i);
  }
}

void DifferentiableActivityInfo::setVariedAndPropagateToUsers(
    SILValue value, unsigned independentVariableIndex) {
  // Skip already-varied values to prevent infinite recursion.
  if (isVaried(value, independentVariableIndex))
    return;
  // Set the value as varied.
  variedValueSets[independentVariableIndex].insert(value);
  if (auto *a = getAsyncLet(value->getDefiningInstruction());
      a && value == a->getBuffer())
    for (auto *address : a->resultAddresses)
      setVariedAndPropagateToUsers(address, independentVariableIndex);
  // Propagate variedness to users.
  for (auto *use : value->getUses())
    propagateVaried(use, independentVariableIndex);
}

void DifferentiableActivityInfo::propagateVaried(
    Operand *operand, unsigned independentVariableIndex) {
  auto *inst = operand->getUser();
  if (auto *a = getAsyncLet(inst)) {
    if (inst == a->start) {
      if (operand->get() == a->start->getArguments()[1])
        propagateVariedInwardsThroughProjections(
            a->getBuffer(), independentVariableIndex);
      return; // The handle is not the numerical result.
    }
    if (inst == a->start->getArguments()[2]->getDefiningInstruction())
      return; // Do not mark the raw pointer varied.
  }
  // Propagate variedness for the given operand.
  // General rule: mark results as varied and recursively propagate variedness
  // to users of results.
  auto i = independentVariableIndex;
  // Handle full apply sites: `apply`, `try_apply`, and `begin_apply`.
  if (FullApplySite::isa(inst)) {
    FullApplySite applySite(inst);
    // Skip non-varying callees.
    if (isWithoutDerivative(applySite.getCallee()))
      return;
    // If operand is varied, set all direct/indirect results and inout arguments
    // as varied.
    if (isVaried(operand->get(), i)) {
      for (auto indRes : applySite.getIndirectSILResults())
        propagateVariedInwardsThroughProjections(indRes, i);
      for (auto semresArg : applySite.getAutoDiffSemanticResultArguments())
        propagateVariedInwardsThroughProjections(semresArg, i);
      // Propagate variedness to apply site direct results.
      forEachApplyDirectResult(applySite, [&](SILValue directResult) {
        setVariedAndPropagateToUsers(directResult, i);
      });
    }
  }
  // Handle store-like instructions:
  //   `store`, `store_borrow`, `copy_addr`, `unconditional_checked_cast`
#define PROPAGATE_VARIED_THROUGH_STORE(INST)                                   \
  else if (auto *si = dyn_cast<INST##Inst>(inst)) {                            \
    if (isVaried(si->getSrc(), i))                                             \
      propagateVariedInwardsThroughProjections(si->getDest(), i);              \
  }
  PROPAGATE_VARIED_THROUGH_STORE(Store)
  PROPAGATE_VARIED_THROUGH_STORE(CopyAddr)
  PROPAGATE_VARIED_THROUGH_STORE(UnconditionalCheckedCastAddr)
#undef PROPAGATE_VARIED_THROUGH_STORE
  else if (auto *sbi = dyn_cast<StoreBorrowInst>(inst)) {
    if (isVaried(sbi->getSrc(), i)) {
      setVariedAndPropagateToUsers(sbi, i);
      propagateVariedInwardsThroughProjections(sbi, i);
    }
  }
  // Handle `tuple_element_addr`.
  else if (auto *teai = dyn_cast<TupleElementAddrInst>(inst)) {
    if (isVaried(teai->getOperand(), i)) {
      // Propagate variedness only if the `tuple_element_addr` result has a
      // tangent space. Otherwise, the result does not need a derivative.
      if (hasTangentSpace(teai))
        setVariedAndPropagateToUsers(teai, i);
    }
  }
  // Handle element projection instructions:
  //   `struct_extract`, `struct_element_addr`, `ref_element_addr`.
  // - If the field is marked `@noDerivative`, do not set the result as
  //   varied because it does not need a derivative.
  // - Otherwise, propagate variedness from operand to result as usual.
#define PROPAGATE_VARIED_FOR_ELEMENT_PROJECTION(INST)                          \
  else if (auto *projInst = dyn_cast<INST##Inst>(inst)) {                      \
    if (isVaried(projInst->getOperand(), i) &&                                 \
        !projInst->getField()->getAttrs().hasAttribute<NoDerivativeAttr>())    \
      setVariedAndPropagateToUsers(projInst, i);                               \
  }
  PROPAGATE_VARIED_FOR_ELEMENT_PROJECTION(StructExtract)
  PROPAGATE_VARIED_FOR_ELEMENT_PROJECTION(StructElementAddr)
  PROPAGATE_VARIED_FOR_ELEMENT_PROJECTION(RefElementAddr)
#undef PROPAGATE_VARIED_FOR_STRUCT_EXTRACTION
  // Handle `br`.
  else if (auto *bi = dyn_cast<BranchInst>(inst)) {
    if (isVaried(operand->get(), i))
      setVariedAndPropagateToUsers(bi->getArgForOperand(operand), i);
  }
  // Handle `cond_br`.
  else if (isa<CondBranchInst>(inst)) {
    // A cond_br only uses its condition operand and passes no branch arguments,
    // so there is no destination block argument to propagate variedness to.
  }
  // Handle `checked_cast_addr_br`.
  // Propagate variedness from source operand to destination operand, in
  // addition to all successor block arguments.
  else if (auto *ccabi = dyn_cast<CheckedCastAddrBranchInst>(inst)) {
    if (isVaried(ccabi->getSrc(), i)) {
      // A test_only cast has no destination to propagate to.
      if (ccabi->hasDest())
        setVariedAndPropagateToUsers(ccabi->getDest(), i);
      for (auto *succBB : ccabi->getSuccessorBlocks())
        for (auto *arg : succBB->getArguments())
          setVariedAndPropagateToUsers(arg, i);
    }
  }
  // Handle all other terminators: if any operand is active, propagate
  // variedness to all successor block arguments. This logic may be incorrect
  // for some terminator instructions, so special cases must be defined above.
  else if (auto *termInst = dyn_cast<TermInst>(inst)) {
    for (auto &op : termInst->getAllOperands())
      if (isVaried(op.get(), i))
        for (auto *succBB : termInst->getSuccessorBlocks())
          for (auto *arg : succBB->getArguments())
            setVariedAndPropagateToUsers(arg, i);
  }
  // Handle everything else.
  else {
    for (auto &op : inst->getAllOperands()) {
      if (isVaried(op.get(), i))
        for (auto result : inst->getResults())
          setVariedAndPropagateToUsers(result, i);
    }
  }
}

/// Returns the accessor kind of the given SIL function, if it is a lowered
/// accessor. Otherwise, return `None`.
static std::optional<AccessorKind> getAccessorKind(SILFunction *fn) {
  auto *dc = fn->getDeclContext();
  if (!dc)
    return std::nullopt;
  auto *accessor = dyn_cast_or_null<AccessorDecl>(dc->getAsDecl());
  if (!accessor)
    return std::nullopt;
  return accessor->getAccessorKind();
}

void DifferentiableActivityInfo::propagateVariedInwardsThroughProjections(
    SILValue value, unsigned independentVariableIndex) {
  auto i = independentVariableIndex;
  // Skip `@noDerivative` projections.
#define SKIP_NODERIVATIVE(INST)                                                \
  if (auto *projInst = dyn_cast<INST##Inst>(value))                            \
    if (projInst->getField()->getAttrs().hasAttribute<NoDerivativeAttr>())     \
      return;
  SKIP_NODERIVATIVE(StructExtract)
  SKIP_NODERIVATIVE(StructElementAddr)
  SKIP_NODERIVATIVE(RefElementAddr)
#undef SKIP_NODERIVATIVE
  // Set value as varied and propagate to users.
  setVariedAndPropagateToUsers(value, i);
  auto *inst = value->getDefiningInstruction();
  if (!inst)
    return;
  if (ApplySite::isa(inst)) {
    ApplySite applySite(inst);
    // If callee is non-varying, skip.
    if (isWithoutDerivative(applySite.getCallee()))
      return;
    // If callee is a `modify` accessor, propagate variedness from yielded
    // addresses to `inout` arguments. Semantically, yielded addresses can be
    // viewed as a projection into the `inout` argument.
    // Note: the assumption that yielded addresses are always a projection into
    // the `inout` argument is a safe over-approximation but not always true.
    if (auto *bai = dyn_cast<BeginApplyInst>(inst)) {
      if (auto *calleeFn = bai->getCalleeFunction()) {
        auto kind = getAccessorKind(calleeFn);
        if (kind && isYieldingMutableAccessor(*kind))
          for (auto inoutArg : bai->getInoutArguments())
            propagateVariedInwardsThroughProjections(inoutArg, i);
      }
    }
    return;
  }
  // Default: propagate variedness through projections to the operands of their
  // defining instructions. This handles projections like:
  // - `struct_element_addr`
  // - `tuple_element_addr`
  for (auto &op : inst->getAllOperands())
    propagateVariedInwardsThroughProjections(op.get(), i);
}

void DifferentiableActivityInfo::setUseful(SILValue value,
                                           unsigned dependentVariableIndex) {
  usefulValueSets[dependentVariableIndex].insert(value);
  setUsefulThroughArrayInitialization(value, dependentVariableIndex);
}

void DifferentiableActivityInfo::setUsefulAndPropagateToOperands(
    SILValue value, unsigned dependentVariableIndex) {
  // Skip already-useful values to prevent infinite recursion.
  if (isUseful(value, dependentVariableIndex))
    return;
  if (auto *a = getAsyncLet(value->getDefiningInstruction());
      a && (value == a->getBuffer() ||
            a->isResultAddress(value->getDefiningInstruction()))) {
    setUseful(a->getBuffer(), dependentVariableIndex);
    for (auto *address : a->resultAddresses)
      setUseful(address, dependentVariableIndex);
    setUsefulAndPropagateToOperands(a->start->getArguments()[1],
                                   dependentVariableIndex);
    return;
  }
  if (value->getType().isAddress() ||
      value->getType().getClassOrBoundGenericClass()) {
    propagateUsefulThroughAddress(value, dependentVariableIndex);
    return;
  }
  setUseful(value, dependentVariableIndex);

  // If the given value is a basic block argument, propagate usefulness to
  // incoming values.
  if (auto *bbArg = dyn_cast<SILPhiArgument>(value)) {
    SmallVector<SILValue, 4> incomingValues;
    if (bbArg->getSingleTerminatorOperands(incomingValues)) {
      for (auto incomingValue : incomingValues)
        setUsefulAndPropagateToOperands(incomingValue, dependentVariableIndex);
      return;
    }

    if (bbArg->isTerminatorResult()) {
      if (TryApplyInst *tai = dyn_cast<TryApplyInst>(bbArg->getTerminatorForResult())) {
        propagateUseful(tai, dependentVariableIndex);
        return;
      }
      llvm::report_fatal_error("unknown terminator with result");
    }

    llvm::report_fatal_error("do not know how to handle this incoming bb argument");
  }
  
  auto *inst = value->getDefiningInstruction();
  if (!inst)
    return;
  propagateUseful(inst, dependentVariableIndex);
}

void DifferentiableActivityInfo::propagateUseful(
    SILInstruction *inst, unsigned dependentVariableIndex) {
  // Propagate usefulness for the given instruction: mark operands as useful and
  // recursively propagate usefulness to defining instructions of operands.
  auto i = dependentVariableIndex;
  // Handle full apply sites: `apply`, `try_apply`, and `begin_apply`.
  if (FullApplySite::isa(inst)) {
    FullApplySite applySite(inst);
    // If callee is non-varying, skip.
    if (isWithoutDerivative(applySite.getCallee()))
      return;
    // If callee is a `modify` accessor, propagate usefulness through yielded
    // addresses. Semantically, yielded addresses can be viewed as a projection
    // into the `inout` argument.
    // Note: the assumption that yielded addresses are always a projection into
    // the `inout` argument is a safe over-approximation but not always true.
    if (auto *bai = dyn_cast<BeginApplyInst>(inst)) {
      if (auto *calleeFn = bai->getCalleeFunction()) {
        auto kind = getAccessorKind(calleeFn);
        if (kind && isYieldingMutableAccessor(*kind))
          for (auto yield : bai->getYieldedValues())
            setUsefulAndPropagateToOperands(yield, i);
      }
    }
    // Propagate usefulness through apply site arguments.
    for (auto arg : applySite.getArgumentsWithoutIndirectResults())
      setUsefulAndPropagateToOperands(arg, i);
  }
  // Handle store-like instructions:
  //   `store`, `store_borrow`, `copy_addr`, `unconditional_checked_cast`
#define PROPAGATE_USEFUL_THROUGH_STORE(INST)                                   \
  else if (auto *si = dyn_cast<INST##Inst>(inst)) {                            \
    setUsefulAndPropagateToOperands(si->getSrc(), i);                          \
  }
  PROPAGATE_USEFUL_THROUGH_STORE(Store)
  PROPAGATE_USEFUL_THROUGH_STORE(StoreBorrow)
  PROPAGATE_USEFUL_THROUGH_STORE(CopyAddr)
  PROPAGATE_USEFUL_THROUGH_STORE(UnconditionalCheckedCastAddr)
#undef PROPAGATE_USEFUL_THROUGH_STORE
  // Handle element projections, skipping `@noDerivative` fields:
  //   `struct_extract`, `struct_element_addr`, `ref_element_addr`.
#define PROPAGATE_USEFUL_THROUGH_ELEMENT_PROJECTION(INST)                      \
  else if (auto *projInst = dyn_cast<INST##Inst>(inst)) {                      \
    if (!projInst->getField()->getAttrs().hasAttribute<NoDerivativeAttr>())    \
      setUsefulAndPropagateToOperands(projInst->getOperand(), i);              \
  }
  PROPAGATE_USEFUL_THROUGH_ELEMENT_PROJECTION(StructExtract)
  PROPAGATE_USEFUL_THROUGH_ELEMENT_PROJECTION(StructElementAddr)
  PROPAGATE_USEFUL_THROUGH_ELEMENT_PROJECTION(RefElementAddr)
#undef PROPAGATE_USEFUL_THROUGH_STRUCT_EXTRACTION
  // Handle everything else.
  else {
    for (auto &op : inst->getAllOperands())
      setUsefulAndPropagateToOperands(op.get(), i);
  }
}

void DifferentiableActivityInfo::propagateUsefulThroughAddress(
    SILValue value, unsigned dependentVariableIndex) {
  assert(value->getType().isAddress() ||
         value->getType().getClassOrBoundGenericClass());
  // Skip already-useful values to prevent infinite recursion.
  if (isUseful(value, dependentVariableIndex))
    return;
  setUseful(value, dependentVariableIndex);
  if (auto *inst = value->getDefiningInstruction())
    propagateUseful(inst, dependentVariableIndex);
  // Recursively propagate usefulness through users that are projections or
  // `begin_access` instructions.
  for (auto use : value->getUses()) {
    // Propagate usefulness through user's operands.
    propagateUseful(use->getUser(), dependentVariableIndex);
    for (auto res : use->getUser()->getResults()) {
#define SKIP_NODERIVATIVE(INST)                                                \
  if (auto *projInst = dyn_cast<INST##Inst>(res))                              \
    if (projInst->getField()->getAttrs().hasAttribute<NoDerivativeAttr>())     \
      continue;
      SKIP_NODERIVATIVE(StructExtract)
      SKIP_NODERIVATIVE(StructElementAddr)
      SKIP_NODERIVATIVE(RefElementAddr)
#undef SKIP_NODERIVATIVE
      if (Projection::isAddressProjection(res) || isa<BeginAccessInst>(res) ||
          isa<BeginBorrowInst>(res) || isa<InitEnumDataAddrInst>(res))
        propagateUsefulThroughAddress(res, dependentVariableIndex);
    }
  }
}

void DifferentiableActivityInfo::setUsefulThroughArrayInitialization(
    SILValue value, unsigned dependentVariableIndex) {
  // Array initializer syntax is lowered to an intrinsic and one or more
  // stores to a `RawPointer` returned by the intrinsic.
  ArraySemanticsCall uninitCall(value,
                                semantics::ARRAY_UNINITIALIZED_INTRINSIC);
  ApplyInst *uai = uninitCall;
  if (!uai)
    return;
  for (auto use : value->getUses()) {
    auto *dti = dyn_cast<DestructureTupleInst>(use->getUser());
    if (!dti)
      continue;

    ValueWorklist worklist(dti->getResult(0));

    while (SILValue v = worklist.pop()) {
      for (auto use : v->getUses()) {
        switch (use->getUser()->getKind()) {
          case SILInstructionKind::UncheckedRefCastInst:
          case SILInstructionKind::StructExtractInst:
          case SILInstructionKind::BeginBorrowInst:
            worklist.pushIfNotVisited(cast<SingleValueInstruction>(use->getUser()));
            break;
          case SILInstructionKind::RefTailAddrInst: {
            auto *rta = cast<RefTailAddrInst>(use->getUser());
            setUseful(rta, dependentVariableIndex);
            // Propagate usefulness through array element addresses:
            // `pointer_to_address` and `index_addr` instructions.
            //
            // - Set all array element addresses as useful.
            // - Find instructions with array element addresses as "result":
            //   - `store` and `copy_addr` with array element address as destination.
            //   - `apply` with array element address as an indirect result.
            // - For each instruction, propagate usefulness through "arguments":
            //   - `store` and `copy_addr`: propagate to source.
            //   - `apply`: propagate to arguments.
            //
            // NOTE: `propagateUseful(use->getUser(), ...)` is intentionally not used
            // because it marks more values than necessary as useful, including:
            // - The `RawPointer` result of the intrinsic.
            // - `integer_literal` operands to `index_addr` for indexing the
            //   `RawPointer`.
            // It is also blocked by TF-1032: control flow differentiation crash for
            // active values with no tangent space.
            for (auto use : rta->getUses()) {
              auto *user = use->getUser();
              if (auto *si = dyn_cast<StoreInst>(user)) {
                setUseful(si->getDest(), dependentVariableIndex);
                setUsefulAndPropagateToOperands(si->getSrc(), dependentVariableIndex);
              } else if (auto *cai = dyn_cast<CopyAddrInst>(user)) {
                setUseful(cai->getDest(), dependentVariableIndex);
                setUsefulAndPropagateToOperands(cai->getSrc(),
                                                dependentVariableIndex);
              } else if (auto *ai = dyn_cast<ApplyInst>(user)) {
                if (FullApplySite(ai).isIndirectResultOperand(*use))
                  for (auto arg : ai->getArgumentsWithoutIndirectResults())
                    setUsefulAndPropagateToOperands(arg, dependentVariableIndex);
              } else if (auto *iai = dyn_cast<IndexAddrInst>(user)) {
                setUseful(iai, dependentVariableIndex);
                for (auto use : iai->getUses()) {
                  auto *user = use->getUser();
                  if (auto si = dyn_cast<StoreInst>(user)) {
                    setUseful(si->getDest(), dependentVariableIndex);
                    setUsefulAndPropagateToOperands(si->getSrc(),
                                                    dependentVariableIndex);
                  } else if (auto *cai = dyn_cast<CopyAddrInst>(user)) {
                    setUseful(cai->getDest(), dependentVariableIndex);
                    setUsefulAndPropagateToOperands(cai->getSrc(),
                                                    dependentVariableIndex);
                  } else if (auto *ai = dyn_cast<ApplyInst>(user)) {
                    if (FullApplySite(ai).isIndirectResultOperand(*use))
                      for (auto arg : ai->getArgumentsWithoutIndirectResults())
                        setUsefulAndPropagateToOperands(arg, dependentVariableIndex);
                  }
                }
              }
            }
            break;
          }
          default:
            break;
        }
      }
    }
  }
}

bool DifferentiableActivityInfo::isVaried(
    SILValue value, unsigned independentVariableIndex) const {
  assert(independentVariableIndex < variedValueSets.size() &&
         "Independent variable index out of range");
  auto &set = variedValueSets[independentVariableIndex];
  return set.count(value);
}

bool DifferentiableActivityInfo::isVaried(
    SILValue value, IndexSubset *independentVariableIndices) const {
  for (auto i : independentVariableIndices->getIndices())
    if (isVaried(value, i))
      return true;
  return false;
}

bool DifferentiableActivityInfo::isUseful(
    SILValue value, unsigned dependentVariableIndex) const {
  assert(dependentVariableIndex < usefulValueSets.size() &&
         "Dependent variable index out of range");
  auto &set = usefulValueSets[dependentVariableIndex];
  return set.count(value);
}

bool DifferentiableActivityInfo::isUseful(
    SILValue value, IndexSubset *dependentVariableIndices) const {
  for (auto i : dependentVariableIndices->getIndices())
    if (isUseful(value, i))
      return true;
  return false;
}

bool DifferentiableActivityInfo::isActive(
    SILValue value, IndexSubset *parameterIndices,
    IndexSubset *resultIndices) const {
  return isVaried(value, parameterIndices) && isUseful(value, resultIndices);
}

Activity DifferentiableActivityInfo::getActivity(
    SILValue value, IndexSubset *parameterIndices,
    IndexSubset *resultIndices) const {
  Activity activity;
  if (isVaried(value, parameterIndices))
    activity |= ActivityFlags::Varied;
  if (isUseful(value, resultIndices))
    activity |= ActivityFlags::Useful;
  return activity;
}

void DifferentiableActivityInfo::dump(
    SILValue value, IndexSubset *parameterIndices, IndexSubset *resultIndices,
    llvm::raw_ostream &s) const {
  s << '[';
  auto activity = getActivity(value, parameterIndices, resultIndices);
  switch (activity.toRaw()) {
  case 0:
    s << "NONE";
    break;
  case (unsigned)ActivityFlags::Varied:
    s << "VARIED";
    break;
  case (unsigned)ActivityFlags::Useful:
    s << "USEFUL";
    break;
  case (unsigned)ActivityFlags::Active:
    s << "ACTIVE";
    break;
  }
  s << "] " << value;
}

void DifferentiableActivityInfo::dump(
    IndexSubset *parameterIndices, IndexSubset *resultIndices,
    llvm::raw_ostream &s) const {
  SILFunction &fn = getFunction();
  s << "Activity info for " << fn.getName() << " at parameter indices (";
  llvm::interleaveComma(parameterIndices->getIndices(), s);
  s << ") and result indices (";
  llvm::interleaveComma(resultIndices->getIndices(), s);
  s << "):\n";
  for (auto &bb : fn) {
    s << "bb" << bb.getDebugID() << ":\n";
    for (auto *arg : bb.getArguments())
      dump(arg, parameterIndices, resultIndices, s);
    for (auto &inst : bb)
      for (auto res : inst.getResults())
        dump(res, parameterIndices, resultIndices, s);
    if (std::next(bb.getIterator()) != fn.end())
      s << '\n';
  }
  s << "End activity info for " << fn.getName() << '\n';
}
