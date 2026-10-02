//===--- AsyncLetDifferentiation.cpp - Async-let VJP transport-------------===//
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

#define DEBUG_TYPE "async-let-differentiation"

#include "swift/SILOptimizer/Differentiation/AsyncLetDifferentiation.h"
#include "swift/AST/ConformanceLookup.h"
#include "swift/SIL/SILDebugScope.h"
#include "swift/SIL/SILDifferentiabilityWitness.h"
#include "swift/SILOptimizer/Analysis/DifferentiableActivityAnalysis.h"
#include "swift/SILOptimizer/Differentiation/ADContext.h"
#include "swift/SILOptimizer/Differentiation/Common.h"
#include "swift/SILOptimizer/Differentiation/LinearMapInfo.h"
#include "swift/SILOptimizer/Differentiation/Thunk.h"
#include "swift/SILOptimizer/Utils/SILOptFunctionBuilder.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace swift;
using namespace swift::autodiff;

namespace {
bool isNilExecutor(SILValue value) {
  auto *e = dyn_cast<EnumInst>(value);
  return e && !e->hasOperand() &&
         e->getElement() ==
             e->getFunction()->getASTContext().getOptionalNoneDecl();
}

// Validate only the effect change performed by helper construction. Numerical
// operations, control flow and derivative availability are ordinary AD's job.
// Transferability of the actual saved residuals is checked after synthesis.
std::string validateNumericalHelper(SILFunction &function) {
  if (function.getActorIsolation().isActorIsolated())
    return "an actor-isolated child needs an executor-aware pullback contract";
  for (auto &block : function)
    for (auto &inst : block) {
      if (isa<ThrowInst>(&inst))
        return "a throwing initializer needs error-path augmented-result "
               "cleanup";
      if (auto *hop = dyn_cast<HopToExecutorInst>(&inst))
        if (!isNilExecutor(hop->getTargetExecutor()))
          return "an actor-bound child needs an executor-aware pullback "
                 "contract";
    }
  return {};
}

SILFunction *createHelper(ADContext &context, SILFunction &source,
                          StringRef name, CanSILFunctionType type) {
  SILOptFunctionBuilder functions(context.getTransform());
  auto *f = functions.createFunction(
      SILLinkage::Private, context.getASTContext().getIdentifier(name).str(),
      type, source.getActorIsolation(),
      /*genericEnv*/ nullptr, source.getLocation(), source.isBare(),
      IsNotTransparent, IsNotSerialized, source.isDynamicallyReplaceable(),
      source.isDistributed(), source.isRuntimeAccessible());
  f->setDebugScope(new (context.getModule())
                       SILDebugScope(source.getLocation(), f));
  f->setHasLoweredAddresses(source.hasLoweredAddresses());
  context.recordGeneratedFunction(f);
  return f;
}

// The numerical helper has no error result, unlike its task-entry wrapper.
// TypeSubstCloner remaps the wrapper's debug scopes to the helper, but it does
// not know that we removed the error result from the signature. Drop only the
// now-inapplicable error-slot record. Keep ordinary capture debug information
// and any records belonging to genuinely inlined functions.
static void removeErrorResultDebugValues(SILFunction &body) {
  assert(!body.getLoweredFunctionType()->hasErrorResult());
  for (auto &block : body) {
    for (auto &inst : llvm::make_early_inc_range(block)) {
      auto *debug = dyn_cast<DebugValueInst>(&inst);
      if (!debug)
        continue;
      auto varInfo = debug->getVarInfo();
      if (!varInfo || varInfo->Name != "$error" || !varInfo->Scope ||
          varInfo->Scope->getInlinedFunction() != &body)
        continue;
      LLVM_DEBUG(llvm::dbgs()
                 << "[async-let-differentiation] removing obsolete $error "
                    "debug record from nonthrowing helper @"
                 << body.getName() << '\n');
      debug->eraseFromParent();
    }
  }
}

SILType lowerCheckpointType(SILFunction &f, Type type) {
  auto signature = f.getLoweredFunctionType()->getSubstGenericSignature();
  Lowering::AbstractionPattern pattern(signature,
                                       type->getReducedType(signature));
  return f.getLoweredType(pattern, type);
}
} // namespace

SILValue swift::autodiff::reabstractAsyncLetPullback(
    ADContext &context, SILBuilder &builder, SILLocation loc, SILValue function,
    CanSILFunctionType target) {
  if (function->getType().castTo<SILFunctionType>() == target)
    return function;

  SILOptFunctionBuilder functions(context.getTransform());
  return reabstractFunction(
      builder, functions, loc, function, target,
      [](SubstitutionMap substitutions) { return substitutions; });
}

IndexSubset *swift::autodiff::getAsyncLetParameterIndices(
    const AsyncLetAccess &a, const DifferentiableActivityInfo &activity,
    const AutoDiffConfig &config) {
  SmallVector<unsigned, 4> indices;
  for (auto indexed : llvm::enumerate(a.captures)) {
    auto capture = indexed.value();
    if (activity.isActive(capture, config) &&
        capture->getType().getASTType()->getAutoDiffTangentSpace(
            LookUpConformanceInModule()))
      indices.push_back(indexed.index());
  }
  return IndexSubset::get(a.start->getFunction()->getASTContext(),
                          a.captures.size(), indices);
}

FunctionType *swift::autodiff::getAsyncLetPullbackASTType(
    const AsyncLetAccess &a, const DifferentiableActivityInfo &activity,
    const AutoDiffConfig &config) {
  auto &ctx = a.start->getFunction()->getASTContext();
  auto resultSpace = a.getValueType().getASTType()->getAutoDiffTangentSpace(
      LookUpConformanceInModule());
  auto *indices = getAsyncLetParameterIndices(a, activity, config);
  if (!resultSpace || indices->getNumIndices() == 0)
    return nullptr;
  SmallVector<TupleTypeElt, 4> gradientTypes;
  for (auto index : indices->getIndices()) {
    auto space =
        a.captures[index]->getType().getASTType()->getAutoDiffTangentSpace(
            LookUpConformanceInModule());
    assert(space);
    gradientTypes.emplace_back(space->getType());
  }
  Type result = gradientTypes.size() == 1 ? gradientTypes.front().getType()
                                          : TupleType::get(gradientTypes, ctx);
  AnyFunctionType::Param seed(resultSpace->getType(), Identifier(),
                              ParameterTypeFlags());
  return FunctionType::get({seed}, /*yields*/ {}, result,
                           FunctionType::ExtInfo());
}

CanSILFunctionType swift::autodiff::getAsyncLetPullbackBufferType(
    const AsyncLetAccess &a, const DifferentiableActivityInfo &activity,
    const AutoDiffConfig &config) {
  auto *astType = getAsyncLetPullbackASTType(a, activity, config);
  assert(astType && "invalid async-let derivative signature");
  auto &function = *a.start->getFunction();
  auto seedType =
      function.getLoweredType(astType->getParams().front().getPlainType());
  auto resultType = function.getLoweredType(astType->getResult());
  SILParameterInfo seed(seedType.getASTType(),
                        ParameterConvention::Indirect_In_Guaranteed);
  SILResultInfo result(resultType.getASTType(), ResultConvention::Indirect);
  return SILFunctionType::get(
      nullptr,
      SILFunctionType::ExtInfo().withRepresentation(
          SILFunctionTypeRepresentation::Thick),
      SILCoroutineKind::None, ParameterConvention::Direct_Guaranteed, {seed},
      {}, {result}, std::nullopt, {}, {}, function.getASTContext());
}

bool swift::autodiff::diagnoseUnsupportedAsyncLets(
    ADContext &context, const DifferentiableActivityInfo &activity,
    const AutoDiffConfig &config, DifferentiationInvoker invoker) {
  for (auto &a : activity.getAsyncLets().getAccesses()) {
    std::string reason;
    if (!isExperimentalAsyncAutoDiffEnabled()) {
      reason = "enable -Xllvm -enable-experimental-async-autodiff to use the "
               "experimental implementation";
    } else if (!a->isSupported()) {
      reason = a->unsupportedReason;
    } else if (llvm::any_of(a->reads, [&](SILInstruction *read) {
                 return activity.isActive(getAsyncLetReadValue(read), config);
               })) {
      if (!getAsyncLetPullbackASTType(*a, activity, config))
        reason = "the active result and selected captures need tangent spaces";
      else
        reason = validateNumericalHelper(*a->initializer);
      if (reason.empty()) {
        auto &parent = *a->start->getFunction();
        if (parent.getActorIsolation().isActorIsolated())
          reason = "actor-isolated async-let parents are not yet supported";
        for (auto &block : parent)
          for (auto &inst : block)
            if (auto *hop = dyn_cast<HopToExecutorInst>(&inst))
              if (!isNilExecutor(hop->getTargetExecutor()))
                reason = "actor-bound async-let parents are not yet supported";
      }
    }
    if (!reason.empty()) {
      context.emitNondifferentiabilityError(
          a->start, invoker, diag::autodiff_async_let_unsupported,
          StringRef(reason));
      return true;
    }
  }
  return false;
}

AsyncLetVJPEmitter::AsyncLetVJPEmitter(
    ADContext &context, SILFunction &vjp,
    const DifferentiableActivityInfo &activity, const AutoDiffConfig &config,
    LinearMapInfo &linearMaps)
    : context(context), vjp(vjp), activity(activity), config(config),
      linearMaps(linearMaps) {
  for (auto &a : activity.getAsyncLets().getAccesses()) {
    assert(a->isSupported());
    if (!llvm::any_of(a->reads, [&](SILInstruction *read) {
          return activity.isActive(getAsyncLetReadValue(read), config);
        }))
      continue;
    LLVM_DEBUG(llvm::dbgs() << "[async-let-differentiation] augmenting child @"
                            << a->initializer->getName() << " in @"
                            << vjp.getName() << '\n');
    State state;
    state.access = a.get();
    llvm::SmallVector<TupleTypeElt, 2> fields = {
        a->getValueType().getASTType(),
        TupleTypeElt(getAsyncLetPullbackASTType(*a, activity, config))};
    state.payloadAST = TupleType::get(fields, context.getASTContext());
    // A generic task result is stored at opaque abstraction. In particular its
    // closure field can have an indirect seed AND indirect gradient results.
    state.payloadType =
        vjp.getLoweredType(Lowering::AbstractionPattern::getOpaque(),
                           state.payloadAST)
            .getObjectType();
    states.push_back(std::move(state));
  }
}

SILFunction *AsyncLetVJPEmitter::createTaskEntry(const AsyncLetAccess &a,
                                                 SILType payloadType,
                                                 unsigned ordinal) {
  auto &ctx = context.getASTContext();
  auto originalType =
      a.initializer->getLoweredFunctionType()->getUnsubstitutedType(
          context.getModule());
  // Do not append arbitrary suffixes to a Swift mangled symbol. The existing
  // differentiation mangler has an explicit path for unmangled original names.
  auto baseName =
      (Twine("__swift_async_ad_") + vjp.getName() + "_" + Twine(ordinal)).str();

  // Normalize only the already validated nonthrowing numerical body. Keep its
  // indirect primal result. Captures are explicit differentiable parameters.
  auto bodyType = SILFunctionType::get(
      nullptr, originalType->getExtInfo().withSendable(false),
      SILCoroutineKind::None, originalType->getCalleeConvention(),
      originalType->getParameters(), {}, originalType->getResults(),
      std::nullopt, {}, {}, ctx);
  auto *body =
      createHelper(context, *a.initializer, baseName + ".body", bodyType);
  BasicTypeSubstCloner(a.initializer, body).cloneFunction();
  removeErrorResultDebugValues(*body);
#ifndef NDEBUG
  // Catch helper-construction errors here, before they propagate into its VJP.
  body->verify();
#endif

  SILResultInfo result(payloadType.getASTType(), ResultConvention::Indirect);
  auto entryType = SILFunctionType::get(
      nullptr, originalType->getExtInfo(), SILCoroutineKind::None,
      originalType->getCalleeConvention(), originalType->getParameters(), {},
      {result}, originalType->getOptionalErrorResult(), {}, {}, ctx);
  auto *entry =
      createHelper(context, *a.initializer, baseName + ".entry", entryType);
  entry->createBasicBlock();
  createEntryArguments(entry);
  SILBuilder builder(entry->getEntryBlock(), entry->getDebugScope());
  auto loc = a.start->getLoc();
  auto arguments = entry->getArguments();
  auto out = arguments.front();
  auto valueAddress = builder.createTupleElementAddr(loc, out, 0);
  auto pullbackAddress = builder.createTupleElementAddr(loc, out, 1);

  // Preserve the child's actual nil-executor hop, including its operand type.
  SILValue executor;
  bool mandatory = false;
  for (auto &inst : a.initializer->front())
    if (auto *hop = dyn_cast<HopToExecutorInst>(&inst)) {
      auto *nil = cast<EnumInst>(hop->getTargetExecutor());
      executor = builder.createEnum(loc, SILValue(), nil->getElement(),
                                    nil->getType());
      mandatory = hop->isMandatory();
      builder.createHopToExecutor(loc, executor, mandatory);
      break;
    }

  auto *parameters = getAsyncLetParameterIndices(a, activity, config);
  auto *results = IndexSubset::get(ctx, 1, {0});
  auto *ref = builder.createFunctionRef(loc, body);
  context.recordGeneratedFunctionReference(ref);
  auto *dfi = context.createDifferentiableFunction(builder, loc, parameters,
                                                   results, ref);
  context.getDifferentiableFunctionInstWorklist().push_back(dfi);
  SILValue derivative;
  builder.emitScopedBorrowOperation(loc, dfi, [&](SILValue borrowed) {
    auto *v = builder.createDifferentiableFunctionExtract(
        loc, NormalDifferentiableFunctionTypeComponent::VJP, borrowed);
    derivative = builder.emitCopyValueOperation(loc, v);
  });
  builder.emitDestroyValueOperation(loc, dfi);
  SmallVector<SILValue, 8> callArguments{valueAddress};
  for (auto argument : entry->getArgumentsWithoutIndirectResults())
    callArguments.push_back(argument);
  auto *call = builder.createApply(loc, derivative, {}, callArguments);
  builder.emitDestroyValueOperation(loc, derivative);
  if (executor)
    builder.createHopToExecutor(loc, executor, mandatory);

  // No second primal execution. The VJP has initialized field 0 and returned a
  // PB with an indirect seed. Adapt it to the generic field-1 storage ABI.
  auto pbType =
      pullbackAddress->getType().getObjectType().castTo<SILFunctionType>();
  auto pb = reabstractAsyncLetPullback(context, builder, loc, call, pbType);
  builder.emitStoreValueOperation(loc, pb, pullbackAddress,
                                  StoreOwnershipQualifier::Init);
  auto unit = builder.createTuple(loc, SILType::getEmptyTupleType(ctx), {});
  builder.createReturn(loc, unit);
  LLVM_DEBUG(llvm::dbgs() << "[async-let-differentiation] numerical helper:\n"
                          << *body << "task entry:\n"
                          << *entry);
  return entry;
}

bool AsyncLetVJPEmitter::clone(
    SILInstruction *inst, SILBuilder &builder,
    llvm::function_ref<SILValue(SILValue)> remap,
    llvm::function_ref<void(SILValue, SILValue)> map,
    llvm::function_ref<void(SILValue)> savePullback) {
  for (unsigned ordinal = 0; ordinal != states.size(); ++ordinal) {
    auto &state = states[ordinal];
    const auto &a = *state.access;
    auto loc = inst->getLoc();
    if (inst == a.buffer) {
      state.entry = createTaskEntry(a, state.payloadType, ordinal);
      state.buffer = builder.createAllocStack(
          RegularLocation::getAutoGeneratedLocation(), state.payloadType);
      state.valueAddress = builder.createTupleElementAddr(loc, state.buffer, 0);
      state.pullbackAddress =
          builder.createTupleElementAddr(loc, state.buffer, 1);
      map(a.buffer, state.valueAddress);
      return true;
    }
    if (inst == a.pointer) {
      auto *pointer = builder.createAddressToPointer(
          loc, state.buffer, a.pointer->getType(),
          a.pointer->needsStackProtection());
      map(a.pointer, pointer);
      return true;
    }
    if (llvm::is_contained(a.addresses, inst)) {
      map(cast<PointerToAddressInst>(inst), state.valueAddress);
      return true;
    }
    if (inst == a.closure) {
      // Replace the original partial_apply at the original capture point.
      // In particular, never consume an owned array/struct capture in BOTH
      // the old initializer closure and a newly constructed task closure.
      auto *ref = builder.createFunctionRef(loc, state.entry);
      context.recordGeneratedFunctionReference(ref);
      SmallVector<SILValue, 4> captures;
      for (auto capture : a.captures)
        captures.push_back(remap(capture));
      auto *closure = builder.createPartialApply(
          loc, ref, {}, captures, a.closure->getCalleeConvention(),
          a.closure->getResultIsolation(), a.closure->isCalledOnce(),
          a.closure->isOnStack(), a.closure->isStackAllocationNested());
      map(a.closure, closure);
      return true;
    }
    if (!a.closure && inst == a.initializerRef) {
      auto *ref = builder.createFunctionRef(loc, state.entry);
      context.recordGeneratedFunctionReference(ref);
      map(a.initializerRef, ref);
      return true;
    }
    if (llvm::is_contained(a.closureConversions, inst)) {
      auto *conversion = cast<SingleValueInstruction>(inst);
      SILValue operand = remap(conversion->getOperand(0));
      auto target =
          operand->getType().castTo<SILFunctionType>()->getWithExtInfo(
              conversion->getType().castTo<SILFunctionType>()->getExtInfo());
      auto targetType = SILType::getPrimitiveObjectType(target);
      SILValue converted;
      if (auto *i = dyn_cast<ConvertFunctionInst>(inst))
        converted = builder.createConvertFunction(loc, operand, targetType,
                                                  i->withoutActuallyEscaping());
      else if (auto *i = dyn_cast<ConvertEscapeToNoEscapeInst>(inst))
        converted = builder.createConvertEscapeToNoEscape(
            loc, operand, targetType, i->isLifetimeGuaranteed());
      else if (isa<ThinToThickFunctionInst>(inst))
        converted = builder.createThinToThickFunction(loc, operand, targetType);
      else if (isa<CopyValueInst>(inst))
        converted = builder.emitCopyValueOperation(loc, operand);
      else if (isa<BeginBorrowInst>(inst))
        converted = builder.emitBeginBorrowOperation(loc, operand);
      else
        llvm_unreachable("unhandled validated closure conversion");
      map(conversion, converted);
      return true;
    }
    if (inst == a.start) {
      auto substitutions = SubstitutionMap::get(
          a.start->getSubstitutions().getGenericSignature(),
          ArrayRef<Type>{state.payloadAST}, ArrayRef<ProtocolConformanceRef>{});
      auto args = a.start->getArguments();
      auto *start = builder.createBuiltin(
          loc, a.start->getName(), a.start->getType(), substitutions,
          {remap(args[0]), remap(args[1]), remap(a.pointer)});
      map(a.start, start);
      return true;
    }
    if (llvm::is_contained(a.reads, inst)) {
      if (auto *load = dyn_cast<LoadInst>(inst)) {
        auto *value = builder.createLoad(loc, state.valueAddress,
                                         load->getOwnershipQualifier());
        map(load, value);
      } else if (auto *borrow = dyn_cast<LoadBorrowInst>(inst)) {
        map(borrow, builder.emitLoadBorrowOperation(loc, state.valueAddress));
      } else {
        auto *copy = cast<CopyAddrInst>(inst);
        builder.createCopyAddr(loc, state.valueAddress, remap(copy->getDest()),
                               IsNotTake, IsInitialization);
      }
      if (activity.isActive(getAsyncLetReadValue(inst), config)) {
        // The existing per-block checkpoint collection owns this copy. Its
        // standard branching trace/loop context determines the dynamic read
        // instance. Do not keep a single mutable PB slot for all iterations.
        SILValue pb = builder.emitLoadValueOperation(
            loc, state.pullbackAddress, LoadOwnershipQualifier::Copy);
        auto target = lowerCheckpointType(
                          vjp, linearMaps.lookUpAsyncLetPullbackType(inst))
                          .castTo<SILFunctionType>();
        savePullback(
            reabstractAsyncLetPullback(context, builder, loc, pb, target));
      }
      return true;
    }
    if (llvm::is_contained(a.deallocations, inst)) {
      builder.createDeallocStack(loc, state.buffer);
      return true;
    }
    // The original finish, end, destroy and borrow-end instructions are cloned
    // normally, using the remapped task pointer/closure values. No duplicate
    // closure cleanup is inserted: ownership is the original ownership graph.
  }
  return false;
}
