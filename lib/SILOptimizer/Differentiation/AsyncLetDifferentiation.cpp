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
#include "swift/SILOptimizer/Utils/SILOptFunctionBuilder.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace swift;
using namespace swift::autodiff;

namespace {
/// The initial transfer proof is deliberately syntactic: scalar-only, local,
/// acyclic straight-line computations and these audited scalar primitives.
/// Merely testing primal argument/result Sendability would NOT prove tape
/// safety.
bool isScalarPrimitive(SILFunction *f) {
  return llvm::StringSwitch<bool>(f->getName())
      .Cases({"$sSf1poiyS2f_SftFZ", "$sSf1soiyS2f_SftFZ"}, true)
      .Case("$sSf1moiyS2f_SftFZ", true)
      .Cases({"$sSd1poiyS2d_SdtFZ", "$sSd1soiyS2d_SdtFZ"}, true)
      .Case("$sSd1moiyS2d_SdtFZ", true)
      .Default(false);
}

bool isNilExecutor(SILValue value) {
  auto *e = dyn_cast<EnumInst>(value);
  return e && !e->hasOperand() &&
         e->getElement()->getName().getBaseIdentifier().str() == "none";
}

std::string validateScalarBody(ADContext &context, SILFunction *f,
                               bool initializer,
                               llvm::SmallPtrSetImpl<SILFunction *> &visiting,
                               llvm::SmallPtrSetImpl<SILFunction *> &checked) {
  if (!initializer && isScalarPrimitive(f))
    return {};
  if (checked.count(f))
    return {};
  if (!visiting.insert(f).second)
    return "recursive child computations are not yet supported";
  if (f->isExternalDeclaration() || f->size() != 1)
    return "child computations must have available single-block bodies";
  auto type = f->getLoweredFunctionType()->getUnsubstitutedType(f->getModule());
  if (type->getSubstGenericSignature() || type->isCoroutine() ||
      (!initializer && type->hasErrorResult()) || type->getNumResults() != 1 ||
      !isAsyncAutoDiffScalar(
          type->getSingleResult().getSILStorageInterfaceType().getObjectType()))
    return "generic, throwing, coroutine, or nonscalar child signatures are "
           "not supported";
  for (auto param : type->getParameters())
    if (!isAsyncAutoDiffScalar(param.getSILStorageInterfaceType()) ||
        param.isFormalIndirect())
      return "child parameters must be by-value Float or Double";

  // Reject existing user derivative registrations. The residual argument relies
  // on the body being differentiated by this pass, not on an arbitrary custom
  // PB.
  for (auto &witness : f->getModule().getDifferentiabilityWitnesses()) {
    if (witness.getOriginalFunction() != f)
      continue;
    if (isa_and_nonnull<DerivativeAttr>(witness.getAttribute()) ||
        (witness.getVJP() && !context.isGeneratedFunction(witness.getVJP())))
      return "custom or pre-existing opaque derivatives of child computations "
             "are outside the initial scope";
  }

  unsigned storesToResult = 0;
  for (auto &inst : f->front()) {
    if (isa<DebugValueInst, FunctionRefInst, ReturnInst, MetatypeInst>(&inst))
      continue;
    if (auto *hop = dyn_cast<HopToExecutorInst>(&inst)) {
      if (!isNilExecutor(hop->getTargetExecutor()))
        return "actor-bound child computations require a separate "
               "residual-isolation design";
      continue;
    }
    if (auto *e = dyn_cast<EnumInst>(&inst)) {
      if (!isNilExecutor(e))
        return "only the nil executor enum is supported in a child computation";
      continue;
    }
    if (auto *tuple = dyn_cast<TupleInst>(&inst)) {
      if (tuple->getElements().empty())
        continue;
    }
    if (auto *store = dyn_cast<StoreInst>(&inst)) {
      if (type->getNumIndirectFormalResults() != 1 ||
          store->getDest() != f->getArguments().front() ||
          !isAsyncAutoDiffScalar(store->getSrc()->getType()) ||
          ++storesToResult != 1)
        return "only initialization of the scalar indirect result is supported";
      continue;
    }
    if (auto *apply = dyn_cast<ApplyInst>(&inst)) {
      auto *ref = dyn_cast<FunctionRefInst>(apply->getCallee());
      if (!ref || !apply->getSubstitutionMap().empty() ||
          apply->getSubstCalleeType()->hasErrorResult())
        return "child calls must resolve directly to nonthrowing functions";
      auto reason = validateScalarBody(context, ref->getReferencedFunction(),
                                       false, visiting, checked);
      if (!reason.empty())
        return reason;
      continue;
    }
    return "unsupported instruction in the scalar-only child residual domain";
  }
  if (type->getNumIndirectFormalResults() && storesToResult != 1)
    return "the child must initialize its indirect result exactly once";
  visiting.erase(f);
  checked.insert(f);
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

// These helpers deliberately operate only on the already-validated scalar
// async-let domain. They are NOT general Swift function reabstraction.
static bool appendScalarLeafTypes(SILType type,
                                  SmallVectorImpl<SILType> &leaves) {
  type = type.getObjectType();
  if (auto tuple = type.getAs<TupleType>()) {
    for (unsigned i = 0; i != tuple->getNumElements(); ++i)
      if (!appendScalarLeafTypes(type.getTupleElementType(i), leaves))
        return false;
    return true;
  }
  if (!isAsyncAutoDiffScalar(type))
    return false;
  leaves.push_back(type);
  return true;
}

static bool isScalarPullback(CanSILFunctionType type,
                             SmallVectorImpl<SILType> &resultLeaves) {
  if (type->isAsync() || type->hasErrorResult() || type->isCoroutine() ||
      type->getSubstGenericSignature() || type->getCombinedSubstitutions() ||
      type->getRepresentation() != SILFunctionTypeRepresentation::Thick ||
      type->getExtInfo().isNoEscape() || type->getNumParameters() != 1 ||
      type->getCalleeConvention() != ParameterConvention::Direct_Guaranteed)
    return false;
  auto seed = type->getParameters().front();
  if (!isAsyncAutoDiffScalar(
          seed.getSILStorageInterfaceType().getObjectType()) ||
      (seed.getConvention() != ParameterConvention::Direct_Unowned &&
       seed.getConvention() != ParameterConvention::Indirect_In_Guaranteed))
    return false;
  for (auto result : type->getResults()) {
    if (result.getConvention() != ResultConvention::Indirect &&
        result.getConvention() != ResultConvention::Unowned &&
        result.getConvention() != ResultConvention::Owned)
      return false;
    if (!appendScalarLeafTypes(result.getSILStorageInterfaceType(),
                               resultLeaves))
      return false;
  }
  return true;
}

// Flatten the value of ONE formal result. A formal result may itself be a
// tuple, independently of the tuple used to represent multiple direct results.
static void appendScalarResultValues(SILBuilder &builder, SILLocation loc,
                                     SILValue value,
                                     SmallVectorImpl<SILValue> &leaves) {
  if (auto tuple = value->getType().getAs<TupleType>()) {
    for (unsigned i = 0; i != tuple->getNumElements(); ++i) {
      SILValue element;
      if (value->getType().isAddress())
        element = builder.createTupleElementAddr(loc, value, i);
      else
        element = builder.createTupleExtract(loc, value, i);
      appendScalarResultValues(builder, loc, element, leaves);
    }
    return;
  }
  assert(isAsyncAutoDiffScalar(value->getType().getObjectType()));
  if (value->getType().isAddress())
    value = builder.createLoad(loc, value, LoadOwnershipQualifier::Trivial);
  leaves.push_back(value);
}

// Rebuild the value of ONE destination formal result from ordered scalar
// leaves. In particular, (Float, Float) can become one indirect tuple result.
static SILValue buildScalarResult(SILBuilder &builder, SILLocation loc,
                                  SILType type, ArrayRef<SILValue> &leaves) {
  type = type.getObjectType();
  if (auto tuple = type.getAs<TupleType>()) {
    SmallVector<SILValue, 4> elements;
    for (unsigned i = 0; i != tuple->getNumElements(); ++i)
      elements.push_back(
          buildScalarResult(builder, loc, type.getTupleElementType(i), leaves));
    return builder.createTuple(loc, type, elements);
  }
  assert(!leaves.empty() && leaves.front()->getType() == type);
  auto value = leaves.front();
  leaves = leaves.drop_front();
  return value;
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
  auto original = function->getType().castTo<SILFunctionType>();
  if (original == target)
    return function;

  auto &module = context.getModule();
  auto fromType = original->getUnsubstitutedType(module);
  auto toType = target->getUnsubstitutedType(module);
  LLVM_DEBUG(llvm::dbgs() << "[async-let-differentiation] pullback transport\n"
                          << "  from: " << fromType << '\n'
                          << "  to:   " << toType << '\n'
                          << "  formal results: " << fromType->getNumResults()
                          << " -> " << toType->getNumResults() << '\n');

  if (original != fromType)
    function = builder.createConvertFunction(
        loc, function, SILType::getPrimitiveObjectType(fromType), false);
  if (fromType == toType) {
    if (toType != target)
      function = builder.createConvertFunction(
          loc, function, SILType::getPrimitiveObjectType(target), false);
    return function;
  }

  // The generic AD reabstraction helper zips formal results one-to-one. That
  // cannot convert two formal Float results to ONE @out (Float, Float) result.
  // Compare the flattened numerical leaves instead, then pack/unpack them.
  SmallVector<SILType, 4> fromLeaves, toLeaves;
  if (!isScalarPullback(fromType, fromLeaves) ||
      !isScalarPullback(toType, toLeaves) || fromLeaves != toLeaves ||
      fromType->getParameters()
              .front()
              .getSILStorageInterfaceType()
              .getObjectType() != toType->getParameters()
                                      .front()
                                      .getSILStorageInterfaceType()
                                      .getObjectType())
    llvm::report_fatal_error(
        "unsupported async-let scalar pullback reabstraction");

  auto &caller = builder.getFunction();
  assert(caller.hasLoweredAddresses() &&
         "async-let pullback transport is emitted after address lowering");
  auto &ctx = context.getASTContext();
  SmallVector<SILParameterInfo, 2> parameters;
  parameters.append(toType->getParameters().begin(),
                    toType->getParameters().end());
  // Each invocation borrows the captured PB. partial_apply owns the capture,
  // so invoking the adapted PB does not consume its saved derivative state.
  parameters.emplace_back(fromType, ParameterConvention::Direct_Guaranteed);
  auto thunkType = SILFunctionType::get(
      nullptr,
      toType->getExtInfo().withRepresentation(
          SILFunctionTypeRepresentation::Thin),
      SILCoroutineKind::None, ParameterConvention::Direct_Unowned, parameters,
      {}, toType->getResults(), std::nullopt, {}, {}, ctx);

  // Private, deterministic per-caller names avoid colliding with a generic
  // reabstraction thunk created elsewhere for the same two types.
  auto stem =
      (Twine("__swift_async_ad_pb_reabstraction_") + caller.getName() + "_")
          .str();
  unsigned ordinal = 0;
  auto name = (Twine(stem) + Twine(ordinal)).str();
  while (module.lookUpFunction(name))
    name = (Twine(stem) + Twine(++ordinal)).str();
  SILOptFunctionBuilder functions(context.getTransform());
  auto *thunk = functions.createFunction(
      SILLinkage::Private, ctx.getIdentifier(name).str(), thunkType,
      ActorIsolation::forUnspecified(), /*genericEnv*/ nullptr,
      caller.getLocation(), IsBare, IsNotTransparent, IsNotSerialized,
      IsNotDynamic, IsNotDistributed, IsNotRuntimeAccessible);
  thunk->setDebugScope(new (module) SILDebugScope(caller.getLocation(), thunk));
  thunk->setHasLoweredAddresses(caller.hasLoweredAddresses());
  thunk->setThunk(IsReabstractionThunk);
  context.recordGeneratedFunction(thunk);
  thunk->createBasicBlock();
  createEntryArguments(thunk);
  SILBuilder body(thunk->getEntryBlock(), thunk->getDebugScope());

  SmallVector<AllocStackInst *, 4> temporaries;
  SmallVector<SILValue, 4> indirectResults;
  SmallVector<SILValue, 8> arguments;
  for (auto result : fromType->getResults()) {
    if (!result.isFormalIndirect())
      continue;
    auto *buffer = body.createAllocStack(
        loc, result.getSILStorageInterfaceType().getObjectType());
    temporaries.push_back(buffer);
    indirectResults.push_back(buffer);
    arguments.push_back(buffer);
  }

  auto inputs = thunk->getArgumentsWithoutIndirectResults();
  SILValue seed = inputs.front();
  auto seedType =
      fromType->getParameters().front().getSILStorageInterfaceType();
  if (seedType.isAddress() && seed->getType().isObject()) {
    auto *buffer = body.createAllocStack(loc, seedType.getObjectType());
    temporaries.push_back(buffer);
    body.createStore(loc, seed, buffer, StoreOwnershipQualifier::Trivial);
    seed = buffer;
  } else if (seedType.isObject() && seed->getType().isAddress()) {
    seed = body.createLoad(loc, seed, LoadOwnershipQualifier::Trivial);
  }
  arguments.push_back(seed);
  auto *call =
      body.createApply(loc, inputs.back(), SubstitutionMap(), arguments);

  // Keep formal-result ordering even if direct and indirect results are mixed.
  SmallVector<SILValue, 4> values;
  unsigned directIndex = 0, indirectIndex = 0;
  for (auto result : fromType->getResults()) {
    SILValue value;
    if (result.isFormalIndirect()) {
      value = indirectResults[indirectIndex++];
    } else if (fromType->getNumDirectFormalResults() == 1) {
      value = call;
      ++directIndex;
    } else {
      value = body.createTupleExtract(loc, call, directIndex++);
    }
    appendScalarResultValues(body, loc, value, values);
  }
  assert(values.size() == toLeaves.size());

  ArrayRef<SILValue> remaining(values);
  SmallVector<SILValue, 4> directResults;
  auto indirect = thunk->getIndirectResults().begin();
  for (auto result : toType->getResults()) {
    auto value = buildScalarResult(
        body, loc, result.getSILStorageInterfaceType(), remaining);
    if (result.isFormalIndirect()) {
      assert(indirect != thunk->getIndirectResults().end());
      body.createStore(loc, value, *indirect++,
                       StoreOwnershipQualifier::Trivial);
    } else {
      directResults.push_back(value);
    }
  }
  assert(remaining.empty() && indirect == thunk->getIndirectResults().end());
  // Every temporary contains only scalar/tuple data. No residual buffer escapes
  // an invocation, and there is no owned payload to destroy inside these slots.
  for (auto *temporary : llvm::reverse(temporaries))
    body.createDeallocStack(loc, temporary);
  body.createReturn(loc, joinElements(directResults, body, loc));
  LLVM_DEBUG(llvm::dbgs() << *thunk);

  auto *ref = builder.createFunctionRef(loc, thunk);
  context.recordGeneratedFunctionReference(ref);
  function = builder.createPartialApply(loc, ref, SubstitutionMap(), {function},
                                        toType->getCalleeConvention());
  if (toType != target)
    function = builder.createConvertFunction(
        loc, function, SILType::getPrimitiveObjectType(target), false);
  return function;
}

FunctionType *
swift::autodiff::getAsyncLetPullbackASTType(const AsyncLetAccess &a) {
  auto &ctx = a.start->getFunction()->getASTContext();
  SmallVector<TupleTypeElt, 4> results;
  for (auto capture : a.captures)
    results.emplace_back(capture->getType().getASTType());
  Type result;
  if (results.size() == 1)
    result = results.front().getType();
  else
    result = TupleType::get(results, ctx);
  AnyFunctionType::Param seed(a.getValueType().getASTType(), Identifier(),
                              ParameterTypeFlags());
  return FunctionType::get({seed}, /*yields*/ {}, result,
                           FunctionType::ExtInfo());
}

CanSILFunctionType
swift::autodiff::getAsyncLetDirectPullbackType(const AsyncLetAccess &a) {
  SILParameterInfo seed(a.getValueType().getASTType(),
                        ParameterConvention::Direct_Unowned);
  SmallVector<SILResultInfo, 4> results;
  for (auto capture : a.captures)
    results.emplace_back(capture->getType().getASTType(),
                         ResultConvention::Unowned);
  return SILFunctionType::get(nullptr,
                              SILFunctionType::ExtInfo().withRepresentation(
                                  SILFunctionTypeRepresentation::Thick),
                              SILCoroutineKind::None,
                              ParameterConvention::Direct_Guaranteed, {seed},
                              {}, results, std::nullopt, {}, {},
                              a.start->getFunction()->getASTContext());
}

bool swift::autodiff::diagnoseUnsupportedAsyncLets(
    ADContext &context, const DifferentiableActivityInfo &activity,
    DifferentiationInvoker invoker) {
  auto &accesses = activity.getAsyncLets();
  for (auto &a : accesses.getAccesses()) {
    std::string reason;
    if (!isExperimentalAsyncAutoDiffEnabled())
      reason = "enable -Xllvm -enable-experimental-async-autodiff to use the "
               "experimental implementation";
    else if (!a->isSupported())
      reason = a->unsupportedReason;
    else {
      for (auto &inst : a->start->getFunction()->front())
        if (auto *hop = dyn_cast<HopToExecutorInst>(&inst))
          if (!isNilExecutor(hop->getTargetExecutor()))
            reason = "actor-bound parents are not supported by the initial "
                     "async-let implementation";
      if (reason.empty()) {
        llvm::SmallPtrSet<SILFunction *, 16> visiting, checked;
        reason = validateScalarBody(context, a->initializer, true, visiting,
                                    checked);
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
    if (!llvm::any_of(a->reads, [&](LoadInst *read) {
          return activity.isActive(read, config);
        }))
      continue;
    LLVM_DEBUG(llvm::dbgs() << "[async-let-differentiation] augmenting child @"
                            << a->initializer->getName() << " in @"
                            << vjp.getName() << '\n');
    State state;
    state.access = a.get();
    llvm::SmallVector<TupleTypeElt, 2> fields = {
        a->getValueType().getASTType(),
        TupleTypeElt(getAsyncLetPullbackASTType(*a))};
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

  SmallVector<unsigned, 4> indices;
  for (unsigned i = 0; i != a.captures.size(); ++i)
    indices.push_back(i);
  auto *parameters = IndexSubset::get(ctx, a.captures.size(), indices);
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
      // Use an artificial location: the larger allocation is not a source
      // VarDecl.
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
          loc, state.buffer, a.pointer->getType(), false);
      map(a.pointer, pointer);
      return true;
    }
    if (llvm::is_contained(a.addresses, inst)) {
      map(cast<PointerToAddressInst>(inst), state.valueAddress);
      return true;
    }
    if (inst == a.start) {
      auto *ref = builder.createFunctionRef(loc, state.entry);
      context.recordGeneratedFunctionReference(ref);
      SmallVector<SILValue, 4> captures;
      for (auto capture : a.captures)
        captures.push_back(remap(capture));
      auto *closure = builder.createPartialApply(
          loc, ref, {}, captures, ParameterConvention::Direct_Guaranteed);
      auto escapingType = closure->getType().castTo<SILFunctionType>();
      escapingType = escapingType->getWithExtInfo(
          escapingType->getExtInfo().withSendable(false));
      state.escapingClosure = builder.createConvertFunction(
          loc, closure, SILType::getPrimitiveObjectType(escapingType), false);
      auto noescapeType = escapingType->getWithExtInfo(
          escapingType->getExtInfo().withNoEscape(true));
      state.noescapeClosure = builder.createConvertEscapeToNoEscape(
          loc, state.escapingClosure,
          SILType::getPrimitiveObjectType(noescapeType), false);
      auto substitutions = SubstitutionMap::get(
          a.start->getSubstitutions().getGenericSignature(),
          ArrayRef<Type>{state.payloadAST}, ArrayRef<ProtocolConformanceRef>{});
      auto args = a.start->getArguments();
      auto *start = builder.createBuiltin(
          loc, a.start->getName(), a.start->getType(), substitutions,
          {remap(args[0]), state.noescapeClosure, remap(a.pointer)});
      map(a.start, start);
      return true;
    }
    if (auto *load = dyn_cast<LoadInst>(inst)) {
      if (!llvm::is_contained(a.reads, load))
        continue;
      auto *value = builder.createLoad(loc, state.valueAddress,
                                       LoadOwnershipQualifier::Trivial);
      map(load, value);
      if (activity.isActive(load, config)) {
        LLVM_DEBUG(llvm::dbgs() << "[async-let-differentiation] checkpoint for "
                                   "result read:\n"
                                << *load);
        SILValue pb = builder.emitLoadValueOperation(
            loc, state.pullbackAddress, LoadOwnershipQualifier::Copy);
        auto target = lowerCheckpointType(
                          vjp, linearMaps.lookUpAsyncLetPullbackType(load))
                          .castTo<SILFunctionType>();
        pb = reabstractAsyncLetPullback(context, builder, loc, pb, target);
        savePullback(pb);
      }
      return true;
    }
    if (inst == a.end) {
      SmallVector<SILValue, 2> args;
      for (auto arg : a.end->getArguments())
        args.push_back(remap(arg));
      auto *end = builder.createBuiltin(loc, a.end->getName(), a.end->getType(),
                                        a.end->getSubstitutions(), args);
      map(a.end, end);
      // finish has already joined and destroyed the task-owned payload. Each
      // checkpoint owns a copy; no parent PB retains this allocation/task
      // handle.
      builder.emitDestroyValueOperation(loc, state.noescapeClosure);
      builder.emitDestroyValueOperation(loc, state.escapingClosure);
      return true;
    }
    if (inst == a.deallocation) {
      builder.createDeallocStack(loc, state.buffer);
      return true;
    }
  }
  return false;
}
