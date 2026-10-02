//===--- AsyncLetResiduals.cpp - Check generated residual transfers
//--------===//
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
//
// Check the values actually retained by a child pullback, after the normal AD
// worklists have resolved its derivative calls. This is a conservative proof,
// not a second concurrency checker: unknown/opaque provenance is diagnosed.
//
// Use ordinary Sendable conformance lookup for leaf values. Follow aggregate
// construction, closure captures, and available derivative implementations for
// non-Sendable closure representations. Generated branch enums and loop context
// allocations are the EXISTING AD storage; they carry proof dependencies on
// their initialized payloads, not a nominal exemption for a type-name prefix.
//
// No generated closure is declared @unchecked Sendable. In particular, the
// TangentVector associated type is not required to be Sendable merely because
// its primal is transferred; only forward values actually retained are checked.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "async-let-differentiation"

#include "swift/SIL/SILDifferentiabilityWitness.h"
#include "swift/SIL/SILFunction.h"
#include "swift/SIL/SILModule.h"
#include "swift/SILOptimizer/Differentiation/ADContext.h"
#include "swift/SILOptimizer/Differentiation/AsyncLetDifferentiation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

using namespace swift;
using namespace swift::autodiff;

namespace {
struct Frame;
struct Binding {
  SILValue value;
  const Frame *frame;
};
struct Frame {
  SILFunction *function;
  const Frame *parent = nullptr;
  SubstitutionMap substitutions;
  SmallVector<Binding, 8> arguments;
};

// A path selects tuple components of a returned value. It is important not to
// inspect the unrelated primal fields of (primal, pullback) as residuals.
struct Query {
  SILValue value;
  const Frame *frame;
  SmallVector<unsigned, 4> path;
};

class ResidualTransferChecker {
  ADContext &context;
  SILFunction *rootFunction;
  SmallVector<Query, 32> visiting;
  SmallPtrSet<SILFunction *, 16> activeCalls;
  SILValue failure;
  std::string reason;

  bool fail(SILValue value, StringRef message) {
    if (reason.empty()) {
      failure = value;
      reason = message.str();
    }
    return false;
  }

  SILType instantiate(SILType type, const Frame *frame) {
    for (; frame; frame = frame->parent) {
      if (!frame->substitutions.empty())
        type = type.mapTypeOutOfEnvironment().subst(context.getModule(),
                                                    frame->substitutions);
    }
    return type;
  }

  // Ownership-preserving value forwarding. Do not look through arbitrary
  // pointer casts or reinterpret casts to manufacture a transfer proof.
  Binding strip(Binding binding) {
    while (true) {
      auto value = binding.value;
      if (auto *arg = dyn_cast<SILFunctionArgument>(value)) {
        if (arg->getFunction() == binding.frame->function &&
            arg->getIndex() < binding.frame->arguments.size()) {
          binding = binding.frame->arguments[arg->getIndex()];
          continue;
        }
      }
      if (isa<CopyValueInst>(value) || isa<BeginBorrowInst>(value) ||
          isa<ConvertFunctionInst>(value) ||
          isa<ConvertEscapeToNoEscapeInst>(value) ||
          isa<ThinToThickFunctionInst>(value)) {
        binding.value = value->getDefiningInstruction()->getOperand(0);
        continue;
      }
      if (auto *extract = dyn_cast<DifferentiableFunctionExtractInst>(value)) {
        auto bundle = strip({extract->getOperand(), binding.frame});
        if (auto *dfi = dyn_cast<DifferentiableFunctionInst>(bundle.value)) {
          if (dfi->hasExtractee(extract->getExtractee())) {
            binding = {dfi->getExtractee(extract->getExtractee()),
                       bundle.frame};
            continue;
          }
        }
      }
      return binding;
    }
  }

  SILFunction *resolveFunction(Binding binding) {
    binding = strip(binding);
    if (auto *ref = dyn_cast<FunctionRefInst>(binding.value))
      return ref->getReferencedFunction();
    if (auto *ref =
            dyn_cast<DifferentiabilityWitnessFunctionInst>(binding.value)) {
      auto kind = ref->getWitnessKind().getAsDerivativeFunctionKind();
      if (!kind)
        return nullptr;
      auto *witness = ref->getWitness();
      if (witness->isDeclaration())
        context.getModule().loadDifferentiabilityWitness(witness);
      return witness->getDerivative(*kind);
    }
    return nullptr;
  }

  bool checkCodeIsolation(SILFunction *function, SILValue origin) {
    if (!function)
      return fail(origin, "the saved pullback has an unresolved code target");
    if (function->getActorIsolation().isActorIsolated())
      return fail(origin, "the saved pullback requires an actor executor");
    return true;
  }

  bool checkCallResult(ApplyInst *call, const Frame *frame,
                       ArrayRef<unsigned> path) {
    auto callee = strip({call->getCallee(), frame});
    SmallVector<Binding, 8> captures;
    SubstitutionMap substitutions = call->getSubstitutionMap();
    if (auto *partial = dyn_cast<PartialApplyInst>(callee.value)) {
      // A call to a reabstraction/forwarding closure. The trailing arguments
      // belong to its captured environment, not the explicit invocation.
      if (substitutions.empty())
        substitutions = partial->getSubstitutionMap();
      else if (!partial->getSubstitutionMap().empty())
        return fail(
            call,
            "composed generic closure substitutions need a transfer summary");
      for (auto capture : partial->getArguments())
        captures.push_back({capture, callee.frame});
      callee = strip({partial->getCallee(), callee.frame});
    }
    auto *function = resolveFunction(callee);
    if (!checkCodeIsolation(function, call))
      return false;
    if (function->isExternalDeclaration())
      context.getModule().loadFunction(function,
                                       SILModule::LinkingMode::LinkNormal);
    if (function->isExternalDeclaration())
      return fail(
          call,
          "the derivative implementation has no inspectable residual summary");
    if (!activeCalls.insert(function).second)
      return fail(
          call,
          "recursive derivative producers need a residual-transfer summary");
    auto resetCall =
        llvm::make_scope_exit([&] { activeCalls.erase(function); });

    Frame child{function, frame, substitutions, {}};
    for (auto arg : call->getArguments())
      child.arguments.push_back({arg, frame});
    child.arguments.append(captures.begin(), captures.end());
    if (child.arguments.size() != function->getArguments().size())
      return fail(call, "the derivative argument mapping is not available for "
                        "transfer checking");

    bool hasReturn = false;
    for (auto &block : *function)
      if (auto *ret = dyn_cast<ReturnInst>(block.getTerminator())) {
        hasReturn = true;
        if (!check(ret->getOperand(), &child, path))
          return false;
      }
    return hasReturn ||
           fail(call, "the derivative has no normal return to inspect");
  }

  // Find the existing AD context associated with a typed subcontext address.
  // A raw pointer is only accepted when it is this pass's allocation token;
  // arbitrary raw pointers are never accepted on grounds of being trivial.
  SILValue contextForAddress(SILValue address, const Frame *frame) {
    auto *typed = dyn_cast<PointerToAddressInst>(address);
    if (!typed)
      return {};
    auto *builtin = dyn_cast<BuiltinInst>(typed->getOperand());
    if (!builtin ||
        !(isBuiltinInst(builtin,
                        BuiltinValueKind::AutoDiffAllocateSubcontextWithType) ||
          isBuiltinInst(builtin,
                        BuiltinValueKind::AutoDiffProjectTopLevelSubcontext)))
      return {};
    return strip({builtin->getArguments().front(), frame}).value;
  }

  bool checkContext(BuiltinInst *allocation, const Frame *frame) {
    if (!context.isGeneratedFunction(allocation->getFunction()))
      return fail(
          allocation,
          "an opaque context allocation has no generated-payload proof");
    bool hasPayload = false;
    for (auto &block : *allocation->getFunction())
      for (auto &inst : block)
        if (auto *store = dyn_cast<StoreInst>(&inst)) {
          if (contextForAddress(store->getDest(), frame) != allocation)
            continue;
          hasPayload = true;
          if (!check(store->getSrc(), frame, {}))
            return false;
        }
    return hasPayload ||
           fail(allocation,
                "the AD context has no visible initialized payload");
  }

  // A non-Sendable closure in local storage is accepted only if every write to
  // that exact storage has provable provenance and the address cannot escape.
  // General alias/region inference is deliberately left to a future integration
  // with RegionAnalysis rather than guessed here.
  bool checkAddress(SILValue address, const Frame *frame,
                    ArrayRef<unsigned> path) {
    if (auto *element = dyn_cast<TupleElementAddrInst>(address)) {
      SmallVector<unsigned, 4> nextPath{element->getFieldIndex()};
      nextPath.append(path.begin(), path.end());
      return checkAddress(element->getOperand(), frame, nextPath);
    }
    if (auto *access = dyn_cast<BeginAccessInst>(address))
      return checkAddress(access->getSource(), frame, path);
    if (auto *arg = dyn_cast<SILFunctionArgument>(address)) {
      if (arg->getFunction() == frame->function &&
          arg->getIndex() < frame->arguments.size()) {
        auto bound = frame->arguments[arg->getIndex()];
        return check(bound.value, bound.frame, path);
      }
    }
    if (!isa<AllocStackInst>(address))
      return fail(
          address,
          "a saved non-Sendable value has unresolved memory provenance");

    bool hasWrite = false;
    SmallVector<SILValue, 8> addresses{address};
    SmallPtrSet<SILInstruction *, 16> examined;
    for (unsigned next = 0; next != addresses.size(); ++next) {
      SILValue current = addresses[next];
      for (auto *use : current->getUses()) {
        auto *user = use->getUser();
        if (!examined.insert(user).second)
          continue;
        if (auto *store = dyn_cast<StoreInst>(user)) {
          if (store->getDest() != address)
            return fail(
                address,
                "partial initialization needs a residual-transfer summary");
          hasWrite = true;
          if (!check(store->getSrc(), frame, path))
            return false;
        } else if (auto *copy = dyn_cast<CopyAddrInst>(user)) {
          if (copy->getDest() == address) {
            hasWrite = true;
            if (!check(copy->getSrc(), frame, path))
              return false;
          } else {
            return fail(
                address,
                "an escaping residual buffer needs region-transfer analysis");
          }
        } else if (isa<LoadInst>(user) || isa<LoadBorrowInst>(user) ||
                   isa<DestroyAddrInst>(user) || isa<DeallocStackInst>(user) ||
                   isa<DebugValueInst>(user) || isa<EndAccessInst>(user)) {
          continue;
        } else if (auto *element = dyn_cast<TupleElementAddrInst>(user)) {
          addresses.push_back(element);
        } else if (auto *access = dyn_cast<BeginAccessInst>(user)) {
          addresses.push_back(access);
        } else {
          return fail(address,
                      "the saved residual buffer has an unmodeled use");
        }
      }
    }
    return hasWrite ||
           fail(address,
                "the saved residual buffer has no visible initializer");
  }

  bool checkImpl(SILValue value, const Frame *frame, ArrayRef<unsigned> path) {
    auto bound = strip({value, frame});
    if (bound.value != value || bound.frame != frame)
      return check(bound.value, bound.frame, path);

    if (auto *extract = dyn_cast<TupleExtractInst>(value)) {
      SmallVector<unsigned, 4> next{extract->getFieldIndex()};
      next.append(path.begin(), path.end());
      return check(extract->getOperand(), frame, next);
    }
    if (auto *destructure = dyn_cast_or_null<DestructureTupleInst>(
            value->getDefiningInstruction())) {
      unsigned index = 0;
      for (auto result : destructure->getResults()) {
        if (result == value)
          break;
        ++index;
      }
      SmallVector<unsigned, 4> next{index};
      next.append(path.begin(), path.end());
      return check(destructure->getOperand(), frame, next);
    }
    if (auto *tuple = dyn_cast<TupleInst>(value)) {
      if (!path.empty())
        return check(tuple->getElement(path.front()), frame, path.drop_front());
      for (auto element : tuple->getElements())
        if (!check(element, frame, {}))
          return false;
      return true;
    }

    auto type = instantiate(value->getType().getObjectType(), frame);
    for (unsigned index : path) {
      if (!type.is<TupleType>())
        return fail(value,
                    "the saved residual has incompatible result projections");
      type = type.getTupleElementType(index);
    }
    // A Sendable function still needs a suitable execution/isolation contract.
    // Inspect concrete code/captures below rather than bypassing that check.
    if (!type.is<SILFunctionType>() &&
        (type.isSendable(rootFunction) ||
         !type.isNonTrivialOrContainsRawPointer(rootFunction)))
      return true;
    if (value->getType().isAddress())
      return checkAddress(value, frame, path);
    if (auto *load = dyn_cast<LoadInst>(value))
      return check(load->getOperand(), frame, path);
    if (auto *load = dyn_cast<LoadBorrowInst>(value))
      return check(load->getOperand(), frame, path);
    if (auto *call = dyn_cast<ApplyInst>(value))
      return checkCallResult(call, frame, path);
    if (auto *arg = dyn_cast<SILPhiArgument>(value)) {
      SmallVector<SILValue, 4> incoming;
      if (!arg->getIncomingPhiValues(incoming) || incoming.empty())
        return fail(value,
                    "a residual block argument has unresolved incoming values");
      for (auto input : incoming)
        if (!check(input, frame, path))
          return false;
      return true;
    }
    if (!path.empty())
      return fail(value,
                  "the producer of a selected residual result is opaque");

    if (auto *partial = dyn_cast<PartialApplyInst>(value)) {
      if (!checkCodeIsolation(resolveFunction({partial->getCallee(), frame}),
                              value))
        return false;
      auto parameters = partial->getSubstCalleeType()->getParameters();
      auto captures = partial->getArguments();
      if (captures.size() > parameters.size())
        return fail(value,
                    "the residual closure has an invalid capture mapping");
      auto capturedParameters = parameters.take_back(captures.size());
      for (auto pair : llvm::zip(captures, capturedParameters)) {
        if (std::get<1>(pair).isIndirectInOut())
          return fail(value,
                      "a saved pullback retains mutable borrowed storage");
        if (!check(std::get<0>(pair), frame, {}))
          return false;
      }
      return true;
    }
    if (isa<FunctionRefInst>(value) ||
        isa<DifferentiabilityWitnessFunctionInst>(value))
      return checkCodeIsolation(resolveFunction({value, frame}), value);
    if (auto *e = dyn_cast<EnumInst>(value))
      return !e->hasOperand() || check(e->getOperand(), frame, {});
    if (auto *s = dyn_cast<StructInst>(value)) {
      if (type.mayHaveCustomDeinit(*frame->function))
        return fail(
            value,
            "a saved value with custom destruction needs a transfer contract");
      for (auto element : s->getElements())
        if (!check(element, frame, {}))
          return false;
      return true;
    }
    if (auto *builtin = dyn_cast<BuiltinInst>(value)) {
      if (isBuiltinInst(
              builtin,
              BuiltinValueKind::AutoDiffCreateLinearMapContextWithType))
        return checkContext(builtin, frame);
      if (isBuiltinInst(builtin,
                        BuiltinValueKind::AutoDiffAllocateSubcontextWithType) ||
          isBuiltinInst(builtin,
                        BuiltinValueKind::AutoDiffProjectTopLevelSubcontext)) {
        if (!context.isGeneratedFunction(builtin->getFunction()))
          return fail(value,
                      "an opaque subcontext pointer has no payload proof");
        return check(builtin->getArguments().front(), frame, {});
      }
    }
    return fail(value, "cannot prove that this saved derivative value can "
                       "cross the child-task boundary");
  }

public:
  ResidualTransferChecker(ADContext &context, SILFunction *root)
      : context(context), rootFunction(root) {}

  bool check(SILValue value, const Frame *frame, ArrayRef<unsigned> path) {
    // Loop-phi/context edges can be cyclic. Checking all outgoing dependencies
    // proves an invariant; a backedge is not another dynamic tape instance.
    for (const auto &query : visiting)
      if (query.value == value && query.frame == frame &&
          ArrayRef<unsigned>(query.path) == path)
        return true;
    visiting.push_back({value, frame, SmallVector<unsigned, 4>(path)});
    auto pop = llvm::make_scope_exit([&] { visiting.pop_back(); });
    return checkImpl(value, frame, path);
  }

  bool run(SILValue root) {
    Frame frame{rootFunction, nullptr, {}, {}};
    bool result = check(root, &frame, {});
    if (!result) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "[async-let-differentiation] residual-transfer proof failed: "
          << reason << '\n'
          << failure << '\n');
    }
    return result;
  }
  StringRef getReason() const { return reason; }
};
} // namespace

bool swift::autodiff::diagnoseAsyncLetResidualTransfers(ADContext &context) {
  bool error = false;
  for (const auto &transfer : context.getAsyncLetTransfers()) {
    ResidualTransferChecker checker(context, transfer.store->getFunction());
    if (checker.run(transfer.store->getSrc()))
      continue;
    context.emitNondifferentiabilityError(transfer.source, transfer.invoker,
                                          diag::autodiff_async_let_unsupported,
                                          checker.getReason());
    error = true;
  }
  return error;
}
