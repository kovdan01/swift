//===--- AsyncLetActivity.cpp - Logical async-let dataflow-----------------===//
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
#include "swift/SILOptimizer/Analysis/AsyncLetActivity.h"
#include "swift/AST/ASTContext.h"
#include "swift/AST/Decl.h"
#include "swift/SIL/SILFunction.h"
#include "swift/SIL/SILModule.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"

using namespace swift;

static llvm::cl::opt<bool> EnableExperimentalAsyncAutoDiff(
    "enable-experimental-async-autodiff", llvm::cl::Hidden,
    llvm::cl::init(false),
    llvm::cl::desc(
        "Enable experimental async VJPs with synchronous pullbacks"));

bool swift::isExperimentalAsyncAutoDiffEnabled() {
  return EnableExperimentalAsyncAutoDiff;
}

bool swift::isAsyncAutoDiffScalar(SILType type) {
  if (!type.isObject())
    return false;
  auto *decl = type.getASTType()->getStructOrBoundGenericStruct();
  if (!decl || !decl->getModuleContext()->isStdlibModule())
    return false;
  return decl->getName().str() == "Float" || decl->getName().str() == "Double";
}

namespace {
StringRef getCalleeName(ApplyInst *apply) {
  auto *ref = dyn_cast<FunctionRefInst>(apply->getCallee());
  return ref ? ref->getReferencedFunction()->getName() : StringRef();
}

bool isDebug(SILInstruction *inst) { return isa<DebugValueInst>(inst); }

/// Do NOT peer through partial_apply here: its operands are the environment.
SILValue stripConversions(SILValue value, AsyncLetAccess &access) {
  while (true) {
    auto *inst = value->getDefiningInstruction();
    SILValue operand;
    if (auto *i = dyn_cast_or_null<ConvertFunctionInst>(inst))
      operand = i->getOperand();
    else if (auto *i = dyn_cast_or_null<ConvertEscapeToNoEscapeInst>(inst))
      operand = i->getOperand();
    else if (auto *i = dyn_cast_or_null<ThinToThickFunctionInst>(inst))
      operand = i->getOperand();
    else if (auto *i = dyn_cast_or_null<CopyValueInst>(inst))
      operand = i->getOperand();
    else if (auto *i = dyn_cast_or_null<BeginBorrowInst>(inst))
      operand = i->getOperand();
    else
      return value;
    access.transport.push_back(inst);
    value = operand;
  }
}

std::string recognize(AsyncLetAccess &a, SILFunction &f) {
  if (f.size() != 1)
    return "the initial implementation requires a single-block parent";
  if (f.getLoweredFunctionType()->getSubstGenericSignature())
    return "generic parents are not yet supported";
  if (f.isSerialized())
    return "serialized/inlinable async-let derivatives are not yet supported";
  if (a.start->getName().str() != "startAsyncLetWithLocalBuffer")
    return "only startAsyncLetWithLocalBuffer is implemented";
  auto args = a.start->getArguments();
  if (args.size() != 3)
    return "unexpected async-let start operands";
  a.pointer = dyn_cast<AddressToPointerInst>(args[2]);
  if (!a.pointer ||
      !(a.buffer = dyn_cast<AllocStackInst>(a.pointer->getOperand())))
    return "the task result must have a unique local stack buffer";
  if (!isAsyncAutoDiffScalar(a.getValueType()))
    return "the initial task-result domain is Float and Double";

  auto closure = stripConversions(args[1], a);
  a.closure = dyn_cast<PartialApplyInst>(closure);
  SILValue callee = closure;
  if (a.closure) {
    if (!a.closure->getSubstitutionMap().empty())
      return "generic closure captures are not yet supported";
    a.captures.append(a.closure->getArguments().begin(),
                      a.closure->getArguments().end());
    a.transport.push_back(a.closure);
    callee = a.closure->getCallee();
  }
  // A conversion/capture node shared with another invocation must not be
  // hidden from activity analysis. Only this task and lifetime/debug uses are
  // allowed to consume the compiler-created closure chain.
  for (auto *node : a.transport)
    for (auto result : node->getResults())
      for (auto *use : result->getUses()) {
        auto *user = use->getUser();
        if (user == a.start || llvm::is_contained(a.transport, user) ||
            isa<DestroyValueInst, EndBorrowInst, DebugValueInst>(user))
          continue;
        return "the initializer closure is shared with another use";
      }
  auto *ref = dyn_cast<FunctionRefInst>(callee);
  if (!ref)
    return "the initializer must be a statically known local function";
  a.initializer = ref->getReferencedFunction();
  if (a.initializer->isExternalDeclaration())
    return "the initializer body must be available for synthesis";
  auto type = a.initializer->getLoweredFunctionType()->getUnsubstitutedType(
      f.getModule());
  if (type->getSubstGenericSignature() || !type->isAsync() ||
      type->getNumParameters() != a.captures.size() ||
      type->getNumResults() != 1 || type->getNumIndirectFormalResults() != 1 ||
      type->getSingleResult().getSILStorageInterfaceType().getObjectType() !=
          a.getValueType())
    return "unsupported initializer signature or reabstraction thunk";
  for (auto capture : a.captures)
    if (!isAsyncAutoDiffScalar(capture->getType()))
      return "the initial capture domain is by-value Float and Double";
  for (auto param : type->getParameters())
    if (param.isFormalIndirect())
      return "indirect captures are not yet supported";
  auto subs = a.start->getSubstitutions();
  if (subs.getReplacementTypes().size() != 1 || !subs.getConformances().empty())
    return "unsupported async-let builtin substitution requirements";

  // Establish ALL users of the result storage; do not guess pointer aliasing.
  for (auto *use : a.buffer->getUses()) {
    auto *user = use->getUser();
    if (user == a.pointer || isDebug(user))
      continue;
    if (auto *dealloc = dyn_cast<DeallocStackInst>(user)) {
      if (a.deallocation)
        return "multiple result-buffer deallocations are not yet supported";
      a.deallocation = dealloc;
      continue;
    }
    return "unrecognized use of the async-let result allocation";
  }
  for (auto *use : a.pointer->getUses()) {
    auto *user = use->getUser();
    if (user == a.start)
      continue;
    if (auto *addr = dyn_cast<PointerToAddressInst>(user)) {
      if (addr->getType() != a.buffer->getType())
        return "the async-let result pointer is reinterpreted";
      a.addresses.push_back(addr);
      for (auto *addressUse : addr->getUses()) {
        auto *load = dyn_cast<LoadInst>(addressUse->getUser());
        if (!load ||
            load->getOwnershipQualifier() != LoadOwnershipQualifier::Trivial)
          return "only trivial scalar result loads are implemented";
        a.reads.push_back(load);
      }
      continue;
    }
    if (auto *apply = dyn_cast<ApplyInst>(user)) {
      auto name = getCalleeName(apply);
      auto callArgs = apply->getArguments();
      if (callArgs.size() != 2 || callArgs[0] != a.start ||
          callArgs[1] != a.pointer)
        return "unrecognized async-let runtime argument association";
      if (name == "swift_asyncLet_get") {
        a.gets.push_back(apply);
        continue;
      }
      if (name == "swift_asyncLet_finish" && !a.finish) {
        a.finish = apply;
        continue;
      }
    }
    return "unrecognized use of the async-let result pointer";
  }
  for (auto *use : a.start->getUses()) {
    auto *user = use->getUser();
    if (user == a.finish || llvm::is_contained(a.gets, user))
      continue;
    if (auto *builtin = dyn_cast<BuiltinInst>(user)) {
      if (builtin->getName().str() == "endAsyncLetLifetime" &&
          builtin->getArguments().size() == 1 && !a.end) {
        a.end = builtin;
        continue;
      }
    }
    return "unrecognized use of the async-let task handle";
  }
  if (!a.finish || !a.end || !a.deallocation)
    return "missing finish, lifetime end, or buffer deallocation";

  llvm::DenseMap<SILInstruction *, unsigned> order;
  unsigned next = 0;
  for (auto &inst : f.front())
    order[&inst] = next++;
  if (!(order[a.buffer] < order[a.start] && order[a.start] < order[a.finish] &&
        order[a.finish] < order[a.end] && order[a.end] < order[a.deallocation]))
    return "unsupported async-let lifetime ordering";
  for (auto *get : a.gets)
    if (!(order[a.start] < order[get] && order[get] < order[a.finish]))
      return "unsupported async-let wait ordering";
  for (auto *read : a.reads) {
    if (order[read] >= order[a.finish] ||
        !llvm::any_of(a.gets,
                      [&](ApplyInst *get) { return order[get] < order[read]; }))
      return "the result must be read after a wait and before finish";
  }
  a.transport.push_back(a.buffer);
  a.transport.push_back(a.pointer);
  a.transport.push_back(a.start);
  a.transport.push_back(a.finish);
  a.transport.push_back(a.end);
  a.transport.push_back(a.deallocation);
  a.transport.append(a.addresses.begin(), a.addresses.end());
  a.transport.append(a.gets.begin(), a.gets.end());
  return {};
}
} // namespace

AsyncLetActivity::AsyncLetActivity(SILFunction &function) {
  for (auto &bb : function)
    for (auto &inst : bb)
      if (auto *builtin = dyn_cast<BuiltinInst>(&inst))
        if (builtin->getName().str().starts_with("startAsyncLet")) {
          auto access = std::make_unique<AsyncLetAccess>();
          access->start = builtin;
          access->unsupportedReason = recognize(*access, function);
          accesses.push_back(std::move(access));
        }

  // Shared result allocations would invalidate the one-task/one-payload model.
  llvm::DenseMap<AllocStackInst *, AsyncLetAccess *> buffers;
  for (auto &a : accesses) {
    if (!a->isSupported())
      continue;
    auto inserted = buffers.try_emplace(a->buffer, a.get());
    if (!inserted.second) {
      a->unsupportedReason = "the result buffer is shared by multiple tasks";
      inserted.first->second->unsupportedReason = a->unsupportedReason;
    }
  }
  for (auto &a : accesses) {
    if (!a->isSupported())
      continue;
    for (auto *read : a->reads)
      reads[read] = a.get();
    transport.insert(a->transport.begin(), a->transport.end());
  }
}
