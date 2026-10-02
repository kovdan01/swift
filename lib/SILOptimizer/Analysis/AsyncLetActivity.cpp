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
#define DEBUG_TYPE "async-let-activity"

#include "swift/SILOptimizer/Analysis/AsyncLetActivity.h"
#include "swift/AST/ASTContext.h"
#include "swift/AST/SemanticAttrs.h"
#include "swift/SIL/SILFunction.h"
#include "swift/SIL/SILModule.h"
#include "swift/SILOptimizer/Analysis/DominanceAnalysis.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace swift;

static llvm::cl::opt<bool> EnableExperimentalAsyncAutoDiff(
    "enable-experimental-async-autodiff", llvm::cl::Hidden,
    llvm::cl::init(false),
    llvm::cl::desc(
        "Enable experimental async VJPs with synchronous pullbacks"));

bool swift::isExperimentalAsyncAutoDiffEnabled() {
  return EnableExperimentalAsyncAutoDiff;
}

SILValue swift::getAsyncLetReadValue(SILInstruction *read) {
  if (auto *copy = dyn_cast<CopyAddrInst>(read))
    return copy->getDest();
  assert(isa<LoadInst, LoadBorrowInst>(read));
  return cast<SingleValueInstruction>(read);
}

namespace {
bool hasRuntimeSemantics(ApplyInst *apply, StringRef semantics) {
  auto *callee = apply->getCalleeFunction();
  return callee && callee->hasSemanticsAttr(semantics);
}

/// Closure conversion is transport, not a numerical operation. The actual
/// capture operands remain visible on partial_apply.
SILValue stripConversions(SILValue value, AsyncLetAccess &a) {
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
    a.closureConversions.push_back(inst);
    value = operand;
  }
}

/// Compile-time lifetime verification, NOT an AD execution trace. Standard
/// LinearMapInfo/PullbackCloner remain responsible for branches and loops.
///
/// A union at CFG joins represents all states that can reach that point. An
/// operation must be valid for every incoming state. Thus waits on both arms
/// can establish readiness at a join without any single wait dominating it.
/// Loop backedges reuse the same transfer function; a loop-local child has to
/// finish/end/deallocate before its allocation executes again.
std::string validateLifetime(const AsyncLetAccess &a, SILFunction &function) {
  enum : unsigned {
    Unallocated = 1,
    Allocated = 2,
    Running = 4,
    Ready = 8,
    Finished = 16,
    Ended = 32
  };
  llvm::DenseMap<SILBasicBlock *, unsigned> incoming;
  SmallVector<SILBasicBlock *, 16> worklist;
  incoming[function.getEntryBlock()] = Unallocated;
  worklist.push_back(function.getEntryBlock());
  while (!worklist.empty()) {
    auto *block = worklist.pop_back_val();
    unsigned state = incoming.lookup(block);
    for (auto &inst : *block) {
      unsigned allowed = 0, next = 0;
      if (&inst == a.buffer) {
        allowed = Unallocated;
        next = Allocated;
      } else if (&inst == a.start) {
        allowed = Allocated;
        next = Running;
      } else if (llvm::is_contained(a.gets, &inst)) {
        allowed = Running | Ready;
        next = Ready;
      } else if (llvm::is_contained(a.reads, &inst)) {
        allowed = Ready;
        next = Ready;
      } else if (llvm::is_contained(a.finishes, &inst)) {
        allowed = Running | Ready;
        next = Finished;
      } else if (llvm::is_contained(a.ends, &inst)) {
        allowed = Finished;
        next = Ended;
      } else if (llvm::is_contained(a.deallocations, &inst)) {
        allowed = Ended;
        next = Unallocated;
      } else if (isa<ReturnInst, ThrowInst>(&inst)) {
        allowed = Unallocated;
        next = Unallocated;
      }
      if (!allowed)
        continue;
      if (state & ~allowed) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[async-let-activity] invalid lifetime state " << state
                   << " at:\n"
                   << inst);
        return "an async-let read or cleanup is not valid on every incoming "
               "control-flow path";
      }
      state = next;
    }
    for (auto *successor : block->getSuccessorBlocks()) {
      unsigned joined = incoming.lookup(successor) | state;
      if (joined != incoming.lookup(successor)) {
        incoming[successor] = joined;
        worklist.push_back(successor);
      }
    }
  }
  return {};
}

std::string recognize(AsyncLetAccess &a, SILFunction &f) {
  // These are representation/serialization restrictions, not numerical type
  // restrictions. Substituted library calls inside a concrete child are fine.
  if (f.getLoweredFunctionType()->getSubstGenericSignature())
    return "generic async-let parents require contextual helper specialization";
  if (f.isSerialized())
    return "serialized/inlinable async-let derivatives are not yet supported";
  auto args = a.start->getArguments();
  if (args.size() != 3)
    return "unexpected async-let start operands";
  a.pointer = dyn_cast<AddressToPointerInst>(args[2]);
  if (!a.pointer ||
      !(a.buffer = dyn_cast<AllocStackInst>(a.pointer->getOperand())))
    return "the task result must have a unique local stack buffer";

  auto closure = stripConversions(args[1], a);
  a.closure = dyn_cast<PartialApplyInst>(closure);
  SILValue callee = closure;
  if (a.closure) {
    if (!a.closure->getSubstitutionMap().empty())
      return "generic initializer specialization is not yet implemented";
    a.captures.append(a.closure->getArguments().begin(),
                      a.closure->getArguments().end());
    callee = a.closure->getCallee();
  }
  a.initializerRef = dyn_cast<FunctionRefInst>(callee);
  if (!a.initializerRef)
    return "the initializer must resolve to a statically known function";
  a.initializer = a.initializerRef->getReferencedFunction();
  if (a.initializer->isExternalDeclaration())
    return "the initializer body must be available for synthesis";
  auto type = a.initializer->getLoweredFunctionType()->getUnsubstitutedType(
      f.getModule());
  if (type->getSubstGenericSignature() || !type->isAsync() ||
      type->isCoroutine() || type->getNumParameters() != a.captures.size() ||
      type->getNumResults() != 1 || type->getNumIndirectFormalResults() != 1 ||
      type->getSingleResult().getSILStorageInterfaceType().getObjectType() !=
          a.getValueType())
    return "unsupported initializer result convention or reabstraction";
  for (auto parameter : type->getParameters())
    if (parameter.isAutoDiffSemanticResult())
      return "mutable inout captures are not supported in an async-let helper";
  auto subs = a.start->getSubstitutions();
  if (subs.getReplacementTypes().size() != 1 || !subs.getConformances().empty())
    return "unsupported async-let builtin substitution requirements";

  // The augmented closure replaces the original closure at its construction,
  // consuming each capture only once. Validate that the whole conversion chain
  // is private to this child before changing its result type.
  SmallVector<SILInstruction *, 8> chain(a.closureConversions.begin(),
                                         a.closureConversions.end());
  chain.push_back(a.closure ? static_cast<SILInstruction *>(a.closure)
                            : a.initializerRef);
  for (auto *node : chain)
    for (auto result : node->getResults())
      for (auto *use : result->getUses()) {
        auto *user = use->getUser();
        if (user == a.start || llvm::is_contained(chain, user) ||
            isa<DestroyValueInst, EndBorrowInst, DebugValueInst>(user) ||
            isBuiltinInst(user, BuiltinValueKind::EndAsyncLetLifetime))
          continue;
        return "the initializer closure is shared with another invocation";
      }
  DominanceInfo dominance(&f);
  if (!dominance.dominates(a.buffer, chain.back()))
    return "the async-let result allocation must dominate closure construction";

  // Account for every storage use. Typed copies/borrows are numerical reads;
  // raw pointers remain opaque transport and may not escape this association.
  for (auto *use : a.buffer->getUses()) {
    auto *user = use->getUser();
    if (user == a.pointer || isa<DebugValueInst>(user))
      continue;
    if (auto *deallocation = dyn_cast<DeallocStackInst>(user)) {
      a.deallocations.push_back(deallocation);
      continue;
    }
    return "unrecognized use of the async-let result allocation";
  }
  for (auto *use : a.pointer->getUses()) {
    auto *user = use->getUser();
    if (user == a.start)
      continue;
    if (auto *address = dyn_cast<PointerToAddressInst>(user)) {
      if (address->getType() != a.buffer->getType())
        return "the async-let result pointer is reinterpreted";
      a.addresses.push_back(address);
      for (auto *addressUse : address->getUses()) {
        auto *read = addressUse->getUser();
        if (auto *load = dyn_cast<LoadInst>(read)) {
          if (load->getOwnershipQualifier() == LoadOwnershipQualifier::Take)
            return "consuming async-let result access needs field-aware "
                   "cleanup";
        } else if (auto *copy = dyn_cast<CopyAddrInst>(read)) {
          if (copy->getSrc() != address || copy->isTakeOfSrc())
            return "the task result may only be copied, not overwritten or "
                   "taken";
          if (!copy->isInitializationOfDest())
            return "async-let address materialization must initialize its "
                   "target";
        } else if (!isa<LoadBorrowInst>(read)) {
          if (isa<DebugValueInst>(read))
            continue;
          return "unsupported projection or escape of the async-let result; "
                 "materialize the complete value before projecting it";
        }
        a.reads.push_back(read);
      }
      continue;
    }
    if (auto *finish = isBuiltinInst(user, BuiltinValueKind::FinishAsyncLet)) {
      auto operands = finish->getArguments();
      if (operands.size() != 2 || operands[0] != a.start ||
          operands[1] != a.pointer)
        return "invalid async-let finish operand association";
      a.finishes.push_back(finish);
      continue;
    }
    if (auto *apply = dyn_cast<ApplyInst>(user)) {
      auto operands = apply->getArguments();
      if (operands.size() != 2 || operands[0] != a.start ||
          operands[1] != a.pointer)
        return "invalid async-let runtime operand association";
      if (hasRuntimeSemantics(apply, semantics::ASYNC_LET_GET)) {
        a.gets.push_back(apply);
        continue;
      }
      if (hasRuntimeSemantics(apply, semantics::ASYNC_LET_FINISH)) {
        a.finishes.push_back(apply);
        continue;
      }
    }
    return "unrecognized async-let result-pointer use or missing runtime "
           "semantics; use a matching rebuilt concurrency module";
  }
  for (auto *use : a.start->getUses()) {
    auto *user = use->getUser();
    if (llvm::is_contained(a.finishes, user) ||
        llvm::is_contained(a.gets, user))
      continue;
    if (auto *end =
            isBuiltinInst(user, BuiltinValueKind::EndAsyncLetLifetime)) {
      auto operands = end->getArguments();
      if (operands.empty() || operands[0] != a.start || operands.size() > 2)
        return "invalid async-let lifetime-end operands";
      if (operands.size() == 2 &&
          !llvm::is_contained(chain, operands[1]->getDefiningInstruction()))
        return "unexpected async-let closure lifetime operand";
      a.ends.push_back(end);
      continue;
    }
    return "unrecognized use of the async-let task handle";
  }
  if (a.finishes.empty() || a.ends.empty() || a.deallocations.empty())
    return "missing async-let finish, lifetime end, or buffer deallocation";
  if (auto reason = validateLifetime(a, f); !reason.empty())
    return reason;

  a.transport.append(chain.begin(), chain.end());
  a.transport.push_back(a.buffer);
  a.transport.push_back(a.pointer);
  a.transport.push_back(a.start);
  a.transport.append(a.finishes.begin(), a.finishes.end());
  a.transport.append(a.ends.begin(), a.ends.end());
  a.transport.append(a.deallocations.begin(), a.deallocations.end());
  a.transport.append(a.addresses.begin(), a.addresses.end());
  a.transport.append(a.gets.begin(), a.gets.end());
  return {};
}
} // namespace

AsyncLetActivity::AsyncLetActivity(SILFunction &function) {
  for (auto &block : function)
    for (auto &inst : block)
      if (auto *start = isBuiltinInst(
              &inst, BuiltinValueKind::StartAsyncLetWithLocalBuffer)) {
        auto access = std::make_unique<AsyncLetAccess>();
        access->start = start;
        access->unsupportedReason = recognize(*access, function);
        accesses.push_back(std::move(access));
      }

  llvm::DenseMap<AllocStackInst *, AsyncLetAccess *> buffers;
  for (auto &a : accesses) {
    if (!a->isSupported())
      continue;
    auto inserted = buffers.try_emplace(a->buffer, a.get());
    if (!inserted.second) {
      a->unsupportedReason =
          "the result allocation is shared by distinct tasks";
      inserted.first->second->unsupportedReason = a->unsupportedReason;
    }
  }
  for (auto &a : accesses) {
    LLVM_DEBUG({
      llvm::dbgs() << "[async-let-activity] @" << function.getName() << ": ";
      if (!a->isSupported()) {
        llvm::dbgs() << a->unsupportedReason << '\n';
      } else {
        llvm::dbgs() << "recognized " << a->captures.size() << " captures, "
                     << a->reads.size() << " reads, " << a->finishes.size()
                     << " cleanup sites across the CFG\n";
      }
    });
    if (!a->isSupported())
      continue;
    for (auto *read : a->reads)
      reads[read] = a.get();
    transport.insert(a->transport.begin(), a->transport.end());
  }
}
