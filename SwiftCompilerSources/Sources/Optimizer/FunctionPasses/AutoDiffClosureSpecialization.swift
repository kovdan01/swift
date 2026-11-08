//===--- AutoDiffClosureSpecialization.swift ---------------------------===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2024 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===-----------------------------------------------------------------------===//

import AST
import SIL

private let verbose = false

private func log(prefix: Bool = true, _ message: @autoclosure () -> String) {
  if verbose {
    debugLog(prefix: prefix, "[ADCS] " + message())
  }
}

/// AutoDiff closure specialization removes intermediate pullback contexts from
/// the reverse-mode tape. A VJP returns `(value, pullback)`; the pullback captures
/// the saved values and closures needed by the reverse pass.
///
/// For straight-line tapes, reuse the general closure specializer after VJP
/// inlining exposes the captured closures. With control flow, the tape is a
/// branch-tracing enum whose cases hold `(predecessor, saved values...)` tuples:
///
/// 1. Validate the VJP, pullback, payload uses and callable correspondence.
/// 2. Intern enum schemas that replace selected closure fields with captures.
/// 3. Clone the pullback, remap its tape types and rewrite projected field uses.
/// 4. Rewrite the VJP to construct the new payloads and capture the new tape.
///
/// Validation precedes enum synthesis and function mutation. Loop tapes and
/// unexpected payload-use shapes are declined rather than partially rewritten.
let autodiffClosureSpecialization = FunctionPass(name: "autodiff-closure-specialization") {
  (function: Function, context: FunctionPassContext) in

  guard function.hasOwnership else {
    return
  }

  guard !function.isDefinedExternally,
    function.isAutodiffVJP
  else {
    return
  }

  let isSingleBB = function.blocks.singleElement != nil
  var bteSpecEligibility = BTESpecEligibility.ineligible

  if !isSingleBB {
    log(
      "\n\nTrying to run AutoDiff Closure Specialization pass on \(function.name.string)")
    bteSpecEligibility = checkIfCanRun(vjp: function, context: context)
    if bteSpecEligibility != .ineligible {
      log(
        "The VJP \(function.name.string) has passed the preliminary check. Proceeding to running the pass"
      )
    }
  }

  var remainingSpecializationRounds = 5

  repeat {
    var changed = false

    guard let returnInst = function.returnInstruction,
      let tupleInst = returnInst.returnedValue.definingInstruction as? TupleInst,
      let lastTupleOp = tupleInst.operands.last,
      let partialApplyInst = lastTupleOp.value.definingInstruction as? PartialApplyInst,
      partialApplyInst.uses.singleUse?.instruction == tupleInst
    else {
      break
    }

    if trySpecialize(apply: partialApplyInst, isRootInvocation: true, context) {
      changed = true
    }

    if context.needFixStackNesting {
      context.fixStackNesting(in: function)
    }

    if !changed {
      break
    }

    remainingSpecializationRounds -= 1
  } while remainingSpecializationRounds > 0

  if !isSingleBB && bteSpecEligibility == .eligible {
    guard let closureAnalysis = PullbackClosureAnalysis(vjp: function) else { return }

    guard !closureAnalysis.closuresInBTE.isEmpty else {
      log(
        "Unable to detect closures to be specialized in \(function.name.string), skipping the pass")
      return
    }

    guard validatePayloadMerges(
      in: closureAnalysis.pullback, vjp: function,
      closureInfos: closureAnalysis.closuresInBTE)
    else { return }

    guard let specializedBTEDict = synthesizeSpecializedEnums(
      from: closureAnalysis, context: context)
    else { return }

    specializeAgainstBTE(
      closureAnalysis: closureAnalysis,
      specializedBTEDict: specializedBTEDict,
      function: function,
      context: context)
  }
}

private enum BTESpecEligibility {
  case noBTE
  case eligible
  case ineligible
}

private func validatePullbackBTEUsage(pb: Function, bteArg: Argument, vjp: Function, prefixFail: String) -> Bool {
  if pb.blocks.singleElement != nil {
    guard let _ = pb.entryBlock.terminator as? ReturnInst else {
      log(
        prefixFail + "unexpected terminator instruction in the entry block of the pullback "
          + pb.name.string
          + " (expected return inst for single-bb pullback)")
      log("  terminator: " + pb.entryBlock.terminator.description)
      log("  parent block begin")
      log("  " + pb.entryBlock.description)
      log("  parent block end")
      return false
    }
    log(
      "Pullback: single-bb; \(bteArg.uses.count) uses of branch tracing enum pullback argument found."
    )
    if !bteArg.uses.isEmpty {
      log(
        prefixFail + "single-bb pullback has uses of BTE arg")
      var needBreak = true
      if bteArg.uses.singleElement != nil {
        let useInst = bteArg.uses.singleUse!.instruction
        if useInst as? ApplyInst != nil {
          log("Single use of BTE arg is apply inst")
        }
        if useInst as? DestroyValueInst != nil {
          log("Single use of BTE arg is destroy_value inst")
          needBreak = false
        }
      }
      if needBreak {
        return false
      }
    }
  } else {
    if bteArg.uses.isEmpty {
      log(prefixFail + "no uses of pullback bte arg found")
      return false
    }
    if bteArg.uses.singleElement == nil {
      log(prefixFail + "multiple uses of pullback bte arg found")
      for (idx, use) in bteArg.uses.enumerated() {
        log("use \(idx): \(use)")
      }
      return false
    }

    guard bteArg.uses.singleUse!.instruction as? SwitchEnumInst != nil else {
      log(
        prefixFail + "unexpected use of BTE argument of pullback " + pb.name.string
          + " (only switch_enum_inst is supported)")
      log("  use: \(bteArg.uses.singleUse!.instruction)")
      log("  parent block begin")
      log("  \(bteArg.uses.singleUse!.instruction.parentBlock)")
      log("  parent block end")
      return false
    }
  }
  return true
}

private func validatePullbackSwitchEnumTerminators(pb: Function, prefixFail: String) -> Bool {
  for pbBB in pb.blocks {
    guard let sei = pbBB.terminator as? SwitchEnumInst else {
      continue
    }
    if sei.getSuccessorForDefault() != nil {
      log(
        prefixFail + "switch_enum_inst from the \(pbBB.shortDescription) of the pullback "
          + pb.name.string
          + " has default destination set, which is not supported")
      return false
    }
  }
  return true
}

private func validateVJPBlockBTEArgs(vjp: Function, prefixFail: String) -> Bool {
  for vjpBB in vjp.blocks {
    if vjpBB.arguments.filter ({ $0.type.isBranchTracingEnum(in: vjp) }).count > 1 {
      log(
        prefixFail + "several arguments of VJP " + vjp.name.string + " basic block "
          + vjpBB.shortDescription + " are branch tracing enums, but not more than 1 is supported"
      )
      return false
    }
  }
  return true
}

private func validatePullbackPayloadBlocks(pb: Function, vjp: Function, prefixFail: String) -> Bool {
  for pbBB in pb.blocks {
    var unwrappedPayloadCount = 0
    for arg in pbBB.arguments {
      if let incoming = getUnwrappedPayloadIncoming(arg) {
        unwrappedPayloadCount += 1
        guard validateUnwrappedPayloadPhi(arg, incoming: incoming, vjp: vjp),
              checkIfCanRunForPayloadValues(
                results: [arg], prefixFail: prefixFail, pb: pb, pbBB: pbBB)
        else { return false }
      }
    }
    let payloadInfo = getBTEPayloadArgOfPbBBInfo(pbBB, vjp: vjp)
    // Payload rewriting handles one payload tuple per basic block.
    if unwrappedPayloadCount > 1 || (unwrappedPayloadCount == 1 && payloadInfo != nil) {
      return false
    }
    guard let (argOfPbBB, _, throwingSuccessor) = payloadInfo else {
      continue
    }
    // A throwing source apply tapes Optional<Pullback>. Its function value is
    // projected as the some successor argument rather than a tuple field here.
    if let successor = throwingSuccessor {
      guard let callback = successor.arguments.singleElement,
            callback.type.isLoweredFunction,
            checkIfCanRunForPayloadValues(
              results: [callback], prefixFail: prefixFail, pb: pb, pbBB: successor)
      else { return false }
    }
    let payloadValues = getPayloadValues(payload: argOfPbBB, vjp: vjp)
    switch payloadValues {
    case .zeroUses:
      ()
    case .unsupported:
      return false
    case .destructureTuple(let results):
      if !checkIfCanRunForPayloadValues(
        results: results, prefixFail: prefixFail, pb: pb, pbBB: pbBB)
      {
        return false
      }
    case .tupleExtract(let results):
      if !checkIfCanRunForPayloadValues(
        results: results, prefixFail: prefixFail, pb: pb, pbBB: pbBB)
      {
        return false
      }
    }
  }
  return true
}

private func checkIfCanRun(vjp: Function, context: FunctionPassContext) -> BTESpecEligibility {
  assert(vjp.blocks.singleElement == nil)

  let prefixFail = "Cannot run AutoDiff Closure Specialization on \(vjp.name.string): "
  guard let paiOfPb = getPartialApplyOfPullbackInExitVJPBB(vjp: vjp) else {
    log(
      "\(prefixFail)partial_apply of pullback not found in exit basic block of VJP")
    return .ineligible
  }
  var branchTracingEnumArgCounter = 0
  for arg in paiOfPb.arguments {
    if arg.type.isBranchTracingEnum(in: vjp) {
      branchTracingEnumArgCounter += 1
    }
  }

  // Check for loops before evaluating the BTE arg count.
  for inst in vjp.instructions {
    guard let builtinInst = inst as? BuiltinInst else {
      continue
    }
    if builtinInst.name.string == "autoDiffProjectTopLevelSubcontext" {
      log(
        "\(prefixFail)VJP seems to contain a loop (builtin autoDiffProjectTopLevelSubcontext detected), this is not supported"
      )
      return .ineligible
    }
  }

  if branchTracingEnumArgCounter == 0 {
    log("This is multi-BB case which would be handled as single-BB case")
    return .noBTE
  }

  guard branchTracingEnumArgCounter == 1 else {
    log(
      "\(prefixFail)partial_apply of pullback in exit basic block of VJP has \(branchTracingEnumArgCounter) branch tracing enum arguments, but exactly 1 is expected"
    )
    return .ineligible
  }

  guard let pb = paiOfPb.referencedFunction else {
    log(
      "\(prefixFail)cannot obtain pullback function reference from the partial_apply of pullback in exit basic block of VJP"
    )
    return .ineligible
  }
  guard computeTopologicalBlockOrder(of: pb) != nil else { return .ineligible }
  guard let bteArgOfPb = pb.entryBlock.getBranchTracingEnumArg(vjp: vjp) else {
    log(
      "\(prefixFail)cannot get branch tracing enum argument of the pullback function \(pb.name.string)"
    )
    return .ineligible
  }

  guard validatePullbackBTEUsage(pb: pb, bteArg: bteArgOfPb, vjp: vjp, prefixFail: prefixFail)
  else {
    return .ineligible
  }

  guard ensureEnumPayloadsAreTupleInst(vjp: vjp) else {
    log("\(prefixFail)branch tracing enum payload is not defined by a TupleInst")
    return .ineligible
  }

  guard validatePullbackSwitchEnumTerminators(pb: pb, prefixFail: prefixFail) else {
    return .ineligible
  }

  guard validateVJPBlockBTEArgs(vjp: vjp, prefixFail: prefixFail) else {
    return .ineligible
  }

  guard validatePullbackPayloadBlocks(pb: pb, vjp: vjp, prefixFail: prefixFail) else {
    return .ineligible
  }

  return .eligible
}

extension UseList {
  var count : Int {
    var n = 0
    for _ in self {
      n += 1
    }
    return n
  }

  var isAtLeastTwo : Bool {
    !self.isEmpty && self.singleElement == nil
  }

  var isExactlyTwo : Bool {
    var n = 0
    for _ in self {
      n += 1
      if n > 2 {
        return false
      }
    }
    return n == 2
  }
}

extension BasicBlockList {
  var count : Int {
    var n = 0
    for _ in self {
      n += 1
    }
    return n
  }
}

private func ensureEnumPayloadsAreTupleInst(vjp: Function) -> Bool {
  for inst in vjp.instructions {
    guard let ei = inst as? EnumInst else {
      continue
    }
    if !ei.type.isBranchTracingEnum(in: vjp) {
      continue
    }
    guard let _ = ei.payload?.definingInstruction as? TupleInst else {
      log("Branch tracing enum payload is not defined by a TupleInst: \(ei)")
      return false
    }
  }
  return true
}

private enum PayloadValues {
  case destructureTuple([Value])
  case tupleExtract([Value])
  case zeroUses
  case unsupported
}

private enum PayloadUseShape {
  case apply(ApplyInst)
  case destroyValue(DestroyValueInst)
  case uncheckedEnumData(UncheckedEnumDataInst)
  case beginBorrow(BeginBorrowInst)
  case switchEnum(SwitchEnumInst)
}

private enum ClosurePayloadUseShape {
  case apply(ApplyInst)
  case beginBorrow(applyUse: Operand, endBorrowUse: Operand)
}

// Phi simplification can hoist the projection of a singleton BTE payload tuple
// into its predecessors. Restore that tuple in the cloned pullback so the BTE
// specialization below can keep handling payloads as tuples.
private struct UnwrappedPayloadIncoming {
  let operand: Operand
  let projection: Instruction
  let enumData: UncheckedEnumDataInst
}

private func getUnwrappedPayloadIncoming(_ arg: Argument) -> [UnwrappedPayloadIncoming]? {
  guard !arg.isReborrow else { return nil }
  var incoming = [UnwrappedPayloadIncoming]()
  var payloadType: Type? = nil
  for pred in arg.parentBlock.predecessors {
    guard let br = pred.terminator as? BranchInst else { return nil }
    let operand = br.operands[arg.index]
    guard let projection = operand.value.definingInstruction,
          operand.value.uses.singleUse == operand
    else { return nil }

    let tuple: Value
    if let dti = projection as? DestructureTupleInst, dti.results.count == 1 {
      tuple = dti.tuple
    } else if let tei = projection as? TupleExtractInst, tei.fieldIndex == 0,
              tei.tuple.type.isTrivial(in: arg.parentFunction) {
      tuple = tei.tuple
    } else {
      return nil
    }
    guard tuple.type.tupleElements.count == 1,
          let uedi = tuple.definingInstruction as? UncheckedEnumDataInst,
          uedi.uses.singleUse?.instruction == projection,
          payloadType == nil || payloadType == uedi.type
    else { return nil }
    payloadType = uedi.type
    incoming.append(UnwrappedPayloadIncoming(operand: operand, projection: projection, enumData: uedi))
  }
  return incoming.isEmpty ? nil : incoming
}

private func validateUnwrappedPayloadPhi(
  _ arg: Argument, incoming: [UnwrappedPayloadIncoming], vjp: Function
) -> Bool {
  guard incoming.allSatisfy({ $0.enumData.enum.type.isBranchTracingEnum(in: vjp) }) else {
    return false
  }
  if arg.type.isBranchTracingEnum(in: vjp) {
    return !arg.uses.isAtLeastTwo
  }
  guard arg.type.isLoweredFunction else { return false }

  // A closure phi must represent the same closure on every edge, since the
  // existing payload specialization selects one VJP payload to rebuild its uses.
  let enumToPayload = findEnumsAndPayloadsInVjp(vjp: vjp)
  var closure: Value? = nil
  for edge in incoming {
    var foundPayload = false
    for (enumInst, payload) in enumToPayload
      where enumInst.type == edge.enumData.enum.type && enumInst.caseIndex == edge.enumData.caseIndex
    {
      guard payload.operands.count == 1 else { return false }
      let value = payload.operands[0].value
      if let closure, closure != value { return false }
      closure = value
      foundPayload = true
    }
    if !foundPayload { return false }
  }
  return true
}

private func restoreUnwrappedPayloadPhis(in pullback: Function, vjp: Function, _ context: FunctionPassContext) {
  for bb in pullback.blocks {
    for arg in Array(bb.arguments) {
      guard let incoming = getUnwrappedPayloadIncoming(arg),
            incoming.allSatisfy({ $0.enumData.enum.type.isBranchTracingEnum(in: vjp) })
      else { continue }
      let newArg = bb.insertPhiArgument(
        atPosition: arg.index, type: incoming[0].enumData.type, ownership: arg.ownership, context)
      let builder = Builder(atBeginOf: bb, context)
      let replacement: Value
      if incoming[0].projection is DestructureTupleInst {
        replacement = builder.createDestructureTuple(tuple: newArg).results[0]
      } else {
        replacement = builder.createTupleExtract(tuple: newArg, elementIndex: 0)
      }
      arg.uses.replaceAll(with: replacement, context)
      bb.eraseArgument(at: arg.index, context)
      for edge in incoming {
        edge.operand.set(to: edge.enumData, context)
        context.erase(instruction: edge.projection)
      }
    }
  }
}

private func classifyUncheckedEnumDataPayloadUse(_ uedi: UncheckedEnumDataInst) -> Operand? {
  if uedi.uses.isAtLeastTwo {
    return nil
  }
  if let singleUse = uedi.uses.singleUse {
    if singleUse.instruction is BranchInst {
      return singleUse
    }
    if let projection = singleUse.instruction as? DestructureTupleInst,
       let result = projection.results.singleElement,
       let branchUse = result.uses.singleUse,
       let phi = Phi(using: branchUse),
       getUnwrappedPayloadIncoming(phi.value) != nil {
      return singleUse
    }
    if let projection = singleUse.instruction as? TupleExtractInst,
       let branchUse = projection.uses.singleUse,
       let phi = Phi(using: branchUse),
       getUnwrappedPayloadIncoming(phi.value) != nil {
      return singleUse
    }
    return nil
  }
  return nil
}

private func classifyBeginBorrowPayloadUse(_ bbi: BeginBorrowInst)
  -> (applyUse: Operand, endBorrowUse: Operand)?
{
  guard bbi.uses.isExactlyTwo else {
    return nil
  }

  var applyUse: Operand? = nil
  var endBorrowUse: Operand? = nil
  for bbiUse in bbi.uses {
    switch bbiUse.instruction {
    case let ai as ApplyInst:
      guard bbiUse == ai.operands[ApplyOperandConventions.calleeIndex],
            applyUse == nil else {
        return nil
      }
      applyUse = bbiUse
    case _ as EndBorrowInst:
      guard endBorrowUse == nil else {
        return nil
      }
      endBorrowUse = bbiUse
    default:
      return nil
    }
  }

  assert(applyUse != nil)
  assert(endBorrowUse != nil)
  return (applyUse!, endBorrowUse!)
}

private func classifyClosurePayloadUse(_ use: Operand) -> ClosurePayloadUseShape? {
  switch use.instruction {
  case let ai as ApplyInst:
    guard use == ai.operands[ApplyOperandConventions.calleeIndex] else { return nil }
    return .apply(ai)
  case let bbi as BeginBorrowInst:
    guard let classified = classifyBeginBorrowPayloadUse(bbi) else {
      return nil
    }
    return .beginBorrow(
      applyUse: classified.applyUse,
      endBorrowUse: classified.endBorrowUse)
  default:
    return nil
  }
}

private func classifyPayloadUse(_ use: Operand) -> PayloadUseShape? {
  switch use.instruction {
  case let ai as ApplyInst:
    guard use == ai.operands[ApplyOperandConventions.calleeIndex] else { return nil }
    return .apply(ai)
  case let dvi as DestroyValueInst:
    return .destroyValue(dvi)
  case let uedi as UncheckedEnumDataInst:
    guard uedi.uses.isEmpty || classifyUncheckedEnumDataPayloadUse(uedi) != nil else { return nil }
    return .uncheckedEnumData(uedi)
  case is ConvertFunctionInst:
    // Generated taped callbacks are invoked with their stored function type.
    // Do not synthesize conversions for an unexpected payload callable.
    return nil
  case let bbi as BeginBorrowInst:
    guard classifyBeginBorrowPayloadUse(bbi) != nil else { return nil }
    return .beginBorrow(bbi)
  case let sei as SwitchEnumInst:
    return .switchEnum(sei)
  default:
    return nil
  }
}

private func getPayloadValues(payload: Argument, vjp: Function) -> PayloadValues {
  if payload.uses.isEmpty {
    return PayloadValues.zeroUses
  }

  var results = [Value]()

  if let singleUse = payload.uses.singleUse,
     let dti = singleUse.instruction as? DestructureTupleInst
  {
    // TODO: do we need to check that results is not empty?
    if dti.operands[0].value.type.tupleElements.count != 0
      && dti.results[0].type.isBranchTracingEnum(in: vjp) && dti.results[0].uses.isAtLeastTwo
    {
      return PayloadValues.unsupported
    }
    for result in dti.results {
      results.append(result)
    }
    return PayloadValues.destructureTuple(results)
  }

  var idxs = [Int]()
  for use in payload.uses {
    guard let tei = use.instruction as? TupleExtractInst else {
      return PayloadValues.unsupported
    }
    if idxs.contains(tei.fieldIndex) {
      return PayloadValues.unsupported
    }
    if tei.fieldIndex == 0 && tei.type.isBranchTracingEnum(in: vjp)
      && tei.results[0].uses.isAtLeastTwo
    {
      return PayloadValues.unsupported
    }
    idxs.append(tei.fieldIndex)
    results.append(tei)
  }
  return PayloadValues.tupleExtract(results)
}

// PullbackCloner emits one invocation per active source apply. Multiple uses
// exposed by inlining custom derivatives belong to nested capture tuples, not
// to these directly taped function fields.
private func hasSingleTapedInvocation(_ value: Value) -> Bool {
  var invocationCount = 0
  for use in value.uses {
    if use.instruction is DestroyValueInst { continue }
    guard classifyClosurePayloadUse(use) != nil else { return false }
    invocationCount += 1
  }
  return invocationCount == 1
}

private func checkIfCanRunForPayloadValues(
  results: [Value], prefixFail: String, pb: Function, pbBB: BasicBlock
) -> Bool {
  for result in results {
    if result.type.isLoweredFunction && !hasSingleTapedInvocation(result) {
      return false
    }
    for use in result.uses {
      guard classifyPayloadUse(use) != nil else {
        log(
          prefixFail + "unexpected use of an element of the tuple being argument of pullback "
            + pb.name.string + " basic block " + pbBB.shortDescription)
        log("  result: \(result)")
        log("  use.instruction: \(use.instruction)")
        return false
      }


    }
  }
  return true
}

private func logSpecializationStats(
  closuresSet: Set<SingleValueInstruction>,
  totalSupportedClosures: Int,
  function: Function
) {
  var totalClosures: Int = 0
  for inst in function.instructions {
    let paiOpt = inst as? PartialApplyInst
    let tttfOpt = inst as? ThinToThickFunctionInst
    if paiOpt != nil || tttfOpt != nil {
      totalClosures += 1
    }
  }

  let specializedClosures = totalSupportedClosures - closuresSet.count
  var msg =
    "Specialized \(specializedClosures) out of \(totalSupportedClosures) supported closures "
  msg += "(rate \(Float(specializedClosures) / Float(totalSupportedClosures))). "
  msg += "Total number of closures is \(totalClosures)"
  log(msg)
}

private func synthesizeSpecializedEnums(
  from closureAnalysis: PullbackClosureAnalysis,
  context: FunctionPassContext
) -> SpecializedBTEMap? {
  getSpecializedBTEDict(
    closuresInBTE: closureAnalysis.closuresInBTE,
    paiOfPullback: closureAnalysis.paiOfPullback,
    context)
}

private func eraseDeadSpecializedClosures(
  from closuresInBTE: [ClosureInBTE],
  context: FunctionPassContext
) -> Set<SingleValueInstruction> {
  var closuresSet = Set<SingleValueInstruction>()
  var allocStacks = Set<AllocStackInst>()

  for closureInfo in closuresInBTE {
    if let pai = closureInfo.closure as? PartialApplyInst {
      for op in pai.argumentOperands {
        guard op.value.type.isAddress else {
          continue
        }
        guard !pai.convention(of: op)!.isInout else {
          continue
        }
        // These invariants hold due to prior checks in Instruction.asSupportedClosure
        let allocStack = op.value as! AllocStackInst
        assert(allocStack.uses.count == 3)
        allocStacks.insert(allocStack)
      }
    }

    closuresSet.insert(closureInfo.closure)
    for reabstraction in closureInfo.reabstractions {
      closuresSet.insert(reabstraction)
    }
  }

  var oldSetSize = 0
  repeat {
    oldSetSize = closuresSet.count
    let closures = Array(closuresSet)
    for closure in closures {
      if closure.uses.isEmpty {
        context.erase(instruction: closure)
        closuresSet.remove(closure)
      }
    }
  } while oldSetSize != closuresSet.count

  for allocStack in allocStacks {
    assert(allocStack.uses.count == 2)
    for use in allocStack.uses {
      assert(use.instruction is DeallocStackInst || use.instruction is StoreInst)
      context.erase(instruction: use.instruction)
    }
    context.erase(instruction: allocStack)
  }

  return closuresSet
}

private func specializeAgainstBTE(
  closureAnalysis: PullbackClosureAnalysis,
  specializedBTEDict: SpecializedBTEMap,
  function: Function,
  context: FunctionPassContext
) {
  let totalSupportedClosures = Set(closureAnalysis.closuresInBTE.map { $0.closure }).count

  let autodiffSpecializationInfo = AutoDiffSpecializationInfo(
    closureAnalysis: closureAnalysis,
    specializedBTEDict: specializedBTEDict)

  let specInfo = PullbackSpecializationAgainstBTE()

  let (specializedFunction, alreadyExists) =
    specInfo.getOrCreatePullbackSpecializedAgainstBTE(
      basedOn: autodiffSpecializationInfo,
      enumDict: autodiffSpecializationInfo.specializedBTEDict,
      context)

  if !alreadyExists {
    context.notifyNewFunction(
      function: specializedFunction, derivedFrom: autodiffSpecializationInfo.pullback)
  }

  rewriteVJPForBTEPullbackSpecialization(
    using: specializedFunction,
    autodiffSpecializationInfo: autodiffSpecializationInfo,
    enumDict: autodiffSpecializationInfo.specializedBTEDict,
    context: context)

  let remainingClosures = eraseDeadSpecializedClosures(
    from: closureAnalysis.closuresInBTE, context: context)

  logSpecializationStats(
    closuresSet: remainingClosures,
    totalSupportedClosures: totalSupportedClosures,
    function: function)
}

private func replaceEnumInstructionsWithSpecializedTypes(
  in vjp: Function, enumDict: SpecializedBTEMap, context: FunctionPassContext
) {
  for inst in vjp.instructions {
    guard let ei = inst as? EnumInst else {
      continue
    }
    guard let newEnumType = enumDict[ei.results[0].type] else {
      continue
    }

    let builder = Builder(before: ei, context)
    let newEI = builder.createEnum(
      caseIndex: ei.caseIndex, payload: ei.payload, enumType: newEnumType)
    ei.replace(with: newEI, context)
  }
}

private func specializeBTEBlockArgsInVJP(
  vjp: Function, enumDict: SpecializedBTEMap, context: FunctionPassContext
) {
  for bb in vjp.blocks {
    guard let arg = bb.getBranchTracingEnumArg(vjp: vjp) else {
      continue
    }
    if enumDict[arg.type] == nil {
      continue
    }
    let newArg = specializeBranchTracingEnumBBArgInVJP(
      arg: arg, specializedBTEDict: enumDict, context: context)
    arg.uses.replaceAll(with: newArg, context)
    bb.eraseArgument(at: arg.index, context)
  }
}

private func replacePullbackPartialApply(
  pai: PartialApplyInst, specializedCallee: Function, context: FunctionPassContext
) -> PartialApplyInst {
  let builderSucc = Builder(
    before: pai,
    location: pai.parentBlock.instructions.last!.location, context)

  let newFunctionRefInst = builderSucc.createFunctionRef(specializedCallee)
  let newPai: PartialApplyInst = builderSucc.createPartialApply(
    function: newFunctionRefInst, substitutionMap: pai.substitutionMap,
    capturedArguments: Array(pai.arguments), calleeConvention: pai.calleeConvention,
    hasUnknownResultIsolation: pai.hasUnknownResultIsolation, isOnStack: pai.isOnStack, isNested: pai.isNested,
    executionSemantics: pai.executionSemantics)

  pai.replace(with: newPai, context)
  return newPai
}

private func buildReplacementElement(
  capturedArgs: [Value], isOptionalSome: Bool?,
  tupleType: Type, vjp: Function, builder: Builder
) -> Value {
  if let isOptionalSome = isOptionalSome {
    let optionalTupleType = tupleType.rawType.optionalType.loweredType(in: vjp)
    if isOptionalSome {
      let tuple = builder.createTuple(type: tupleType, elements: capturedArgs)
      return builder.createOptionalSome(operand: tuple, type: optionalTupleType)
    } else {
      return builder.createOptionalNone(type: optionalTupleType)
    }
  } else {
    return builder.createTuple(type: tupleType, elements: capturedArgs)
  }
}

private func rewritePayloadTuplesInVJP(
  vjp: Function, closureInfos: inout [ClosureInBTE], context: FunctionPassContext
) {
  let enumToPayload = findEnumsAndPayloadsInVjp(vjp: vjp)
  let payloads = Set<TupleInst>(enumToPayload.values)

  for payload in payloads {
    let ti = payload
    if ti.operands.count == 0 {
      continue
    }

    var indexesToExclude = [Int]()

    var tupleIdxToCapturedArgs = [Int: (values: [Value], isOptionalSome: Bool?)]()
    for (idx, closureInfo) in closureInfos.enumerated() {
      if closureInfo.payloadTuple != ti {
        continue
      }
      indexesToExclude.append(idx)
      let idxInTuple = closureInfo.indexInPayload
      assert(ti.operands[idxInTuple].value == closureInfo.valueInPayload)
      var isOptionalSome: Bool? = nil
      if closureInfo.optionalWrapper != nil {
        isOptionalSome = !closureInfo.optionalWrapper!.isOptionalNone
      }
      tupleIdxToCapturedArgs[idxInTuple] = (
        values: closureInfo.capturedArgs, isOptionalSome: isOptionalSome
      )
    }

    for idx in indexesToExclude.reversed() {
      closureInfos.remove(at: idx)
    }

    var newPayloadValues = [Value]()
    for (opIdx, op) in ti.operands.enumerated() {
      if tupleIdxToCapturedArgs[opIdx] == nil {
        newPayloadValues.append(op.value)
        continue
      }

      if tupleIdxToCapturedArgs[opIdx]!.isOptionalSome != nil {
        assert(opIdx + 1 == ti.operands.count)
      }
      let builderPred = Builder(before: ti, context)
      let entry = tupleIdxToCapturedArgs[opIdx]!
      let tupleType = context.getTupleType(
        elements: entry.values.map { $0.type }
      ).loweredType(in: vjp)
      let replacement = buildReplacementElement(
        capturedArgs: entry.values, isOptionalSome: entry.isOptionalSome,
        tupleType: tupleType, vjp: vjp, builder: builderPred)
      newPayloadValues.append(replacement)
    }

    let oldElements = ti.type.tupleElements
    var newLabeledElements = [(label: Identifier, type: AST.`Type`)]()
    for (idx, value) in newPayloadValues.enumerated() {
      newLabeledElements.append(
        (
          label: oldElements.label(at: idx),
          type: value.type.rawType
        ))
    }
    let newTupleType = context.getTupleType(elements: newLabeledElements)
      .loweredType(in: vjp)

    let builderPred = Builder(before: ti, context)
    let newPayload = builderPred.createTuple(type: newTupleType, elements: newPayloadValues)
    ti.replace(with: newPayload, context)
  }
}

private func cleanupDeadOptionalEnums(in vjp: Function, context: FunctionPassContext) {
  var wasUpdated = false
  repeat {
    wasUpdated = false
    for inst in vjp.instructions {
      guard let enumOpt = inst as? EnumInst else {
        continue
      }
      if enumOpt.type.isOptional && enumOpt.uses.isEmpty {
        context.erase(instruction: enumOpt)
        wasUpdated = true
      }
    }
  } while wasUpdated
}

private func rewriteVJPForBTEPullbackSpecialization(
  using specializedCallee: Function, autodiffSpecializationInfo: AutoDiffSpecializationInfo,
  enumDict: SpecializedBTEMap,
  context: FunctionPassContext
) {
  let vjp = autodiffSpecializationInfo.paiOfPullback.parentFunction
  var closureInfos = autodiffSpecializationInfo.closuresInBTE

  replaceEnumInstructionsWithSpecializedTypes(in: vjp, enumDict: enumDict, context: context)

  specializeBTEBlockArgsInVJP(vjp: vjp, enumDict: enumDict, context: context)

  let _ = replacePullbackPartialApply(
    pai: autodiffSpecializationInfo.paiOfPullback, specializedCallee: specializedCallee,
    context: context)

  rewritePayloadTuplesInVJP(vjp: vjp, closureInfos: &closureInfos, context: context)

  cleanupDeadOptionalEnums(in: vjp, context: context)
}

private func findEnumsAndPayloadsInVjp(vjp: Function) -> [EnumInst: TupleInst] {
  var dict = [EnumInst: TupleInst]()
  for inst in vjp.instructions {
    guard let ei = inst as? EnumInst else {
      continue
    }
    if !ei.type.isBranchTracingEnum(in: vjp) {
      continue
    }
    let ti = ei.operands[0].value.definingInstruction as! TupleInst
    dict[ei] = ti
  }
  return dict
}

private func getParametersForPullbackSpecializedAgainstBTE(
  basedOn autodiffSpecializationInfo: AutoDiffSpecializationInfo, pb: Function, enumType: Type,
  enumDict: SpecializedBTEMap,
  _ context: FunctionPassContext
) -> [ParameterInfo] {
  let applySiteCallee = autodiffSpecializationInfo.pullback
  var specializedParamInfoList: [ParameterInfo] = []
  var foundBranchTracingEnumParam = false
  // Start by adding all original parameters except for the closure parameters.
  for paramInfo in applySiteCallee.convention.parameters {
    if paramInfo.type != enumType.rawType.mapOutOfEnvironment().canonical {
      specializedParamInfoList.append(paramInfo)
      continue
    }
    assert(!foundBranchTracingEnumParam)
    foundBranchTracingEnumParam = true
    let newParamInfo = ParameterInfo(
      type: enumDict[enumType]!.rawType.mapOutOfEnvironment().canonical,
      convention: paramInfo.convention,
      options: paramInfo.options, hasLoweredAddresses: paramInfo.hasLoweredAddresses)
    specializedParamInfoList.append(newParamInfo)
  }
  assert(foundBranchTracingEnumParam)
  return specializedParamInfoList
}

private func getEnumCasesForSwitchEnumInst(_ sei: SwitchEnumInst) -> [(Int, BasicBlock)] {
  (0..<sei.numCases).compactMap { i in
    sei.getUniqueSuccessor(forCaseIndex: i).map { (i, $0) }
  }
}

private struct PayloadRewriteContext {
  let closureInfoArray: [ClosureInBTE]
  let useTei: Bool
  let throwingSuccessor: BasicBlock?
}

private func findMatchingClosureInfo(
  in closureInfoArray: [ClosureInBTE], forPayloadIndex index: Int
) -> ClosureInBTE? {
  var result: ClosureInBTE? = nil
  for closureInfo in closureInfoArray {
    if closureInfo.indexInPayload == index {
      if result != nil {
        assert(result!.closure == closureInfo.closure)
        assert(result!.payloadTuple == closureInfo.payloadTuple)
      } else {
        result = closureInfo
      }
    }
  }
  return result
}

private func insertLifetimeEndIfNeeded(
  for value: Value, before insertionPoint: Instruction, _ context: FunctionPassContext
) {
  if value.type.isTrivial(in: value.parentFunction) {
    return
  }
  for use in value.uses {
    if use.endsLifetime {
      return
    }
  }
  let builder = Builder(before: insertionPoint, context)
  if value.parentFunction.hasOwnership {
    builder.createDestroyValue(operand: value)
  } else {
    // TODO: we are OSSA-only? So delete this?
    builder.createReleaseValue(operand: value)
  }
}

private func extractTupleElements(
  from tuple: Value, useTupleExtract: Bool, builder: Builder
) -> [Value] {
  var elements = [Value]()
  if useTupleExtract {
    for (tupleIdx, _) in tuple.type.tupleElements.enumerated() {
      elements.append(builder.createTupleExtract(tuple: tuple, elementIndex: tupleIdx))
    }
  } else {
    for result in builder.createDestructureTuple(tuple: tuple).results {
      elements.append(result)
    }
  }
  return elements
}

private func rewriteBeginBorrowUse(
  bbi: BeginBorrowInst, resultIdx: Int,
  result: Value, rewriteCtx: PayloadRewriteContext, _ context: FunctionPassContext
) {
  if findMatchingClosureInfo(in: rewriteCtx.closureInfoArray, forPayloadIndex: resultIdx) != nil {
    guard let classified = classifyBeginBorrowPayloadUse(bbi) else {
      assert(false)
      return
    }
    context.erase(instruction: classified.endBorrowUse.instruction)
    rewriteUsesOfPayloadItem(
      use: classified.applyUse, resultIdx: resultIdx,
      result: result, rewriteCtx: rewriteCtx, context)
    context.erase(instruction: bbi)
  } else {
    let builder = Builder(before: bbi, context)
    let newBBI = builder.createBeginBorrow(
      of: result,
      isLexical: bbi.isLexical,
      hasPointerEscape: bbi.hasPointerEscape,
      isFromVarDecl: bbi.isFromVarDecl)
    bbi.replace(with: newBBI, context)
  }
}

private func rewriteApplyDirectClosure(
  ai: ApplyInst, closureInfo: ClosureInBTE, extractedElements: [Value],
  builder: Builder, _ context: FunctionPassContext
) {
  if closureInfo.reabstractions.isEmpty {
    var capturedArguments = extractedElements
    var indirectCaptures = [(allocation: AllocStackInst, convention: ArgumentConvention, isTrivial: Bool)]()
    if let partialApply = closureInfo.closure as? PartialApplyInst {
      for (index, operand) in partialApply.argumentOperands.enumerated() {
        let convention = partialApply.convention(of: operand)!
        if operand.value.type.isAddress && !convention.isInout {
          let capture = capturedArguments[index]
          let allocation = builder.createAllocStack(capture.type)
          builder.createStore(source: capture, destination: allocation, ownership: .initialize)
          capturedArguments[index] = allocation
          indirectCaptures.append((allocation, convention, capture.type.isTrivial(in: ai.parentFunction)))
        }
      }
    }
    var newArgs = Array(ai.arguments)
    newArgs.append(contentsOf: capturedArguments)
    let vjpFn = closureInfo.closure.asSupportedClosureFn!
    let newFri = builder.createFunctionRef(vjpFn)
    // A thick closure application has already applied the root function's
    // generic substitutions. Recover them from the closure being eliminated.
    let substitutions = (closureInfo.closure as? PartialApplyInst)?.substitutionMap ?? ai.substitutionMap
    let newAi = builder.createApply(
      function: newFri, substitutions.substitute(with: ai.parentFunction.forwardingSubstitutionMap),
      arguments: newArgs)
    // An indirect capture is part of the closure context, but a direct call
    // needs address storage lasting through the call itself. Only consuming
    // conventions transfer destruction of the stored capture to the callee.
    for capture in indirectCaptures.reversed() {
      if !capture.convention.isConsumed && !capture.isTrivial {
        builder.createDestroyAddr(address: capture.allocation)
      }
      builder.createDeallocStack(capture.allocation)
    }
    ai.replace(with: newAi, context)

    // TODO: maybe we can set insertion point earlier
    for res in extractedElements {
      insertLifetimeEndIfNeeded(for: res, before: newAi.parentBlock.terminator, context)
    }

    return
  }

  var extractedElements = extractedElements

  let vjpFn = closureInfo.closure.asSupportedClosureFn!
  let newFri = builder.createFunctionRef(vjpFn)
  var indirectArgs = [(idx: Int, allocStack: AllocStackInst, store: StoreInst)]()
  let rootClosure : SingleValueInstruction
  if let pai = closureInfo.closure as? PartialApplyInst {
    for idx in 0..<extractedElements.count {
      if pai.arguments[idx].type.isAddress && !pai.convention(of: pai.argumentOperands[idx])!.isInout {
        let allocStack = builder.createAllocStack(extractedElements[idx].type)
        let store = builder.createStore(source: extractedElements[idx], destination: allocStack, ownership: StoreInst.StoreOwnership.initialize)
        indirectArgs.append((idx: idx, allocStack: allocStack, store: store))
        extractedElements[idx] = allocStack
      }
    }

    rootClosure = builder.createPartialApply(
      function: newFri,
      substitutionMap: pai.substitutionMap.substitute(with: ai.parentFunction.forwardingSubstitutionMap),
      capturedArguments: extractedElements,
      calleeConvention: pai.calleeConvention,
      hasUnknownResultIsolation: pai.hasUnknownResultIsolation,
      isOnStack: pai.isOnStack,
      isNested: pai.isNested,
      executionSemantics: pai.executionSemantics)

    for (idx, allocStack, store) in indirectArgs.reversed() {
      extractedElements[idx] = store.source
      builder.createDeallocStack(allocStack)
    }
  } else {
    assert(closureInfo.closure is ThinToThickFunctionInst)
    let type = ai.parentFunction.mapTypeIntoEnvironment(
      closureInfo.closure.type.rawType.mapOutOfEnvironment()).canonical.loweredType(in: ai.parentFunction)
    rootClosure = builder.createThinToThickFunction(thinFunction: newFri, resultType: type)
  }

  var allNewClosures : [SingleValueInstruction] = [rootClosure]

  var currentClosure = rootClosure
  for reabstraction in closureInfo.reabstractions.dropLast() {
    let reabstractionFn = reabstraction.asSupportedClosureFn!
    let reabstractionFri = builder.createFunctionRef(reabstractionFn)
    let newReabstraction = builder.createPartialApply(
      function: reabstractionFri,
      substitutionMap: reabstraction.substitutionMap.substitute(with: ai.parentFunction.forwardingSubstitutionMap),
      capturedArguments: [currentClosure],
      calleeConvention: reabstraction.calleeConvention,
      hasUnknownResultIsolation: reabstraction.hasUnknownResultIsolation,
      isOnStack: reabstraction.isOnStack,
      isNested: reabstraction.isNested,
      executionSemantics: reabstraction.executionSemantics)
    currentClosure = newReabstraction
    allNewClosures.append(newReabstraction)
  }

  var newArgs = Array(ai.arguments)
  newArgs.append(currentClosure)
  let fn = closureInfo.reabstractions.last!.asSupportedClosureFn!
  let newFriForAi = builder.createFunctionRef(fn)
  let newAi = builder.createApply(
    function: newFriForAi,
    closureInfo.reabstractions.last!.substitutionMap.substitute(with: ai.parentFunction.forwardingSubstitutionMap),
    arguments: newArgs)
  ai.replace(with: newAi, context)

  for newClosure in allNewClosures.reversed() {
    insertLifetimeEndIfNeeded(for: newClosure, before: newAi.parentBlock.terminator, context)
  }

  // TODO: maybe we can set insertion point earlier
  for res in extractedElements {
    insertLifetimeEndIfNeeded(for: res, before: newAi.parentBlock.terminator, context)
  }
}

private func rewriteApplyUse(
  ai: ApplyInst, resultIdx: Int,
  result: Value, rewriteCtx: PayloadRewriteContext, _ context: FunctionPassContext
) {
  let builder = Builder(before: ai, context)
  if let closureInfo = findMatchingClosureInfo(
    in: rewriteCtx.closureInfoArray, forPayloadIndex: resultIdx)
  {
    let extractedElements = extractTupleElements(
      from: result, useTupleExtract: rewriteCtx.useTei, builder: builder)
    rewriteApplyDirectClosure(
      ai: ai, closureInfo: closureInfo, extractedElements: extractedElements,
      builder: builder, context)
  } else {
    let newAi = builder.createApply(
      function: result, ai.substitutionMap, arguments: Array(ai.arguments))
    ai.replace(with: newAi, context)
  }
}

private func rewriteDestroyValueUse(
  dvi: DestroyValueInst, resultIdx: Int,
  result: Value, rewriteCtx: PayloadRewriteContext, _ context: FunctionPassContext
) {
  let isClosurePayload = rewriteCtx.closureInfoArray.contains { $0.indexInPayload == resultIdx }
  if !isClosurePayload {
    let builder = Builder(before: dvi, context)
    if dvi.parentFunction.hasOwnership {
      builder.createDestroyValue(operand: result)
    } else {
      // TODO: we are OSSA-only? So delete this?
      builder.createReleaseValue(operand: result)
    }
  }
  context.erase(instruction: dvi)
}

private func rewriteUsesOfPayloadItem(
  use: Operand, resultIdx: Int,
  result: Value, rewriteCtx: PayloadRewriteContext, _ context: FunctionPassContext
) {
  guard let payloadUseShape = classifyPayloadUse(use) else {
    assert(false)
    return
  }

  switch payloadUseShape {
  case .beginBorrow(let bbi):
    rewriteBeginBorrowUse(
      bbi: bbi, resultIdx: resultIdx,
      result: result, rewriteCtx: rewriteCtx, context)

  case .apply(let ai):
    rewriteApplyUse(
      ai: ai, resultIdx: resultIdx,
      result: result, rewriteCtx: rewriteCtx, context)

  case .destroyValue(let dvi):
    rewriteDestroyValueUse(
      dvi: dvi, resultIdx: resultIdx,
      result: result, rewriteCtx: rewriteCtx, context)

  case .uncheckedEnumData(let uedi):
    let builder = Builder(before: uedi, context)
    let newUedi = builder.createUncheckedEnumData(
      enum: result, caseIndex: uedi.caseIndex,
      resultType: result.type.getEnumCases(in: uedi.parentFunction)![uedi.caseIndex]!.payload!)
    uedi.replace(with: newUedi, context)

  case .switchEnum(let sei):
    let builder = Builder(before: sei, context)
    builder.createSwitchEnum(
      enum: result, cases: getEnumCasesForSwitchEnumInst(sei))
    context.erase(instruction: sei)

    if let successor = rewriteCtx.throwingSuccessor {
      let arg = successor.arguments.singleElement!
      let argUses = Array(arg.uses)
      for argUse in argUses {
        rewriteUsesOfPayloadItem(
          use: argUse, resultIdx: resultIdx,
          result: arg, rewriteCtx: rewriteCtx, context)
      }
    }
  }
}

// ===================== Utility functions and extensions ===================== //

private extension EnumInst {
  var caseName: StringRef {
    return self.type.getEnumCases(in: self.parentFunction)![self.caseIndex]!.name
  }

  var isOptionalSome: Bool {
    assert(self.type.isOptional)
    assert(self.caseIndex == Builder.optionalNoneCaseIndex || self.caseIndex == Builder.optionalSomeCaseIndex)
    return self.caseIndex == Builder.optionalSomeCaseIndex
  }

  var isOptionalNone: Bool {
    assert(self.type.isOptional)
    assert(self.caseIndex == Builder.optionalNoneCaseIndex || self.caseIndex == Builder.optionalSomeCaseIndex)
    return self.caseIndex == Builder.optionalNoneCaseIndex
  }
}

private extension Collection {
  var singleElementAssumingAtMostOne : Element? {
    assert(self.count <= 1)
    return self.singleElement
  }
}

private extension BasicBlock {
  func getBranchTracingEnumArg(vjp: Function) -> Argument? {
    return self.arguments.filter { $0.type.isBranchTracingEnum(in: vjp) }.singleElementAssumingAtMostOne
  }
}

// Consider the code snippet below. This is what returning basic block of VJP usually ends with.
//   // function_ref pullback of foo(_:)
//   %pullbackOfFoo = function_ref @$pullbackOfFoo : $@convention(thin) (Float, _AD__$foo_bbA__Pred__src_0_wrt_0) -> Float
//   %paiOfPullback = partial_apply [callee_guaranteed] %pullbackOfFoo(%bte) : $@convention(thin) (Float, _AD__$foo_bbA__Pred__src_0_wrt_0) -> Float
//   %resWithPullback = tuple (%result, %paiOfPullback)
//   return %resWithPullback
//
// The function below finds %paiOfPullback in terms of the example above.
private func getPartialApplyOfPullbackInExitVJPBB(vjp: Function) -> PartialApplyInst? {
  log("getPartialApplyOfPullbackInExitVJPBB: running for VJP \(vjp.name)")
  guard let exitBB = vjp.blocks.filter({ $0.terminator as? ReturnInst != nil }).singleElementAssumingAtMostOne
  else {
    log("getPartialApplyOfPullbackInExitVJPBB: exit BB not found, aborting")
    return nil
  }

  let ri = exitBB.terminator as! ReturnInst
  guard let retValDefiningInstr = ri.returnedValue.definingInstruction else {
    log("getPartialApplyOfPullbackInExitVJPBB: return value is not defined by an instruction, aborting")
    return nil
  }

  func handleConvertFunctionOrPartialApply(inst: Instruction) -> PartialApplyInst? {
    if let pai = inst as? PartialApplyInst {
      log("getPartialApplyOfPullbackInExitVJPBB: success")
      return pai
    }
    if let cfi = inst as? ConvertFunctionInst {
      if let pai = cfi.fromFunction as? PartialApplyInst {
        log("getPartialApplyOfPullbackInExitVJPBB: success")
        return pai
      }
      log("getPartialApplyOfPullbackInExitVJPBB: fromFunction operand of convert_function instruction is not defined by partial_apply instruction, aborting")
      return nil
    }
    log("getPartialApplyOfPullbackInExitVJPBB: unexpected instruction type, aborting")
    return nil
  }

  if let ti = retValDefiningInstr as? TupleInst {
    log("getPartialApplyOfPullbackInExitVJPBB: return value is defined by tuple instruction")
    if ti.operands.count < 2 {
      log("getPartialApplyOfPullbackInExitVJPBB: tuple instruction has \(ti.operands.count) operands, but at least 2 expected, aborting")
      return nil
    }
    guard let lastTupleElemDefiningInst = ti.operands.last!.value.definingInstruction else {
      log("getPartialApplyOfPullbackInExitVJPBB: last tuple element is not defined by an instruction, aborting")
      return nil
    }
    return handleConvertFunctionOrPartialApply(inst: lastTupleElemDefiningInst)
  }

  return handleConvertFunctionOrPartialApply(inst: retValDefiningInstr)
}

// Consider the following code in a VJP when handling a throwing function:
//   bbA:
//     %closure = ...
//     %optionalWrapper1 = enum $Optional<ClosureType>, #Optional.some!enumelt, %closure
//     %payloadTuple1 = tuple (..., %optionalWrapper1)
//     %bteWithSome = enum $_AD__$xxx_bbN__Pred__xxx, #_AD__$xxx_bbN__Pred__xxx.bbK!enumelt, %payloadTuple1
//   bbB:
//     %optionalWrapper2 = enum $Optional<ClosureType>, #Optional.none!enumelt
//     %payloadTuple2 = tuple (..., %optionalWrapper2)
//     %bteWithNone = enum $_AD__$xxx_bbM__Pred__xxx, #_AD__$xxx_bbM__Pred__xxx.bbK!enumelt, %payloadTuple2
//
// During specialization, we need to change the underlying type for 'none' optional value %optionalWrapper2.
// The code below for each 'some' optional value (%optionalWrapper1 in the example above) finds the corresponding
// 'none' optional value and saves info about found value to allow further specialization. Pay attention to
// exact same names for branch tracing enum cases used for payload tuples in both cases (bbK in the example above).
private func findOptionalNoneMatchingOptionalSome(in vjp: Function, closuresInBTE: [ClosureInBTE]) -> [ClosureInBTE] {
  let branchTracingEnumInstructions: [EnumInst] = vjp.instructions.filter { $0 is EnumInst }.map {
    $0 as! EnumInst
  }.filter { $0.type.isBranchTracingEnum(in: vjp) }

  var closuresInBTEForOptionalNone = [ClosureInBTE]()

  for closureInBTE in closuresInBTE {
    if closureInBTE.optionalWrapper == nil {
      continue
    }
    assert(closureInBTE.optionalWrapper!.isOptionalSome)
    // In terms of the example above, we've found %closure and %optionalWrapper1.

    let enumCase = closureInBTE.enumCase
    // In terms of the example above, enumCase.name is now equal to "bbK".

    let bteWithNoneArr = branchTracingEnumInstructions.filter {
      $0.caseName == enumCase.name && $0.type != enumCase.enumType(in: vjp)
    }
    guard let bteWithNone = bteWithNoneArr.singleElement else {
      assert(bteWithNoneArr.isEmpty)
      continue
    }

    let payloadTuple = bteWithNone.operands.singleElement!.value as! TupleInst
    let optionalNone = payloadTuple.operands.last!.value as! EnumInst
    assert(optionalNone.isOptionalNone)

    closuresInBTEForOptionalNone.append(ClosureInBTE(
      closure: closureInBTE.closure,
      reabstractions: closureInBTE.reabstractions,
      optionalWrapper: optionalNone,
      useInPayload: payloadTuple.operands.last!,
      enumCase: bteWithNone.type.getEnumCases(in: vjp)![bteWithNone.caseIndex]!
    ))
  }

  return closuresInBTEForOptionalNone
}

typealias BTEPayloadArgOfPbBBInfo = (arg: Argument, enumCase: EnumCase, throwingSuccessor: BasicBlock?)

// Consider a basic block argument which is a payload tuple of a branch tracing enum payload case.
// The last element type of this tuple might be not a closure type, but an optional of a closure type.
// Such optionals are unwrapped with switch_enum instruction. This function finds the successor basic
// block for 'some' case of the switch_enum accepting this optional closure (given that the optional
// closure comes from the last element of branch tracing enum payload tuple BB argument).
//
// For the example below, find bbM successor when given %arg argument:
//
//   bbN(..., %arg : $(predecessor: _AD__$xxx_bbA__Pred__xxx, ClosureType1, ..., Optional<ClosureTypeNum>)):
//     // ...
//     %last_closure = tuple_extract %arg, num
//     // Alternatively, we might have (..., %last_closure) = destructure_tuple %arg
//     switch_enum %last_closure, case #Optional.some!enumelt: bbM, case #Optional.none!enumelt: bbK
//
//   bbM(%closure: ClosureTypeNum):
//     // ...
private func getSuccessorForOptionalSome(arg: Argument) -> BasicBlock? {
  let bb = arg.parentBlock
  guard let sei = bb.terminator as? SwitchEnumInst,
        sei.enumOp.type.isOptional
  else {
    return nil
  }
  assert(bb.successors.count == 2)

  if let tei = sei.enumOp.definingInstruction as? TupleExtractInst {
    if tei.tuple != arg {
      return nil
    }
    assert(tei.fieldIndex + 1 == arg.type.tupleElements.count)
  } else if let dti = sei.enumOp.definingInstruction as? DestructureTupleInst {
    if dti.tuple != arg {
      return nil
    }
    assert(dti.results.last! == sei.enumOp)
  } else {
    return nil
  }

  return sei.getUniqueSuccessor(forCaseIndex: Builder.optionalSomeCaseIndex)!
}

/// BTE payload-layout invariant used by the multi-basic-block specialization path.
///
/// For a pullback block reached by dispatching on a branch tracing enum, the corresponding payload tuple
/// is interpreted as:
///   1. payload element 0 is the predecessor branch tracing enum, if present;
///   2. later payload elements may store closures;
///   3. the last closure may be wrapped in `Optional`, for throwing-function support.
///
/// The specialized form keeps the same control-flow shape, but replaces closure payload elements with
/// tuples of captured arguments.
///
/// Before:
///   bbPred:
///     %bte = ...
///     switch_enum %bte, case #...bbK!enumelt: bbK, ...
///
///   bbK(%payload: $(predecessor: _AD__$...Pred..., ClosureType1, Optional<ClosureType2>)):
///     %closure1 = tuple_extract %payload, 1
///     %closure2opt = tuple_extract %payload, 2
///
/// After:
///   bbPred:
///     %bte_spec = ...
///     switch_enum %bte_spec, case #...bbK!enumelt: bbK, ...
///
///   bbK(%payload: $(predecessor: _AD__$...Pred..._spec, Captures1, Optional<Captures2>)):
///     %captures1 = tuple_extract %payload, 1
///     %captures2opt = tuple_extract %payload, 2
///
/// The helpers below rely on this invariant when:
/// - identifying which pullback block argument is a BTE payload tuple;
/// - mapping pullback payload blocks back to the corresponding VJP payload tuple;
/// - rewriting uses of specialized payload elements.
private func predecessorEnumDispatch(into bb: BasicBlock, argIndex: Int)
  -> (enumValue: Value, enumType: Type, caseIndex: Int)?
{
  guard let pred = bb.predecessors.first else { return nil }
  return predecessorEnumDispatch(from: pred, into: bb, argIndex: argIndex)
}

private func predecessorEnumDispatch(
  from pred: BasicBlock, into bb: BasicBlock, argIndex: Int
) -> (enumValue: Value, enumType: Type, caseIndex: Int)? {
  switch pred.terminator {
  case let br as BranchInst:
    guard let uedi = br.operands[argIndex].value.definingInstruction as? UncheckedEnumDataInst
    else { return nil }
    return (uedi.enum, uedi.enum.type, uedi.caseIndex)
  case let sei as SwitchEnumInst:
    return (sei.enumOp, sei.enumOp.type, sei.getUniqueCase(forSuccessor: bb)!)
  default:
    return nil
  }
}

// If the pullback's basic block has an argument which is a payload tuple of the
// branch tracing enum corresponding to the given VJP, return this argument and any valid combination
// of a branch tracing enum type and its case index having the same payload tuple type as the argument.
// The function assumes that no more than one such argument is present.
//
// To find the payload tuple argument, we look at any of the predecessor blocks and see how
// a particular argument is calculated. We consider the argument to be a payload tuple of a branch
// tracing enum if it is calculated as a result of unchecked_enum_data or switch_enum instruction
// accepting a branch tracing enum.
//
// For the example below, find %argB and %argD arguments in bbB and bbD basic blocks correspondingly.
//
//   bbA(...):
//     // Consider %bteA having branch tracing enum type in given vjp, e.g. _AD__$xxx_bbA__Pred__xxx
//     %payloadA = unchecked_enum_data %bteA, #_AD__$xxx_bbA__Pred__xxx.bbX!enumelt
//     br bbB(..., %payloadA, ...)
//
//   bbB(..., %argB : $(..., ..., ...), ...):
//     // ...
//
//   bbC(...):
//     // Consider %bteC having branch tracing enum type in given vjp, e.g. _AD__$xxx_bbC__Pred__xxx
//     switch_enum %bteC, #_AD__$xxx_bbC__Pred__xxx.bbY!enumelt: bbD, ...
//
//   bbD(%argD : $(..., ..., ...)):
//     // ...
private func getBTEPayloadArgOfPbBBInfo(_ bb: BasicBlock, vjp: Function)
  -> BTEPayloadArgOfPbBBInfo?
{
  log("getBTEPayloadArgOfPbBBInfo: start iterating over bb args")
  for arg in bb.arguments {
    log("getBTEPayloadArgOfPbBBInfo: \(arg)")
    if !arg.type.isTuple {
      log("getBTEPayloadArgOfPbBBInfo: arg is not a tuple, skipping")
      continue
    }

    guard let enumInfo = predecessorEnumDispatch(into: bb, argIndex: arg.index) else {
      continue
    }

    if !enumInfo.enumType.isBranchTracingEnum(in: vjp) {
      log("getBTEPayloadArgOfPbBBInfo: enum type \(enumInfo.enumType) is not a branch tracing enum in VJP \(vjp.name)")
      continue
    }

    log("getBTEPayloadArgOfPbBBInfo: success")
    return BTEPayloadArgOfPbBBInfo(
      arg: arg,
      enumCase: enumInfo.enumType.getEnumCases(in: vjp)![enumInfo.caseIndex]!,
      throwingSuccessor: getSuccessorForOptionalSome(arg: arg)
    )
  }

  log("getBTEPayloadArgOfPbBBInfo: finish iterating over bb args; branch tracing enum arg not found")
  return nil
}

extension ClosureInBTE {
  var capturedArgs : [Value] {
    if let pai = self.closure as? PartialApplyInst {
      var paiArgs = Array(pai.arguments)
      for idx in 0..<paiArgs.count {
        // TODO: preliminary check; also check that store is init
        if paiArgs[idx].type.isAddress && !pai.convention(of: pai.argumentOperands[idx])!.isInout {
          assert(paiArgs[idx].uses.count == 3)
          let store = paiArgs[idx].uses.filter { $0.instruction is StoreInst }.map{ $0.instruction as! StoreInst }.singleElement!
          paiArgs[idx] = store.source
        }
      }
      return paiArgs
    }
    guard self.closure is ThinToThickFunctionInst else {
      fatalError("unexpected closure type")
    }
    return []
  }
}

private extension Instruction {
  var asSupportedClosure: SingleValueInstruction? {
    switch self {
    case let tttf as ThinToThickFunctionInst where tttf.callee is FunctionRefInst:
      return tttf
    // TODO: figure out what to do with non-inout indirect arguments
    // https://forums.swift.org/t/non-inout-indirect-types-not-supported-in-closure-specialization-optimization/70826
    case let pai as PartialApplyInst
    where pai.callee is FunctionRefInst:
      // BTE specialization loads and removes non-inout captures, so it requires
      // a dedicated stack allocation even for conventions supported by the general pass.
      for op in pai.argumentOperands {
        guard op.value.type.isAddress else {
          continue
        }
        guard !pai.convention(of: op)!.isInout else {
          continue
        }
        guard let allocStack = op.value as? AllocStackInst else {
          return nil
        }
        guard allocStack.uses.count == 3 else {
          return nil
        }
        guard let storeUse = allocStack.uses.filter({ $0.instruction is StoreInst }).singleElement else {
          return nil
        }
        let store = storeUse.instruction as! StoreInst
        guard case .initialize = store.storeOwnership else {
          return nil
        }
        guard let paiUse = allocStack.uses.filter({ $0.instruction is PartialApplyInst }).singleElement else {
          return nil
        }
        assert(paiUse.instruction == pai)
        guard allocStack.uses.filter({ $0.instruction is DeallocStackInst }).singleElement != nil else {
          return nil
        }
        continue
      }
      return pai
    default:
      return nil
    }
  }

  var asSupportedClosureFn: Function? {
    let callee: Value?
    switch asSupportedClosure {
    case let tttf as ThinToThickFunctionInst:
      callee = tttf.callee
    case let pai as PartialApplyInst:
      callee = pai.callee
    default:
      callee = nil
    }
    return (callee as? FunctionRefInst)?.referencedFunction
  }

  var asOptionalWrapper: EnumInst? {
    guard let ei = self as? EnumInst,
          ei.operands.singleElement != nil,
          ei.type.isOptional
    else {
      return nil
    }
    return ei
  }
}

// For a closure which might potentially be a part of a branch tracing enum payload tuple
// (or a part of multiple such tuples) find actual uses in these tuples. Note that at this point
// we do not support cases when closure has any unexpected uses which are not branch tracing
// enum payloads. So, if any non-BTE payload use is found, we consider the given closure
// non-specializable and return no uses as for now.
//
// Also note that the closure might be used in a BTE payload not directly but after
// "wrapping" it in either an optional or an autodiff subset parameters thunk. If such a "wrapper"
// is detected, we only allow it as a single direct use of closure and then search for BTE payload
// uses of that wrapper.
//
// Considering the code below, the function will find:
// - for %closure1: uses in %payload11 and %payload12;
// - for %closure2: uses in %payload21 and %payload22 with %subsetThunk
// - for %closure3: uses in %payload31 and %payload32 with %optionalWrapper
//
//   %closure1 = partial_apply %foo1(...)
//   %payload11 = tuple (..., %closure1, ...)
//   %bte11 = enum $_AD__$xxx, #_AD__$xxx.bbXXX!enumelt, %payload11
//   %payload12 = tuple (..., %closure1, ...)
//   %bte12 = enum $_AD__$xxx, #_AD__$xxx.bbXXX!enumelt, %payload12
//
//   %closure2 = partial_apply %foo2(...) // user: %subsetThunk
//   // function_ref autodiff subset parameters thunk for ...
//   %subsetThunkFn = function_ref @$xxx
//   %subsetThunk = partial_apply %subsetThunkFn(%closure2)
//   %payload21 = tuple (..., %subsetThunk, ...)
//   %bte21 = enum $_AD__$xxx, #_AD__$xxx.bbXXX!enumelt, %payload21
//   %payload22 = tuple (..., %subsetThunk, ...)
//   %bte22 = enum $_AD__$xxx, #_AD__$xxx.bbXXX!enumelt, %payload22
//
//   %closure3 = thin_to_thick_function %foo3
//   %optionalWrapper12 = enum $Optional<...>, #Optional.some!enumelt, %closure3
//   %payload31 = tuple (..., %optionalWrapper, ...)
//   %bte31 = enum $_AD__$xxx, #_AD__$xxx.bbXXX!enumelt, %payload31
//   %payload32 = tuple (..., %optionalWrapper, ...)
//   %bte32 = enum $_AD__$xxx, #_AD__$xxx.bbXXX!enumelt, %payload32
private func findBTEUses(for rootClosure: SingleValueInstruction) -> [ClosureInBTE]? {
  log("findBTEUses: running for \(rootClosure)")
  let vjp = rootClosure.parentFunction
  var closuresInBTE = [ClosureInBTE]()

  var reabstractions = [PartialApplyInst]()
  var currentClosure = rootClosure
  while let singleUse = currentClosure.uses.singleElement {
    // TODO: handle other stuff like convert_function, thin_to_thick_function, etc
    guard let pai = singleUse.instruction as? PartialApplyInst else {
      break
    }

    guard pai.numArguments == 1,
      pai.arguments[0].type.isLoweredFunction,
      pai.arguments[0].type.isReferenceCounted(in: pai.parentFunction)
        || pai.callee.type.isThickFunction
    else {
      break
    }

    // Source closure expressions and subset/reabstraction thunks have static
    // entry points. Decline an unknown adapter before any enum is synthesized.
    guard pai.callee is FunctionRefInst else { return nil }

    currentClosure = pai
    reabstractions.append(pai)
  }

  let optionalWrapper = currentClosure.uses.singleElement?.instruction.asOptionalWrapper
  let closure = optionalWrapper ?? currentClosure

  for use in closure.uses {
    guard let ti = use.instruction as? TupleInst else {
      log("findBTEUses: unexpected use of closure, aborting: \(use)")
      return []
    }
    var closureInBTEForPayloadArr = [ClosureInBTE]()
    for tiUse in ti.uses {
      guard let ei = tiUse.instruction as? EnumInst else {
        log("findBTEUses: unexpected use of payload tuple, aborting: \(tiUse)")
        return []
      }
      guard ei.type.isBranchTracingEnum(in: vjp) else {
        log("findBTEUses: enum type \(ei.type) is not a " +
            "branch tracing enum in VJP \(vjp.name), aborting")
        return []
      }
      let enumCase = ei.type.getEnumCases(in: vjp)![ei.caseIndex]!
      let closureInBTE = ClosureInBTE(
        closure: rootClosure,
        reabstractions: reabstractions,
        optionalWrapper: optionalWrapper,
        useInPayload: use,
        enumCase: enumCase
      )
      // Avoid duplicates in case when multiple identical enum instructions accept `ti` as payload
      if !closureInBTEForPayloadArr.contains(closureInBTE) {
        closureInBTEForPayloadArr.append(closureInBTE)
      }
    }
    closuresInBTE.append(contentsOf: closureInBTEForPayloadArr)
  }
  log("findBTEUses: found \(closuresInBTE.count) uses of the following closure in branch tracing enums: \(rootClosure)")
  return closuresInBTE
}

private func findClosuresInBTE(paiOfPullback: PartialApplyInst) -> [ClosureInBTE]? {
  let vjp = paiOfPullback.parentFunction
  var reabstractions = Set<SingleValueInstruction>()
  var closuresInBTE = [ClosureInBTE]()
  for inst in vjp.instructions {
    guard inst != paiOfPullback,
          let rootClosure = inst.asSupportedClosure,
          !reabstractions.contains(rootClosure)
    else {
      continue
    }

    guard let currentClosuresInBTE = findBTEUses(for: rootClosure) else { return nil }
    closuresInBTE.append(contentsOf: currentClosuresInBTE)
    reabstractions.formUnion(currentClosuresInBTE.flatMap(\.reabstractions))
  }

  closuresInBTE.append(contentsOf:
    findOptionalNoneMatchingOptionalSome(in: vjp, closuresInBTE: closuresInBTE))

  return closuresInBTE
}

private func getSpecializedBTEDict(closuresInBTE: [ClosureInBTE], paiOfPullback: PartialApplyInst, _ context: FunctionPassContext) -> [Type: Type]? {
  guard !closuresInBTE.isEmpty else {
    return [:]
  }

  let vjp = paiOfPullback.parentFunction
  let pullback = paiOfPullback.referencedFunction!
  let enumTypeOfEntryBBArg = pullback.entryBlock.getBranchTracingEnumArg(vjp: vjp)!.type

  return autodiffSpecializeBranchTracingEnums(
    topVJP: vjp, topBTE: enumTypeOfEntryBBArg,
    closuresInBTE: closuresInBTE, context: context)
}

private typealias SpecializedBTEMap = [Type: Type]

private struct PullbackClosureAnalysis {
  let paiOfPullback: PartialApplyInst
  let closuresInBTE: [ClosureInBTE]

  var vjp: Function { paiOfPullback.parentFunction }
  var pullback: Function { paiOfPullback.referencedFunction! }

  init?(vjp: Function) {
    guard let paiOfPullback = getPartialApplyOfPullbackInExitVJPBB(vjp: vjp) else { return nil }

    // A pullback case identifies one VJP payload tuple. Distinct constructors
    // sharing that case are only supported when CFG simplification duplicated
    // an identical ordered list of saved SSA values. Decline different plans
    // before synthesizing any specialized enum declarations.
    var payloadsByCase = [Type: [Int: TupleInst]]()
    for (enumInst, payload) in findEnumsAndPayloadsInVjp(vjp: vjp) {
      let enumType = remapBranchTracingEnumType(enumInst.type, into: vjp)
      if let previous = payloadsByCase[enumType]?[enumInst.caseIndex],
         !previous.hasSamePayloadValues(as: payload) {
        return nil
      }
      payloadsByCase[enumType, default: [:]][enumInst.caseIndex] = payload
    }

    self.paiOfPullback = paiOfPullback
    guard let closuresInBTE = findClosuresInBTE(paiOfPullback: paiOfPullback) else { return nil }
    self.closuresInBTE = closuresInBTE
  }
}

private struct AutoDiffSpecializationInfo {
  let closureAnalysis: PullbackClosureAnalysis
  let specializedBTEDict: [Type: Type]

  var paiOfPullback: PartialApplyInst { closureAnalysis.paiOfPullback }
  var closuresInBTE: [ClosureInBTE] { closureAnalysis.closuresInBTE }
  var vjp: Function { closureAnalysis.vjp }
  var pullback: Function { closureAnalysis.pullback }

  init(closureAnalysis: PullbackClosureAnalysis, specializedBTEDict: [Type: Type]) {
    self.closureAnalysis = closureAnalysis
    self.specializedBTEDict = specializedBTEDict
  }

  init?(vjp: Function, _ context: FunctionPassContext) {
    guard let closureAnalysis = PullbackClosureAnalysis(vjp: vjp),
          let specializedBTEDict = synthesizeSpecializedEnums(
            from: closureAnalysis, context: context)
    else { return nil }
    self.init(closureAnalysis: closureAnalysis, specializedBTEDict: specializedBTEDict)
  }

  func pullbackNameSpecializedAgainstBTE(_ context: FunctionPassContext) -> String {
    let operand = paiOfPullback.argumentOperands.filter {
      $0.value.type.isBranchTracingEnum(in: vjp)
    }.singleElement!
    let originalType = remapBranchTracingEnumType(operand.value.type, into: vjp)
    return context.mangle(
      withSpecializedBranchTracingEnum: specializedBTEDict[originalType]!,
      argIdx: paiOfPullback.calleeArgumentIndex(of: operand)!,
      closuresInBTE: closuresInBTE, from: pullback)
  }
}

private func computeTopologicalBlockOrder(of function: Function) -> [BasicBlock]? {
  var visited: Set<BasicBlock> = [function.entryBlock]
  var order = [function.entryBlock]
  while order.count < function.blocks.count {
    let previousCount = order.count
    for bb in function.blocks where !visited.contains(bb) {
      if bb.predecessors.allSatisfy({ visited.contains($0) }) {
        visited.insert(bb)
        order.append(bb)
      }
    }
    // Source loop tapes are rejected separately. Still decline an unexpected
    // cyclic pullback instead of spinning forever on its unvisited blocks.
    guard order.count > previousCount else { return nil }
  }
  return order
}

// Generated tape tuples use a fixed callable order for each original block.
// A merge with different callable plans is unexpected; reject it before enum
// synthesis instead of selecting one predecessor's plan for every incoming edge.
private func validatePayloadMerges(
  in pullback: Function, vjp: Function, closureInfos: [ClosureInBTE]
) -> Bool {
  let enumToPayload = findEnumsAndPayloadsInVjp(vjp: vjp)
  for bb in pullback.blocks {
    for arg in bb.arguments where arg.type.isTuple {
      let hasEnumIncoming = bb.predecessors.contains { pred in
        predecessorEnumDispatch(from: pred, into: bb, argIndex: arg.index)?
          .enumType.isBranchTracingEnum(in: vjp) == true
      }
      guard hasEnumIncoming else { continue }
      var firstPlan: [ClosureInBTE]? = nil
      for pred in bb.predecessors {
        guard let dispatch = predecessorEnumDispatch(from: pred, into: bb, argIndex: arg.index)
        else { return false }
        let enumType = remapBranchTracingEnumType(dispatch.enumType, into: vjp)
        let payload = enumToPayload.first { enumInst, _ in
          enumInst.type == enumType && enumInst.caseIndex == dispatch.caseIndex
        }?.value
        let plan: [ClosureInBTE]
        if let payload {
          plan = collectMatchingClosureInfos(forPayload: payload, allClosureInfos: closureInfos)
        } else {
          // A case pruned from the VJP can remain in its pullback unchanged.
          guard !closureInfos.contains(where: {
            $0.enumCase.enumType(in: vjp) == enumType && $0.enumCase.index == dispatch.caseIndex
          }) else { return false }
          plan = []
        }
        if let firstPlan {
          guard firstPlan.count == plan.count,
                plan.allSatisfy({ info in firstPlan.contains(where: {
                  $0.indexInPayload == info.indexInPayload && $0.closure == info.closure
                    && $0.reabstractions == info.reabstractions
                    && $0.optionalWrapper == info.optionalWrapper
                }) })
          else { return false }
        } else {
          firstPlan = plan
        }
      }
    }
  }
  return true
}

private func findCorrespondingVJPPayloadTuple(
  forBB bb: BasicBlock, argIndex: Int, enumToPayload: [EnumInst: TupleInst],
  enumDict: SpecializedBTEMap, vjp: Function
) -> TupleInst? {
  guard let enumInfo = predecessorEnumDispatch(into: bb, argIndex: argIndex) else {
    return nil
  }
  for (enumInst, payload) in enumToPayload {
    if enumDict[enumInst.type] == remapBranchTracingEnumType(enumInfo.enumType, into: vjp)
      && enumInst.caseIndex == enumInfo.caseIndex {
      return payload
    }
  }
  return nil
}

extension ClosureInBTE {
  /// The value that actually appears as the payload-tuple operand for this closure:
  /// the subset thunk if present, otherwise the optional-wrapper enum, otherwise the raw closure.
  var valueInPayload: Value {
    if let optionalWrapper { return optionalWrapper }
    if let reabstraction = reabstractions.last { return reabstraction }
    return closure
  }
}

private func collectMatchingClosureInfos(
  forPayload ti: TupleInst, allClosureInfos: [ClosureInBTE]
) -> [ClosureInBTE] {
  allClosureInfos.filter { info in
    info.payloadTuple == ti && ti.operands.contains { $0.value == info.valueInPayload }
  }
}

private struct PullbackSpecializationAgainstBTE {
  typealias Cloner = SIL.Cloner<FunctionPassContext>

  func getOrCreatePullbackSpecializedAgainstBTE(
    basedOn autodiffSpecializationInfo: AutoDiffSpecializationInfo,
    enumDict: SpecializedBTEMap,
    _ context: FunctionPassContext
  )
    -> (function: Function, alreadyExists: Bool)
  {
    let pb = autodiffSpecializationInfo.pullback
    let vjp = autodiffSpecializationInfo.paiOfPullback.parentFunction

    let specializedPbName = autodiffSpecializationInfo.pullbackNameSpecializedAgainstBTE(context)
    if let specializedPb = context.lookupFunction(name: specializedPbName) {
      return (specializedPb, true)
    }

    let enumTypeOfEntryBBArg = remapBranchTracingEnumType(
      pb.entryBlock.getBranchTracingEnumArg(vjp: vjp)!.type, into: vjp)

    let specializedParameters = getParametersForPullbackSpecializedAgainstBTE(
      basedOn: autodiffSpecializationInfo, pb: pb, enumType: enumTypeOfEntryBBArg,
      enumDict: enumDict,
      context)

    let specializedPb =
      context.createSpecializedFunctionDeclaration(
        from: pb, withName: specializedPbName,
        withParams: specializedParameters,
        makeBare: true)

    context.buildSpecializedFunction(
      specializedFunction: specializedPb,
      buildFn: { (specializedPb, specializedContext) in
        var cloner = Cloner(cloneToEmptyFunction: specializedPb, specializedContext)
        defer { cloner.deinitialize() }

        cloneAndSpecializePullbackAgainstBTE(
          using: &cloner, autodiffSpecializationInfo: autodiffSpecializationInfo, enumDict: enumDict
        )
        // Cloning a whole function, even if it contains an `unreachable`, doesn't require lifetime completion.
        specializedContext.setNeedCompleteLifetimes(to: false)
      })

    return (specializedPb, false)
  }

  private typealias PayloadProjection = (
    resultIdx: Int, oldResultUses: [Operand], replacementValue: Value
  )

  private func rebuildPayloadProjections(
    of payload: Argument, context: FunctionPassContext
  ) -> (values: [PayloadProjection], useTei: Bool, obsoleteDestructure: DestructureTupleInst?) {
    if let singleUse = payload.uses.singleUse,
       let oldDti = singleUse.instruction as? DestructureTupleInst {
      let builder = Builder(before: oldDti, context)
      let newDti = builder.createDestructureTuple(tuple: oldDti.tuple)
      let projections = oldDti.results.enumerated().map { resultIdx, result in
        (
          resultIdx: resultIdx,
          oldResultUses: Array(result.uses),
          replacementValue: newDti.results[resultIdx] as Value
        )
      }
      // Keep the old consuming projection until all its result uses are rewritten.
      return (projections, false, oldDti)
    }

    var projections = [PayloadProjection]()
    for use in Array(payload.uses) {
      let oldTei = use.instruction as! TupleExtractInst
      let builder = Builder(before: oldTei, context)
      let newTei = builder.createTupleExtract(tuple: oldTei.tuple, elementIndex: oldTei.fieldIndex)
      projections.append((
        resultIdx: oldTei.fieldIndex,
        oldResultUses: Array(oldTei.results[0].uses),
        replacementValue: newTei.results[0]))
      oldTei.replace(with: newTei, context)
    }
    return (projections, true, nil)
  }

  private func rewriteProjectedPayloadValues(
    projections: [PayloadProjection],
    rewriteCtx: PayloadRewriteContext,
    context: FunctionPassContext
  ) {
    for projection in projections {
      for use in projection.oldResultUses {
        rewriteUsesOfPayloadItem(
          use: use,
          resultIdx: projection.resultIdx,
          result: projection.replacementValue,
          rewriteCtx: rewriteCtx,
          context)
      }
    }
  }

  func cloneAndSpecializePullbackAgainstBTE(
    using cloner: inout Cloner, autodiffSpecializationInfo: AutoDiffSpecializationInfo,
    enumDict: SpecializedBTEMap
  ) {
    let closureInfos = autodiffSpecializationInfo.closuresInBTE
    self.cloneEntryBlockArgsForPullbackSpecializedAgainstBTE(
      using: &cloner, usingOrigCalleeAt: autodiffSpecializationInfo, enumDict: enumDict)

    cloner.cloneFunctionBody(
      from: autodiffSpecializationInfo.pullback,
      entryBlockArguments: Array(cloner.targetFunction.entryBlock.arguments))

    let vjp = autodiffSpecializationInfo.paiOfPullback.parentFunction
    restoreUnwrappedPayloadPhis(in: cloner.targetFunction, vjp: vjp, cloner.context)
    let bbQueue = computeTopologicalBlockOrder(of: cloner.targetFunction)!
    let enumToPayload = findEnumsAndPayloadsInVjp(vjp: vjp)

    for bb in bbQueue {
      // With single-bb, we've ensured that there are no uses of BTE arg, so no manipulation required
      if bb == cloner.targetFunction.entryBlock && cloner.targetFunction.blocks.singleElement == nil
      {
        let bteArg = bb.getBranchTracingEnumArg(
          vjp: autodiffSpecializationInfo.paiOfPullback.parentFunction)!
        let sei = bteArg.uses.singleUse!.instruction as! SwitchEnumInst
        let builderEntry = Builder(before: sei, cloner.context)

        builderEntry.createSwitchEnum(
          enum: sei.enumOp, cases: getEnumCasesForSwitchEnumInst(sei))
        cloner.context.erase(instruction: sei)

        continue
      }

      guard
        let (arg, enumCase, throwingSuccessor) = getBTEPayloadArgOfPbBBInfo(
          bb, vjp: autodiffSpecializationInfo.paiOfPullback.parentFunction)
      else {
        continue
      }

      let tiInVjp = findCorrespondingVJPPayloadTuple(
        forBB: bb, argIndex: arg.index, enumToPayload: enumToPayload, enumDict: enumDict, vjp: vjp)

      let closureInfoArray: [ClosureInBTE] = {
        if let tiInVjp = tiInVjp {
          return collectMatchingClosureInfos(forPayload: tiInVjp, allClosureInfos: closureInfos)
        }
        return []
      }()

      let newArg = specializePayloadTupleBBArgInPullback(
        arg: arg, enumCase: enumCase, context: cloner.context)
      arg.uses.replaceAll(with: newArg, cloner.context)
      bb.eraseArgument(at: arg.index, cloner.context)

      if newArg.uses.isEmpty {
        continue
      }

      if let successor = throwingSuccessor {
        let oldArg = successor.arguments.singleElement!
        let newArg = specializeOptionalBBArgInPullback(
          bb: successor, newOptionalType: enumCase.payload!.tupleElements.last!,
          context: cloner.context)
        oldArg.uses.replaceAll(with: newArg, cloner.context)
        successor.eraseArgument(at: oldArg.index, cloner.context)
      }

      let projections = rebuildPayloadProjections(of: newArg, context: cloner.context)
      let rewriteCtx = PayloadRewriteContext(
        closureInfoArray: closureInfoArray,
        useTei: projections.useTei,
        throwingSuccessor: throwingSuccessor)
      rewriteProjectedPayloadValues(
        projections: projections.values,
        rewriteCtx: rewriteCtx,
        context: cloner.context)
      if let oldDti = projections.obsoleteDestructure {
        cloner.context.erase(instruction: oldDti)
      }
    }
  }

  private func cloneEntryBlockArgsForPullbackSpecializedAgainstBTE(
    using cloner: inout Cloner,
    usingOrigCalleeAt autodiffSpecializationInfo: AutoDiffSpecializationInfo,
    enumDict: SpecializedBTEMap
  ) {
    let pb = autodiffSpecializationInfo.pullback
    let vjp = autodiffSpecializationInfo.vjp
    let enumType = remapBranchTracingEnumType(
      pb.entryBlock.getBranchTracingEnumArg(vjp: vjp)!.type, into: vjp)

    let originalEntryBlock = autodiffSpecializationInfo.pullback.entryBlock
    let clonedFunction = cloner.targetFunction
    let clonedEntryBlock = cloner.getOrCreateEntryBlock()

    for arg in originalEntryBlock.arguments {
      var clonedEntryBlockArgType = arg.type.getLoweredType(in: clonedFunction)
      if remapBranchTracingEnumType(arg.type, into: vjp) == enumType {
        // The enum schema is analyzed in the VJP environment. The cloned
        // pullback has its own archetypes, even if their printed names agree.
        clonedEntryBlockArgType = remapBranchTracingEnumType(
          enumDict[enumType]!, into: clonedFunction)
      }
      let clonedEntryBlockArg = clonedEntryBlock.addFunctionArgument(
        type: clonedEntryBlockArgType, cloner.context)
      clonedEntryBlockArg.copyFlags(from: arg as! FunctionArgument, cloner.context)
    }
  }
}

let getAutoDiffSpecializationInfoTest = FunctionTest("autodiff_get_specialization_info") {
  function, arguments, context in
  guard let autodiffSpecializationInfo = AutoDiffSpecializationInfo(vjp: function, context) else {
    print("Unsupported branch tracing enum specialization")
    return
  }
  print("Run getAutoDiffSpecializationInfo for VJP \(function.name): autodiffSpecializationInfo = (")
  print("  pullback = \(autodiffSpecializationInfo.pullback.name)")
  print("  closuresInBTE = [")
  for closureInBTE in autodiffSpecializationInfo.closuresInBTE {
    print("    ClosureInBTE(")
    print("      closure: \(closureInBTE.closure)")
    print("      reabstractions: " + (closureInBTE.reabstractions.isEmpty ? "[]" : "["))
    for reabstraction in closureInBTE.reabstractions {
      print("        \(reabstraction)")
    }
    if !closureInBTE.reabstractions.isEmpty {
      print("    ]")
    }
    print("      optionalWrapper: " + (closureInBTE.optionalWrapper == nil ? "nil" : "\(closureInBTE.optionalWrapper!)"))
    print("      useInPayload: \(closureInBTE.useInPayload)")
    let enumCase = closureInBTE.enumCase
    print("      enumCase: \(enumCase.enumType(in: function)).\(enumCase.name)")
    print("    )")
  }
  print("  ]\n)\n")
}

private func specializeBranchTracingEnumBBArgInVJP(
  arg: Argument, specializedBTEDict: [Type: Type], context: FunctionPassContext
) -> Argument {
  let bb = arg.parentBlock
  assert(specializedBTEDict[arg.type] != nil)
  let newType = specializedBTEDict[arg.type]!
  return bb.insertPhiArgument(
    atPosition: arg.index, type: newType, ownership: arg.ownership, context)
}

let specializeBranchTracingEnums = FunctionTest("autodiff_specialize_branch_tracing_enums") {
  function, arguments, context in
  guard let autodiffSpecializationInfo = AutoDiffSpecializationInfo(vjp: function, context) else {
    print("Unsupported branch tracing enum specialization")
    return
  }
  let specializedBTEDict = autodiffSpecializationInfo.specializedBTEDict
  print("Specialized branch tracing enum dict for VJP \(function.name) contains \(specializedBTEDict.count) elements:")

  var keys = [Type](specializedBTEDict.keys)

  func compareTypes(lhs: Type, rhs: Type) -> Bool { "\(lhs)" < "\(rhs)" }
  keys.sort(by: compareTypes)

  for (idx, key) in keys.enumerated() {
    print("non-specialized BTE \(idx): \(key.nominal!.description)")
    print("specialized BTE \(idx): \(specializedBTEDict[key]!.nominal!.description)")
  }
  print("")
}

let specializeBTEArgInVjpBB = FunctionTest("autodiff_specialize_bte_arg_in_vjp_bb") {
  function, arguments, context in
  guard let autodiffSpecializationInfo = AutoDiffSpecializationInfo(vjp: function, context) else {
    print("Unsupported branch tracing enum specialization")
    return
  }
  let specializedBTEDict = autodiffSpecializationInfo.specializedBTEDict

  print("Specialized BTE arguments of basic blocks in VJP \(function.name):")
  for bb in function.blocks {
    guard let arg = bb.getBranchTracingEnumArg(vjp: function) else {
      continue
    }
    let newArg = specializeBranchTracingEnumBBArgInVJP(
      arg: arg, specializedBTEDict: specializedBTEDict, context: context)
    print("\(newArg)")
    bb.eraseArgument(at: newArg.index, context)
  }
  print("")
}

private func specializePayloadTupleBBArgInPullback(
  arg: Argument,
  enumCase: EnumCase,
  context: FunctionPassContext
) -> Argument {
  let bb = arg.parentBlock
  let newPayloadTupleTy = enumCase.payload!

  return bb.insertPhiArgument(
    atPosition: arg.index, type: newPayloadTupleTy, ownership: arg.ownership, context)
}

let specializePayloadArgInPullbackBB = FunctionTest("autodiff_specialize_payload_arg_in_pb_bb") {
  function, arguments, context in
  guard let autodiffSpecializationInfo = AutoDiffSpecializationInfo(vjp: function, context) else {
    print("Unsupported branch tracing enum specialization")
    return
  }
  let specializedBTEDict = autodiffSpecializationInfo.specializedBTEDict

  let pullback = autodiffSpecializationInfo.pullback
  print("Specialized BTE payload arguments of basic blocks in pullback \(pullback.name):")
  for bb in pullback.blocks {
    guard
      let (arg, enumCase, throwingSuccessor) = getBTEPayloadArgOfPbBBInfo(bb, vjp: function)
    else {
      continue
    }

    let enumType = specializedBTEDict[enumCase.enumType(in: function)]!
    let newArg = specializePayloadTupleBBArgInPullback(
      arg: arg,
      enumCase: enumType.getEnumCases(in: function)![enumCase.index]!,
      context: context)
    print("\(newArg)")
    bb.eraseArgument(at: newArg.index, context)

    if let successor = throwingSuccessor {
      let newArg = specializeOptionalBBArgInPullback(
        bb: successor,
        newOptionalType: enumType.getEnumCases(in: pullback)![enumCase.index]!.payload!.tupleElements.last!,
        context: context)
      print("\(newArg)")
      successor.eraseArgument(at: newArg.index, context)
    }
  }
  print("")
}

private func specializeOptionalBBArgInPullback(
  bb: BasicBlock,
  newOptionalType: Type,
  context: FunctionPassContext
) -> Argument {
  let arg = bb.arguments.singleElement!

  let underlyingType = newOptionalType.rawType.optionalObjectType.loweredTypeWithAbstractionPattern(
    in: bb.parentFunction)

  return bb.insertPhiArgument(
    atPosition: arg.index, type: underlyingType, ownership: arg.ownership, context)
}
