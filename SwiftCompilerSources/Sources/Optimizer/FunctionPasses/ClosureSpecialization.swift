//===--- ClosureSpecialization.swift ---------------------------===//
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

/// This file contains the closure-specialization optimizations for general and differentiable Swift.

/// General Closure Specialization
/// ------------------------------------
/// TODO: Add description when the functionality is added.

/// AutoDiff Closure Specialization
/// -------------------------------
/// This optimization performs closure specialization tailored for the patterns seen in Swift Autodiff. In principle,
/// the optimization does the same thing as the existing closure specialization pass. However, it is tailored to the
/// patterns of Swift Autodiff.
///
/// The compiler performs reverse-mode differentiation on functions marked with `@differentiable(reverse)`. In doing so,
/// it generates corresponding VJP and Pullback functions, which perform the forward and reverse pass respectively. You
/// can think of VJPs as functions that "differentiate" an original function and Pullbacks as the calculated
/// "derivative" of the original function.
///
/// VJPs always return a tuple of 2 values -- the original result and the Pullback. Pullbacks are essentially a chain
/// of closures, where the closure-contexts are implicitly used as the so-called "tape" during the reverse
/// differentiation process. It is this chain of closures contained within the Pullbacks that this optimization aims
/// to optimize via closure specialization.
///
/// The code patterns that this optimization targets, look similar to the one below:
/// ``` swift
///
/// // Since `foo` is marked with the `differentiable(reverse)` attribute the compiler
/// // will generate corresponding VJP and Pullback functions in SIL. Let's assume that
/// // these functions are called `vjp_foo` and `pb_foo` respectively.
/// @differentiable(reverse)
/// func foo(_ x: Float) -> Float {
///   return sin(x)
/// }
///
/// //============== Before closure specialization ==============//
/// // VJP of `foo`. Returns the original result and the Pullback of `foo`.
/// sil @vjp_foo: $(Float) -> (originalResult: Float, pullback: (Float) -> Float) {
/// bb0(%0: $Float):
///   // __Inlined__ `vjp_sin`: It is important for all intermediate VJPs to have
///   // been inlined in `vjp_foo`, otherwise `vjp_foo` will not be able to determine
///   // that `pb_foo` is closing over other closures and no specialization will happen.
///                                                                               \
///   %originalResult = apply @sin(%0): $(Float) -> Float                          \__ Inlined `vjp_sin`
///   %partially_applied_pb_sin = partial_apply pb_sin(%0): $(Float) -> Float      /
///                                                                               /
///
///   %pb_foo = function_ref @pb_foo: $@convention(thin) (Float, (Float) -> Float) -> Float
///   %partially_applied_pb_foo = partial_apply %pb_foo(%partially_applied_pb_sin): $(Float, (Float) -> Float) -> Float
///
///   return (%originalResult, %partially_applied_pb_foo)
/// }
///
/// // Pullback of `foo`.
/// //
/// // It receives what are called as intermediate closures that represent
/// // the calculations that the Pullback needs to perform to calculate a function's
/// // derivative.
/// //
/// // The intermediate closures may themselves contain intermediate closures and
/// // that is why the Pullback for a function differentiated at the "top" level
/// // may end up being a "chain" of closures.
/// sil @pb_foo: $(Float, (Float) -> Float) -> Float {
/// bb0(%0: $Float, %pb_sin: $(Float) -> Float):
///   %derivative_of_sin = apply %pb_sin(%0): $(Float) -> Float
///   return %derivative_of_sin: Float
/// }
///
/// //============== After closure specialization ==============//
/// sil @vjp_foo: $(Float) -> (originalResult: Float, pullback: (Float) -> Float) {
/// bb0(%0: $Float):
///   %originalResult = apply @sin(%0): $(Float) -> Float
///
///   // Before the optimization, pullback of `foo` used to take a closure for computing
///   // pullback of `sin`. Now, the specialized pullback of `foo` takes the arguments that
///   // pullback of `sin` used to close over and pullback of `sin` is instead copied over
///   // inside pullback of `foo`.
///   %specialized_pb_foo = function_ref @specialized_pb_foo: $@convention(thin) (Float, Float) -> Float
///   %partially_applied_pb_foo = partial_apply %specialized_pb_foo(%0): $(Float, Float) -> Float
///
///   return (%originalResult, %partially_applied_pb_foo)
/// }
///
/// sil @specialized_pb_foo: $(Float, Float) -> Float {
/// bb0(%0: $Float, %1: $Float):
///   %2 = partial_apply @pb_sin(%1): $(Float) -> Float
///   %3 = apply %2(): $() -> Float
///   return %3: $Float
/// }
/// ```

import AST
import SIL
import SILBridging

private let verbose = false

// TODO: unify existing and new logging
private let needLogADCS = true
private var passRunCount = 0
private func logADCS(prefix: String = "", msg: String) {
  if !needLogADCS {
    return
  }
  assert(passRunCount == 1 || passRunCount == 2)
  let allLinesPrefix = "[ADCS][" + String(passRunCount) + "] "
  let linesArray = msg.split(separator: "\n")
  for (idx, line) in linesArray.enumerated() {
    if idx == 0 {
      debugLog(allLinesPrefix + prefix + line)
    } else {
      debugLog(allLinesPrefix + line)
    }
  }
}

private func log(prefix: Bool = true, _ message: @autoclosure () -> String) {
  if verbose {
    debugLog(prefix: prefix, message())
  }
}

// =========== Entry point =========== //
let generalClosureSpecialization = FunctionPass(
  name: "experimental-swift-based-closure-specialization"
) {
  (function: Function, context: FunctionPassContext) in
  // TODO: Implement general closure specialization optimization
  print("NOT IMPLEMENTED")
}

struct EnumTypeAndCase {
  var enumType: Type
  var caseIdx: Int
}

// TODO: proper hash
extension EnumTypeAndCase: Hashable {
  public func hash(into hasher: inout Hasher) {
    hasher.combine(1)
  }
}

extension Type: Hashable {
  public func hash(into hasher: inout Hasher) {
    hasher.combine(1)
  }
}

typealias ClosureInfoCFG = (
  closure: SingleValueInstruction,
  subsetThunk: PartialApplyInst?,
  idxInEnumPayload: Int, capturedArgs: [Value],
  enumTypeAndCase: EnumTypeAndCase, payloadTuple: TupleInst
)

extension Type {
  var isBranchTracingEnum: Bool {
    self.isEnum && self.description.hasPrefix("$_AD__")
  }
}

private func dumpVJP(vjp: Function) {
  logADCS(msg: "VJP dump begin")
  logADCS(msg: vjp.description)
  logADCS(msg: "VJP dump end")
}

private func dumpPB(pb: Function) {
  logADCS(msg: "PB dump begin")
  logADCS(msg: pb.description)
  logADCS(msg: "PB dump end")
}

private func dumpVJPAndPB(vjp: Function, pb: Function) {
  dumpVJP(vjp: vjp)
  dumpPB(pb: pb)
}

var isMultiBBWithoutBranchTracingEnumPullbackArg: Bool = false

private func checkIfCanRun(vjp: Function, context: FunctionPassContext) -> Bool {
  assert(vjp.blocks.singleElement == nil)

  let prefixFail = "Cannot run AutoDiff Closure Specialization on " + vjp.name.string + ": "
  guard let paiOfPb = getPartialApplyOfPullbackInExitVJPBB(vjp: vjp) else {
    logADCS(
      prefix: prefixFail, msg: "partial_apply of pullback not found in exit basic block of VJP")
    return false
  }
  var branchTracingEnumArgCounter = 0
  for arg in paiOfPb.arguments {
    if arg.type.isBranchTracingEnum {
      branchTracingEnumArgCounter += 1
    }
  }
  if branchTracingEnumArgCounter != 1 {
    let pbOpt = paiOfPb.referencedFunction
    assert(pbOpt != nil)
    let pb = pbOpt!
    for inst in vjp.instructions {
      let builtinInstOpt = inst as? BuiltinInst
      if builtinInstOpt == nil {
        continue
      }
      let builtinInst = builtinInstOpt!
      if builtinInst.name.string == "autoDiffProjectTopLevelSubcontext" {
        logADCS(
          prefix: prefixFail,
          msg:
            "VJP seems to contain a loop (builtin autoDiffProjectTopLevelSubcontext detected), this is not supported"
        )
        return false
      }
    }
    if branchTracingEnumArgCounter == 0 {
      isMultiBBWithoutBranchTracingEnumPullbackArg = true
      logADCS(msg: "This is multi-BB case which would be handled as single-BB case")
      return true
    }
    logADCS(
      prefix: prefixFail,
      msg: "partial_apply of pullback in exit basic block of VJP has "
        + String(branchTracingEnumArgCounter)
        + " branch tracing enum arguments, but exactly 1 is expected")
    dumpVJPAndPB(vjp: vjp, pb: pb)
    return false
  }

  guard let pb = paiOfPb.referencedFunction else {
    logADCS(
      prefix: prefixFail,
      msg:
        "cannot obtain pullback function reference from the partial_apply of pullback in exit basic block of VJP"
    )
    return false
  }
  guard getEnumArgOfEntryPbBB(pb.entryBlock) != nil else {
    logADCS(
      prefix: prefixFail,
      msg: "cannot get branch tracing enum argument of the pullback function " + pb.name.string)
    return false
  }

  guard pb.entryBlock.terminator as? SwitchEnumInst != nil else {
    logADCS(
      prefix: prefixFail,
      msg: "unexpected terminator instruction in the entry block of the pullback " + pb.name.string
        + " (only switch_enum_inst is supported)")
    logADCS(msg: "  terminator: " + pb.entryBlock.terminator.description)
    logADCS(msg: "  parent block begin")
    logADCS(msg: "  " + pb.entryBlock.description)
    logADCS(msg: "  parent block end")
    return false
  }
  for pbBB in pb.blocks {
    guard let sei = pbBB.terminator as? SwitchEnumInst else {
      continue
    }
    if sei.bridged.SwitchEnumInst_getSuccessorForDefault().block != nil {
      logADCS(
        prefix: prefixFail,
        msg: "switch_enum_inst from the entry block of the pullback " + pb.name.string
          + " has default destination set, which is not supported")
      return false
    }
  }

  for vjpBB in vjp.blocks {
    if getEnumArgOfVJPBB(vjpBB) != nil {
      break
    }
    for arg in vjpBB.arguments {
      if arg.type.isBranchTracingEnum {
        logADCS(
          prefix: prefixFail,
          msg: "several arguments of VJP " + vjp.name.string + " basic block "
            + vjpBB.shortDescription + " are branch tracing enums, but not more than 1 is supported"
        )
        return false
      }
    }
  }

  guard let vjpBBToPbBBMap = getVjpBBToPbBBMap(vjp: vjp, pb: pb) else {
    logADCS(
      prefix: prefixFail,
      msg: "cannot create bijection mapping between VJP and pullback basic blocks")
    return false
  }
  guard let vjpBBToTupleInstMap = getVjpBBToTupleInstMap(vjp: vjp) else {
    logADCS(
      prefix: prefixFail,
      msg:
        "cannot create mapping from VJP basic blocks to branch tracing enum payload tuples defined in them"
    )
    return false
  }

  for (vjpBB, pbBB) in vjpBBToPbBBMap {
    guard let argOfPbBB = getEnumPayloadArgOfPbBB(pbBB) else {
      for arg in pbBB.arguments {
        if arg.type.isTuple {
          // MYTODO: what if we have tuple which is not payload but just tuple?
          logADCS(
            prefix: prefixFail,
            msg: "several arguments of pullback " + pb.name.string + " basic block "
              + pbBB.shortDescription
              + " are tuples (assuming as payload tuples of branch tracing enums), but not more than 1 is supported"
          )
          return false
        }
      }
      continue
    }
    if argOfPbBB.uses.count > 1 {
      logADCS(
        prefix: prefixFail,
        msg: "tuple argument of pullback " + pb.name.string + " basic block "
          + pbBB.shortDescription + " has more than 1 uses")
      return false
    }
    if argOfPbBB.uses.singleUse != nil {
      guard let dti = argOfPbBB.uses.singleUse!.instruction as? DestructureTupleInst else {
        logADCS(
          prefix: prefixFail,
          msg: "tuple argument of pullback " + pb.name.string + " basic block "
            + pbBB.shortDescription + " single use is not a destructure_tuple_inst")
        logADCS(msg: "  use: " + argOfPbBB.uses.singleUse!.description)
        logADCS(msg: "  instruction: " + argOfPbBB.uses.singleUse!.instruction.description)
        logADCS(msg: "  parent block begin")
        logADCS(msg: "  " + argOfPbBB.uses.singleUse!.instruction.parentBlock.description)
        logADCS(msg: "  parent block end")
        return false
      }
      // TODO: do we need to check that results is not empty?
      if dti.operands[0].value.type.tupleElements.count != 0
        && dti.results[0].type.isBranchTracingEnum && dti.results[0].uses.count > 1
      {
        logADCS(
          prefix: prefixFail,
          msg: "predecessor element of the tuple being argument of pullback " + pb.name.string
            + " basic block " + pbBB.shortDescription + " has more than 1 use")
        return false
      }
      for result in dti.results {
        for use in result.uses {
          switch use.instruction {
          case _ as ApplyInst:
            ()
          case _ as DestroyValueInst:
            ()
          case _ as UncheckedEnumDataInst:
            ()
          case _ as SwitchEnumInst:
            ()
          default:
            logADCS(
              prefix: prefixFail,
              msg: "unexpected use of an element of the tuple being argument of pullback "
                + pb.name.string + " basic block " + pbBB.shortDescription)
            return false
          }
        }
      }
    }
    guard let ti = vjpBBToTupleInstMap[vjpBB] else {
      logADCS(
        prefix: prefixFail,
        msg:
          "when pullback basic block has a payload tuple argument, the corresponding VJP basic block is expected to have a mapping to the related tuple_inst, but not such mapping was found"
      )
      return false
    }
    if argOfPbBB.type != ti.type {
      logADCS(
        prefix: prefixFail,
        msg:
          "type mismatch between pullback basic block payload tuple and the tuple_inst from the corresponding VJP basic block "
      )
      return false
    }
  }

  return true
}

private func getVjpBBToTupleInstMap(vjp: Function) -> [BasicBlock: TupleInst]? {
  let prefix = "getVjpBBToTupleInstMap: failure reason "
  var vjpBBToTupleInstMap = [BasicBlock: TupleInst]()
  for bb in vjp.blocks {
    var tiOpt = TupleInst?(nil)
    for inst in bb.instructions {
      guard let ti = inst as? TupleInst else {
        continue
      }
      var useCountBranchTracingEnum = 0
      var useCountNonBranchTracingEnum = 0
      for use in ti.uses {
        guard let ei = use.instruction as? EnumInst else {
          useCountNonBranchTracingEnum += 1
          continue
        }
        if ei.type.isBranchTracingEnum {
          useCountBranchTracingEnum += 1
        } else {
          useCountNonBranchTracingEnum += 1
        }
      }
      if useCountNonBranchTracingEnum != 0 && useCountBranchTracingEnum != 0 {
        logADCS(prefix: prefix, msg: "0")
        logADCS(msg: "  useCountBranchTracingEnum: " + String(useCountBranchTracingEnum))
        logADCS(msg: "  useCountNonBranchTracingEnum: " + String(useCountNonBranchTracingEnum))
        logADCS(msg: "  ti: " + ti.description)
        logADCS(msg: "  parent block begin")
        logADCS(msg: "  " + ti.parentBlock.description)
        logADCS(msg: "  parent block end")
        return nil
      }
      if useCountBranchTracingEnum != 0 {
        if tiOpt != nil {
          logADCS(prefix: prefix, msg: "1")
          logADCS(msg: "  useCountBranchTracingEnum: " + String(useCountBranchTracingEnum))
          logADCS(msg: "  useCountNonBranchTracingEnum: " + String(useCountNonBranchTracingEnum))
          logADCS(msg: "  ti: " + ti.description)
          logADCS(msg: "  tiOpt!: " + tiOpt!.description)
          logADCS(msg: "  parent block begin")
          logADCS(msg: "  " + ti.parentBlock.description)
          logADCS(msg: "  parent block end")
          return nil
        }
        tiOpt = ti
      }
    }
    if tiOpt != nil {
      vjpBBToTupleInstMap[bb] = tiOpt!
    }
  }
  return vjpBBToTupleInstMap
}

private func multiBBHelper(
  callSite: CallSite, function: Function, enumDict: inout EnumDict, context: FunctionPassContext
) {
  var closuresSet = Set<SingleValueInstruction>()
  for closureInfo in callSite.closureInfosCFG {
    closuresSet.insert(closureInfo.closure)
  }
  let totalSupportedClosures = closuresSet.count

  var totalClosures: Int = 0
  for inst in function.instructions {
    let paiOpt = inst as? PartialApplyInst
    let tttfOpt = inst as? ThinToThickFunctionInst
    if paiOpt != nil || tttfOpt != nil {
      totalClosures += 1
    }
  }

  let (specializedFunction, alreadyExists) =
    getOrCreateSpecializedFunctionCFG(
      basedOn: callSite, enumDict: &enumDict, context)

  if !alreadyExists {
    context.notifyNewFunction(function: specializedFunction, derivedFrom: callSite.applyCallee)
  }

  rewriteApplyInstructionCFG(
    using: specializedFunction, callSite: callSite,
    enumDict: enumDict, context: context)

  var specializedClosures: Int = 0
  for closure in closuresSet {
    if closure.uses.count == 0 {
      specializedClosures += 1
      // TODO: do we need to manually delete the related function_ref instruction?
      closure.parentBlock.eraseInstruction(closure)
    }
  }

  var msg =
    "Specialized " + String(specializedClosures) + " out of " + String(totalSupportedClosures)
    + " supported closures "
  msg += "(rate " + String(Float(specializedClosures) / Float(totalSupportedClosures)) + "). "
  msg += "Total number of closures is " + String(totalClosures)
  logADCS(msg: msg)
}

let autodiffClosureSpecialization1 = FunctionPass(name: "autodiff-closure-specialization1") {
  (function: Function, context: FunctionPassContext) in
  passRunCount = 1
  autodiffClosureSpecialization(function: function, context: context)
}

let autodiffClosureSpecialization2 = FunctionPass(name: "autodiff-closure-specialization2") {
  (function: Function, context: FunctionPassContext) in
  passRunCount = 2
  autodiffClosureSpecialization(function: function, context: context)
}

func autodiffClosureSpecialization(function: Function, context: FunctionPassContext) {

  guard !function.isDefinedExternally,
    function.isAutodiffVJP
  else {
    return
  }

  let isSingleBB = function.blocks.singleElement != nil
  isMultiBBWithoutBranchTracingEnumPullbackArg = false
  var canRunMultiBB = false

  if !isSingleBB {
    logADCS(
      msg:
        "Trying to run AutoDiff Closure Specialization pass on " + function.name.string)
    canRunMultiBB = checkIfCanRun(vjp: function, context: context)
    if canRunMultiBB {
      logADCS(
        msg:
          "The VJP " + function.name.string
          + " has passed the preliminary check. Proceeding to running the pass")
      if isMultiBBWithoutBranchTracingEnumPullbackArg {
        logADCS(msg: "Dumping VJP and PB before pass run begin")
        dumpVJPAndPB(
          vjp: function,
          pb: getPartialApplyOfPullbackInExitVJPBB(vjp: function)!.referencedFunction!)
        logADCS(msg: "Dumping VJP and PB before pass run end")
      }
    }
  }

  var remainingSpecializationRounds = 5

  repeat {
    // TODO: Names here are pretty misleading. We are looking for a place where
    // the pullback closure is created (so for `partial_apply` instruction).
    let callSiteOpt = gatherCallSite(in: function, context)
    if callSiteOpt == nil {
      break
    }

    let callSite = callSiteOpt!

    let (specializedFunction, alreadyExists) = getOrCreateSpecializedFunction(
      basedOn: callSite, context)

    if !alreadyExists {
      context.notifyNewFunction(function: specializedFunction, derivedFrom: callSite.applyCallee)
    }

    rewriteApplyInstruction(using: specializedFunction, callSite: callSite, context)

    // TODO avoid this array
    let callSites = [callSite]
    var deadClosures: InstructionWorklist = callSites.reduce(into: InstructionWorklist(context)) {
      deadClosures, callSite in
      callSite.closureArgDescriptors
        .map { $0.closure }
        .forEach { deadClosures.pushIfNotVisited($0) }
    }

    defer {
      deadClosures.deinitialize()
    }

    while let deadClosure = deadClosures.pop() {
      let isDeleted = context.tryDeleteDeadClosure(
        closure: deadClosure as! SingleValueInstruction)
      if isDeleted {
        context.notifyInvalidatedStackNesting()
      }
    }

    if context.needFixStackNesting {
      function.fixStackNesting(context)
    }

    if isMultiBBWithoutBranchTracingEnumPullbackArg {
      logADCS(msg: "Dumping VJP and PB at round \(remainingSpecializationRounds) begin")
      dumpVJPAndPB(vjp: function, pb: specializedFunction)
      logADCS(msg: "Dumping VJP and PB at round \(remainingSpecializationRounds) end")
    }

    remainingSpecializationRounds -= 1
  } while remainingSpecializationRounds > 0

  if !isSingleBB && canRunMultiBB && !isMultiBBWithoutBranchTracingEnumPullbackArg {
    var enumDict: EnumDict = [:]
    var adcsHelper = BridgedAutoDiffClosureSpecializationHelper()
    defer {
      adcsHelper.clearEnumDict()
    }

    remainingSpecializationRounds = 5
    repeat {
      logADCS(msg: "Remaining specialization rounds: " + String(remainingSpecializationRounds))
      // TODO: Names here are pretty misleading. We are looking for a place where
      // the pullback closure is created (so for `partial_apply` instruction).
      let callSiteOpt = gatherCallSiteCFG(in: function, context)
      if callSiteOpt == nil {
        // TODO: it looks like that we do not have more than 1 round, at least for multi BB case
        logADCS(
          msg:
            "Unable to detect closures to be specialized in " + function.name.string
            + ", skipping the pass")
        break
      }

      let callSite = callSiteOpt!

      multiBBHelper(callSite: callSite, function: function, enumDict: &enumDict, context: context)

      remainingSpecializationRounds -= 1
    } while remainingSpecializationRounds > 0
  }
}

// =========== Top-level functions ========== //

private let specializationLevelLimit = 2

private func getPartialApplyOfPullbackInExitVJPBB(vjp: Function) -> PartialApplyInst? {
  let prefix = "getPartialApplyOfPullbackInExitVJPBB: failure reason "
  var exitBBOpt = BasicBlock?(nil)
  for block in vjp.blocks {
    if block.isReachableExitBlock {
      guard block.terminator as? ReturnInst != nil else {
        continue
      }
      if exitBBOpt != nil {
        logADCS(prefix: prefix, msg: "0")
        return nil
      }
      exitBBOpt = block
    }
  }
  if exitBBOpt == nil {
    logADCS(prefix: prefix, msg: "1")
    return nil
  }
  let ri = exitBBOpt!.terminator as! ReturnInst
  if ri.returnedValue.definingInstruction == nil {
    logADCS(prefix: prefix, msg: "2")
    return nil
  }

  func handleConvertFunctionOrPartialApply(inst: Instruction) -> PartialApplyInst? {
    let paiOpt = inst as? PartialApplyInst
    let cfOpt = inst as? ConvertFunctionInst
    if paiOpt != nil {
      return paiOpt!
    }
    if cfOpt == nil {
      logADCS(prefix: prefix, msg: "3")
      logADCS(msg: "  instruction: " + inst.description)
      logADCS(msg: "  parent block begin")
      logADCS(msg: "  " + inst.parentBlock.description)
      logADCS(msg: "  parent block end")
      return nil
    }
    let pai = cfOpt!.operands[0].value as? PartialApplyInst
    if pai == nil {
      logADCS(prefix: prefix, msg: "4")
      logADCS(msg: "  value: " + cfOpt!.operands[0].value.description)
      logADCS(msg: "  parent block begin")
      logADCS(msg: "  " + cfOpt!.parentBlock.description)
      logADCS(msg: "  parent block end")
    }
    return pai
  }

  let tiOpt = ri.returnedValue.definingInstruction as? TupleInst
  if tiOpt == nil {
    return handleConvertFunctionOrPartialApply(inst: ri.returnedValue.definingInstruction!)
  }
  let ti = tiOpt!
  if ti.operands.count != 2 {
    logADCS(prefix: prefix, msg: "5")
    logADCS(msg: "  ti: " + ti.description)
    logADCS(msg: "  parent block begin")
    logADCS(msg: "  " + ti.parentBlock.description)
    logADCS(msg: "  parent block end")
    return nil
  }
  if ti.operands[1].value.definingInstruction == nil {
    logADCS(prefix: prefix, msg: "6")
    logADCS(msg: "  ti: " + ti.description)
    logADCS(msg: "  value: " + ti.operands[1].value.description)
    logADCS(msg: "  parent block begin")
    logADCS(msg: "  " + ti.parentBlock.description)
    logADCS(msg: "  parent block end")
    return nil
  }
  return handleConvertFunctionOrPartialApply(inst: ti.operands[1].value.definingInstruction!)
}

private func gatherCallSiteCFG(in caller: Function, _ context: FunctionPassContext) -> CallSite? {
  var callSiteOpt = CallSite?(nil)
  var supportedClosuresCount = 0
  var subsetThunkArr = [SingleValueInstruction]()

  for inst in caller.instructions {
    if let rootClosure = inst.asSupportedClosure {
      supportedClosuresCount += 1

      if rootClosure == getPartialApplyOfPullbackInExitVJPBB(vjp: rootClosure.parentFunction)! {
        continue
      }
      if subsetThunkArr.contains(rootClosure) {
        continue
      }
      let closureInfoArr = handleNonAppliesCFG(for: rootClosure, context)
      logADCS(msg: "closureInfoArr.count = " + String(closureInfoArr.count))
      if closureInfoArr.count == 0 {
        continue
      }
      if callSiteOpt == nil {
        callSiteOpt = CallSite(applySite: getPartialApplyOfPullbackInExitVJPBB(vjp: caller)!)
      }
      for closureInfo in closureInfoArr {
        callSiteOpt!.closureInfosCFG.append(closureInfo)
        if closureInfo.subsetThunk != nil {
          subsetThunkArr.append(closureInfo.subsetThunk!)
        }
      }

    }
  }

  if supportedClosuresCount == 0 {
    logADCS(msg: "No supported closures found in " + caller.name.string)
  }

  return callSiteOpt
}

private func gatherCallSite(in caller: Function, _ context: FunctionPassContext) -> CallSite? {
  /// __Root__ closures created via `partial_apply` or `thin_to_thick_function` may be converted and reabstracted
  /// before finally being used at an apply site. We do not want to handle these intermediate closures separately
  /// as they are handled and cloned into the specialized function as part of the root closures. Therefore, we keep
  /// track of these intermediate closures in a set.
  ///
  /// This set is populated via the `markConvertedAndReabstractedClosuresAsUsed` function which is called when we're
  /// handling the different uses of our root closures.
  ///
  /// Below SIL example illustrates the above point.
  /// ```
  /// // The below set of a "root" closure and its reabstractions/conversions
  /// // will be handled as a unit and the entire set will be copied over
  /// // in the specialized version of `takesClosure` if we determine that we
  /// // can specialize `takesClosure` against its closure argument.
  ///                                                                                                          __
  /// %someFunction = function_ref @someFunction: $@convention(thin) (Int, Int) -> Int                            \
  /// %rootClosure = partial_apply [callee_guaranteed] %someFunction (%someInt): $(Int, Int) -> Int                \
  /// %thunk = function_ref @reabstractionThunk : $@convention(thin) (@callee_guaranteed (Int) -> Int) -> @out Int /
  /// %reabstractedClosure = partial_apply [callee_guaranteed] %thunk(%rootClosure) :                             /
  ///                        $@convention(thin) (@callee_guaranteed (Int) -> Int) -> @out Int                  __/
  ///
  /// %takesClosure = function_ref @takesClosure : $@convention(thin) (@owned @callee_guaranteed (Int) -> @out Int) -> Int
  /// %result = partial_apply %takesClosure(%reabstractedClosure) : $@convention(thin) (@owned @callee_guaranteed () -> @out Int) -> Int
  /// ret %result
  /// ```
  var convertedAndReabstractedClosures = InstructionSet(context)

  defer {
    convertedAndReabstractedClosures.deinitialize()
  }

  var callSiteOpt = CallSite?(nil)

  for inst in caller.instructions {
    if !convertedAndReabstractedClosures.contains(inst),
      let rootClosure = inst.asSupportedClosure
    {
      updateCallSite(
        for: rootClosure, in: &callSiteOpt,
        convertedAndReabstractedClosures: &convertedAndReabstractedClosures, context)
    }
  }

  return callSiteOpt
}

private func getOrCreateSpecializedFunction(
  basedOn callSite: CallSite, _ context: FunctionPassContext
)
  -> (function: Function, alreadyExists: Bool)
{
  if isMultiBBWithoutBranchTracingEnumPullbackArg {
    logADCS(msg: "getOrCreateSpecializedFunction: callSite begin")
    logADCS(msg: "\(callSite)")
    logADCS(msg: "getOrCreateSpecializedFunction: callSite end")
  }
  let specializedFunctionName = callSite.specializedCalleeName(context)
  if let specializedFunction = context.lookupFunction(name: specializedFunctionName) {
    return (specializedFunction, true)
  }

  let applySiteCallee = callSite.applyCallee
  let specializedParameters = applySiteCallee.convention.getSpecializedParameters(basedOn: callSite)

  let specializedFunction =
    context.createFunctionForClosureSpecialization(
      from: applySiteCallee, withName: specializedFunctionName,
      withParams: specializedParameters,
      withSerialization: applySiteCallee.isSerialized)

  context.buildSpecializedFunction(
    specializedFunction: specializedFunction,
    buildFn: { (emptySpecializedFunction, functionPassContext) in
      let closureSpecCloner = SpecializationCloner(
        emptySpecializedFunction: emptySpecializedFunction, functionPassContext)
      closureSpecCloner.cloneAndSpecializeFunctionBody(using: callSite)
    })

  return (specializedFunction, false)
}

private func getOrCreateSpecializedFunctionCFG(
  basedOn callSite: CallSite, enumDict: inout EnumDict,
  _ context: FunctionPassContext
)
  -> (function: Function, alreadyExists: Bool)
{
  assert(callSite.closureArgDescriptors.count == 0)
  let pb = callSite.applyCallee
  let vjp = callSite.applySite.parentFunction

  let specializedPbName = callSite.specializedCalleeNameCFG(vjp: vjp, context)
  if let specializedPb = context.lookupFunction(name: specializedPbName) {
    return (specializedPb, true)
  }

  let closureInfos = callSite.closureInfosCFG
  var bbVisited = [BasicBlock: Bool]()
  var bbWorklist = [callSite.applyCallee.entryBlock]
  var enumTypesReverseQueue = [Type]()
  let enumTypeOfEntryBBArg = getEnumArgOfEntryPbBB(pb.entryBlock)!.type
  while bbWorklist.count != 0 {
    let block = bbWorklist.first!
    bbVisited[block] = true
    bbWorklist.removeFirst()
    var currentEnumTypeOpt = Type?(nil)
    if block == block.parentFunction.entryBlock {
      currentEnumTypeOpt = enumTypeOfEntryBBArg
    } else {
      let argOpt = getEnumPayloadArgOfPbBB(block)
      if argOpt != nil && argOpt!.uses.singleUse != nil {
        let dti = argOpt!.uses.singleUse!.instruction as! DestructureTupleInst
        if dti.results[0].type.isBranchTracingEnum {
          assert(dti.results[0].uses.count <= 1)
          if dti.results[0].uses.singleUse != nil {
            currentEnumTypeOpt = dti.results[0].type
          }
        }
      }
    }
    if currentEnumTypeOpt != nil && !enumTypesReverseQueue.contains(currentEnumTypeOpt!) {
      enumTypesReverseQueue.append(currentEnumTypeOpt!)
    }
    for succ in block.successors {
      if bbVisited[succ] != true {
        bbWorklist.append(succ)
      }
    }
  }

  for enumType in enumTypesReverseQueue.reversed() {
    var adcsHelper = BridgedAutoDiffClosureSpecializationHelper()
    defer {
      adcsHelper.clearClosuresBuffer()
    }
    for closureInfoCFG in closureInfos {
      if enumType != closureInfoCFG.enumTypeAndCase.enumType {
        continue
      }
      adcsHelper.appendToClosuresBuffer(
        closureInfoCFG.enumTypeAndCase.enumType.bridged,
        closureInfoCFG.enumTypeAndCase.caseIdx,
        closureInfoCFG.closure.bridged,
        closureInfoCFG.idxInEnumPayload)
    }
    enumDict[enumType] =
      adcsHelper.rewriteBranchTracingEnum( /*enumType: */
        enumType.bridged,
        /*topVjp: */callSite.applySite.parentFunction.bridged
      ).type
  }

  let specializedParameters = getSpecializedParametersCFG(
    basedOn: callSite, pb: pb, enumType: enumTypeOfEntryBBArg, enumDict: enumDict, context)

  let specializedPb =
    context.createFunctionForClosureSpecialization(
      from: pb, withName: specializedPbName,
      withParams: specializedParameters,
      withSerialization: pb.isSerialized)

  context.buildSpecializedFunction(
    specializedFunction: specializedPb,
    buildFn: { (emptySpecializedFunction, functionPassContext) in
      let closureSpecCloner = SpecializationCloner(
        emptySpecializedFunction: emptySpecializedFunction, functionPassContext)
      closureSpecCloner.cloneAndSpecializeFunctionBodyCFG(
        using: callSite, enumDict: enumDict)
    })

  return (specializedPb, false)
}

private func rewriteApplyInstructionCFG(
  using specializedCallee: Function, callSite: CallSite,
  enumDict: EnumDict,
  context: FunctionPassContext
) {
  let vjp = callSite.applySite.parentFunction
  let vjpExitBB = callSite.applySite.parentBlock
  let closureInfos = callSite.closureInfosCFG

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

  for bb in vjp.blocks {
    guard let arg = getEnumArgOfVJPBB(bb) else {
      continue
    }
    if enumDict[arg.type] == nil {
      continue
    }
    bb.bridged.recreateEnumBlockArgument(arg.bridged)
  }
  let pai = callSite.applySite as! PartialApplyInst

  let builderSucc = Builder(
    before: pai,
    location: callSite.applySite.parentBlock.instructions.last!.location, context)

  let paiConvention = pai.calleeConvention
  let paiHasUnknownResultIsolation = pai.hasUnknownResultIsolation
  let paiSubstitutionMap = SubstitutionMap(bridged: pai.bridged.getSubstitutionMap())
  let paiIsOnStack = pai.isOnStack

  // TODO assert that PAI is on index 1 in tuple

  let newFunctionRefInst = builderSucc.createFunctionRef(specializedCallee)
  var newCapturedArgs = [Value]()
  for paiArg in pai.arguments {
    newCapturedArgs.append(paiArg)
  }
  let newPai: PartialApplyInst = builderSucc.createPartialApply(
    function: newFunctionRefInst, substitutionMap: paiSubstitutionMap,
    capturedArguments: newCapturedArgs, calleeConvention: paiConvention,
    hasUnknownResultIsolation: paiHasUnknownResultIsolation, isOnStack: paiIsOnStack)

  pai.replace(with: newPai, context)

  let vjpBBToTupleInstMap = getVjpBBToTupleInstMap(vjp: vjp)!

  for bb in vjp.blocks {
    if bb == vjpExitBB {
      // Already handled before separately
      continue
    }
    guard let ti = vjpBBToTupleInstMap[bb] else {
      continue
    }
    if ti.operands.count == 0 {
      continue
    }

    var tupleIdxToCapturedArgs = [Int: [Value]]()
    for closureInfo in closureInfos {
      if closureInfo.payloadTuple != ti {
        continue
      }
      let idxInTuple = closureInfo.idxInEnumPayload
      assert(
        (closureInfo.subsetThunk == nil && ti.operands[idxInTuple].value == closureInfo.closure)
          || (closureInfo.subsetThunk != nil
            && ti.operands[idxInTuple].value == closureInfo.subsetThunk!)
      )
      tupleIdxToCapturedArgs[idxInTuple] = closureInfo.capturedArgs
    }

    var newPayloadValues = [Value]()
    for (opIdx, op) in ti.operands.enumerated() {
      if tupleIdxToCapturedArgs[opIdx] == nil {
        newPayloadValues.append(op.value)
        continue
      }

      let builderPred = Builder(before: ti, context)
      let tuple = builderPred.createTuple(elements: tupleIdxToCapturedArgs[opIdx]!)

      newPayloadValues.append(tuple)
    }
    let builderPred = Builder(before: ti, context)
    let newPayload = builderPred.createPayloadTupleForBranchTracingEnum(
      elements: newPayloadValues, tupleWithLabels: ti.type)
    ti.replace(with: newPayload, context)
  }
}

private func rewriteApplyInstruction(
  using specializedCallee: Function, callSite: CallSite,
  _ context: FunctionPassContext
) {
  let newApplyArgs = callSite.getArgumentsForSpecializedApply(of: specializedCallee)

  for newApplyArg in newApplyArgs {
    if case let .PreviouslyCaptured(capturedArg, needsRetain, parentClosureArgIndex) = newApplyArg,
      needsRetain
    {
      let closureArgDesc = callSite.closureArgDesc(at: parentClosureArgIndex)!
      var builder = Builder(before: closureArgDesc.closure, context)

      // TODO: Support only OSSA instructions once the OSSA elimination pass is moved after all function optimization
      // passes.
      if callSite.applySite.parentBlock != closureArgDesc.closure.parentBlock {
        // Emit the retain and release that keeps the argument live across the callee using the closure.
        builder.createRetainValue(operand: capturedArg)

        for instr in closureArgDesc.lifetimeFrontier {
          builder = Builder(before: instr, context)
          builder.createReleaseValue(operand: capturedArg)
        }

        // Emit the retain that matches the captured argument by the partial_apply in the callee that is consumed by
        // the partial_apply.
        builder = Builder(before: callSite.applySite, context)
        builder.createRetainValue(operand: capturedArg)
      } else {
        builder.createRetainValue(operand: capturedArg)
      }
    }
  }

  // Rewrite apply instruction
  var builder = Builder(before: callSite.applySite, context)
  let oldApply = callSite.applySite as! PartialApplyInst
  let funcRef = builder.createFunctionRef(specializedCallee)
  let capturedArgs = Array(newApplyArgs.map { $0.value })

  let newApply = builder.createPartialApply(
    function: funcRef, substitutionMap: SubstitutionMap(),
    capturedArguments: capturedArgs, calleeConvention: oldApply.calleeConvention,
    hasUnknownResultIsolation: oldApply.hasUnknownResultIsolation,
    isOnStack: oldApply.isOnStack)

  builder = Builder(before: callSite.applySite.next!, context)
  // TODO: Support only OSSA instructions once the OSSA elimination pass is moved after all function optimization
  // passes.
  for closureArgDesc in callSite.closureArgDescriptors {
    if closureArgDesc.isClosureConsumed,
      !closureArgDesc.isPartialApplyOnStack,
      !closureArgDesc.parameterInfo.isTrivialNoescapeClosure
    {
      builder.createReleaseValue(operand: closureArgDesc.closure)
    }
  }

  oldApply.replace(with: newApply, context)
}

// ===================== Utility functions and extensions ===================== //

private func updateCallSite(
  for rootClosure: SingleValueInstruction,
  in callSiteOpt: inout CallSite?,
  convertedAndReabstractedClosures: inout InstructionSet,
  _ context: FunctionPassContext
) {
  // A "root" closure undergoing conversions and/or reabstractions has additional restrictions placed upon it, in order
  // for a call site to be specialized against it. We handle conversion/reabstraction uses before we handle apply uses
  // to gather the parameters required to evaluate these restrictions or to skip call site uses of "unsupported"
  // closures altogether.
  //
  // There are currently 2 restrictions that are evaluated prior to specializing a callsite against a converted and/or
  // reabstracted closure -
  // 1. A reabstracted root closure can only be specialized against, if the reabstracted closure is ultimately passed
  //    trivially (as a noescape+thick function) into the call site.
  //
  // 2. A root closure may be a partial_apply [stack], in which case we need to make sure that all mark_dependence
  //    bases for it will be available in the specialized callee in case the call site is specialized against this root
  //    closure.

  var rootClosurePossibleLiveRange = InstructionRange(begin: rootClosure, context)
  defer {
    rootClosurePossibleLiveRange.deinitialize()
  }

  var rootClosureApplies = OperandWorklist(context)
  defer {
    rootClosureApplies.deinitialize()
  }

  let (foundUnexpectedUse, haveUsedReabstraction) =
    handleNonApplies(
      for: rootClosure, rootClosureApplies: &rootClosureApplies,
      rootClosurePossibleLiveRange: &rootClosurePossibleLiveRange, context)

  if foundUnexpectedUse {
    return
  }

  let intermediateClosureArgDescriptorData =
    handleApplies(
      for: rootClosure, callSiteOpt: &callSiteOpt, rootClosureApplies: &rootClosureApplies,
      rootClosurePossibleLiveRange: &rootClosurePossibleLiveRange,
      convertedAndReabstractedClosures: &convertedAndReabstractedClosures,
      haveUsedReabstraction: haveUsedReabstraction, context)

  if callSiteOpt == nil {
    return
  }

  finalizeCallSite(
    for: rootClosure, in: &callSiteOpt,
    rootClosurePossibleLiveRange: rootClosurePossibleLiveRange,
    intermediateClosureArgDescriptorData: intermediateClosureArgDescriptorData, context)
}

private func getEnumArgOfEntryPbBB(_ bb: BasicBlock) -> Argument? {
  assert(bb.parentFunction.entryBlock == bb)
  var argOpt = Argument?(nil)
  for arg in bb.arguments {
    if arg.type.isBranchTracingEnum {
      if argOpt != nil {
        return nil
      }
      argOpt = arg
    }
  }
  return argOpt
}

private func getEnumArgOfVJPBB(_ bb: BasicBlock) -> Argument? {
  var argOpt = Argument?(nil)
  for arg in bb.arguments {
    if arg.type.isBranchTracingEnum {
      if argOpt != nil {
        return nil
      }
      argOpt = arg
    }
  }
  return argOpt
}

private func getEnumPayloadArgOfPbBB(_ bb: BasicBlock) -> Argument? {
  // TODO: now we just assume that if we have exactly one tuple argument,
  //       this is what we need. This is not always true.
  var argOpt = Argument?(nil)
  for arg in bb.arguments {
    if !arg.type.isTuple {
      continue
    }
    if argOpt != nil {
      return nil
    }
    argOpt = arg
  }
  return argOpt
}

extension BasicBlockList {
  var count: Int {
    var n = 0
    for _ in self {
      n += 1
    }
    return n
  }
}

extension UseList {
  var count: Int {
    var n = 0
    for _ in self {
      n += 1
    }
    return n
  }
}

private func getVjpBBToPbBBMap(vjp: Function, pb: Function) -> [BasicBlock: BasicBlock]? {
  let prefix = "getVjpBBToPbBBMap: failure reason "
  if vjp.blocks.count != pb.blocks.count {
    logADCS(prefix: prefix, msg: "00")
    logADCS(
      msg: "vjp.blocks.count = " + String(vjp.blocks.count)
        + ", pb.blocks.count = " + String(pb.blocks.count))
    func printBlocks(f: Function) {
      for bb in f.blocks {
        var msg = bb.shortDescription + " -> "
        let riOpt = bb.terminator as? ReturnInst
        if riOpt != nil {
          msg = "exit " + msg
        }
        for succBB in bb.successors {
          msg += succBB.shortDescription + " "
        }
        logADCS(msg: msg)
      }
    }
    logADCS(msg: "VJP BBs begin")
    printBlocks(f: vjp)
    logADCS(msg: "VJP BBs end")
    logADCS(msg: "PB BBs begin")
    printBlocks(f: pb)
    logADCS(msg: "PB BBs end")
    return nil
  }
  var dict = [BasicBlock: BasicBlock]()

  let vjpBB = vjp.entryBlock
  var pbBBOpt = BasicBlock?(nil)
  for pbBB in pb.blocks {
    if pbBB.isReachableExitBlock {
      if pbBBOpt != nil {
        logADCS(prefix: prefix, msg: "01")
        return nil
      }
      pbBBOpt = pbBB
    }
  }
  if pbBBOpt == nil {
    logADCS(prefix: prefix, msg: "02")
    return nil
  }
  let pbBB = pbBBOpt!

  func dfs(vjpBBArg: BasicBlock, pbBBArg: BasicBlock) -> Bool {
    if dict[vjpBBArg] != nil {
      return true
    }
    dict[vjpBBArg] = pbBBArg
    if (vjpBBArg.singleSuccessor == nil) != (pbBBArg.singlePredecessor == nil) {
      logADCS(prefix: prefix, msg: "03")
      return false
    }
    if vjpBBArg.singleSuccessor != nil {
      return dfs(vjpBBArg: vjpBBArg.singleSuccessor!, pbBBArg: pbBBArg.singlePredecessor!)
    }
    var remainingVjpBB = BasicBlock?(nil)
    var predPbBBToSuccVjpBB = [BasicBlock: BasicBlock]()
    var vjpSuccTotalCount = 0
    var vjpSuccOKCount = 0
    for vjpSuccBB in vjpBBArg.successors {
      vjpSuccTotalCount += 1
      if vjpSuccBB.isReachableExitBlock {
        vjpSuccOKCount += 1
        predPbBBToSuccVjpBB[pb.entryBlock] = vjpSuccBB
        if !dfs(vjpBBArg: vjpSuccBB, pbBBArg: pb.entryBlock) {
          logADCS(prefix: prefix, msg: "04")
          return false
        }
        continue
      }
      var eiOpt = EnumInst?(nil)
      // TODO: support cases when this is not the first enum inst
      for inst in vjpSuccBB.instructions {
        eiOpt = inst as? EnumInst
        if eiOpt == nil {
          continue
        }
        if !eiOpt!.results[0].type.isBranchTracingEnum {
          eiOpt = nil
          continue
        }
        break
      }
      if eiOpt == nil {
        logADCS(prefix: prefix, msg: "05")
        return false
      }
      let enumType = eiOpt!.results[0].type
      assert(enumType.isBranchTracingEnum)
      var newPredBB = BasicBlock?(nil)
      for pbPredBB in pbBBArg.predecessors {
        guard let pbPredBBArg = getEnumPayloadArgOfPbBB(pbPredBB) else {
          continue
        }
        if pbPredBBArg.type.tupleElements.count != 0
          && pbPredBBArg.type.tupleElements[0] == enumType
        {
          if newPredBB != nil {
            logADCS(prefix: prefix, msg: "06")
            return false
          }
          newPredBB = pbPredBB
        }
      }
      if newPredBB == nil {
        if remainingVjpBB != nil {
          logADCS(prefix: prefix, msg: "07")
          return false
        }
        remainingVjpBB = vjpSuccBB
        continue
      }
      vjpSuccOKCount += 1
      predPbBBToSuccVjpBB[newPredBB!] = vjpSuccBB
      if !dfs(vjpBBArg: vjpSuccBB, pbBBArg: newPredBB!) {
        logADCS(prefix: prefix, msg: "08")
        return false
      }
    }
    if vjpSuccOKCount + 1 == vjpSuccTotalCount {
      assert(remainingVjpBB != nil)
      var remainingPbBB = BasicBlock?(nil)
      for pbPredBB in pbBBArg.predecessors {
        if predPbBBToSuccVjpBB[pbPredBB] == nil {
          assert(remainingPbBB == nil)
          remainingPbBB = pbPredBB
        }
      }
      assert(remainingPbBB != nil)
      if !dfs(vjpBBArg: remainingVjpBB!, pbBBArg: remainingPbBB!) {
        logADCS(prefix: prefix, msg: "09")
        return false
      }
    }
    return true
  }

  let status = dfs(vjpBBArg: vjpBB, pbBBArg: pbBB)
  if status {
    assert(dict.count == vjp.blocks.count)
    return dict
  }
  logADCS(prefix: prefix, msg: "10")
  return nil
}

private func handleNonAppliesCFG(
  for rootClosure: SingleValueInstruction,
  _ context: FunctionPassContext
)
  -> [ClosureInfoCFG]
{
  var closureInfoArr = [ClosureInfoCFG]()

  var closure = rootClosure
  var subsetThunkOpt = PartialApplyInst?(nil)
  if rootClosure.uses.singleElement != nil {
    logADCS(
      msg: "handleNonAppliesCFG: root closure has single use, checking if it's a subset thunk")
    let maybeSubsetThunkOpt = closure.uses.singleElement!.instruction as? PartialApplyInst
    if maybeSubsetThunkOpt != nil && maybeSubsetThunkOpt!.argumentOperands.count == 1
      && maybeSubsetThunkOpt!.referencedFunction != nil
      && maybeSubsetThunkOpt!.referencedFunction!.description.starts(
        with: "// autodiff subset parameters thunk for")
    {
      logADCS(msg: "handleNonAppliesCFG: subset thunk detected:")
      logADCS(msg: "  pai: \(maybeSubsetThunkOpt!)")
      logADCS(msg: "  fri: \(maybeSubsetThunkOpt!.referencedFunction!)")
      subsetThunkOpt = maybeSubsetThunkOpt
      closure = subsetThunkOpt!
    } else {
      if maybeSubsetThunkOpt != nil {
        logADCS(
          msg:
            "handleNonAppliesCFG: not a subset thunk:")
        logADCS(msg: "  argumentOperands.count = \(maybeSubsetThunkOpt!.argumentOperands.count)")
        logADCS(msg: "  referencedFunction = \(maybeSubsetThunkOpt!.referencedFunction)")
      } else {
        logADCS(msg: "handleNonAppliesCFG: not a subset thunk, unexpected instruction type: \(closure.uses.singleElement!.instruction)")
      }
    }
  }

  for use in closure.uses {
    guard let ti = use.instruction as? TupleInst else {
      let paiOfPbInExitVjpBB = getPartialApplyOfPullbackInExitVJPBB(
        vjp: rootClosure.parentFunction)!
      let paiOpt = rootClosure as? PartialApplyInst
      assert(paiOpt != paiOfPbInExitVjpBB)
      logADCS(msg: "handleNonAppliesCFG: unexpected use of closure")
      logADCS(msg: "handleNonAppliesCFG:   root closure: " + rootClosure.description)
      logADCS(msg: "handleNonAppliesCFG:   closure: " + closure.description)
      logADCS(msg: "handleNonAppliesCFG:   use.instruction: " + use.instruction.description)
      logADCS(
        msg: "handleNonAppliesCFG:   root closure use count: " + String(rootClosure.uses.count))
      logADCS(msg: "handleNonAppliesCFG:   parent block of use begin")
      logADCS(msg: "handleNonAppliesCFG:   " + use.instruction.parentBlock.description)
      logADCS(msg: "handleNonAppliesCFG:   parent block of use end")
      return []
    }
    for tiUse in ti.uses {
      guard let ei = tiUse.instruction as? EnumInst else {
        let paiOfPbInExitVjpBB = getPartialApplyOfPullbackInExitVJPBB(
          vjp: rootClosure.parentFunction)!
        let paiOpt = rootClosure as? PartialApplyInst
        assert(paiOpt != paiOfPbInExitVjpBB)
        logADCS(msg: "handleNonAppliesCFG: unexpected use of tuple")
        logADCS(msg: "handleNonAppliesCFG:   root closure: " + rootClosure.description)
        logADCS(msg: "handleNonAppliesCFG:   closure.uses.count: " + String(closure.uses.count))
        logADCS(msg: "handleNonAppliesCFG:   tuple: " + ti.description)
        logADCS(msg: "handleNonAppliesCFG:   tiUse.instruction: " + tiUse.instruction.description)
        return []
      }
      if !ei.type.isBranchTracingEnum {
        logADCS(msg: "handleNonAppliesCFG: unexpected enum type:" + ei.type.description)
        return []
      }
      var capturedArgs = [Value]()
      let paiOpt = rootClosure as? PartialApplyInst
      if paiOpt != nil {
        for argOp in paiOpt!.argumentOperands {
          capturedArgs.append(argOp.value)
        }
      }
      let enumTypeAndCase = EnumTypeAndCase(enumType: ei.type, caseIdx: ei.caseIndex)
      closureInfoArr.append(
        ClosureInfoCFG(
          closure: rootClosure,
          subsetThunk: subsetThunkOpt,
          idxInEnumPayload: use.index,
          capturedArgs: capturedArgs,
          enumTypeAndCase: enumTypeAndCase, payloadTuple: ti
        ))
    }
  }
  if closureInfoArr.count == 0 {
    logADCS(msg: "handleNonAppliesCFG: returning empty closure info array")
  }
  return closureInfoArr
}

/// Handles all non-apply direct and transitive uses of `rootClosure`.
///
/// Returns:
/// haveUsedReabstraction - whether the root closure is reabstracted via a thunk
/// foundUnexpectedUse - whether the root closure is directly or transitively used in an instruction that we don't know
///                      how to handle. If true, then `rootClosure` should not be specialized against.
private func handleNonApplies(
  for rootClosure: SingleValueInstruction,
  rootClosureApplies: inout OperandWorklist,
  rootClosurePossibleLiveRange: inout InstructionRange,
  _ context: FunctionPassContext
)
  -> (foundUnexpectedUse: Bool, haveUsedReabstraction: Bool)
{
  var foundUnexpectedUse = false
  var haveUsedReabstraction = false

  /// The root closure or an intermediate closure created by reabstracting the root closure may be a `partial_apply
  /// [stack]` and we need to make sure that all `mark_dependence` bases for this `onStack` closure will be available in
  /// the specialized callee, in case the call site is specialized against this root closure.
  ///
  /// `possibleMarkDependenceBases` keeps track of all potential values that may be used as bases for creating
  /// `mark_dependence`s for our `onStack` root/reabstracted closures. For root closures these values are non-trivial
  /// closure captures (which are always available as function arguments in the specialized callee). For reabstracted
  /// closures these values may be the root closure or its conversions (below is a short SIL example representing this
  /// case).
  /// ```
  /// %someFunction = function_ref @someFunction : $@convention(thin) (Int) -> Int
  /// %rootClosure = partial_apply [callee_guaranteed] %someFunction(%someInt) : $@convention(thin) (Int) -> Int
  /// %noescapeRootClosure = convert_escape_to_noescape %rootClosure : $@callee_guaranteed () -> Int to $@noescape @callee_guaranteed () -> Int
  /// %thunk = function_ref @reabstractionThunk : $@convention(thin) (@noescape @callee_guaranteed () -> Int) -> @out Int
  /// %thunkedRootClosure = partial_apply [callee_guaranteed] [on_stack] %thunk(%noescapeRootClosure) : $@convention(thin) (@noescape @callee_guaranteed () -> Int) -> @out Int
  /// %dependency = mark_dependence %thunkedRootClosure : $@noescape @callee_guaranteed () -> @out Int on %noescapeClosure : $@noescape @callee_guaranteed () -> Int
  /// %takesClosure = function_ref @takesClosure : $@convention(thin) (@owned @noescape @callee_guaranteed () -> @out Int)
  /// %ret = apply %takesClosure(%dependency) : $@convention(thin) (@owned @noescape @callee_guaranteed () -> @out Int)
  /// ```
  ///
  /// Any value outside of the aforementioned values is not going to be available in the specialized callee and a
  /// `mark_dependence` of the root closure on such a value means that we cannot specialize the call site against it.
  var possibleMarkDependenceBases = ValueSet(context)
  defer {
    possibleMarkDependenceBases.deinitialize()
  }

  var rootClosureConversionsAndReabstractions = OperandWorklist(context)
  rootClosureConversionsAndReabstractions.pushIfNotVisited(contentsOf: rootClosure.uses)
  defer {
    rootClosureConversionsAndReabstractions.deinitialize()
  }

  if let pai = rootClosure as? PartialApplyInst {
    for arg in pai.arguments {
      possibleMarkDependenceBases.insert(arg)
    }
  }

  while let use = rootClosureConversionsAndReabstractions.pop() {
    switch use.instruction {
    case let cfi as ConvertFunctionInst:
      rootClosureConversionsAndReabstractions.pushIfNotVisited(contentsOf: cfi.uses)
      possibleMarkDependenceBases.insert(cfi)
      rootClosurePossibleLiveRange.insert(use.instruction)

    case let cvt as ConvertEscapeToNoEscapeInst:
      rootClosureConversionsAndReabstractions.pushIfNotVisited(contentsOf: cvt.uses)
      possibleMarkDependenceBases.insert(cvt)
      rootClosurePossibleLiveRange.insert(use.instruction)

    case let pai as PartialApplyInst:
      if !pai.isPullbackInResultOfAutodiffVJP,
        pai.isSupportedClosure,
        pai.isPartialApplyOfThunk,
        // Argument must be a closure
        pai.arguments[0].type.isThickFunction
      {
        rootClosureConversionsAndReabstractions.pushIfNotVisited(contentsOf: pai.uses)
        possibleMarkDependenceBases.insert(pai)
        rootClosurePossibleLiveRange.insert(use.instruction)
        haveUsedReabstraction = true
      } else if pai.isPullbackInResultOfAutodiffVJP {
        rootClosureApplies.pushIfNotVisited(use)
      }

    case let mv as MoveValueInst:
      rootClosureConversionsAndReabstractions.pushIfNotVisited(contentsOf: mv.uses)
      possibleMarkDependenceBases.insert(mv)
      rootClosurePossibleLiveRange.insert(use.instruction)

    case let mdi as MarkDependenceInst:
      if possibleMarkDependenceBases.contains(mdi.base),
        mdi.value == use.value,
        mdi.value.type.isNoEscapeFunction,
        mdi.value.type.isThickFunction
      {
        rootClosureConversionsAndReabstractions.pushIfNotVisited(contentsOf: mdi.uses)
        rootClosurePossibleLiveRange.insert(use.instruction)
      }

    case is CopyValueInst,
      is DestroyValueInst,
      is RetainValueInst,
      is ReleaseValueInst,
      is StrongRetainInst,
      is StrongReleaseInst:
      rootClosurePossibleLiveRange.insert(use.instruction)

    case let ti as TupleInst:
      if ti.parentFunction.isAutodiffVJP,
        let returnInst = ti.parentFunction.returnInstruction,
        ti == returnInst.returnedValue
      {
        // This is the pullback closure returned from an Autodiff VJP and we don't need to handle it.
      } else {
        fallthrough
      }

    default:
      foundUnexpectedUse = true
      log("Found unexpected direct or transitive user of root closure: \(use.instruction)")
      return (foundUnexpectedUse, haveUsedReabstraction)
    }
  }

  return (foundUnexpectedUse, haveUsedReabstraction)
}

private typealias IntermediateClosureArgDescriptorDatum = (
  applySite: SingleValueInstruction, closureArgIndex: Int, paramInfo: ParameterInfo
)

private func handleApplies(
  for rootClosure: SingleValueInstruction, callSiteOpt: inout CallSite?,
  rootClosureApplies: inout OperandWorklist,
  rootClosurePossibleLiveRange: inout InstructionRange,
  convertedAndReabstractedClosures: inout InstructionSet, haveUsedReabstraction: Bool,
  _ context: FunctionPassContext
) -> [IntermediateClosureArgDescriptorDatum] {
  var intermediateClosureArgDescriptorData: [IntermediateClosureArgDescriptorDatum] = []

  while let use = rootClosureApplies.pop() {
    rootClosurePossibleLiveRange.insert(use.instruction)

    // TODO [extend to general swift]: Handle full apply sites
    guard let pai = use.instruction as? PartialApplyInst else {
      continue
    }

    // TODO: Handling generic closures may be possible but is not yet implemented
    if pai.hasSubstitutions || !pai.calleeIsDynamicFunctionRef
      || !pai.isPullbackInResultOfAutodiffVJP
    {
      continue
    }

    guard let callee = pai.referencedFunction else {
      continue
    }

    if callee.isDefinedExternally {
      continue
    }

    // Don't specialize non-fragile (read as non-serialized) callees if the caller is fragile; the specialized callee
    // will have shared linkage, and thus cannot be referenced from the fragile caller.
    let caller = rootClosure.parentFunction
    if caller.isSerialized && !callee.isSerialized {
      continue
    }

    // If the callee uses a dynamic Self, we cannot specialize it, since the resulting specialization might no longer
    // have 'self' as the last parameter.
    //
    // TODO: We could fix this by inserting new arguments more carefully, or changing how we model dynamic Self
    // altogether.
    if callee.mayBindDynamicSelf {
      continue
    }

    // Proceed if the closure is passed as an argument (and not called). If it is called we have nothing to do.
    //
    // `closureArgumentIndex` is the index of the closure in the callee's argument list.
    guard let closureArgumentIndex = pai.calleeArgumentIndex(of: use) else {
      continue
    }

    // Ok, we know that we can perform the optimization but not whether or not the optimization is profitable. Check if
    // the closure is actually called in the callee (or in a function called by the callee).
    if !isClosureApplied(in: callee, closureArgIndex: closureArgumentIndex) {
      continue
    }

    let onlyHaveThinToThickClosure =
      rootClosure is ThinToThickFunctionInst && !haveUsedReabstraction

    guard let closureParamInfo = pai.operandConventions[parameter: use.index] else {
      fatalError("While handling apply uses, parameter info not found for operand: \(use)!")
    }

    // If we are going to need to release the copied over closure, we must make sure that we understand all the exit
    // blocks, i.e., they terminate with an instruction that clearly indicates whether to release the copied over
    // closure or leak it.
    if closureParamInfo.convention.isGuaranteed,
      !onlyHaveThinToThickClosure,
      !callee.blocks.allSatisfy({ $0.isReachableExitBlock || $0.terminator is UnreachableInst })
    {
      continue
    }

    // Functions with a readnone, readonly or releasenone effect and a nontrivial context cannot be specialized.
    // Inserting a release in such a function results in miscompilation after other optimizations. For now, the
    // specialization is disabled.
    //
    // TODO: A @noescape closure should never be converted to an @owned argument regardless of the function's effect
    // attribute.
    if !callee.effectAllowsSpecialization && !onlyHaveThinToThickClosure {
      continue
    }

    // Avoid an infinite specialization loop caused by repeated runs of ClosureSpecializer and CapturePropagation.
    // CapturePropagation propagates constant function-literals. Such function specializations can then be optimized
    // again by the ClosureSpecializer and so on. This happens if a closure argument is called _and_ referenced in
    // another closure, which is passed to a recursive call. E.g.
    //
    // func foo(_ c: @escaping () -> ()) {
    //  c() foo({ c() })
    // }
    //
    // A limit of 2 is good enough and will not be exceed in "regular" optimization scenarios.
    let closureCallee =
      rootClosure is PartialApplyInst
      ? (rootClosure as! PartialApplyInst).referencedFunction!
      : (rootClosure as! ThinToThickFunctionInst).referencedFunction!

    if closureCallee.specializationLevel > specializationLevelLimit {
      continue
    }

    if haveUsedReabstraction {
      markConvertedAndReabstractedClosuresAsUsed(
        rootClosure: rootClosure, convertedAndReabstractedClosure: use.value,
        convertedAndReabstractedClosures: &convertedAndReabstractedClosures)
    }

    if callSiteOpt == nil {
      callSiteOpt = CallSite(applySite: pai)
    } else {
      assert(callSiteOpt!.applySite == pai)
    }

    intermediateClosureArgDescriptorData
      .append((applySite: pai, closureArgIndex: closureArgumentIndex, paramInfo: closureParamInfo))
  }

  return intermediateClosureArgDescriptorData
}

/// Finalizes the call sites for a given root closure by adding a corresponding `ClosureArgDescriptor`
/// to all call sites where the closure is ultimately passed as an argument.
private func finalizeCallSite(
  for rootClosure: SingleValueInstruction, in callSiteOpt: inout CallSite?,
  rootClosurePossibleLiveRange: InstructionRange,
  intermediateClosureArgDescriptorData: [IntermediateClosureArgDescriptorDatum],
  _ context: FunctionPassContext
) {
  let closureInfo = ClosureInfo(
    closure: rootClosure, lifetimeFrontier: Array(rootClosurePossibleLiveRange.ends))

  for (applySite, closureArgumentIndex, parameterInfo) in intermediateClosureArgDescriptorData {
    if callSiteOpt!.applySite != applySite {
      fatalError(
        "While finalizing call sites, call site descriptor not found for call site: \(applySite)!")
    }
    let closureArgDesc = ClosureArgDescriptor(
      closureInfo: closureInfo, closureArgumentIndex: closureArgumentIndex,
      parameterInfo: parameterInfo)
    callSiteOpt!.appendClosureArgDescriptor(closureArgDesc)
  }
}

private func isClosureApplied(in callee: Function, closureArgIndex index: Int) -> Bool {
  func inner(_ callee: Function, _ index: Int, _ handledFuncs: inout Set<Function>) -> Bool {
    let closureArg = callee.argument(at: index)

    for use in closureArg.uses {
      if let fai = use.instruction as? ApplySite {
        if fai.callee == closureArg {
          return true
        }

        if let faiCallee = fai.referencedFunction,
          !faiCallee.blocks.isEmpty,
          handledFuncs.insert(faiCallee).inserted,
          handledFuncs.count <= recursionBudget
        {
          if inner(faiCallee, fai.calleeArgumentIndex(of: use)!, &handledFuncs) {
            return true
          }
        }
      }
    }

    return false
  }

  // Limit the number of recursive calls to not go into exponential behavior in corner cases.
  let recursionBudget = 8
  var handledFuncs: Set<Function> = []
  return inner(callee, index, &handledFuncs)
}

/// Marks any converted/reabstracted closures, corresponding to a given root closure as used. We do not want to
/// look at such closures separately as during function specialization they will be handled as part of the root closure.
private func markConvertedAndReabstractedClosuresAsUsed(
  rootClosure: Value, convertedAndReabstractedClosure: Value,
  convertedAndReabstractedClosures: inout InstructionSet
) {
  if convertedAndReabstractedClosure != rootClosure {
    switch convertedAndReabstractedClosure {
    case let pai as PartialApplyInst:
      convertedAndReabstractedClosures.insert(pai)
      return
        markConvertedAndReabstractedClosuresAsUsed(
          rootClosure: rootClosure,
          convertedAndReabstractedClosure: pai.arguments[0],
          convertedAndReabstractedClosures: &convertedAndReabstractedClosures)
    case let cvt as ConvertFunctionInst:
      convertedAndReabstractedClosures.insert(cvt)
      return
        markConvertedAndReabstractedClosuresAsUsed(
          rootClosure: rootClosure,
          convertedAndReabstractedClosure: cvt.fromFunction,
          convertedAndReabstractedClosures: &convertedAndReabstractedClosures)
    case let cvt as ConvertEscapeToNoEscapeInst:
      convertedAndReabstractedClosures.insert(cvt)
      return
        markConvertedAndReabstractedClosuresAsUsed(
          rootClosure: rootClosure,
          convertedAndReabstractedClosure: cvt.fromFunction,
          convertedAndReabstractedClosures: &convertedAndReabstractedClosures)
    case let mdi as MarkDependenceInst:
      convertedAndReabstractedClosures.insert(mdi)
      return
        markConvertedAndReabstractedClosuresAsUsed(
          rootClosure: rootClosure, convertedAndReabstractedClosure: mdi.value,
          convertedAndReabstractedClosures: &convertedAndReabstractedClosures)
    default:
      log("Parent function of callSite: \(rootClosure.parentFunction)")
      log("Root closure: \(rootClosure)")
      log("Converted/reabstracted closure: \(convertedAndReabstractedClosure)")
      fatalError(
        "While marking converted/reabstracted closures as used, found unexpected instruction: \(convertedAndReabstractedClosure)"
      )
    }
  }
}

private func rewriteUsesOfPayloadItem(
  use: Operand, resultIdx: Int, closureInfoArray: [ClosureInfoCFG],
  newDti: DestructureTupleInst, context: FunctionPassContext
) {
  switch use.instruction {
  case let ai as ApplyInst:
    let builder = Builder(before: ai, context)
    var closureInfoOpt = ClosureInfoCFG?(nil)
    for closureInfo in closureInfoArray {
      if closureInfo.idxInEnumPayload == resultIdx {
        if closureInfoOpt != nil {
          assert(closureInfoOpt!.closure == closureInfo.closure)
          assert(closureInfoOpt!.payloadTuple == closureInfo.payloadTuple)
        } else {
          closureInfoOpt = closureInfo
        }
      }
    }
    if closureInfoOpt != nil {
      let dtiOfCapturedArgsTuple = builder.createDestructureTuple(
        tuple: newDti.results[resultIdx])
      if closureInfoOpt!.subsetThunk == nil {
        var newArgs = [Value]()
        for op in ai.argumentOperands {
          newArgs.append(op.value)
        }
        for res in dtiOfCapturedArgsTuple.results {
          newArgs.append(res)
        }
        let vjpFn = closureInfoOpt!.closure.asSupportedClosureFn!
        let newFri = builder.createFunctionRef(vjpFn)
        let newAi = builder.createApply(
          function: newFri, ai.substitutionMap, arguments: newArgs)
        ai.replace(with: newAi, context)
        // MYTODO: maybe we can set insertion point earlier
        let newBuilder = Builder(before: newAi.parentBlock.terminator, context)
        for res in dtiOfCapturedArgsTuple.results {
          if !res.type.isTrivial(in: res.parentFunction) {
            newBuilder.createDestroyValue(operand: res)
          }
        }
      } else {
        var newClosure = SingleValueInstruction?(nil)
        let maybePai = closureInfoOpt!.closure as? PartialApplyInst
        if maybePai != nil {
          var newArgs = [Value]()
          for res in dtiOfCapturedArgsTuple.results {
            newArgs.append(res)
          }
          let vjpFn = closureInfoOpt!.closure.asSupportedClosureFn!
          let newFri = builder.createFunctionRef(vjpFn)
          let newPai = builder.createPartialApply(
            function: newFri, substitutionMap: maybePai!.substitutionMap,
            capturedArguments: newArgs, calleeConvention: maybePai!.calleeConvention,
            hasUnknownResultIsolation: maybePai!.hasUnknownResultIsolation,
            isOnStack: maybePai!.isOnStack)
          newClosure = newPai
          // MYTODO: maybe we can set insertion point earlier
          let newBuilder = Builder(before: newPai.parentBlock.terminator, context)
          for res in dtiOfCapturedArgsTuple.results {
            if !res.type.isTrivial(in: res.parentFunction) {
              newBuilder.createDestroyValue(operand: res)
            }
          }
        } else {
          let maybeTttfi = closureInfoOpt!.closure as? ThinToThickFunctionInst
          assert(maybeTttfi != nil)
          let vjpFn = closureInfoOpt!.closure.asSupportedClosureFn!
          let newFri = builder.createFunctionRef(vjpFn)
          let newTttfi = builder.createThinToThickFunction(
            thinFunction: newFri, resultType: maybeTttfi!.type)
          newClosure = newTttfi
        }
        assert(newClosure != nil)
        let subsetThunkFn = closureInfoOpt!.subsetThunk!.referencedFunction!
        let newFri = builder.createFunctionRef(subsetThunkFn)

        var newArgs = [Value]()
        for op in ai.argumentOperands {
          newArgs.append(op.value)
        }
        newArgs.append(newClosure!)
        let newAi = builder.createApply(
          function: newFri, ai.substitutionMap, arguments: newArgs)
        ai.replace(with: newAi, context)
        let newBuilder = Builder(before: newAi.parentBlock.terminator, context)
        if !newClosure!.type.isTrivial(in: newAi.parentFunction) {
          newBuilder.createDestroyValue(operand: newClosure!)
        }
      }
    } else {
      var newArgs = [Value]()
      for op in ai.argumentOperands {
        newArgs.append(op.value)
      }
      let newAi = builder.createApply(
        function: newDti.results[resultIdx], ai.substitutionMap, arguments: newArgs)
      ai.replace(with: newAi, context)
    }

  case let dvi as DestroyValueInst:
    var needDestroyValue = true
    for closureInfo in closureInfoArray {
      if closureInfo.idxInEnumPayload == resultIdx {
        needDestroyValue = false
      }
    }
    if needDestroyValue {
      let builder = Builder(before: dvi, context)
      builder.createDestroyValue(operand: newDti.results[resultIdx])
    }
    dvi.parentBlock.eraseInstruction(dvi)

  case let uedi as UncheckedEnumDataInst:
    let builder = Builder(before: uedi, context)
    let newUedi = builder.createUncheckedEnumData(
      enum: newDti.results[resultIdx], caseIndex: uedi.caseIndex,
      resultType: newDti.results[resultIdx].type.bridged.getEnumCasePayload(
        uedi.caseIndex, uedi.parentFunction.bridged
      ).type)
    uedi.replace(with: newUedi, context)

  case let sei as SwitchEnumInst:
    let builder = Builder(before: sei, context)
    let newSEI = builder.createSwitchEnum(
      enum: newDti.results[resultIdx], cases: getEnumCasesForSwitchEnumInst(sei))
    newSEI.parentBlock.eraseInstruction(sei)

  default:
    assert(false)
  }
}

private func getEnumCasesForSwitchEnumInst(_ sei: SwitchEnumInst) -> [(Int, BasicBlock)] {
  var enumCases = [(Int, BasicBlock)]()
  for i in 0...sei.bridged.SwitchEnumInst_getNumCases() {
    let bbForCase = sei.getUniqueSuccessor(forCaseIndex: i)
    if bbForCase != nil {
      enumCases.append((i, bbForCase!))
    }
  }
  return enumCases
}

private func getPbBBToVjpBBMap(_ vjpBBToPbBBMap: [BasicBlock: BasicBlock]) -> [BasicBlock:
  BasicBlock]
{
  var pbBBToVjpBBMap = [BasicBlock: BasicBlock]()
  for (vjpBB, pbBB) in vjpBBToPbBBMap {
    pbBBToVjpBBMap[pbBB] = vjpBB
  }
  assert(pbBBToVjpBBMap.count == vjpBBToPbBBMap.count)
  return pbBBToVjpBBMap
}

extension BasicBlock {
  func eraseInstruction(_ inst: Instruction) {
    self.bridged.eraseInstruction(inst.bridged)
  }
}

extension SpecializationCloner {
  fileprivate func cloneAndSpecializeFunctionBodyCFG(
    using callSite: CallSite, enumDict: EnumDict
  ) {
    let closureInfos = callSite.closureInfosCFG
    self.cloneEntryBlockArgsWithoutOrigClosuresCFG(
      usingOrigCalleeAt: callSite, enumDict: enumDict)

    var args = [Value]()
    for arg in self.entryBlock.arguments {
      args.append(arg)
    }

    self.cloneFunctionBody(from: callSite.applyCallee, entryBlockArguments: args)

    let clonedPbBBToVjpBBMap = getPbBBToVjpBBMap(
      getVjpBBToPbBBMap(vjp: callSite.applySite.parentFunction, pb: self.cloned)!)
    let vjpBBToTupleInstMap = getVjpBBToTupleInstMap(vjp: callSite.applySite.parentFunction)!

    for bb in self.cloned.blocks {
      if bb == self.cloned.entryBlock {
        let sei = bb.terminator as! SwitchEnumInst
        let builderEntry = Builder(before: sei, self.context)

        builderEntry.createSwitchEnum(
          enum: sei.enumOp, cases: getEnumCasesForSwitchEnumInst(sei))
        bb.eraseInstruction(sei)

        continue
      }

      guard let arg = getEnumPayloadArgOfPbBB(bb) else {
        continue
      }

      guard let ti = vjpBBToTupleInstMap[clonedPbBBToVjpBBMap[bb]!] else {
        continue
      }

      var closureInfoArray = [ClosureInfoCFG]()
      var adcsHelper = BridgedAutoDiffClosureSpecializationHelper()
      defer {
        adcsHelper.clearClosuresBufferForPb()
      }
      for (opIdx, op) in ti.operands.enumerated() {
        let val = op.value
        for closureInfo in closureInfos {
          if ((closureInfo.subsetThunk == nil && closureInfo.closure == val)
            || (closureInfo.subsetThunk != nil && closureInfo.subsetThunk! == val))
            && closureInfo.payloadTuple == ti
          {
            assert(closureInfo.idxInEnumPayload == opIdx)
            closureInfoArray.append(closureInfo)
            adcsHelper.appendToClosuresBufferForPb(
              closureInfo.closure.bridged,
              closureInfo.idxInEnumPayload)
          }
        }
      }
      let newArg = bb.bridged.recreateTupleBlockArgument(arg.bridged).argument

      assert(newArg.uses.count <= 1)
      if newArg.uses.singleUse == nil {
        continue
      }
      let oldDti = newArg.uses.singleUse!.instruction as! DestructureTupleInst
      let builderBeforeOldDti = Builder(before: oldDti, self.context)
      let newDti = builderBeforeOldDti.createDestructureTuple(tuple: oldDti.tuple)

      for (resultIdx, result) in oldDti.results.enumerated() {
        for use in result.uses {
          rewriteUsesOfPayloadItem(
            use: use, resultIdx: resultIdx, closureInfoArray: closureInfoArray, newDti: newDti,
            context: self.context)
        }
      }

      oldDti.parentBlock.eraseInstruction(oldDti)
    }
  }

  private func cloneEntryBlockArgsWithoutOrigClosuresCFG(
    usingOrigCalleeAt callSite: CallSite, enumDict: EnumDict
  ) {
    let pb = callSite.applyCallee
    let enumType = getEnumArgOfEntryPbBB(pb.entryBlock)!.type

    let originalEntryBlock = callSite.applyCallee.entryBlock
    let clonedFunction = self.cloned
    let clonedEntryBlock = self.entryBlock

    for arg in originalEntryBlock.arguments {
      var clonedEntryBlockArgType = arg.type.getLoweredType(in: clonedFunction)
      if clonedEntryBlockArgType == enumType {
        // This should always hold since we have at least 1 closure (otherwise, we wouldn't go here).
        // It causes re-write of the corresponding branch tracing enum, and the top enum type will be re-written transitively.
        assert(enumDict[enumType] != nil)
        clonedEntryBlockArgType = enumDict[enumType]!
      }
      let clonedEntryBlockArg = clonedEntryBlock.addFunctionArgument(
        type: clonedEntryBlockArgType, self.context)
      clonedEntryBlockArg.copyFlags(from: arg as! FunctionArgument)
    }
  }

  fileprivate func cloneAndSpecializeFunctionBody(using callSite: CallSite) {
    self.cloneEntryBlockArgsWithoutOrigClosures(usingOrigCalleeAt: callSite)

    let (allSpecializedEntryBlockArgs, closureArgIndexToAllClonedReleasableClosures) =
      cloneAllClosures(at: callSite)

    self.cloneFunctionBody(
      from: callSite.applyCallee, entryBlockArguments: allSpecializedEntryBlockArgs)

    self.insertCleanupCodeForClonedReleasableClosures(
      from: callSite,
      closureArgIndexToAllClonedReleasableClosures: closureArgIndexToAllClonedReleasableClosures)
  }

  private func cloneEntryBlockArgsWithoutOrigClosures(usingOrigCalleeAt callSite: CallSite) {
    let originalEntryBlock = callSite.applyCallee.entryBlock
    let clonedFunction = self.cloned
    let clonedEntryBlock = self.entryBlock

    originalEntryBlock.arguments
      .enumerated()
      .filter { index, _ in !callSite.hasClosureArg(at: index) }
      .forEach { _, arg in
        let clonedEntryBlockArgType = arg.type.getLoweredType(in: clonedFunction)
        let clonedEntryBlockArg = clonedEntryBlock.addFunctionArgument(
          type: clonedEntryBlockArgType, self.context)
        clonedEntryBlockArg.copyFlags(from: arg as! FunctionArgument)
      }
  }

  /// Clones all closures, originally passed to the callee at the given callSite, into the specialized function.
  ///
  /// Returns the following -
  /// - allSpecializedEntryBlockArgs: Complete list of entry block arguments for the specialized function. This includes
  ///   the original arguments to the function (minus the closure arguments) and the arguments representing the values
  ///   originally captured by the skipped closure arguments.
  ///
  /// - closureArgIndexToAllClonedReleasableClosures: Mapping from a closure's argument index at `callSite` to the list
  ///   of corresponding releasable closures cloned into the specialized function. We have a "list" because we clone
  ///   "closure chains", which consist of a "root" closure and its conversions/reabstractions. This map is used to
  ///   generate cleanup code for the cloned closures in the specialized function.
  private func cloneAllClosures(at callSite: CallSite)
    -> (
      allSpecializedEntryBlockArgs: [Value],
      closureArgIndexToAllClonedReleasableClosures: [Int: [SingleValueInstruction]]
    )
  {
    func entryBlockArgsWithOrigClosuresSkipped() -> [Value?] {
      var clonedNonClosureEntryBlockArgs = self.entryBlock.arguments.makeIterator()

      return callSite.applyCallee
        .entryBlock
        .arguments
        .enumerated()
        .reduce(into: []) { result, origArgTuple in
          let (index, _) = origArgTuple
          if !callSite.hasClosureArg(at: index) {
            result.append(clonedNonClosureEntryBlockArgs.next())
          } else {
            result.append(Optional.none)
          }
        }
    }

    var entryBlockArgs: [Value?] = entryBlockArgsWithOrigClosuresSkipped()
    var closureArgIndexToAllClonedReleasableClosures: [Int: [SingleValueInstruction]] = [:]

    for closureArgDesc in callSite.closureArgDescriptors {
      let (finalClonedReabstractedClosure, allClonedReleasableClosures) =
        self.cloneClosureChain(representedBy: closureArgDesc, at: callSite)

      entryBlockArgs[closureArgDesc.closureArgIndex] = finalClonedReabstractedClosure
      closureArgIndexToAllClonedReleasableClosures[closureArgDesc.closureArgIndex] =
        allClonedReleasableClosures
    }

    return (entryBlockArgs.map { $0! }, closureArgIndexToAllClonedReleasableClosures)
  }

  private func cloneClosureChain(
    representedBy closureArgDesc: ClosureArgDescriptor, at callSite: CallSite
  )
    -> (
      finalClonedReabstractedClosure: SingleValueInstruction,
      allClonedReleasableClosures: [SingleValueInstruction]
    )
  {
    let (origToClonedValueMap, capturedArgRange) = self.addEntryBlockArgs(
      forValuesCapturedBy: closureArgDesc)
    let clonedFunction = self.cloned
    let clonedEntryBlock = self.entryBlock
    let clonedClosureArgs = Array(clonedEntryBlock.arguments[capturedArgRange])

    let builder =
      clonedEntryBlock.instructions.isEmpty
      ? Builder(atStartOf: clonedFunction, self.context)
      : Builder(
        atEndOf: clonedEntryBlock, location: clonedEntryBlock.instructions.last!.location,
        self.context)

    let clonedRootClosure = builder.cloneRootClosure(
      representedBy: closureArgDesc, capturedArguments: clonedClosureArgs)

    let finalClonedReabstractedClosure =
      builder.cloneRootClosureReabstractions(
        rootClosure: closureArgDesc.closure, clonedRootClosure: clonedRootClosure,
        reabstractedClosure: callSite.appliedArgForClosure(at: closureArgDesc.closureArgIndex)!,
        origToClonedValueMap: origToClonedValueMap,
        self.context)

    let allClonedReleasableClosures = [finalClonedReabstractedClosure]
    return (finalClonedReabstractedClosure, allClonedReleasableClosures)
  }

  private func addEntryBlockArgs(forValuesCapturedBy closureArgDesc: ClosureArgDescriptor)
    -> (origToClonedValueMap: [HashableValue: Value], capturedArgRange: Range<Int>)
  {
    var origToClonedValueMap: [HashableValue: Value] = [:]
    let clonedFunction = self.cloned
    let clonedEntryBlock = self.entryBlock

    let capturedArgRangeStart = clonedEntryBlock.arguments.count

    for arg in closureArgDesc.arguments {
      let capturedArg = clonedEntryBlock.addFunctionArgument(
        type: arg.type.getLoweredType(in: clonedFunction),
        self.context)
      origToClonedValueMap[arg] = capturedArg
    }

    let capturedArgRangeEnd = clonedEntryBlock.arguments.count
    let capturedArgRange =
      capturedArgRangeStart == capturedArgRangeEnd
      ? 0..<0
      : capturedArgRangeStart..<capturedArgRangeEnd

    return (origToClonedValueMap, capturedArgRange)
  }

  private func insertCleanupCodeForClonedReleasableClosures(
    from callSite: CallSite,
    closureArgIndexToAllClonedReleasableClosures: [Int: [SingleValueInstruction]]
  ) {
    for closureArgDesc in callSite.closureArgDescriptors {
      let allClonedReleasableClosures = closureArgIndexToAllClonedReleasableClosures[
        closureArgDesc.closureArgIndex]!

      // Insert a `destroy_value`, for all releasable closures, in all reachable exit BBs if the closure was passed as a
      // guaranteed parameter or its type was noescape+thick. This is b/c the closure was passed at +0 originally and we
      // need to balance the initial increment of the newly created closure(s).
      if closureArgDesc.isClosureGuaranteed
        || closureArgDesc.parameterInfo.isTrivialNoescapeClosure,
        !allClonedReleasableClosures.isEmpty
      {
        for exitBlock in callSite.reachableExitBBsInCallee {
          let clonedExitBlock = self.getClonedBlock(for: exitBlock)

          let terminator =
            clonedExitBlock.terminator is UnreachableInst
            ? clonedExitBlock.terminator.previous!
            : clonedExitBlock.terminator

          let builder = Builder(before: terminator, self.context)

          for closure in allClonedReleasableClosures {
            if let pai = closure as? PartialApplyInst {
              builder.destroyPartialApply(pai: pai, self.context)
            }
          }
        }
      }
    }

    if self.context.needFixStackNesting {
      self.cloned.fixStackNesting(self.context)
    }
  }
}

extension [HashableValue: Value] {
  fileprivate subscript(key: Value) -> Value? {
    get {
      self[key.hashable]
    }
    set {
      self[key.hashable] = newValue
    }
  }
}

extension CallSite {
  fileprivate enum NewApplyArg {
    case Original(Value)
    // TODO: This can be simplified in OSSA. We can just do a copy_value for everything - except for addresses???
    case PreviouslyCaptured(
      value: Value, needsRetain: Bool, parentClosureArgIndex: Int)

    var value: Value {
      switch self {
      case let .Original(originalArg):
        return originalArg
      case let .PreviouslyCaptured(capturedArg, _, _):
        return capturedArg
      }
    }
  }

  fileprivate func getArgumentsForSpecializedApply(of specializedCallee: Function) -> [NewApplyArg]
  {
    var newApplyArgs: [NewApplyArg] = []

    // Original arguments
    for (applySiteIndex, arg) in self.applySite.arguments.enumerated() {
      let calleeArgIndex = self.applySite.unappliedArgumentCount + applySiteIndex
      if !self.hasClosureArg(at: calleeArgIndex) {
        newApplyArgs.append(.Original(arg))
      }
    }

    // Previously captured arguments
    for closureArgDesc in self.closureArgDescriptors {
      for (applySiteIndex, capturedArg) in closureArgDesc.arguments.enumerated() {
        let needsRetain = closureArgDesc.isCapturedArgNonTrivialObjectType(
          applySiteIndex: applySiteIndex,
          specializedCallee: specializedCallee)

        newApplyArgs.append(
          .PreviouslyCaptured(
            value: capturedArg, needsRetain: needsRetain,
            parentClosureArgIndex: closureArgDesc.closureArgIndex))
      }
    }

    return newApplyArgs
  }
}

extension ClosureArgDescriptor {
  fileprivate func isCapturedArgNonTrivialObjectType(
    applySiteIndex: Int, specializedCallee: Function
  ) -> Bool {
    precondition(
      self.closure is PartialApplyInst, "ClosureArgDescriptor is not for a partial_apply closure!")

    let capturedArg = self.arguments[applySiteIndex]
    let pai = self.closure as! PartialApplyInst
    let capturedArgIndexInCallee = applySiteIndex + pai.unappliedArgumentCount
    let capturedArgConvention = self.callee.argumentConventions[capturedArgIndexInCallee]

    return !capturedArg.type.isTrivial(in: specializedCallee)
      && !capturedArgConvention.isAllowedIndirectConvForClosureSpec
  }
}

extension Builder {
  fileprivate func cloneRootClosure(
    representedBy closureArgDesc: ClosureArgDescriptor, capturedArguments: [Value]
  )
    -> SingleValueInstruction
  {
    let function = self.createFunctionRef(closureArgDesc.callee)

    if let pai = closureArgDesc.closure as? PartialApplyInst {
      return self.createPartialApply(
        function: function, substitutionMap: SubstitutionMap(),
        capturedArguments: capturedArguments, calleeConvention: pai.calleeConvention,
        hasUnknownResultIsolation: pai.hasUnknownResultIsolation,
        isOnStack: pai.isOnStack)
    } else {
      return self.createThinToThickFunction(
        thinFunction: function, resultType: closureArgDesc.closure.type)
    }
  }

  fileprivate func cloneRootClosureReabstractions(
    rootClosure: Value, clonedRootClosure: Value, reabstractedClosure: Value,
    origToClonedValueMap: [HashableValue: Value], _ context: FunctionPassContext
  )
    -> SingleValueInstruction
  {
    func inner(
      _ rootClosure: Value, _ clonedRootClosure: Value, _ reabstractedClosure: Value,
      _ origToClonedValueMap: inout [HashableValue: Value]
    ) -> Value {
      switch reabstractedClosure {
      case let reabstractedClosure where reabstractedClosure == rootClosure:
        origToClonedValueMap[reabstractedClosure] = clonedRootClosure
        return clonedRootClosure

      case let cvt as ConvertFunctionInst:
        let toBeReabstracted = inner(
          rootClosure, clonedRootClosure, cvt.fromFunction,
          &origToClonedValueMap)
        let reabstracted = self.createConvertFunction(
          originalFunction: toBeReabstracted, resultType: cvt.type,
          withoutActuallyEscaping: cvt.withoutActuallyEscaping)
        origToClonedValueMap[cvt] = reabstracted
        return reabstracted

      case let cvt as ConvertEscapeToNoEscapeInst:
        let toBeReabstracted = inner(
          rootClosure, clonedRootClosure, cvt.fromFunction,
          &origToClonedValueMap)
        let reabstracted = self.createConvertEscapeToNoEscape(
          originalFunction: toBeReabstracted, resultType: cvt.type,
          isLifetimeGuaranteed: true)
        origToClonedValueMap[cvt] = reabstracted
        return reabstracted

      case let pai as PartialApplyInst:
        let toBeReabstracted = inner(
          rootClosure, clonedRootClosure, pai.arguments[0],
          &origToClonedValueMap)

        guard let function = pai.referencedFunction else {
          log("Parent function of callSite: \(rootClosure.parentFunction)")
          log("Root closure: \(rootClosure)")
          log("Unsupported reabstraction closure: \(pai)")
          fatalError("Encountered unsupported reabstraction (via partial_apply) of root closure!")
        }

        let fri = self.createFunctionRef(function)
        let reabstracted = self.createPartialApply(
          function: fri, substitutionMap: SubstitutionMap(),
          capturedArguments: [toBeReabstracted],
          calleeConvention: pai.calleeConvention,
          hasUnknownResultIsolation: pai.hasUnknownResultIsolation,
          isOnStack: pai.isOnStack)
        origToClonedValueMap[pai] = reabstracted
        return reabstracted

      case let mdi as MarkDependenceInst:
        let toBeReabstracted = inner(
          rootClosure, clonedRootClosure, mdi.value, &origToClonedValueMap)
        let base = origToClonedValueMap[mdi.base]!
        let reabstracted = self.createMarkDependence(
          value: toBeReabstracted, base: base, kind: .Escaping)
        origToClonedValueMap[mdi] = reabstracted
        return reabstracted

      default:
        log("Parent function of callSite: \(rootClosure.parentFunction)")
        log("Root closure: \(rootClosure)")
        log("Converted/reabstracted closure: \(reabstractedClosure)")
        fatalError("Encountered unsupported reabstraction of root closure: \(reabstractedClosure)")
      }
    }

    var origToClonedValueMap = origToClonedValueMap
    let finalClonedReabstractedClosure = inner(
      rootClosure, clonedRootClosure, reabstractedClosure,
      &origToClonedValueMap)
    return (finalClonedReabstractedClosure as! SingleValueInstruction)
  }

  fileprivate func destroyPartialApply(pai: PartialApplyInst, _ context: FunctionPassContext) {
    // TODO: Support only OSSA instructions once the OSSA elimination pass is moved after all function optimization
    // passes.

    if pai.isOnStack {
      // for arg in pai.arguments {
      //   self.createDestroyValue(operand: arg)
      // }
      // self.createDestroyValue(operand: pai)

      if pai.parentFunction.hasOwnership {
        // Under OSSA, the closure acts as an owned value whose lifetime is a borrow scope for the captures, so we need to
        // end the borrow scope before ending the lifetimes of the captures themselves.
        self.createDestroyValue(operand: pai)
        self.destroyCapturedArgs(for: pai)
      } else {
        self.destroyCapturedArgs(for: pai)
        self.createDeallocStack(pai)
        context.notifyInvalidatedStackNesting()
      }
    } else {
      if pai.parentFunction.hasOwnership {
        self.createDestroyValue(operand: pai)
      } else {
        self.createReleaseValue(operand: pai)
      }
    }
  }
}

typealias EnumDict = [Type: Type]

private func getSpecializedParametersCFG(
  basedOn callSite: CallSite, pb: Function, enumType: Type, enumDict: EnumDict,
  _ context: FunctionPassContext
) -> [ParameterInfo] {
  let applySiteCallee = callSite.applyCallee
  var specializedParamInfoList: [ParameterInfo] = []
  var foundBranchTracingEnumParam = false
  // Start by adding all original parameters except for the closure parameters.
  for paramInfo in applySiteCallee.convention.parameters {
    // TODO: is this safe to perform such check?
    if paramInfo.type.rawType.bridged.type != enumType.canonicalType.rawType.bridged.type {
      specializedParamInfoList.append(paramInfo)
      continue
    }
    assert(!foundBranchTracingEnumParam)
    foundBranchTracingEnumParam = true
    let newParamInfo = ParameterInfo(
      type: enumDict[enumType]!.canonicalType, convention: paramInfo.convention,
      options: paramInfo.options, hasLoweredAddresses: paramInfo.hasLoweredAddresses)
    specializedParamInfoList.append(newParamInfo)
  }
  assert(foundBranchTracingEnumParam)
  return specializedParamInfoList
}

extension FunctionConvention {
  fileprivate func getSpecializedParameters(basedOn callSite: CallSite) -> [ParameterInfo] {
    let applySiteCallee = callSite.applyCallee
    var specializedParamInfoList: [ParameterInfo] = []

    // Start by adding all original parameters except for the closure parameters.
    let firstParamIndex = applySiteCallee.argumentConventions.firstParameterIndex
    for (index, paramInfo) in applySiteCallee.convention.parameters.enumerated() {
      let argIndex = index + firstParamIndex
      if !callSite.hasClosureArg(at: argIndex) {
        specializedParamInfoList.append(paramInfo)
      }
    }

    // Now, append parameters captured by each of the original closure parameter.
    //
    // Captured parameters are always appended to the function signature. If the argument type of the captured
    // parameter in the callee is:
    // - direct and trivial, pass the new parameter as Direct_Unowned.
    // - direct and non-trivial, pass the new parameter as Direct_Owned.
    // - indirect, pass the new parameter using the same parameter convention as in
    //   the original closure.
    for closureArgDesc in callSite.closureArgDescriptors {
      if let closure = closureArgDesc.closure as? PartialApplyInst {
        let closureCallee = closureArgDesc.callee
        let closureCalleeConvention = closureCallee.convention
        let unappliedArgumentCount =
          closure.unappliedArgumentCount - closureCalleeConvention.indirectSILResultCount

        let prevCapturedParameters =
          closureCalleeConvention
          .parameters[unappliedArgumentCount...]
          .enumerated()
          .map { index, paramInfo in
            let argIndexOfParam =
              closureCallee.argumentConventions.firstParameterIndex + unappliedArgumentCount + index
            let argType = closureCallee.argumentTypes[argIndexOfParam]
            return paramInfo.withSpecializedConvention(
              isArgTypeTrivial: argType.isTrivial(in: closureCallee))
          }

        specializedParamInfoList.append(contentsOf: prevCapturedParameters)
      }
    }

    return specializedParamInfoList
  }
}

extension ParameterInfo {
  fileprivate func withSpecializedConvention(isArgTypeTrivial: Bool) -> Self {
    let specializedParamConvention =
      self.convention.isAllowedIndirectConvForClosureSpec
      ? self.convention
      : isArgTypeTrivial ? ArgumentConvention.directUnowned : ArgumentConvention.directOwned

    return ParameterInfo(
      type: self.type, convention: specializedParamConvention, options: self.options,
      hasLoweredAddresses: self.hasLoweredAddresses)
  }

  fileprivate var isTrivialNoescapeClosure: Bool {
    SILFunctionType_isTrivialNoescape(type.bridged)
  }
}

extension ArgumentConvention {
  fileprivate var isAllowedIndirectConvForClosureSpec: Bool {
    switch self {
    case .indirectInout, .indirectInoutAliasable:
      return true
    default:
      return false
    }
  }
}

extension PartialApplyInst {
  /// True, if the closure obtained from this partial_apply is the
  /// pullback returned from an autodiff VJP
  fileprivate var isPullbackInResultOfAutodiffVJP: Bool {
    if self.parentFunction.isAutodiffVJP,
      let use = self.uses.singleUse,
      let tupleInst = use.instruction as? TupleInst,
      let returnInst = self.parentFunction.returnInstruction,
      tupleInst == returnInst.returnedValue
    {
      return true
    }

    return false
  }

  var isPartialApplyOfThunk: Bool {
    if self.numArguments == 1,
      let fun = self.referencedFunction,
      fun.thunkKind == .reabstractionThunk || fun.thunkKind == .thunk,
      self.arguments[0].type.isLoweredFunction,
      self.arguments[0].type.isReferenceCounted(in: self.parentFunction)
        || self.callee.type.isThickFunction
    {
      return true
    }

    return false
  }

  fileprivate var hasOnlyInoutIndirectArguments: Bool {
    self.argumentOperands
      .filter { !$0.value.type.isObject }
      .allSatisfy { self.convention(of: $0)!.isInout }
  }
}

extension Instruction {
  fileprivate var asSupportedClosure: SingleValueInstruction? {
    switch self {
    case let tttf as ThinToThickFunctionInst where tttf.callee is FunctionRefInst:
      return tttf
    // TODO: figure out what to do with non-inout indirect arguments
    // https://forums.swift.org/t/non-inout-indirect-types-not-supported-in-closure-specialization-optimization/70826
    case let pai as PartialApplyInst
    where pai.callee is FunctionRefInst && pai.hasOnlyInoutIndirectArguments:
      return pai
    default:
      return nil
    }
  }

  fileprivate var asSupportedClosureFn: Function? {
    switch self {
    case let tttf as ThinToThickFunctionInst where tttf.callee is FunctionRefInst:
      let fri = tttf.callee as! FunctionRefInst
      return fri.referencedFunction
    // TODO: figure out what to do with non-inout indirect arguments
    // https://forums.swift.org/t/non-inout-indirect-types-not-supported-in-closure-specialization-optimization/70826
    case let pai as PartialApplyInst
    where pai.callee is FunctionRefInst && pai.hasOnlyInoutIndirectArguments:
      let fri = pai.callee as! FunctionRefInst
      return fri.referencedFunction
    default:
      return nil
    }
  }

  fileprivate var isSupportedClosure: Bool {
    asSupportedClosure != nil
  }
}

extension ApplySite {
  fileprivate var calleeIsDynamicFunctionRef: Bool {
    return !(callee is DynamicFunctionRefInst || callee is PreviousDynamicFunctionRefInst)
  }
}

extension Function {
  fileprivate var effectAllowsSpecialization: Bool {
    switch self.effectAttribute {
    case .readNone, .readOnly, .releaseNone: return false
    default: return true
    }
  }
}

// ===================== Utility Types ===================== //
private struct OrderedDict<Key: Hashable, Value> {
  private var valueIndexDict: [Key: Int] = [:]
  private var entryList: [(Key, Value)] = []

  subscript(key: Key) -> Value? {
    if let index = valueIndexDict[key] {
      return entryList[index].1
    }
    return nil
  }

  mutating func insert(key: Key, value: Value) {
    if valueIndexDict[key] == nil {
      valueIndexDict[key] = entryList.count
      entryList.append((key, value))
    }
  }

  mutating func update(key: Key, value: Value) {
    if let index = valueIndexDict[key] {
      entryList[index].1 = value
    }
  }

  var keys: LazyMapSequence<[(Key, Value)], Key> {
    entryList.lazy.map { $0.0 }
  }

  var values: LazyMapSequence<[(Key, Value)], Value> {
    entryList.lazy.map { $0.1 }
  }
}

/// Represents all the information required to represent a closure in isolation, i.e., outside of a callsite context
/// where the closure may be getting passed as an argument.
///
/// Composed with other information inside a `ClosureArgDescriptor` to represent a closure as an argument at a callsite.
private struct ClosureInfo {
  let closure: SingleValueInstruction
  let lifetimeFrontier: [Instruction]

  init(closure: SingleValueInstruction, lifetimeFrontier: [Instruction]) {
    self.closure = closure
    self.lifetimeFrontier = lifetimeFrontier
  }

}

/// Represents a closure as an argument at a callsite.
private struct ClosureArgDescriptor {
  let closureInfo: ClosureInfo
  /// The index of the closure in the callsite's argument list.
  let closureArgumentIndex: Int
  let parameterInfo: ParameterInfo

  var closure: SingleValueInstruction {
    closureInfo.closure
  }
  var lifetimeFrontier: [Instruction] {
    closureInfo.lifetimeFrontier
  }

  var isPartialApplyOnStack: Bool {
    if let pai = closure as? PartialApplyInst {
      return pai.isOnStack
    }
    return false
  }

  var callee: Function {
    if let pai = closure as? PartialApplyInst {
      return pai.referencedFunction!
    } else {
      return (closure as! ThinToThickFunctionInst).referencedFunction!
    }
  }

  var location: Location {
    closure.location
  }

  var closureArgIndex: Int {
    closureArgumentIndex
  }

  var closureParamInfo: ParameterInfo {
    parameterInfo
  }

  var numArguments: Int {
    if let pai = closure as? PartialApplyInst {
      return pai.numArguments
    } else {
      return 0
    }
  }

  var arguments: LazyMapSequence<OperandArray, Value> {
    if let pai = closure as? PartialApplyInst {
      return pai.arguments
    }

    return OperandArray.empty.lazy.map { $0.value } as LazyMapSequence<OperandArray, Value>
  }

  var isClosureGuaranteed: Bool {
    closureParamInfo.convention.isGuaranteed
  }

  var isClosureConsumed: Bool {
    closureParamInfo.convention.isConsumed
  }
}

/// Represents a callsite containing one or more closure arguments.
private struct CallSite {
  let applySite: ApplySite
  var closureArgDescriptors: [ClosureArgDescriptor] = []
  var closureInfosCFG: [ClosureInfoCFG] = []

  init(applySite: ApplySite) {
    self.applySite = applySite
  }

  mutating func appendClosureArgDescriptor(_ descriptor: ClosureArgDescriptor) {
    self.closureArgDescriptors.append(descriptor)
  }

  var applyCallee: Function {
    applySite.referencedFunction!
  }

  var reachableExitBBsInCallee: [BasicBlock] {
    applyCallee.blocks.filter { $0.isReachableExitBlock }
  }

  func hasClosureArg(at index: Int) -> Bool {
    closureArgDescriptors.contains { $0.closureArgumentIndex == index }
  }

  func closureArgDesc(at index: Int) -> ClosureArgDescriptor? {
    closureArgDescriptors.first { $0.closureArgumentIndex == index }
  }

  func appliedArgForClosure(at index: Int) -> Value? {
    if let closureArgDesc = closureArgDesc(at: index) {
      return applySite.arguments[closureArgDesc.closureArgIndex - applySite.unappliedArgumentCount]
    }

    return nil
  }

  func specializedCalleeName(_ context: FunctionPassContext) -> String {
    let closureArgs = Array(self.closureArgDescriptors.map { $0.closure })
    let closureIndices = Array(self.closureArgDescriptors.map { $0.closureArgIndex })

    return context.mangle(
      withClosureArguments: closureArgs, closureArgIndices: closureIndices,
      from: applyCallee)
  }

  func specializedCalleeNameCFG(vjp: Function, _ context: FunctionPassContext) -> String {
    let paiOfPbInExitVjpBB = getPartialApplyOfPullbackInExitVJPBB(vjp: vjp)!
    var argAndIdxInPbPAI = (arg: Value, idx: Int)?(nil)
    for (argIdx, arg) in paiOfPbInExitVjpBB.arguments.enumerated() {
      if arg.type.isBranchTracingEnum {
        assert(argAndIdxInPbPAI == nil)
        argAndIdxInPbPAI = (arg: arg, idx: argIdx)
      }
    }
    assert(argAndIdxInPbPAI != nil)

    return context.mangle(
      withBranchTracingEnum: argAndIdxInPbPAI!.arg, argIdx: argAndIdxInPbPAI!.idx,
      from: applyCallee)
  }
}

// ===================== Unit tests ===================== //

let gatherCallSiteTest = FunctionTest("closure_specialize_gather_call_site") {
  function, arguments, context in
  print("Specializing closures in function: \(function.name)")
  print("===============================================")
  let callSite = gatherCallSite(in: function, context)!
  // TODO avoid this array
  let callSites = [callSite]

  callSites.forEach { callSite in
    print("PartialApply call site: \(callSite.applySite)")
    print("Passed in closures: ")
    for index in callSite.closureArgDescriptors.indices {
      var closureArgDescriptor = callSite.closureArgDescriptors[index]
      print("\(index+1). \(closureArgDescriptor.closureInfo.closure)")
    }
  }
  print("\n")
}

let specializedFunctionSignatureAndBodyTest = FunctionTest(
  "closure_specialize_specialized_function_signature_and_body"
) { function, arguments, context in

  let callSite = gatherCallSite(in: function, context)!

  let (specializedFunction, _) = getOrCreateSpecializedFunction(basedOn: callSite, context)
  print("Generated specialized function: \(specializedFunction.name)")
  print("\(specializedFunction)\n")
}

let rewrittenCallerBodyTest = FunctionTest("closure_specialize_rewritten_caller_body") {
  function, arguments, context in
  let callSite = gatherCallSite(in: function, context)!

  let (specializedFunction, _) = getOrCreateSpecializedFunction(basedOn: callSite, context)
  rewriteApplyInstruction(using: specializedFunction, callSite: callSite, context)

  print("Rewritten caller body for: \(function.name):")
  print("\(function)\n")
}
