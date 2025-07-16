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
import CxxStdlib
import Cxx

private let verbose = false

// TODO: unify existing and new logging
private let needLogADCS = true
private var passRunCount = 0
private func logADCS(prefix: String = "", msg: String) {
  if !needLogADCS || passRunCount == 0 {
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

extension BridgedType: Hashable {
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
  func isBranchTracingEnum(vjp: Function) -> Bool {
    if !(self.isEnum && self.description.hasPrefix("$_AD__$s")) {
      return false
    }
    assert(vjp.isAutodiffVJP)
    let res : Bool = vjp.bridged.isAutodiffBranchTracingEnumValid(self.bridged)
    if !res {
      logADCS(msg: "Foreign branch tracing enum encountered: vjp = \(vjp.name), enum type = \(self.description)")
    }
    return res
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

enum PayloadValues {
  case DestructureTuple([Value])
  case TupleExtract([Value])
  case ZeroUses
  case Unsupported
}

func getPayloadValues(payload : Argument, vjp : Function) -> PayloadValues {
  if payload.uses.count == 0 {
    return PayloadValues.ZeroUses
  }

  var results = [Value]()

  if payload.uses.singleUse != nil && payload.uses.singleUse!.instruction as? DestructureTupleInst != nil {
    let dti = payload.uses.singleUse!.instruction as! DestructureTupleInst
    // TODO: do we need to check that results is not empty?
    if dti.operands[0].value.type.tupleElements.count != 0
      && dti.results[0].type.isBranchTracingEnum(vjp: vjp) && dti.results[0].uses.count > 1 {
      return PayloadValues.Unsupported
    }
    for result in dti.results {
      results.append(result)
    }
    return PayloadValues.DestructureTuple(results)
  }

  var idxs = [Int]()
  for use in payload.uses {
    guard let tei = use.instruction as? TupleExtractInst else {
      return PayloadValues.Unsupported
    }
    if idxs.contains(tei.fieldIndex) {
      return PayloadValues.Unsupported
    }
    if tei.fieldIndex == 0 &&
      tei.type.isBranchTracingEnum(vjp: vjp) && tei.results[0].uses.count > 1 {
      return PayloadValues.Unsupported
    }
    idxs.append(tei.fieldIndex)
    results.append(tei)
  }
  return PayloadValues.TupleExtract(results)

  return PayloadValues.Unsupported
}

func checkIfCanRunForPayloadValues(results : [Value], prefixFail: String, pb: Function, pbBB: BasicBlock) -> Bool {
  for result in results {
    for use in result.uses {
      switch use.instruction {
      case _ as ApplyInst:
        ()
      case _ as DestroyValueInst:
        ()
      case _ as StrongReleaseInst:
        ()
      case let uedi as UncheckedEnumDataInst:
        if uedi.uses.count > 1 {
          logADCS(
            prefix: prefixFail,
            msg: "unchecked_enum_data instr has \(uedi.uses.count) uses, but no more than 1 is allowed")
          logADCS(msg: "  uedi: \(uedi)")
          logADCS(msg: "  uedi uses begin")
          for uediUse in uedi.uses {
            logADCS(msg: "  uediUse.instruction: \(uediUse.instruction)")
          }
          logADCS(msg: "  uedi uses end")
          return false
        }
        if uedi.uses.singleUse != nil {
          if let bri = uedi.uses.singleUse!.instruction as? BranchInst {
            // All OK
          } else {
            logADCS(
              prefix: prefixFail,
              msg: "unchecked_enum_data instr has unexpected single use")
            logADCS(msg: "  uedi: \(uedi)")
            logADCS(msg: "  uedi use: \(uedi.uses.singleUse!.instruction)")
            for (idx, uediUseResult) in uedi.uses.singleUse!.instruction.results.enumerated() {
              logADCS(msg: "  uedi use result \(idx) uses begin")
              for useOfResult in uediUseResult.uses {
                logADCS(msg: "    uedi use use: \(useOfResult.instruction)")
              }
              logADCS(msg: "  uedi use result \(idx) uses end")
            }
            return false
          }
        }
      case let cfi as ConvertFunctionInst:
        if cfi.uses.count != 2 {
          logADCS(
            prefix: prefixFail,
            msg: "expected exactly 2 uses of convert_function use of payload tuple element, found \(cfi.uses.count)")
          for (idx, cfiUse) in cfi.uses.enumerated() {
            logADCS(msg: "use \(idx): \(cfiUse)")
          }
          return false
        }
        var bbiUse = Operand?(nil)
        var dviUse = Operand?(nil)
        var sriUse = Operand?(nil)
        for cfiUse in cfi.uses {
          switch cfiUse.instruction {
            case let bbi as BeginBorrowInst:
              if bbiUse != nil {
                logADCS(
                  prefix: prefixFail,
                  msg: "multiple begin_borrow uses of convert_function result found, but exactly 1 expected")
                return false
              }
              bbiUse = cfiUse
            case let dvi as DestroyValueInst:
              if dviUse != nil {
                logADCS(
                  prefix: prefixFail,
                  msg: "multiple destroy_value uses of convert_function result found, but exactly 1 expected")
                return false
              }
              dviUse = cfiUse
            case let sri as StrongReleaseInst:
              if sriUse != nil {
                logADCS(
                  prefix: prefixFail,
                  msg: "multiple strong_release uses of convert_function result found, but exactly 1 expected")
                return false
              }
              sriUse = cfiUse
            default:
              logADCS(
                prefix: prefixFail,
                msg: "unexpected use of convert_function result found: \(cfiUse)")
              return false
          }
        }
        assert((dviUse != nil) != (sriUse != nil))
        assert(bbiUse != nil)

      case let bbi as BeginBorrowInst:
        if bbi.uses.count != 2 {
          logADCS(
            prefix: prefixFail,
            msg: "expected exactly 2 uses of begin_borrow use of payload tuple element, found \(bbi.uses.count)")
          for (idx, bbiUse) in bbi.uses.enumerated() {
            logADCS(msg: "use \(idx): \(bbiUse)")
          }
          return false
        }
        var aiUse = Operand?(nil)
        var ebUse = Operand?(nil)
        for bbiUse in bbi.uses {
          switch bbiUse.instruction {
            case let eb as EndBorrowInst:
              if ebUse != nil {
                logADCS(
                  prefix: prefixFail,
                  msg: "multiple end_borrow uses of begin_borrow result found, but exactly 1 expected")
                return false
              }
              ebUse = bbiUse
            case let ai as ApplyInst:
              if aiUse != nil {
                logADCS(
                  prefix: prefixFail,
                  msg: "multiple apply uses of begin_borrow result found, but exactly 1 expected")
                return false
              }
              aiUse = bbiUse
            default:
              logADCS(
                prefix: prefixFail,
                msg: "unexpected use of begin_borrow result found: \(bbiUse)")
              return false
          }
        }
        assert(ebUse != nil)
        assert(aiUse != nil)

      case _ as SwitchEnumInst:
        ()
      case _ as TupleExtractInst:
        ()

      default:
        logADCS(
          prefix: prefixFail,
          msg: "unexpected use of an element of the tuple being argument of pullback "
            + pb.name.string + " basic block " + pbBB.shortDescription)
        logADCS(msg: "  result: \(result)")
        logADCS(msg: "  use.instruction: \(use.instruction)")
        return false
      }
    }
  }
  return true
}

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
    if arg.type.isBranchTracingEnum(vjp: vjp) {
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
  guard let bteArgOfPb = getEnumArgOfEntryPbBB(pb.entryBlock, vjp: vjp) else {
    logADCS(
      prefix: prefixFail,
      msg: "cannot get branch tracing enum argument of the pullback function " + pb.name.string)
    return false
  }

  if pb.blocks.singleElement != nil {
    guard let riTerm = pb.entryBlock.terminator as? ReturnInst else {
      logADCS(
        prefix: prefixFail,
        msg: "unexpected terminator instruction in the entry block of the pullback " + pb.name.string
          + " (expected return inst for single-bb pullback)")
      logADCS(msg: "  terminator: " + pb.entryBlock.terminator.description)
      logADCS(msg: "  parent block begin")
      logADCS(msg: "  " + pb.entryBlock.description)
      logADCS(msg: "  parent block end")
      return false
    }
    logADCS(msg: "Pullback: single-bb; \(bteArgOfPb.uses.count) uses of branch tracing enum pullback argument found.")
    if bteArgOfPb.uses.count != 0 {
      logADCS(
        prefix: prefixFail,
        msg: "single-bb pullback has uses of BTE arg")
      var needBreak = true
      if bteArgOfPb.uses.count == 1 {
        let useInst = bteArgOfPb.uses.singleUse!.instruction
        let aiOpt = useInst as? ApplyInst
        let dviOpt = useInst as? DestroyValueInst
        if aiOpt != nil {
          logADCS(msg: "Single use of BTE arg is apply inst")
        }
        if dviOpt != nil {
          logADCS(msg: "Single use of BTE arg is destroy_value inst")
          needBreak = false
        }
      }
      dumpVJPAndPB(vjp: vjp, pb: pb)
      if needBreak {
        return false
      }
    }
  } else {
    if bteArgOfPb.uses.count == 0 {
      logADCS(prefix: prefixFail, msg: "no uses of pullback bte arg found")
      return false
    }
    if bteArgOfPb.uses.count != 1 {
      logADCS(prefix: prefixFail, msg: "multiple uses of pullback bte arg found")
      for (idx, use) in bteArgOfPb.uses.enumerated() {
        logADCS(msg: "use \(idx): \(use)")
      }
      return false
    }

    guard bteArgOfPb.uses.singleUse!.instruction as? SwitchEnumInst != nil else {
      logADCS(
        prefix: prefixFail,
        msg: "unexpected use of BTE argument of pullback " + pb.name.string
          + " (only switch_enum_inst is supported)")
      logADCS(msg: "  use: \(bteArgOfPb.uses.singleUse!.instruction)")
      logADCS(msg: "  parent block begin")
      logADCS(msg: "  \(bteArgOfPb.uses.singleUse!.instruction.parentBlock)")
      logADCS(msg: "  parent block end")
      return false
    }
  }

  for pbBB in pb.blocks {
    guard let sei = pbBB.terminator as? SwitchEnumInst else {
      continue
    }
    if sei.bridged.SwitchEnumInst_getSuccessorForDefault().block != nil {
      logADCS(
        prefix: prefixFail,
        msg: "switch_enum_inst from the \(pbBB.shortDescription) of the pullback " + pb.name.string
          + " has default destination set, which is not supported")
      return false
    }
  }

  for vjpBB in vjp.blocks {
    if getEnumArgOfVJPBB(vjpBB) != nil {
      break
    }
    for arg in vjpBB.arguments {
      if arg.type.isBranchTracingEnum(vjp: vjp) {
        logADCS(
          prefix: prefixFail,
          msg: "several arguments of VJP " + vjp.name.string + " basic block "
            + vjpBB.shortDescription + " are branch tracing enums, but not more than 1 is supported"
        )
        return false
      }
    }
  }

  for pbBB in pb.blocks {
    guard let argOfPbBB = getEnumPayloadArgOfPbBB(pbBB, vjp: vjp) else {
      continue
    }
    let payloadValues = getPayloadValues(payload: argOfPbBB, vjp: vjp)
    switch payloadValues {
    case .ZeroUses:
      ()
    case .Unsupported:
      return false
    case .DestructureTuple(let results):
      if !checkIfCanRunForPayloadValues(results: results, prefixFail: prefixFail, pb: pb, pbBB: pbBB) {
        return false
      }
    case .TupleExtract(let results):
      if !checkIfCanRunForPayloadValues(results: results, prefixFail: prefixFail, pb: pb, pbBB: pbBB) {
        return false
      }
    }
  }

  var foundBranchTracingEnumParam = false
  for paramInfo in pb.convention.parameters {
    if paramInfo.type.rawType.bridged.type == bteArgOfPb.type.canonicalType.rawType.bridged.type {
      foundBranchTracingEnumParam = true
      break
    }
  }

  if !foundBranchTracingEnumParam {
    logADCS(
      prefix: prefixFail,
      msg: "cannot find pullback param matching branch tracing enum type \(bteArgOfPb.type)")
    return false
  }

  return true
}

private func multiBBHelper(
  pullbackClosureInfo: PullbackClosureInfo, function: Function, enumDict: inout EnumDict, context: FunctionPassContext
) {
  var closuresSet = Set<SingleValueInstruction>()
  for closureInfo in pullbackClosureInfo.closureInfosCFG {
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
      basedOn: pullbackClosureInfo, enumDict: &enumDict, context)

  if !alreadyExists {
    context.notifyNewFunction(function: specializedFunction, derivedFrom: pullbackClosureInfo.pullbackFn)
  }

  rewriteApplyInstructionCFG(
    using: specializedFunction, pullbackClosureInfo: pullbackClosureInfo,
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
      logADCS(msg: "Dumping VJP and PB before pass run begin")
      dumpVJPAndPB(
        vjp: function,
        pb: getPartialApplyOfPullbackInExitVJPBB(vjp: function)!.referencedFunction!)
      logADCS(msg: "Dumping VJP and PB before pass run end")
    }
  }

  var remainingSpecializationRounds = 5

  repeat {
    guard let pullbackClosureInfo = getPullbackClosureInfo(in: function, context) else {
      break
    }

    var (specializedFunction, alreadyExists) = getOrCreateSpecializedFunction(
      basedOn: pullbackClosureInfo, context)

    if !alreadyExists {
      context.notifyNewFunction(function: specializedFunction, derivedFrom: pullbackClosureInfo.pullbackFn)
    }

    rewriteApplyInstruction(using: specializedFunction, pullbackClosureInfo: pullbackClosureInfo, context)

    var deadClosures = InstructionWorklist(context)
    pullbackClosureInfo.closureArgDescriptors
      .map { $0.closure }
      .forEach { deadClosures.pushIfNotVisited($0) }

    defer {
      deadClosures.deinitialize()
    }

    while let deadClosure = deadClosures.pop() {
      let isDeleted = context.tryDeleteDeadClosure(closure: deadClosure as! SingleValueInstruction)
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
    var adcsHelper = BridgedAutoDiffClosureSpecializationHelper()
    var enumDict = EnumDict()
    defer {
      adcsHelper.clearEnumDict()
    }

    remainingSpecializationRounds = 5
    repeat {
      logADCS(msg: "Remaining specialization rounds: " + String(remainingSpecializationRounds))
      // TODO: Names here are pretty misleading. We are looking for a place where
      // the pullback closure is created (so for `partial_apply` instruction).
      guard let pullbackClosureInfo = getPullbackClosureInfoCFG(in: function, context) else {
        // TODO: it looks like that we do not have more than 1 round, at least for multi BB case
        logADCS(
          msg:
            "Unable to detect closures to be specialized in " + function.name.string
            + ", skipping the pass")
        break
      }

      multiBBHelper(pullbackClosureInfo: pullbackClosureInfo, function: function, enumDict: &enumDict, context: context)

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
        logADCS(msg: "getPartialApplyOfPullbackInExitVJPBB: reachable exit block begin")
        logADCS(msg: "\(block)")
        logADCS(msg: "getPartialApplyOfPullbackInExitVJPBB: reachable exit block end")
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
    dumpVJP(vjp: vjp)
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
      let tttfOpt = inst as? ThinToThickFunctionInst
      if tttfOpt != nil {
        logADCS(msg: "  instruction: " + inst.description)
        logADCS(msg: "  parent block begin")
        logADCS(msg: "  " + inst.parentBlock.description)
        logADCS(msg: "  parent block end")
      } else {
        dumpVJP(vjp: vjp)
      }
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
  if ti.operands.count < 2 {
    logADCS(prefix: prefix, msg: "5")
    logADCS(msg: "  ti: " + ti.description)
    logADCS(msg: "  parent block begin")
    logADCS(msg: "  " + ti.parentBlock.description)
    logADCS(msg: "  parent block end")
    return nil
  }
  if ti.operands[ti.operands.count - 1].value.definingInstruction == nil {
    logADCS(prefix: prefix, msg: "6")
    logADCS(msg: "  ti: " + ti.description)
    logADCS(msg: "  value: " + ti.operands[ti.operands.count - 1].value.description)
    logADCS(msg: "  parent block begin")
    logADCS(msg: "  " + ti.parentBlock.description)
    logADCS(msg: "  parent block end")
    return nil
  }
  return handleConvertFunctionOrPartialApply(inst: ti.operands[ti.operands.count - 1].value.definingInstruction!)
}


private func getPullbackClosureInfoCFG(in caller: Function, _ context: FunctionPassContext) -> PullbackClosureInfo? {
  var pullbackClosureInfoOpt = PullbackClosureInfo?(nil)
  var supportedClosuresCount = 0
  var subsetThunkArr = [SingleValueInstruction]()

  for inst in caller.instructions {
    if let rootClosure = inst.asSupportedClosure {
      logADCS(msg: "AAAAAA 00 \(rootClosure)")
      supportedClosuresCount += 1

      if rootClosure == getPartialApplyOfPullbackInExitVJPBB(vjp: rootClosure.parentFunction)! {
        continue
      }
      logADCS(msg: "AAAAAA 01 \(rootClosure)")
      if subsetThunkArr.contains(rootClosure) {
        continue
      }
      logADCS(msg: "AAAAAA 02 \(rootClosure)")
      let closureInfoArr = handleNonAppliesCFG(for: rootClosure, context)
      logADCS(msg: "closureInfoArr.count = " + String(closureInfoArr.count))
      if closureInfoArr.count == 0 {
        continue
      }
      logADCS(msg: "AAAAAA 03 \(rootClosure)")
      if pullbackClosureInfoOpt == nil {
        pullbackClosureInfoOpt = PullbackClosureInfo(paiOfPullback: getPartialApplyOfPullbackInExitVJPBB(vjp: caller)!)
      logADCS(msg: "AAAAAA 04 \(rootClosure)")
      }
      for closureInfo in closureInfoArr {
        pullbackClosureInfoOpt!.closureInfosCFG.append(closureInfo)
        if closureInfo.subsetThunk != nil {
          subsetThunkArr.append(closureInfo.subsetThunk!)
        }
      }

    }
  }

  if supportedClosuresCount == 0 {
    logADCS(msg: "No supported closures found in " + caller.name.string)
  }

  return pullbackClosureInfoOpt
}

private func getPullbackClosureInfo(in caller: Function, _ context: FunctionPassContext) -> PullbackClosureInfo? {
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

  var pullbackClosureInfoOpt = PullbackClosureInfo?(nil)
  let isSingleBB = caller.blocks.singleElement != nil
  var supportedClosuresCount = 0
  var subsetThunkArr = [SingleValueInstruction]()

  for inst in caller.instructions {
    if !convertedAndReabstractedClosures.contains(inst),
      let rootClosure = inst.asSupportedClosure
    {
      updatePullbackClosureInfo(
        for: rootClosure, in: &pullbackClosureInfoOpt,
        convertedAndReabstractedClosures: &convertedAndReabstractedClosures, context)
    }
  }

  return pullbackClosureInfoOpt
}

private func getOrCreateSpecializedFunction(
  basedOn pullbackClosureInfo: PullbackClosureInfo, _ context: FunctionPassContext
)
  -> (function: Function, alreadyExists: Bool)
{
  let specializedFunctionName = pullbackClosureInfo.specializedCalleeName(context)
  if let specializedFunction = context.lookupFunction(name: specializedFunctionName) {
    return (specializedFunction, true)
  }

  let pullbackFn = pullbackClosureInfo.pullbackFn
  let specializedParameters = pullbackFn.convention.getSpecializedParameters(basedOn: pullbackClosureInfo)

  let specializedFunction = 
    context.createSpecializedFunctionDeclaration(from: pullbackFn, withName: specializedFunctionName,
                                                 withParams: specializedParameters,
                                                 makeThin: true, makeBare: true)

  context.buildSpecializedFunction(
    specializedFunction: specializedFunction,
    buildFn: { (emptySpecializedFunction, functionPassContext) in 
      let closureSpecCloner = SpecializationCloner(
        emptySpecializedFunction: emptySpecializedFunction, functionPassContext)
      closureSpecCloner.cloneAndSpecializeFunctionBody(using: pullbackClosureInfo)
    })

  return (specializedFunction, false)
}

private func getOrCreateSpecializedFunctionCFG(
  basedOn pullbackClosureInfo: PullbackClosureInfo, enumDict: inout EnumDict,
  _ context: FunctionPassContext
)
  -> (function: Function, alreadyExists: Bool)
{
  assert(pullbackClosureInfo.closureArgDescriptors.count == 0)
  let pb = pullbackClosureInfo.pullbackFn
  let vjp = pullbackClosureInfo.paiOfPullback.parentFunction

  let specializedPbName = pullbackClosureInfo.specializedCalleeNameCFG(vjp: vjp, context)
  if let specializedPb = context.lookupFunction(name: specializedPbName) {
    return (specializedPb, true)
  }

  let closureInfos = pullbackClosureInfo.closureInfosCFG
  let enumTypeOfEntryBBArg = getEnumArgOfEntryPbBB(pb.entryBlock, vjp: vjp)!.type

  var adcsHelper = BridgedAutoDiffClosureSpecializationHelper()
  defer {
    adcsHelper.clearClosuresBuffer()
  }
  for closureInfoCFG in closureInfos {
    adcsHelper.appendToClosuresBuffer(
      closureInfoCFG.enumTypeAndCase.enumType.bridged,
      closureInfoCFG.enumTypeAndCase.caseIdx,
      closureInfoCFG.closure.bridged,
      closureInfoCFG.idxInEnumPayload)
  }
  enumDict =
      adcsHelper.rewriteAllEnums(
        /*topVjp: */pullbackClosureInfo.paiOfPullback.parentFunction.bridged,
        /*topEnum: */enumTypeOfEntryBBArg.bridged
      )

  let specializedParameters = getSpecializedParametersCFG(
    basedOn: pullbackClosureInfo, pb: pb, enumType: enumTypeOfEntryBBArg, enumDict: enumDict, context)

  let specializedPb =
    context.createSpecializedFunctionDeclaration(from: pb, withName: specializedPbName,
                                                 withParams: specializedParameters,
                                                 makeThin: true, makeBare: true)

  context.buildSpecializedFunction(
    specializedFunction: specializedPb,
    buildFn: { (emptySpecializedFunction, functionPassContext) in
      let closureSpecCloner = SpecializationCloner(
        emptySpecializedFunction: emptySpecializedFunction, functionPassContext)
      closureSpecCloner.cloneAndSpecializeFunctionBodyCFG(
        using: pullbackClosureInfo, enumDict: enumDict)
    })

  return (specializedPb, false)
}

private func findEnumsAndPayloadsInVjp(vjp: Function) -> [EnumInst:TupleInst] {
  var dict = [EnumInst:TupleInst]()
  for inst in vjp.instructions {
    guard let ei = inst as? EnumInst else {
      continue
    }
    if !ei.type.isBranchTracingEnum(vjp: vjp) {
      continue
    }
    let ti = ei.operands[0].value.definingInstruction as! TupleInst
    dict[ei] = ti
  }
  return dict
}

private func rewriteApplyInstructionCFG(
  using specializedCallee: Function, pullbackClosureInfo: PullbackClosureInfo,
  enumDict: EnumDict,
  context: FunctionPassContext
) {
  let vjp = pullbackClosureInfo.paiOfPullback.parentFunction
  let closureInfos = pullbackClosureInfo.closureInfosCFG

  for inst in vjp.instructions {
    guard let ei = inst as? EnumInst else {
      continue
    }
    guard let newEnumType = SILBridging_enumDictGetByKey(enumDict, ei.results[0].type.bridged).typeOrNil else {
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
    if SILBridging_enumDictGetByKey(enumDict, arg.type.bridged).typeOrNil == nil {
      continue
    }
    bb.bridged.recreateEnumBlockArgument(arg.bridged)
  }
  let pai = pullbackClosureInfo.paiOfPullback as! PartialApplyInst

  let builderSucc = Builder(
    before: pai,
    location: pullbackClosureInfo.paiOfPullback.parentBlock.instructions.last!.location, context)

  let paiConvention = pai.calleeConvention
  let paiHasUnknownResultIsolation = pai.hasUnknownResultIsolation
  let paiSubstitutionMap = SubstitutionMap(bridged: pai.bridged.getSubstitutionMap())
  let paiIsOnStack = pai.isOnStack

  // MYTODO assert that PAI is on index 1 in tuple

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

  let enumToPayload = findEnumsAndPayloadsInVjp(vjp: vjp)
  let payloads = Set<TupleInst>(enumToPayload.values)

  for payload in payloads {
    let ti = payload
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
  using specializedCallee: Function, pullbackClosureInfo: PullbackClosureInfo,
  _ context: FunctionPassContext
) {
  let newApplyArgs = pullbackClosureInfo.getArgumentsForSpecializedApply(of: specializedCallee)

  for newApplyArg in newApplyArgs {
    if case let .PreviouslyCaptured(capturedArg, needsRetain, parentClosureArgIndex) = newApplyArg,
      needsRetain
    {
      let closureArgDesc = pullbackClosureInfo.closureArgDesc(at: parentClosureArgIndex)!
      var builder = Builder(before: closureArgDesc.closure, context)

      // TODO: Support only OSSA instructions once the OSSA elimination pass is moved after all function optimization
      // passes.
      if pullbackClosureInfo.paiOfPullback.parentBlock != closureArgDesc.closure.parentBlock {
        // Emit the retain and release that keeps the argument live across the callee using the closure.
        builder.createRetainValue(operand: capturedArg)

        for instr in closureArgDesc.lifetimeFrontier {
          builder = Builder(before: instr, context)
          builder.createReleaseValue(operand: capturedArg)
        }

        // Emit the retain that matches the captured argument by the partial_apply in the callee that is consumed by
        // the partial_apply.
        builder = Builder(before: pullbackClosureInfo.paiOfPullback, context)
        builder.createRetainValue(operand: capturedArg)
      } else {
        builder.createRetainValue(operand: capturedArg)
      }
    }
  }

  // Rewrite apply instruction
  var builder = Builder(before: pullbackClosureInfo.paiOfPullback, context)
  let oldPartialApply = pullbackClosureInfo.paiOfPullback
  let funcRef = builder.createFunctionRef(specializedCallee)
  let capturedArgs = Array(newApplyArgs.map { $0.value })

  let newPartialApply = builder.createPartialApply(
    function: funcRef, substitutionMap: SubstitutionMap(),
    capturedArguments: capturedArgs, calleeConvention: oldPartialApply.calleeConvention,
    hasUnknownResultIsolation: oldPartialApply.hasUnknownResultIsolation,
    isOnStack: oldPartialApply.isOnStack)

  builder = Builder(before: pullbackClosureInfo.paiOfPullback.next!, context)
  // TODO: Support only OSSA instructions once the OSSA elimination pass is moved after all function optimization 
  // passes.
  for closureArgDesc in pullbackClosureInfo.closureArgDescriptors {
    if closureArgDesc.isClosureConsumed,
      !closureArgDesc.isPartialApplyOnStack,
      !closureArgDesc.parameterInfo.isTrivialNoescapeClosure
    {
      builder.createReleaseValue(operand: closureArgDesc.closure)
    }
  }

  oldPartialApply.replace(with: newPartialApply, context)
}

// ===================== Utility functions and extensions ===================== //

private func updatePullbackClosureInfo(
  for rootClosure: SingleValueInstruction,
  in pullbackClosureInfoOpt: inout PullbackClosureInfo?,
  convertedAndReabstractedClosures: inout InstructionSet,
  _ context: FunctionPassContext
) {
  // A "root" closure undergoing conversions and/or reabstractions has additional restrictions placed upon it, in order
  // for a pullback to be specialized against it. We handle conversion/reabstraction uses before we handle apply uses
  // to gather the parameters required to evaluate these restrictions or to skip pullback's uses of "unsupported"
  // closures altogether.
  //
  // There are currently 2 restrictions that are evaluated prior to specializing a pullback against a converted and/or
  // reabstracted closure -
  // 1. A reabstracted root closure can only be specialized against, if the reabstracted closure is ultimately passed
  //    trivially (as a noescape+thick function) as captured argument of pullback's partial_apply.
  //
  // 2. A root closure may be a partial_apply [stack], in which case we need to make sure that all mark_dependence
  //    bases for it will be available in the specialized callee in case the pullback is specialized against this root
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
      for: rootClosure, pullbackClosureInfoOpt: &pullbackClosureInfoOpt, rootClosureApplies: &rootClosureApplies,
      rootClosurePossibleLiveRange: &rootClosurePossibleLiveRange,
      convertedAndReabstractedClosures: &convertedAndReabstractedClosures,
      haveUsedReabstraction: haveUsedReabstraction, context)

  if pullbackClosureInfoOpt == nil {
    return
  }

  finalizePullbackClosureInfo(
    for: rootClosure, in: &pullbackClosureInfoOpt,
    rootClosurePossibleLiveRange: rootClosurePossibleLiveRange,
    intermediateClosureArgDescriptorData: intermediateClosureArgDescriptorData, context)
}

private func getEnumArgOfEntryPbBB(_ bb: BasicBlock, vjp: Function) -> Argument? {
  assert(bb.parentFunction.entryBlock == bb)
  var argOpt = Argument?(nil)
  for arg in bb.arguments {
    if arg.type.isBranchTracingEnum(vjp: vjp) {
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
    if arg.type.isBranchTracingEnum(vjp: bb.parentFunction) {
      if argOpt != nil {
        return nil
      }
      argOpt = arg
    }
  }
  return argOpt
}

private func getEnumPayloadArgOfPbBB(_ bb: BasicBlock, vjp: Function) -> Argument? {
  var argOpt = Argument?(nil)
  for (argIdx, arg) in bb.arguments.enumerated() {
    if !arg.type.isTuple {
      continue
    }

    let predBBOpt = bb.predecessors.first
    if predBBOpt == nil {
      continue
    }
    let predBB = predBBOpt!

    let brInstOpt = predBB.terminator as? BranchInst
    let seInstOpt = predBB.terminator as? SwitchEnumInst
    if brInstOpt == nil && seInstOpt == nil {
      continue
    }
    if brInstOpt != nil {
      let brInst = brInstOpt!
      let possibleUEDI = brInst.operands[argIdx].value.definingInstruction
      let uedi = possibleUEDI as? UncheckedEnumDataInst
      if uedi == nil {
        continue
      }
      let enumType = uedi!.`enum`.type
      if !enumType.isBranchTracingEnum(vjp: vjp) {
        continue
      }
    } else {
      assert(seInstOpt != nil)
      let enumType = seInstOpt!.enumOp.type
      if !enumType.isBranchTracingEnum(vjp: vjp) {
        continue
      }
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

private func handleNonAppliesCFG(
  for rootClosure: SingleValueInstruction,
  _ context: FunctionPassContext
)
  -> [ClosureInfoCFG]
{
  let enumToPayload : [EnumInst:TupleInst] = findEnumsAndPayloadsInVjp(vjp: rootClosure.parentFunction)
  var closureInfoArr = [ClosureInfoCFG]()

  var closure = rootClosure
  var subsetThunkOpt = PartialApplyInst?(nil)
  if rootClosure.uses.singleElement != nil {
    let tiOpt = rootClosure.uses.singleElement!.instruction as? TupleInst
    if tiOpt == nil || !enumToPayload.values.contains(tiOpt!) {
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
      if !ei.type.isBranchTracingEnum(vjp: rootClosure.parentFunction) {
        logADCS(msg: "handleNonAppliesCFG: unexpected enum type:" + ei.type.description)
        return []
      }
      assert(enumToPayload[ei] == ti)
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
  /// the specialized callee, in case the pullback is specialized against this root closure.
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
  /// `mark_dependence` of the root closure on such a value means that we cannot specialize the pullback against it.
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
  for rootClosure: SingleValueInstruction, pullbackClosureInfoOpt: inout PullbackClosureInfo?,
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

    if pullbackClosureInfoOpt == nil {
      pullbackClosureInfoOpt = PullbackClosureInfo(paiOfPullback: pai)
    } else {
      assert(pullbackClosureInfoOpt!.paiOfPullback == pai)
    }

    intermediateClosureArgDescriptorData
      .append((applySite: pai, closureArgIndex: closureArgumentIndex, paramInfo: closureParamInfo))
  }

  return intermediateClosureArgDescriptorData
}

/// Finalizes the pullback closure info for a given root closure by adding a corresponding `ClosureArgDescriptor`
private func finalizePullbackClosureInfo(
  for rootClosure: SingleValueInstruction, in pullbackClosureInfoOpt: inout PullbackClosureInfo?,
  rootClosurePossibleLiveRange: InstructionRange,
  intermediateClosureArgDescriptorData: [IntermediateClosureArgDescriptorDatum],
  _ context: FunctionPassContext
) {
  assert(pullbackClosureInfoOpt != nil)

  let closureInfo = ClosureInfo(closure: rootClosure, lifetimeFrontier: Array(rootClosurePossibleLiveRange.ends))

  for (applySite, closureArgumentIndex, parameterInfo) in intermediateClosureArgDescriptorData {
    if pullbackClosureInfoOpt!.paiOfPullback != applySite {
      fatalError(
        "ClosureArgDescriptor's applySite field is not equal to pullback's partial_apply; got \(applySite)!")
    }
    let closureArgDesc = ClosureArgDescriptor(
      closureInfo: closureInfo, closureArgumentIndex: closureArgumentIndex, 
      parameterInfo: parameterInfo)
    pullbackClosureInfoOpt!.appendClosureArgDescriptor(closureArgDesc)
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
      log("Parent function of pullbackClosureInfo: \(rootClosure.parentFunction)")
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
  result: Value, useTei: Bool, context: FunctionPassContext
) {
  switch use.instruction {
  case let cfi as ConvertFunctionInst:
    let builder = Builder(before: cfi, context)
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
      assert(cfi.uses.count == 2)
      var bbiUse = Operand?(nil)
      var dviUse = Operand?(nil)
      var sriUse = Operand?(nil)
      for cfiUse in cfi.uses {
        switch cfiUse.instruction {
          case let bbi as BeginBorrowInst:
            assert(bbiUse == nil)
            bbiUse = cfiUse
          case let dvi as DestroyValueInst:
            assert(dviUse == nil)
            dviUse = cfiUse
          case let sri as DestroyValueInst:
            assert(sriUse == nil)
            sriUse = cfiUse
          default:
            assert(false)
        }
      }
      assert((dviUse != nil) != (sriUse != nil))
      assert(bbiUse != nil)
      if dviUse != nil {
        dviUse!.instruction.parentBlock.eraseInstruction(dviUse!.instruction)
      } else {
        sriUse!.instruction.parentBlock.eraseInstruction(sriUse!.instruction)
      }
      rewriteUsesOfPayloadItem(use: bbiUse!, resultIdx: resultIdx, closureInfoArray: closureInfoArray, result: result, useTei: useTei, context: context)
      cfi.parentBlock.eraseInstruction(cfi)
    } else {
      let newCFI = builder.createConvertFunction(
        originalFunction: result,
        resultType: cfi.type,
        withoutActuallyEscaping: cfi.withoutActuallyEscaping)
      cfi.replace(with: newCFI, context)
    }

  case let bbi as BeginBorrowInst:
    let builder = Builder(before: bbi, context)
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
      assert(bbi.uses.count == 2)
      var aiUse = Operand?(nil)
      var ebUse = Operand?(nil)
      for bbiUse in bbi.uses {
        switch bbiUse.instruction {
          case let eb as EndBorrowInst:
            assert(ebUse == nil)
            ebUse = bbiUse
          case let ai as ApplyInst:
            assert(aiUse == nil)
            aiUse = bbiUse
          default:
            assert(false)
        }
      }
      assert(ebUse != nil)
      assert(aiUse != nil)
      ebUse!.instruction.parentBlock.eraseInstruction(ebUse!.instruction)
      rewriteUsesOfPayloadItem(use: aiUse!, resultIdx: resultIdx, closureInfoArray: closureInfoArray, result: result, useTei: useTei, context: context)
      bbi.parentBlock.eraseInstruction(bbi)
    } else {
      let newBBI = builder.createBeginBorrow(
        of: result,
        isLexical: bbi.isLexical,
        hasPointerEscape: bbi.hasPointerEscape,
        isFromVarDecl: bbi.isFromVarDecl)
      bbi.replace(with: newBBI, context)
    }

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
      var teiArray = [TupleExtractInst]()
      var dtiOfCapturedArgsTuple = DestructureTupleInst?(nil)
      if useTei {
        for (tupleIdx, tupleElem) in result.type.tupleElements.enumerated() {
          teiArray.append(builder.createTupleExtract(tuple: result, elementIndex: tupleIdx))
        }
      } else {
        dtiOfCapturedArgsTuple = builder.createDestructureTuple(tuple: result)
      }
      if closureInfoOpt!.subsetThunk == nil {
        var newArgs = [Value]()
        for op in ai.argumentOperands {
          newArgs.append(op.value)
        }
        if useTei {
          for res in teiArray {
            newArgs.append(res)
          }
        } else {
          for res in dtiOfCapturedArgsTuple!.results {
            newArgs.append(res)
          }
        }
        let vjpFn = closureInfoOpt!.closure.asSupportedClosureFn!
        let newFri = builder.createFunctionRef(vjpFn)
        let newAi = builder.createApply(
          function: newFri, ai.substitutionMap, arguments: newArgs)
        ai.replace(with: newAi, context)

        // MYTODO: maybe we can set insertion point earlier
        let newBuilder = Builder(before: newAi.parentBlock.terminator, context)
        var resArray = [Value]()
        if useTei {
          resArray = teiArray
        } else {
          for dtiRes in dtiOfCapturedArgsTuple!.results {
            resArray.append(dtiRes)
          }
        }
        for res in resArray {
          if res.type.isTrivial(in: res.parentFunction) {
            continue
          }
          var needDestroy = true
          for resUse in res.uses {
            if resUse.endsLifetime {
              needDestroy = false
              break
            }
          }
          if needDestroy {
            if ai.parentFunction.hasOwnership {
              newBuilder.createDestroyValue(operand: res)
            } else {
              newBuilder.createReleaseValue(operand: res)
            }
          }
        }
      } else {
        var newClosure = SingleValueInstruction?(nil)
        let maybePai = closureInfoOpt!.closure as? PartialApplyInst
        if maybePai != nil {
          var newArgs = [Value]()
          var resArray = [Value]()
          if useTei {
            resArray = teiArray
          } else {
            for dtiRes in dtiOfCapturedArgsTuple!.results {
              resArray.append(dtiRes)
            }
          }
          for res in resArray {
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
          for res in resArray {
            if res.type.isTrivial(in: res.parentFunction) {
              continue
            }
            var needDestroy = true
            for resUse in res.uses {
              if resUse.endsLifetime {
                needDestroy = false
                break
              }
            }
            if needDestroy {
              if ai.parentFunction.hasOwnership {
                newBuilder.createDestroyValue(operand: res)
              } else {
                newBuilder.createReleaseValue(operand: res)
              }
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
        assert(newClosure!.uses.singleUse != nil)
        if !newClosure!.type.isTrivial(in: newAi.parentFunction) && !newClosure!.uses.singleUse!.endsLifetime {
          if ai.parentFunction.hasOwnership {
            newBuilder.createDestroyValue(operand: newClosure!)
          } else {
            newBuilder.createReleaseValue(operand: newClosure!)
          }
        }
      }
    } else {
      var newArgs = [Value]()
      for op in ai.argumentOperands {
        newArgs.append(op.value)
      }
      let newAi = builder.createApply(
        function: result, ai.substitutionMap, arguments: newArgs)
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
      if dvi.parentFunction.hasOwnership {
        builder.createDestroyValue(operand: result)
      } else {
        builder.createReleaseValue(operand: result)
      }
    }
    dvi.parentBlock.eraseInstruction(dvi)

  case let sri as StrongReleaseInst:
    var needDestroyValue = true
    for closureInfo in closureInfoArray {
      if closureInfo.idxInEnumPayload == resultIdx {
        needDestroyValue = false
      }
    }
    if needDestroyValue {
      let builder = Builder(before: sri, context)
      builder.createStrongRelease(operand: result)
    }
    sri.parentBlock.eraseInstruction(sri)

  case let tei as TupleExtractInst:
    let builder = Builder(before: tei, context)
    builder.createTupleExtract(tuple: tei.tuple, elementIndex: tei.fieldIndex)
    tei.parentBlock.eraseInstruction(tei)

  case let uedi as UncheckedEnumDataInst:
    let builder = Builder(before: uedi, context)
    let newUedi = builder.createUncheckedEnumData(
      enum: result, caseIndex: uedi.caseIndex,
      resultType: result.type.bridged.getEnumCasePayload(
        uedi.caseIndex, uedi.parentFunction.bridged
      ).type)
    uedi.replace(with: newUedi, context)

  case let sei as SwitchEnumInst:
    let builder = Builder(before: sei, context)
    let newSEI = builder.createSwitchEnum(
      enum: result, cases: getEnumCasesForSwitchEnumInst(sei))
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

extension BasicBlock {
  func eraseInstruction(_ inst: Instruction) {
    self.bridged.eraseInstruction(inst.bridged)
  }
}

extension SpecializationCloner {
  fileprivate func cloneAndSpecializeFunctionBodyCFG(
    using pullbackClosureInfo: PullbackClosureInfo, enumDict: EnumDict
  ) {
    let closureInfos = pullbackClosureInfo.closureInfosCFG
    self.cloneEntryBlockArgsWithoutOrigClosuresCFG(
      usingOrigCalleeAt: pullbackClosureInfo, enumDict: enumDict)

    var args = [Value]()
    for arg in self.entryBlock.arguments {
      args.append(arg)
    }

    self.cloneFunctionBody(from: pullbackClosureInfo.pullbackFn, entryBlockArguments: args)

    var bbVisited = [BasicBlock: Bool]()
    bbVisited[self.cloned.entryBlock] = true
    var bbQueue = [BasicBlock]()
    bbQueue.append(self.cloned.entryBlock)
    while bbVisited.count != self.cloned.blocks.count {//bbWorklist.count != 0 {
      for bb in self.cloned.blocks {
        if bbVisited[bb] == true {
          continue
        }
        var allPredsVisited = true
        for predBB in bb.predecessors {
          if bbVisited[predBB] != true {
            allPredsVisited = false
            break
          }
        }
        if allPredsVisited {
          bbQueue.append(bb)
          bbVisited[bb] = true
        }
      }
    }

    logADCS(msg: "bbQueue.count = \(bbQueue.count) BEGIN")
    for (idx, bb) in bbQueue.enumerated() {
      logADCS(msg: "\(idx): \(bb.shortDescription)")
    }
    logADCS(msg: "bbQueue.count = \(bbQueue.count) END")

    for bb in bbQueue {
      // With single-bb, we've ensured that there are no uses of BTE arg, so no manipulation required
      if bb == self.cloned.entryBlock && self.cloned.blocks.singleElement == nil {
        let bteArg = getEnumArgOfEntryPbBB(bb, vjp: pullbackClosureInfo.paiOfPullback.parentFunction)!
        let sei = bteArg.uses.singleUse!.instruction as! SwitchEnumInst
        let parentBB = sei.parentBlock
        let builderEntry = Builder(before: sei, self.context)

        builderEntry.createSwitchEnum(
          enum: sei.enumOp, cases: getEnumCasesForSwitchEnumInst(sei))
        parentBB.eraseInstruction(sei)

        continue
      }

      guard let arg = getEnumPayloadArgOfPbBB(bb, vjp: pullbackClosureInfo.paiOfPullback.parentFunction) else {
        continue
      }

      // MYTODO: can we assume that at least one pred is present?
      let predBBOpt = bb.predecessors.first
      assert(predBBOpt != nil)
      let predBB = predBBOpt!
      var argIdx = -1
      for (i, a) in bb.arguments.enumerated() {
        if a == arg {
          argIdx = i
          break
        }
      }
      assert(argIdx != -1)

      let enumToPayload = findEnumsAndPayloadsInVjp(vjp: pullbackClosureInfo.paiOfPullback.parentFunction)

      let brInstOpt = predBB.terminator as? BranchInst
      var tiInVjp = Optional<TupleInst>(nil)
      if brInstOpt != nil {
        let brInst = brInstOpt!
        let possibleUEDI = brInst.operands[argIdx].value.definingInstruction
        let uedi = possibleUEDI as? UncheckedEnumDataInst
        assert(uedi != nil)
        let enumType = uedi!.`enum`.type
        let caseIdx = uedi!.caseIndex
        for (enumInst, payload) in enumToPayload {
          if SILBridging_enumDictGetByKey(enumDict, enumInst.type.bridged) == enumType.bridged && enumInst.caseIndex == caseIdx {
            tiInVjp = payload
            break
          }
        }
      } else {
        let sei = predBB.terminator as? SwitchEnumInst
        assert(sei != nil)
        let enumType = sei!.enumOp.type
        let caseIdx = sei!.getUniqueCase(forSuccessor: bb)!
        for (enumInst, payload) in enumToPayload {
          if SILBridging_enumDictGetByKey(enumDict, enumInst.type.bridged) == enumType.bridged && enumInst.caseIndex == caseIdx {
            tiInVjp = payload
            break
          }
        }
      }

      var closureInfoArray = [ClosureInfoCFG]()
      var adcsHelper = BridgedAutoDiffClosureSpecializationHelper()
      defer {
        adcsHelper.clearClosuresBufferForPb()
      }
      if tiInVjp != nil {
        for (opIdx, op) in tiInVjp!.operands.enumerated() {
          let val = op.value
          for closureInfo in closureInfos {
            if ((closureInfo.subsetThunk == nil && closureInfo.closure == val)
              || (closureInfo.subsetThunk != nil && closureInfo.subsetThunk! == val))
              && closureInfo.payloadTuple == tiInVjp! // MYTODO: is this correct?
            {
              assert(closureInfo.idxInEnumPayload == opIdx)
              closureInfoArray.append(closureInfo)
              adcsHelper.appendToClosuresBufferForPb(
                closureInfo.closure.bridged,
                closureInfo.idxInEnumPayload)
            }
          }
        }
      }
      logADCS(msg: "recreateTupleBlockArgument: \(bb.shortDescription)")
      let newArg = bb.bridged.recreateTupleBlockArgument(arg.bridged).argument

      if newArg.uses.count == 0 {
        continue
      }

      if newArg.uses.count == 1 && newArg.uses.singleUse!.instruction as? DestructureTupleInst != nil {
        let oldDti = newArg.uses.singleUse!.instruction as! DestructureTupleInst
        let builderBeforeOldDti = Builder(before: oldDti, self.context)
        let newDti = builderBeforeOldDti.createDestructureTuple(tuple: oldDti.tuple)

        for (resultIdx, result) in oldDti.results.enumerated() {
          for use in result.uses {
            rewriteUsesOfPayloadItem(
              use: use, resultIdx: resultIdx, closureInfoArray: closureInfoArray, result: newDti.results[resultIdx],
              useTei: false, context: self.context)
          }
        }

        oldDti.parentBlock.eraseInstruction(oldDti)
        continue
      }

      for newArgUse in newArg.uses {
        let oldTei = newArgUse.instruction as! TupleExtractInst
        let builderBeforeOldTei = Builder(before: oldTei, self.context)
        let newTei = builderBeforeOldTei.createTupleExtract(tuple: oldTei.tuple, elementIndex: oldTei.fieldIndex)

        for use in oldTei.results[0].uses {
          rewriteUsesOfPayloadItem(
            use: use, resultIdx: oldTei.fieldIndex, closureInfoArray: closureInfoArray, result: newTei.results[0],
            useTei: true, context: self.context)
        }

        oldTei.parentBlock.eraseInstruction(oldTei)
      }
    }
  }

  private func cloneEntryBlockArgsWithoutOrigClosuresCFG(
    usingOrigCalleeAt pullbackClosureInfo: PullbackClosureInfo, enumDict: EnumDict
  ) {
    let pb = pullbackClosureInfo.pullbackFn
    let enumType = getEnumArgOfEntryPbBB(pb.entryBlock, vjp: pullbackClosureInfo.paiOfPullback.parentFunction)!.type

    let originalEntryBlock = pullbackClosureInfo.pullbackFn.entryBlock
    let clonedFunction = self.cloned
    let clonedEntryBlock = self.entryBlock

    for arg in originalEntryBlock.arguments {
      var clonedEntryBlockArgType = arg.type.getLoweredType(in: clonedFunction)
      if clonedEntryBlockArgType == enumType {
        // This should always hold since we have at least 1 closure (otherwise, we wouldn't go here).
        // It causes re-write of the corresponding branch tracing enum, and the top enum type will be re-written transitively.
        assert(SILBridging_enumDictGetByKey(enumDict, enumType.bridged).typeOrNil != nil)
        clonedEntryBlockArgType = SILBridging_enumDictGetByKey(enumDict, enumType.bridged).typeOrNil!
      }
      let clonedEntryBlockArg = clonedEntryBlock.addFunctionArgument(
        type: clonedEntryBlockArgType, self.context)
      clonedEntryBlockArg.copyFlags(from: arg as! FunctionArgument, self.context)
    }
  }

  fileprivate func cloneAndSpecializeFunctionBody(using pullbackClosureInfo: PullbackClosureInfo) {
    self.cloneEntryBlockArgsWithoutOrigClosures(usingOrigCalleeAt: pullbackClosureInfo)

    let (allSpecializedEntryBlockArgs, closureArgIndexToAllClonedReleasableClosures) =
      cloneAllClosures(at: pullbackClosureInfo)

    self.cloneFunctionBody(
      from: pullbackClosureInfo.pullbackFn, entryBlockArguments: allSpecializedEntryBlockArgs)

    self.insertCleanupCodeForClonedReleasableClosures(
      from: pullbackClosureInfo,
      closureArgIndexToAllClonedReleasableClosures: closureArgIndexToAllClonedReleasableClosures)
  }

  private func cloneEntryBlockArgsWithoutOrigClosures(usingOrigCalleeAt pullbackClosureInfo: PullbackClosureInfo) {
    let originalEntryBlock = pullbackClosureInfo.pullbackFn.entryBlock
    let clonedFunction = self.cloned
    let clonedEntryBlock = self.entryBlock

    originalEntryBlock.arguments
      .enumerated()
      .filter { index, _ in !pullbackClosureInfo.hasClosureArg(at: index) }
      .forEach { _, arg in
        let clonedEntryBlockArgType = arg.type.getLoweredType(in: clonedFunction)
        let clonedEntryBlockArg = clonedEntryBlock.addFunctionArgument(type: clonedEntryBlockArgType, self.context)
        clonedEntryBlockArg.copyFlags(from: arg as! FunctionArgument, self.context)
      }
  }

  /// Clones all closures, originally passed to the callee at the given pullbackClosureInfo, into the specialized function.
  ///
  /// Returns the following -
  /// - allSpecializedEntryBlockArgs: Complete list of entry block arguments for the specialized function. This includes
  ///   the original arguments to the function (minus the closure arguments) and the arguments representing the values
  ///   originally captured by the skipped closure arguments.
  ///
  /// - closureArgIndexToAllClonedReleasableClosures: Mapping from a closure's argument index at `pullbackClosureInfo` to the list
  ///   of corresponding releasable closures cloned into the specialized function. We have a "list" because we clone
  ///   "closure chains", which consist of a "root" closure and its conversions/reabstractions. This map is used to
  ///   generate cleanup code for the cloned closures in the specialized function.
  private func cloneAllClosures(at pullbackClosureInfo: PullbackClosureInfo)
    -> (
      allSpecializedEntryBlockArgs: [Value],
      closureArgIndexToAllClonedReleasableClosures: [Int: [SingleValueInstruction]]
    )
  {
    func entryBlockArgsWithOrigClosuresSkipped() -> [Value?] {
      var clonedNonClosureEntryBlockArgs = self.entryBlock.arguments.makeIterator()

      return pullbackClosureInfo.pullbackFn
        .entryBlock
        .arguments
        .enumerated()
        .reduce(into: []) { result, origArgTuple in
          let (index, _) = origArgTuple
          if !pullbackClosureInfo.hasClosureArg(at: index) {
            result.append(clonedNonClosureEntryBlockArgs.next())
          } else {
            result.append(Optional.none)
          }
        }
    }

    var entryBlockArgs: [Value?] = entryBlockArgsWithOrigClosuresSkipped()
    var closureArgIndexToAllClonedReleasableClosures: [Int: [SingleValueInstruction]] = [:]

    for closureArgDesc in pullbackClosureInfo.closureArgDescriptors {
      let (finalClonedReabstractedClosure, allClonedReleasableClosures) =
        self.cloneClosureChain(representedBy: closureArgDesc, at: pullbackClosureInfo)

      entryBlockArgs[closureArgDesc.closureArgIndex] = finalClonedReabstractedClosure
      closureArgIndexToAllClonedReleasableClosures[closureArgDesc.closureArgIndex] =
        allClonedReleasableClosures
    }

    return (entryBlockArgs.map { $0! }, closureArgIndexToAllClonedReleasableClosures)
  }

  private func cloneClosureChain(
    representedBy closureArgDesc: ClosureArgDescriptor, at pullbackClosureInfo: PullbackClosureInfo
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
        reabstractedClosure: pullbackClosureInfo.appliedArgForClosure(at: closureArgDesc.closureArgIndex)!,
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
    from pullbackClosureInfo: PullbackClosureInfo,
    closureArgIndexToAllClonedReleasableClosures: [Int: [SingleValueInstruction]]
  ) {
    for closureArgDesc in pullbackClosureInfo.closureArgDescriptors {
      let allClonedReleasableClosures = closureArgIndexToAllClonedReleasableClosures[
        closureArgDesc.closureArgIndex]!

      // Insert a `destroy_value`, for all releasable closures, in all reachable exit BBs if the closure was passed as a
      // guaranteed parameter or its type was noescape+thick. This is b/c the closure was passed at +0 originally and we
      // need to balance the initial increment of the newly created closure(s).
      if closureArgDesc.isClosureGuaranteed
        || closureArgDesc.parameterInfo.isTrivialNoescapeClosure,
        !allClonedReleasableClosures.isEmpty
      {
        for exitBlock in pullbackClosureInfo.reachableExitBBsInCallee {
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

private extension PullbackClosureInfo {
  enum NewApplyArg {
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
    for (applySiteIndex, arg) in self.paiOfPullback.arguments.enumerated() {
      let calleeArgIndex = self.paiOfPullback.unappliedArgumentCount + applySiteIndex
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
          log("Parent function of pullbackClosureInfo: \(rootClosure.parentFunction)")
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
        log("Parent function of pullbackClosureInfo: \(rootClosure.parentFunction)")
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

typealias EnumDict = BranchTracingEnumDict

private func getSpecializedParametersCFG(
  basedOn pullbackClosureInfo: PullbackClosureInfo, pb: Function, enumType: Type, enumDict: EnumDict,
  _ context: FunctionPassContext
) -> [ParameterInfo] {
  let applySiteCallee = pullbackClosureInfo.pullbackFn
  var specializedParamInfoList: [ParameterInfo] = []
  var foundBranchTracingEnumParam = false
  var enumDictCopy = enumDict
  // Start by adding all original parameters except for the closure parameters.
  for paramInfo in applySiteCallee.convention.parameters {
    // TODO: is this safe to perform such check?
    if !SILType_equalEnums(paramInfo.type.bridged, enumType.canonicalType.bridged) {
      specializedParamInfoList.append(paramInfo)
      continue
    }
    assert(!foundBranchTracingEnumParam)
    foundBranchTracingEnumParam = true
    let newParamInfo = ParameterInfo(
      type: SILBridging_enumDictGetByKey(enumDictCopy, enumType.bridged).type.canonicalType,
      //type: SILBridging_enumDictGetByKey(enumDict, enumType.bridged]!.type.canonicalType,
      convention: paramInfo.convention,
      options: paramInfo.options, hasLoweredAddresses: paramInfo.hasLoweredAddresses)
    specializedParamInfoList.append(newParamInfo)
  }
  assert(foundBranchTracingEnumParam)
  return specializedParamInfoList
}

private extension FunctionConvention {
  func getSpecializedParameters(basedOn pullbackClosureInfo: PullbackClosureInfo) -> [ParameterInfo] {
    let pullbackFn = pullbackClosureInfo.pullbackFn
    var specializedParamInfoList: [ParameterInfo] = []

    // Start by adding all original parameters except for the closure parameters.
    let firstParamIndex = pullbackFn.argumentConventions.firstParameterIndex
    for (index, paramInfo) in pullbackFn.convention.parameters.enumerated() {
      let argIndex = index + firstParamIndex
      if !pullbackClosureInfo.hasClosureArg(at: argIndex) {
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
    for closureArgDesc in pullbackClosureInfo.closureArgDescriptors {
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

/// Represents all the information required to represent a closure in isolation, i.e., outside of a pullback's partial_apply context
/// where the closure may be getting captured as an argument.
///
/// Composed with other information inside a `ClosureArgDescriptor` to represent a closure as a captured argument of a pullback's partial_apply.
private struct ClosureInfo {
  let closure: SingleValueInstruction
  let lifetimeFrontier: [Instruction]

  init(closure: SingleValueInstruction, lifetimeFrontier: [Instruction]) {
    self.closure = closure
    self.lifetimeFrontier = lifetimeFrontier
  }

}

/// Represents a closure as a captured argument of a pullback's partial_apply.
private struct ClosureArgDescriptor {
  let closureInfo: ClosureInfo
  /// The index of the closure in the pullback's partial_apply argument list.
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

/// Represents a partial_apply of pullback capturing one or more closure arguments.
private struct PullbackClosureInfo {
  let paiOfPullback: PartialApplyInst
  var closureArgDescriptors: [ClosureArgDescriptor] = []
  var closureInfosCFG: [ClosureInfoCFG] = []

  init(paiOfPullback: PartialApplyInst) {
    self.paiOfPullback = paiOfPullback
  }

  mutating func appendClosureArgDescriptor(_ descriptor: ClosureArgDescriptor) {
    self.closureArgDescriptors.append(descriptor)
  }

  var pullbackFn: Function {
    paiOfPullback.referencedFunction!
  }

  var reachableExitBBsInCallee: [BasicBlock] {
    pullbackFn.blocks.filter { $0.isReachableExitBlock }
  }

  func hasClosureArg(at index: Int) -> Bool {
    closureArgDescriptors.contains { $0.closureArgumentIndex == index }
  }

  func closureArgDesc(at index: Int) -> ClosureArgDescriptor? {
    closureArgDescriptors.first { $0.closureArgumentIndex == index }
  }

  func appliedArgForClosure(at index: Int) -> Value? {
    if let closureArgDesc = closureArgDesc(at: index) {
      return paiOfPullback.arguments[closureArgDesc.closureArgIndex - paiOfPullback.unappliedArgumentCount]
    }

    return nil
  }

  func specializedCalleeName(_ context: FunctionPassContext) -> String {
    let closureArgs = Array(self.closureArgDescriptors.map { $0.closure })
    let closureIndices = Array(self.closureArgDescriptors.map { $0.closureArgIndex })

    return context.mangle(withClosureArguments: closureArgs, closureArgIndices: closureIndices, 
                          from: pullbackFn)
  }

  func specializedCalleeNameCFG(vjp: Function, _ context: FunctionPassContext) -> String {
    let paiOfPbInExitVjpBB = getPartialApplyOfPullbackInExitVJPBB(vjp: vjp)!
    var argAndIdxInPbPAI = (arg: Value, idx: Int)?(nil)
    for (argIdx, arg) in paiOfPbInExitVjpBB.arguments.enumerated() {
      if arg.type.isBranchTracingEnum(vjp: vjp) {
        assert(argAndIdxInPbPAI == nil)
        argAndIdxInPbPAI = (arg: arg, idx: argIdx)
      }
    }
    assert(argAndIdxInPbPAI != nil)

    return context.mangle(
      withBranchTracingEnum: argAndIdxInPbPAI!.arg, argIdx: argAndIdxInPbPAI!.idx,
      from: pullbackFn)
  }
}

// ===================== Unit tests ===================== //

let getPullbackClosureInfoTest = FunctionTest("autodiff_closure_specialize_get_pullback_closure_info") { function, arguments, context in
  print("Specializing closures in function: \(function.name)")
  print("===============================================")
  let pullbackClosureInfo = getPullbackClosureInfo(in: function, context)!
  print("PartialApply of pullback: \(pullbackClosureInfo.paiOfPullback)")
  print("Passed in closures: ")
  for index in pullbackClosureInfo.closureArgDescriptors.indices {
    var closureArgDescriptor = pullbackClosureInfo.closureArgDescriptors[index]
    print("\(index+1). \(closureArgDescriptor.closureInfo.closure)")
  }
  print("\n")
}

let specializedFunctionSignatureAndBodyTest = FunctionTest(
  "autodiff_closure_specialize_specialized_function_signature_and_body") { function, arguments, context in

  let pullbackClosureInfo = getPullbackClosureInfo(in: function, context)!

  let (specializedFunction, _) = getOrCreateSpecializedFunction(basedOn: pullbackClosureInfo, context)
  print("Generated specialized function: \(specializedFunction.name)")
  print("\(specializedFunction)\n")
}

let rewrittenCallerBodyTest = FunctionTest("autodiff_closure_specialize_rewritten_caller_body") { function, arguments, context in
  let pullbackClosureInfo = getPullbackClosureInfo(in: function, context)!

  let (specializedFunction, _) = getOrCreateSpecializedFunction(basedOn: pullbackClosureInfo, context)
  rewriteApplyInstruction(using: specializedFunction, pullbackClosureInfo: pullbackClosureInfo, context)

  print("Rewritten caller body for: \(function.name):")
  print("\(function)\n")
}
