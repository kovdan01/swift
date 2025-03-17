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

private func log(_ message: @autoclosure () -> String) {
  if verbose {
    print("### \(message())")
  }
}

// =========== Entry point =========== //
let generalClosureSpecialization = FunctionPass(name: "experimental-swift-based-closure-specialization") {
  (function: Function, context: FunctionPassContext) in
  // TODO: Implement general closure specialization optimization
  print("NOT IMPLEMENTED")
}

func isFunctionSimpleIfElse(function: Function) -> Bool {
  var blocksCount = 0
  for (idx, block) in function.blocks.enumerated() {
    blocksCount += 1
    if idx == 0 {
      let successors = block.successors
      if successors.count != 2 {
        return false
      }
      if !((successors[0].index == 1 && successors[1].index == 2) ||
           (successors[0].index == 2 && successors[1].index == 1)) {
        return false
      }
      let condBranchInst = block.terminator as? CondBranchInst
      if condBranchInst == nil {
        return false
      }
    } else if idx == 1 || idx == 2 {
      if block.singlePredecessor == nil {
        return false
      }
      if block.singlePredecessor!.index != 0 {
        return false
      }
      if block.arguments.count != 0 {
        return false
      }
    } else if idx == 3 {
      var predecessorIndexes = Array<Int>()
      for predecessor in block.predecessors {
        predecessorIndexes.append(predecessor.index)
      }
      if predecessorIndexes.count != 2 {
        return false
      }
      if !((predecessorIndexes[0] == 1 && predecessorIndexes[1] == 2) ||
           (predecessorIndexes[0] == 2 && predecessorIndexes[1] == 1)) {
        return false
      }
      let returnInst = block.terminator as? ReturnInst
      if returnInst == nil {
        return false
      }
      for arg in block.arguments {
        if arg.type.description.hasSuffix("_specialized") {
          return false
        }
      }
    } else {
      return false
    }
  }
  if blocksCount != 4 {
    return false
  }
  return true
}

struct EnumTypeAndCase {
  var enumType : Type
  var caseIdx : Int
}

// MYTODO: proper hash
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

typealias ClosureInfoCFG = (closure: SingleValueInstruction, idxInEnumPayload: Int, capturedArgs: [Value], enumTypeAndCase: EnumTypeAndCase)
typealias ClosureInfoWithApplyCFG = (closureInfo: ClosureInfoCFG, applyInPb: ApplyInst)

let autodiffClosureSpecialization = FunctionPass(name: "autodiff-closure-specialization") {
  (function: Function, context: FunctionPassContext) in

  guard !function.isDefinedExternally,
        function.isAutodiffVJP else {
    return
  }

  debugPrint("AAAAA AUTODIFF PASS BEGIN")
  debugPrint("AAAAA VJP BEFORE BEGIN")
  debugPrint(function)
  debugPrint("AAAAA VJP BEFORE END")

  let isSingleBB = function.blocks.singleElement != nil
  if !isSingleBB && !isFunctionSimpleIfElse(function: function) {
    return
  }
  
  //var remainingSpecializationRounds = 5
  var remainingSpecializationRounds = 1

  var enumDict : EnumDict = [:]

  repeat {
    let callSiteOpt = gatherCallSite(in: function, context)
    if callSiteOpt == nil {
      break
    }

    let callSite = callSiteOpt!

    if isSingleBB {
      var (specializedFunction, alreadyExists) = getOrCreateSpecializedFunction(basedOn: callSite, context)

      if !alreadyExists {
        context.notifyNewFunction(function: specializedFunction, derivedFrom: callSite.applyCallee)
      }

      rewriteApplyInstruction(using: specializedFunction, callSite: callSite, context)

      // MYTODO avoid this array
      let callSites = [callSite]
      var deadClosures: InstructionWorklist = callSites.reduce(into: InstructionWorklist(context)) { deadClosures, callSite in
        callSite.closureArgDescriptors
          .map { $0.closure }
          .forEach { deadClosures.pushIfNotVisited($0) }
      }

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
    } else {
//      if callSite.closureInfosWithApplyCFG.count != 2 {
//        break
//      }
//      if callSite.closureInfosWithApplyCFG[0].closureInfo.enumTypeAndCase.caseIdx ==
//         callSite.closureInfosWithApplyCFG[1].closureInfo.enumTypeAndCase.caseIdx {
//        break
//      }

      debugPrint("AAAAA PB BEFORE BEGIN")
      debugPrint(callSite.applyCallee)
      debugPrint("AAAAA PB BEFORE END")

      var (specializedFunction, alreadyExists) =
          getOrCreateSpecializedFunctionCFG(basedOn: callSite, enumDict: &enumDict, context)

      if !alreadyExists {
        context.notifyNewFunction(function: specializedFunction, derivedFrom: callSite.applyCallee)
      }

      rewriteApplyInstructionCFG(using: specializedFunction, callSite: callSite,
                                 enumType: callSite.closureInfosWithApplyCFG[0].closureInfo.enumTypeAndCase.enumType,
                                 enumDict: enumDict, context: context)
    }

    remainingSpecializationRounds -= 1
  } while remainingSpecializationRounds > 0
  debugPrint("AAAAA AUTODIFF PASS END")
}

// =========== Top-level functions ========== //

private let specializationLevelLimit = 2

private func gatherCallSite(in caller: Function, _ context: FunctionPassContext) -> Optional<CallSite> {
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

  var callSiteOpt = Optional<CallSite>(nil)
  let isSingleBB = caller.blocks.singleElement != nil

  for inst in caller.instructions {
    if !convertedAndReabstractedClosures.contains(inst),
       let rootClosure = inst.asSupportedClosure
    {
      if isSingleBB {
        updateCallSite(for: rootClosure, in: &callSiteOpt,
                       convertedAndReabstractedClosures: &convertedAndReabstractedClosures, context)
      } else {
        updateCallSiteCFG(for: rootClosure, in: &callSiteOpt, context)
      }
    }
  }

  return callSiteOpt
}

private func getOrCreateSpecializedFunction(basedOn callSite: CallSite, _ context: FunctionPassContext)
  -> (function: Function, alreadyExists: Bool)
{
  let specializedFunctionName = callSite.specializedCalleeName(context)
  if let specializedFunction = context.lookupFunction(name: specializedFunctionName) {
    return (specializedFunction, true)
  }

  let applySiteCallee = callSite.applyCallee
  let specializedParameters = applySiteCallee.convention.getSpecializedParameters(basedOn: callSite)

  let specializedFunction = 
    context.createFunctionForClosureSpecialization(from: applySiteCallee, withName: specializedFunctionName, 
                                                   withParams: specializedParameters, 
                                                   withSerialization: applySiteCallee.isSerialized)

  context.buildSpecializedFunction(specializedFunction: specializedFunction,
                                   buildFn: { (emptySpecializedFunction, functionPassContext) in 
                                      let closureSpecCloner = SpecializationCloner(emptySpecializedFunction: emptySpecializedFunction, functionPassContext)
                                      closureSpecCloner.cloneAndSpecializeFunctionBody(using: callSite)
                                   })

  return (specializedFunction, false)
}

private func getOrCreateSpecializedFunctionCFG(basedOn callSite: CallSite, enumDict: inout EnumDict, _ context: FunctionPassContext)
  -> (function: Function, alreadyExists: Bool)
{
  assert(callSite.closureArgDescriptors.count == 0)
  let closureInfos = callSite.closureInfosWithApplyCFG
  //assert(closureInfos.count == 2)
  let enumType = closureInfos[0].closureInfo.enumTypeAndCase.enumType
  for closureInfo in closureInfos {
    assert(closureInfo.closureInfo.enumTypeAndCase.enumType == closureInfos[1].closureInfo.enumTypeAndCase.enumType)
  }
  let specializedPbName = callSite.specializedCalleeNameCFG(context)
  if let specializedPb = context.lookupFunction(name: specializedPbName) {
    return (specializedPb, true)
  }

  let pb = callSite.applyCallee
  let specializedParameters = getSpecializedParametersCFG(basedOn: callSite, pb: pb, enumType: enumType, enumDict: &enumDict, context)

  let specializedPb =
    context.createFunctionForClosureSpecialization(from: pb, withName: specializedPbName,
                                                   withParams: specializedParameters,
                                                   withSerialization: pb.isSerialized)

  context.buildSpecializedFunction(specializedFunction: specializedPb,
                                   buildFn: { (emptySpecializedFunction, functionPassContext) in 
                                      let closureSpecCloner = SpecializationCloner(emptySpecializedFunction: emptySpecializedFunction, functionPassContext)
                                      closureSpecCloner.cloneAndSpecializeFunctionBodyCFG(using: callSite, enumType: enumType, enumDict: &enumDict)
                                   })
  debugPrint("AAAAAA PB AFTER BEGIN")
  debugPrint(specializedPb)
  debugPrint("AAAAAA PB AFTER END")

  return (specializedPb, false)
}

private func rewriteApplyInstructionCFG(using specializedCallee: Function, callSite: CallSite,
                                     enumType: Type, enumDict: EnumDict,
                                     context: FunctionPassContext) {
  var builderSucc = Builder(atEndOf: callSite.applySite.parentBlock, location: callSite.applySite.parentBlock.instructions.last!.location, context)
  // MYTODO: enums set not create each time
  // MYTODO: maybe each enum might be re-written multiple times
  assert(enumDict[enumType] != nil)
  let newEnumType = enumDict[enumType]!

  var vjpToInlineOpt = Optional<Function>(nil)

  var applyArgOpt = Optional<Value>(nil)
  for (argIdx, arg) in callSite.applySite.arguments.enumerated() {
    assert(argIdx == 0)
    assert(applyArgOpt == nil)
    applyArgOpt = arg
  }
  assert(applyArgOpt != nil)
  let applyArg = applyArgOpt!

  let bb = callSite.applySite.parentBlock
  let preds = bb.predecessors

  var vjpsToInline = Array<Function>()

  for pred in preds {
    var brInst = pred.instructions.last! as! BranchInst
    var enumIdxInBranch = Optional<Int>(nil)
    for (targetBBArgIdx, targetBBArg) in brInst.targetBlock.arguments.enumerated() {
      let argType = targetBBArg.bridged.getType().type
      if argType == enumType {
        assert(enumIdxInBranch == nil)
        enumIdxInBranch = targetBBArgIdx
      }
    }
    assert(enumIdxInBranch != nil)
    var enumInstOld = brInst.operands[enumIdxInBranch!].value.definingInstruction! as! EnumInst
    var oldPayload = enumInstOld.payload! as! TupleInst

    // MYTODO: for some reason fail, but this is partial apply
    // MYTODO: Found a null pointer in a value of type
    // MYTODO: thin to thick
    var idxInPayloadArray = Array<Int>()
    for closureInfo in callSite.closureInfosWithApplyCFG {
      if closureInfo.closureInfo.closure.parentBlock == pred {
        //assert(idxInPayload == nil)
        idxInPayloadArray.append(closureInfo.closureInfo.idxInEnumPayload)
      }
    }
//    assert(idxInPayload != nil)
    for idxInPayload in idxInPayloadArray { 
    debugPrint("AAAA rewriteApplyCFG 00 BEGIN")
    debugPrint(idxInPayload)
    debugPrint("AAAA rewriteApplyCFG 00 MIDDLE 1")
    debugPrint(idxInPayloadArray)
    debugPrint("AAAA rewriteApplyCFG 00 MIDDLE 2")
    debugPrint(oldPayload)
    debugPrint("AAAA rewriteApplyCFG 00 MIDDLE 3")
    debugPrint(oldPayload.operands)
    debugPrint("AAAA rewriteApplyCFG 00 END")
    let paiOrThinToThickInstr = oldPayload.operands[idxInPayload].value.definingInstruction!
    debugPrint("AAAA rewriteApplyCFG 01")
    let maybeThinToThickInstr = paiOrThinToThickInstr as? ThinToThickFunctionInst
    debugPrint("AAAA rewriteApplyCFG 02")
    var optionalPAI = Optional<PartialApplyInst>(nil)
    var optionalVJPToInline = Optional<Function>(nil)
    if maybeThinToThickInstr == nil {
      optionalPAI = paiOrThinToThickInstr as! PartialApplyInst
      let fri = optionalPAI!.operands[0].value.definingInstruction! as! FunctionRefInst
      optionalVJPToInline = fri.referencedFunction
    } else {
      let fri = maybeThinToThickInstr!.operands[0].value.definingInstruction! as! FunctionRefInst
      optionalVJPToInline = fri.referencedFunction
//        continue
    }
    debugPrint("AAAA rewriteApplyCFG 03")
    // MYTODO: support thin to thick
    if optionalPAI != nil {
      debugPrint("AAAA rewriteApplyCFG 04")
      let pai = optionalPAI!
      var tupleValues = Array<Value>()
      for (opIdx, op) in pai.operands.enumerated() {
        if opIdx == 0 {
          continue
        }
        tupleValues.append(op.value)
      }
//      let builderPred = Builder(before: pai, context)
      let builderPred = Builder(before: enumInstOld, context)
      let tuple = builderPred.createTuple(elements: tupleValues)

      var newPayloadValues = Array<Value>()
      for (opIdx, op) in oldPayload.operands.enumerated() {
        if opIdx == idxInPayload {
          newPayloadValues.append(tuple)
          continue
        }
        newPayloadValues.append(op.value)
      }
      let newPayload = builderPred.createTupleWithPredecessor(elements: newPayloadValues)
      let enumInstNew = builderPred.createEnum(caseIndex: enumInstOld.caseIndex, payload: newPayload, enumType: newEnumType)

      let vjpToInline = optionalVJPToInline!

      var newBrOperandValues = Array<Value>()
      for op in brInst.operands {
        if brInst.getArgument(for: op).bridged.getType().type == enumType {
          newBrOperandValues.append(enumInstNew)
        } else {
          newBrOperandValues.append(op.value)
        }
      }
      let builderBr = Builder(before: brInst, context)
      let newBrInst = builderBr.createBranch(to: brInst.targetBlock, arguments: newBrOperandValues)
      debugPrint("AAAA rewriteApplyCFG 05")

      vjpsToInline.append(vjpToInline)
      pred.bridged.eraseInstruction(brInst.bridged)
      pred.bridged.eraseInstruction(enumInstOld.bridged)
      pred.bridged.eraseInstruction(oldPayload.bridged)
      pred.bridged.eraseInstruction(pai.bridged)

      enumInstOld = enumInstNew
      brInst = newBrInst
      oldPayload = newPayload
    } else { // thin to thick
      debugPrint("AAAA rewriteApplyCFG 06")
      assert(maybeThinToThickInstr != nil)
      let builderPred = Builder(before: enumInstOld, context)

      var tupleValues = Array<Value>()
      let tuple = builderPred.createTuple(elements: tupleValues)

      var newPayloadValues = Array<Value>()
      for (opIdx, op) in oldPayload.operands.enumerated() {
        if opIdx == idxInPayload {
          newPayloadValues.append(tuple)
          continue
        }
        newPayloadValues.append(op.value)
      }
      debugPrint("AAAA rewriteApplyCFG 07")
      let newPayload = builderPred.createTupleWithPredecessor(elements: newPayloadValues)



      let enumInstNew = builderPred.createEnum(caseIndex: enumInstOld.caseIndex, payload: newPayload, enumType: newEnumType)

      var newBrOperandValues = Array<Value>()
      for op in brInst.operands {
        if brInst.getArgument(for: op).bridged.getType().type == enumType {
          newBrOperandValues.append(enumInstNew)
        } else {
          newBrOperandValues.append(op.value)
        }
      }
      let builderBr = Builder(before: brInst, context)
      let newBrInst = builderBr.createBranch(to: brInst.targetBlock, arguments: newBrOperandValues)
      debugPrint("AAAA rewriteApplyCFG 08")

      vjpsToInline.append(optionalVJPToInline!)
      pred.bridged.eraseInstruction(brInst.bridged)
      pred.bridged.eraseInstruction(enumInstOld.bridged)
      pred.bridged.eraseInstruction(oldPayload.bridged)
      pred.bridged.eraseInstruction(maybeThinToThickInstr!.bridged)

      enumInstOld = enumInstNew
      brInst = newBrInst
      oldPayload = newPayload
    }
    }
  }
//    assert(vjpToInlineOpt == nil)
//    vjpToInlineOpt = vjpsToInline[0]
//    // MYTODO: support arbitrary vjps; is it possible w/o creating multiple basic blocks?
//    for e in vjpsToInline {
//      assert(e == vjpToInlineOpt!)
//    }
  let succBB = callSite.applySite.parentBlock
  // MYTODO function ref to spec new pullback

  let pai = callSite.applySite as! PartialApplyInst
  assert(pai.numArguments == 1)
  let paiFunction = pai.operands[0].value
  let paiConvention = pai.calleeConvention
  let paiHasUnknownResultIsolation = pai.hasUnknownResultIsolation
  let paiSubstitutionMap = SubstitutionMap(bridged: pai.bridged.getSubstitutionMap())
  let paiIsOnStack = pai.isOnStack

  let returnInst = succBB.terminator as! ReturnInst
  let tupleInst = returnInst.returnedValue.definingInstruction as! TupleInst
  let tupleElem = tupleInst.operands[0].value
  let functionRefInst = paiFunction as! FunctionRefInst

  succBB.bridged.eraseInstruction(returnInst.bridged)
// MYTODO: assert no uses
//    assert(tupleInst.uses.makeIterator().currentOpPtr == nil)
  succBB.bridged.eraseInstruction(tupleInst.bridged)
//    assert(pai.uses.makeIterator().currentOpPtr == nil)
  succBB.bridged.eraseInstruction(pai.bridged)
  let newFunctionRefInst = builderSucc.createFunctionRef(specializedCallee)
  functionRefInst.replace(with: newFunctionRefInst, context)
  for (argIndex, arg) in succBB.arguments.enumerated() {
    if arg.type == enumType {
      //assert(argIndex == succBB.arguments.count - 1)
      let newBBArg = succBB.bridged.recreateEnumBlockArgument(argIndex, newEnumType.bridged).argument
      let newPai : PartialApplyInst = builderSucc.createPartialApply(function: newFunctionRefInst, substitutionMap: paiSubstitutionMap,
                                                  capturedArguments: [newBBArg], calleeConvention: paiConvention,
                                                  hasUnknownResultIsolation: paiHasUnknownResultIsolation, isOnStack: paiIsOnStack)
      let newTupleInst = builderSucc.createTuple(elements: [tupleElem, newPai])
      let newReturnInst = builderSucc.createReturn(of: newTupleInst)
      break
    }
  }
//  return vjpToInlineOpt!
}

private func rewriteApplyInstruction(using specializedCallee: Function, callSite: CallSite, 
                                     _ context: FunctionPassContext) {
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

  let newApply = builder.createPartialApply(function: funcRef, substitutionMap: SubstitutionMap(), 
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

private func updateCallSiteCFG(for rootClosure: SingleValueInstruction,
                               in callSiteOpt: inout Optional<CallSite>,
                               _ context: FunctionPassContext) {
  let tupleOpt = handleNonAppliesCFG(for: rootClosure, context)
  if tupleOpt == nil {
    return
  }

  let closureInfo = tupleOpt!.closureInfo
  let pbApplyOperand = tupleOpt!.pbApplyOperand

  guard let pbPAI = pbApplyOperand.instruction as? PartialApplyInst else {
    return
  }

  guard let pb = pbPAI.referencedFunction else {
    return
  }

  let argType = pbApplyOperand.value.type
  var argIdxOpt = Optional<Int>(nil)
  // MYTODO: make argIdx computation less fragile
  for (idx, arg) in pb.arguments.enumerated() {
    if arg.type == argType {
      argIdxOpt = idx
    }
  }

  let arg = pb.argument(at: argIdxOpt!)
  if !arg.bridged.hasOneUse() {
    return
  }
  let argFirstUse = arg.bridged.getFirstUse()
  let possibleSwitchEnumInst = BridgedOperand(op: argFirstUse.op!).getUser().instruction
  let optionalSwitchEnumInst = possibleSwitchEnumInst as? SwitchEnumInst
  if optionalSwitchEnumInst == nil {
    return
  }

  let succBB = optionalSwitchEnumInst!.getUniqueSuccessor(forCaseIndex: closureInfo.enumTypeAndCase.caseIdx)!
  assert(succBB.arguments.count == 1)
  var closureValInPbOpt = Optional<Value>(nil)
  if succBB.arguments[0].bridged.hasOneUse() {
    let possibleDestructureTupleInst = BridgedOperand(op: succBB.arguments[0].bridged.getFirstUse().op!).getUser().instruction
    let optionalDestructureTupleInst = possibleDestructureTupleInst as? DestructureTupleInst
    assert(optionalDestructureTupleInst != nil)
    closureValInPbOpt = optionalDestructureTupleInst!.results[closureInfo.idxInEnumPayload]
  } else {
    for use in succBB.arguments[0].uses {
      let tupleExtractInstOpt = use.instruction as? TupleExtractInst
      if tupleExtractInstOpt == nil {
        continue
      }
      if tupleExtractInstOpt!.fieldIndex == closureInfo.idxInEnumPayload {
        assert(closureValInPbOpt == nil)
        closureValInPbOpt = tupleExtractInstOpt!.results[0]
      }
    }
  }
  assert(closureValInPbOpt != nil)

  var applyInPbOpt = Optional<ApplyInst>(nil)
  for use in closureValInPbOpt!.uses {
    let applyInstOpt = use.instruction as? ApplyInst
    if applyInstOpt == nil {
      continue
    }
    assert(applyInPbOpt == nil)
    applyInPbOpt = applyInstOpt!
  }


  assert(applyInPbOpt != nil)

  if callSiteOpt == nil {
    callSiteOpt = CallSite(applySite: pbPAI)
  } else {
    assert(callSiteOpt!.applySite == pbPAI)
  }

  callSiteOpt!.closureInfosWithApplyCFG.append((closureInfo: closureInfo, applyInPb: applyInPbOpt!))
}

private func updateCallSite(for rootClosure: SingleValueInstruction,
                            in callSiteOpt: inout Optional<CallSite>,
                            convertedAndReabstractedClosures: inout InstructionSet,
                            _ context: FunctionPassContext) {
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
       handleNonApplies(for: rootClosure, rootClosureApplies: &rootClosureApplies,
                     rootClosurePossibleLiveRange: &rootClosurePossibleLiveRange, context);


  if foundUnexpectedUse {
    return
  }

  let intermediateClosureArgDescriptorData = 
    handleApplies(for: rootClosure, callSiteOpt: &callSiteOpt, rootClosureApplies: &rootClosureApplies,
                  rootClosurePossibleLiveRange: &rootClosurePossibleLiveRange, 
                  convertedAndReabstractedClosures: &convertedAndReabstractedClosures,
                  haveUsedReabstraction: haveUsedReabstraction, context)

  if callSiteOpt == nil {
    return
  }

  finalizeCallSite(for: rootClosure, in: &callSiteOpt,
                   rootClosurePossibleLiveRange: rootClosurePossibleLiveRange,
                   intermediateClosureArgDescriptorData: intermediateClosureArgDescriptorData, context)
}

private func handleNonAppliesCFG(for rootClosure: SingleValueInstruction,
                                 _ context: FunctionPassContext)
  -> Optional<(closureInfo: ClosureInfoCFG, pbApplyOperand: Operand)>
{
  let blockIdx = rootClosure.parentBlock.index
  if blockIdx != 1 && blockIdx != 2 {
    return nil
  }

  var rootClosureConversionsAndReabstractions = OperandWorklist(context)
  rootClosureConversionsAndReabstractions.pushIfNotVisited(contentsOf: rootClosure.uses)
  defer {
    rootClosureConversionsAndReabstractions.deinitialize()
  }

  var closureInfoOpt = Optional<ClosureInfoCFG>(nil)
  var pbApplyOperandOpt = Optional<Operand>(nil)

  while let use = rootClosureConversionsAndReabstractions.pop() {
    switch use.instruction {
    case let pai as PartialApplyInst:
      if !pai.isPullbackInResultOfAutodiffVJP {
        return nil
      }
      assert(pbApplyOperandOpt == nil)
      assert(pai.parentBlock.index == 3)
      pbApplyOperandOpt = use

    case let ti as TupleInst:
      if ti.parentFunction.isAutodiffVJP,
         let returnInst = ti.parentFunction.returnInstruction,
         ti == returnInst.returnedValue
      {
        // This is the pullback closure returned from an Autodiff VJP and we don't need to handle it.
      } else if rootClosure.parentFunction.blocks.singleElement == nil {
        if !ti.bridged.hasOneUse() {
          return nil
        }
        let tupleFirstUse = ti.bridged.getFirstUse()
        let possibleEnumInst = BridgedOperand(op: tupleFirstUse.op!).getUser().instruction
        let optionalEI = possibleEnumInst as? EnumInst
        if optionalEI == nil {
          return nil
        }
        let ei = optionalEI!
        if !ei.bridged.hasOneUse() {
          return nil
        }
        let firstEnumUse = ei.bridged.getFirstUse()
        let possibleBranchInst = BridgedOperand(op: firstEnumUse.op!).getUser().instruction
        let optionalBI = possibleBranchInst as? BranchInst
        if optionalBI == nil {
          return nil
        }
        let bi = optionalBI!

        let succBBArg = bi.getArgument(for: Operand(bridged: BridgedOperand(op: firstEnumUse.op!)))

        if use.value != rootClosure {
          return nil
        }
        rootClosureConversionsAndReabstractions.pushIfNotVisited(contentsOf: succBBArg.uses)
        var capturedArgs = Array<Value>()
        var idxInEnumPayload = use.index
        var paiOpt = rootClosure as? PartialApplyInst
        if paiOpt != nil {
          for argOp in paiOpt!.argumentOperands {
            capturedArgs.append(argOp.value)
          }
        }
        assert(closureInfoOpt == nil)
        let enumTypeAndCase = EnumTypeAndCase(enumType: ei.type, caseIdx: ei.caseIndex)
        closureInfoOpt = (closure: rootClosure, idxInEnumPayload: idxInEnumPayload, capturedArgs: capturedArgs, enumTypeAndCase: enumTypeAndCase)
      } else {
        fallthrough
      }

    default:
      return nil
    }
  }
  assert((closureInfoOpt == nil) == (pbApplyOperandOpt == nil))
  if closureInfoOpt == nil {
    return nil
  }
  return (closureInfo: closureInfoOpt!, pbApplyOperand: pbApplyOperandOpt!)
}

/// Handles all non-apply direct and transitive uses of `rootClosure`.
///
/// Returns: 
/// haveUsedReabstraction - whether the root closure is reabstracted via a thunk 
/// foundUnexpectedUse - whether the root closure is directly or transitively used in an instruction that we don't know
///                      how to handle. If true, then `rootClosure` should not be specialized against.
private func handleNonApplies(for rootClosure: SingleValueInstruction, 
                              rootClosureApplies: inout OperandWorklist,
                              rootClosurePossibleLiveRange: inout InstructionRange, 
                              _ context: FunctionPassContext) 
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

private typealias IntermediateClosureArgDescriptorDatum = (applySite: SingleValueInstruction, closureArgIndex: Int, paramInfo: ParameterInfo)

private func handleApplies(for rootClosure: SingleValueInstruction, callSiteOpt: inout Optional<CallSite>,
                           rootClosureApplies: inout OperandWorklist, 
                           rootClosurePossibleLiveRange: inout InstructionRange, 
                           convertedAndReabstractedClosures: inout InstructionSet, haveUsedReabstraction: Bool, 
                           _ context: FunctionPassContext) -> [IntermediateClosureArgDescriptorDatum] 
{
  var intermediateClosureArgDescriptorData: [IntermediateClosureArgDescriptorDatum] = []
  
  while let use = rootClosureApplies.pop() {
    rootClosurePossibleLiveRange.insert(use.instruction)

    // TODO [extend to general swift]: Handle full apply sites
    guard let pai = use.instruction as? PartialApplyInst else {
      continue
    }

    // TODO: Handling generic closures may be possible but is not yet implemented
    if pai.hasSubstitutions || !pai.calleeIsDynamicFunctionRef || !pai.isPullbackInResultOfAutodiffVJP {
      continue
    }

    guard let callee = pai.referencedFunction else {
      continue
    }

    // Workaround for a problem with OSSA: https://github.com/swiftlang/swift/issues/78847
    // TODO: remove this if-statement once the underlying problem is fixed.
    if callee.hasOwnership {
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

    let onlyHaveThinToThickClosure = rootClosure is ThinToThickFunctionInst && !haveUsedReabstraction

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
    let closureCallee = rootClosure is PartialApplyInst 
                        ? (rootClosure as! PartialApplyInst).referencedFunction!
                        : (rootClosure as! ThinToThickFunctionInst).referencedFunction!

    if closureCallee.specializationLevel > specializationLevelLimit {
      continue
    }

    if haveUsedReabstraction {
      markConvertedAndReabstractedClosuresAsUsed(rootClosure: rootClosure, convertedAndReabstractedClosure: use.value, 
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
private func finalizeCallSite(for rootClosure: SingleValueInstruction, in callSiteOpt: inout Optional<CallSite>,
                              rootClosurePossibleLiveRange: InstructionRange, 
                              intermediateClosureArgDescriptorData: [IntermediateClosureArgDescriptorDatum], 
                              _ context: FunctionPassContext) {
  let closureInfo = ClosureInfo(closure: rootClosure, lifetimeFrontier: Array(rootClosurePossibleLiveRange.ends))

  for (applySite, closureArgumentIndex, parameterInfo) in intermediateClosureArgDescriptorData {
    if callSiteOpt!.applySite != applySite {
      fatalError("While finalizing call sites, call site descriptor not found for call site: \(applySite)!")
    }
    let closureArgDesc = ClosureArgDescriptor(closureInfo: closureInfo, closureArgumentIndex: closureArgumentIndex, 
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
private func markConvertedAndReabstractedClosuresAsUsed(rootClosure: Value, convertedAndReabstractedClosure: Value, 
                                                        convertedAndReabstractedClosures: inout InstructionSet) 
{
  if convertedAndReabstractedClosure != rootClosure {
    switch convertedAndReabstractedClosure {
    case let pai as PartialApplyInst:
      convertedAndReabstractedClosures.insert(pai)
      return 
        markConvertedAndReabstractedClosuresAsUsed(rootClosure: rootClosure, 
                                                   convertedAndReabstractedClosure: pai.arguments[0], 
                                                   convertedAndReabstractedClosures: &convertedAndReabstractedClosures)
    case let cvt as ConvertFunctionInst:
      convertedAndReabstractedClosures.insert(cvt)
      return 
        markConvertedAndReabstractedClosuresAsUsed(rootClosure: rootClosure, 
                                                   convertedAndReabstractedClosure: cvt.fromFunction,
                                                   convertedAndReabstractedClosures: &convertedAndReabstractedClosures)
    case let cvt as ConvertEscapeToNoEscapeInst:
      convertedAndReabstractedClosures.insert(cvt)
      return 
        markConvertedAndReabstractedClosuresAsUsed(rootClosure: rootClosure, 
                                                   convertedAndReabstractedClosure: cvt.fromFunction,
                                                   convertedAndReabstractedClosures: &convertedAndReabstractedClosures)
    case let mdi as MarkDependenceInst:
      convertedAndReabstractedClosures.insert(mdi)
      return 
        markConvertedAndReabstractedClosuresAsUsed(rootClosure: rootClosure, convertedAndReabstractedClosure: mdi.value,
                                                   convertedAndReabstractedClosures: &convertedAndReabstractedClosures)
    default:
      log("Parent function of callSite: \(rootClosure.parentFunction)")
      log("Root closure: \(rootClosure)")
      log("Converted/reabstracted closure: \(convertedAndReabstractedClosure)")
      fatalError("While marking converted/reabstracted closures as used, found unexpected instruction: \(convertedAndReabstractedClosure)")
    }
  }
}

private extension SpecializationCloner {
  func cloneAndSpecializeFunctionBodyCFG(using callSite: CallSite, enumType: Type, enumDict: inout EnumDict) {
    self.cloneEntryBlockArgsWithoutOrigClosuresCFG(usingOrigCalleeAt: callSite, enumType: enumType, enumDict: &enumDict)

    var args = Array<Value>()
    for arg in self.entryBlock.arguments {
      args.append(arg)
    }

    self.cloneFunctionBody(from: callSite.applyCallee, entryBlockArguments: args)

    //debugPrint(self.entryBlock.parentFunction)

    let closureInfos = callSite.closureInfosWithApplyCFG
    //assert(closureInfos.count == 2)

    let entrySwitchEnum = self.entryBlock.terminator as! SwitchEnumInst
    let builderEntry = Builder(before: entrySwitchEnum, self.context)

    var enumCases = Array<(Int, BasicBlock)>()
    for i in 0...entrySwitchEnum.bridged.SwitchEnumInst_getNumCases() {
      let bbForCase = entrySwitchEnum.getUniqueSuccessor(forCaseIndex: i)
      // MYTODO: ensure that identical indexes have identical textual values like bb1, bb2, ...
      if bbForCase != nil {
        enumCases.append((i, bbForCase!))
      }
    }
    let newEntrySwitchEnum = builderEntry.createSwitchEnum(enum: entrySwitchEnum.enumOp, cases: enumCases)
    self.entryBlock.bridged.eraseInstruction(entrySwitchEnum.bridged)

    // MYTODO: loop over enum cases with inner loop over VJPs in one enum case
    var closureInfoByEnumCase = Dictionary<Int, Array<ClosureInfoWithApplyCFG>>()
    for closureInfo in closureInfos {
      var tmp : Optional<Array<ClosureInfoWithApplyCFG>> = closureInfoByEnumCase[closureInfo.closureInfo.enumTypeAndCase.caseIdx]
      if tmp == nil {
        closureInfoByEnumCase[closureInfo.closureInfo.enumTypeAndCase.caseIdx] = [closureInfo]
      } else {
        tmp!.append(closureInfo)
        closureInfoByEnumCase[closureInfo.closureInfo.enumTypeAndCase.caseIdx] = tmp!
      }
    }
    for (enumIdx, closureInfoArray) in closureInfoByEnumCase {
      let succBB = newEntrySwitchEnum.getUniqueSuccessor(forCaseIndex: enumIdx)!

      var rewriter = BridgedEnumRewriter()
      for closureInfoWithApplyCFG in closureInfoArray {
        rewriter.appendToClosuresBuffer(closureInfoWithApplyCFG.closureInfo.enumTypeAndCase.caseIdx,
                                        closureInfoWithApplyCFG.closureInfo.closure.bridged,
                                        closureInfoWithApplyCFG.closureInfo.idxInEnumPayload)
      }
      succBB.bridged.recreateTupleBlockArgument(enumIdx)
      rewriter.clearClosuresBuffer()

      var applyInPbArray = Array<ApplyInst>()
      for (closureInfoIdx, closureInfo) in closureInfoArray.enumerated() {
        let applyInPbOriginal = closureInfo.applyInPb
        let pbBbOriginal = applyInPbOriginal.parentBlock
        var applyIdx = Optional<Int>(nil)
        for (instIdx, inst) in pbBbOriginal.instructions.enumerated() {
          if inst == applyInPbOriginal {
            assert(applyIdx == nil)
            applyIdx = instIdx
          }
        }
        assert(applyIdx != nil)
        let pbBbCloned = succBB//self.entryBlock.parentFunction
        var applyInPbX = Optional<Instruction>(nil)
        for (instIdx, inst) in pbBbCloned.instructions.enumerated() {
          if instIdx == applyIdx! {
            assert(applyInPbX == nil)
            applyInPbX = inst// as? ApplyInst
          }
        }

        debugPrint(succBB)
        debugPrint(applyInPbX)
        let applyInPb = applyInPbX as? ApplyInst
        debugPrint(applyInPb)
        assert(applyInPb != nil)
        applyInPbArray.append(applyInPb!)
      }

//      for (closureInfoIdx, closureInfo) in closureInfoArray.enumerated() {

//      let applyInPbOriginal = closureInfo.applyInPb
//      let pbOriginal = applyInPbOriginal.parentFunction
//      var applyIdx = Optional<Int>(nil)
//      for (instIdx, inst) in pbOriginal.instructions.enumerated() {
//        if inst == applyInPbOriginal {
//          assert(applyIdx == nil)
//          applyIdx = instIdx
//        }
//      }
//      assert(applyIdx != nil)
//      let pbCloned = self.entryBlock.parentFunction
//      var applyInPbX = Optional<Instruction>(nil)
//      for (instIdx, inst) in pbCloned.instructions.enumerated() {
//        if instIdx == applyIdx! {
//          assert(applyInPbX == nil)
//          applyInPbX = inst// as? ApplyInst
//        }
//      }
//
//      debugPrint(succBB)
//      debugPrint(applyInPbX)
//      let applyInPb = applyInPbX as? ApplyInst
//      debugPrint(applyInPb)
//      assert(applyInPb != nil)
//      let applyInPb = applyInPbArray[closureInfoIdx]

      let oldDtiOpt = applyInPbArray[0].callee.definingInstruction as? DestructureTupleInst
      if oldDtiOpt != nil {
        let oldDti = oldDtiOpt!
        let builderBeforeOldDti = Builder(before: oldDti, self.context)
        let newDti = builderBeforeOldDti.createDestructureTuple(tuple: oldDti.tuple)
//        let resToChangeOld = oldDti.results[closureInfo.closureInfo.idxInEnumPayload]

        for (resultIdx, result) in oldDti.results.enumerated() {
          for use in result.uses {
            switch use.instruction {
              case let ai as ApplyInst:
                let builder = Builder(before: ai, self.context)
                // MYTODO: other closures?
                var closureInfoOpt = Optional<ClosureInfoWithApplyCFG>(nil)
                for ccc in closureInfoArray {
                  if ccc.closureInfo.idxInEnumPayload == resultIdx {
                    assert(closureInfoOpt == nil)
                    closureInfoOpt = ccc
                  }
                }
                if (closureInfoOpt != nil) {//resultIdx == closureInfo.closureInfo.idxInEnumPayload) {
                  let dtiOfCapturedArgsTuple = builder.createDestructureTuple(tuple: newDti.results[resultIdx])
                  var newArgs = Array<Value>()
                  for op in ai.argumentOperands {
                    newArgs.append(op.value)
                  }
                  for res in dtiOfCapturedArgsTuple.results {
                    newArgs.append(res)
                  }
                  let vjpFnOpt = closureInfoOpt!.closureInfo.closure.asSupportedClosureFn
                  assert(vjpFnOpt != nil)
                  let newFri = builder.createFunctionRef(vjpFnOpt!)
                  debugPrint("AAAAA 00")
                  let newAi = builder.createApply(function: newFri, SubstitutionMap(), arguments: newArgs)
                  debugPrint("AAAAA 01")
                  ai.replace(with: newAi, self.context)
                } else {
                  var newArgs = Array<Value>()
                  for op in ai.argumentOperands {
                    newArgs.append(op.value)
                  }
                  debugPrint("AAAAA 02")
                  debugPrint("AAAAA old BB BEGIN")
                  //debugPrint(closureInfo.applyInPb.parentBlock)
 //                 debugPrint(pbBbOriginal)
                  debugPrint("AAAAA old BB END")
                  debugPrint("AAAAA new BB BEGIN")
                  debugPrint(succBB)
                  debugPrint("AAAAA new BB END")
                  debugPrint(ai)
                  debugPrint(newArgs)
                  let newAi = builder.createApply(function: newDti.results[resultIdx], ai.substitutionMap, arguments: newArgs)
                  debugPrint("AAAAA 03")
                  ai.replace(with: newAi, self.context)
                }

              case let dvi as DestroyValueInst:
                var needDestroyValue = true
                for ccc in closureInfoArray {
                  if ccc.closureInfo.idxInEnumPayload == resultIdx {
                    needDestroyValue = false
                  }
                }
                if needDestroyValue { //resultIdx != closureInfo.closureInfo.idxInEnumPayload {
                  let builder = Builder(before: dvi, self.context)
                  builder.createDestroyValue(operand: newDti.results[resultIdx])
                }
                dvi.parentBlock.bridged.eraseInstruction(dvi.bridged)

              case let uedi as UncheckedEnumDataInst:
                // MYTODO: rewrite this assert
//                assert(resultIdx != closureInfo.closureInfo.idxInEnumPayload)
                let builder = Builder(before: uedi, self.context)
                let newUedi = builder.createUncheckedEnumData(enum: newDti.results[resultIdx], caseIndex: uedi.caseIndex,
                                                              resultType: uedi.type)
                uedi.replace(with: newUedi, self.context)

              default:
                debugPrint("CCCC 00")
                debugPrint(use)
                debugPrint("CCCC 01")
                debugPrint(use.instruction)
                debugPrint("CCCC 02")
                debugPrint(use.instruction.parentFunction)
                debugPrint("CCCC 03")
                assert(false)
            }
          }
        }

        oldDti.parentBlock.bridged.eraseInstruction(oldDti.bridged)
      } else {
//        let oldTeiOpt = applyInPb.callee.definingInstruction as? TupleExtractInst
//        assert(oldTeiOpt != nil)
//        let oldTei = oldTeiOpt!
      }
//      }
    }
  }

  private func cloneEntryBlockArgsWithoutOrigClosuresCFG(usingOrigCalleeAt callSite: CallSite, enumType: Type, enumDict: inout EnumDict) {
    let originalEntryBlock = callSite.applyCallee.entryBlock
    let clonedFunction = self.cloned
    let clonedEntryBlock = self.entryBlock

    for arg in originalEntryBlock.arguments {
      var clonedEntryBlockArgType = arg.type.getLoweredType(in: clonedFunction)
      if clonedEntryBlockArgType == enumType {
        var builder = Builder(atStartOf: clonedFunction, self.context)
        assert(enumDict[enumType] != nil)
        clonedEntryBlockArgType = enumDict[enumType]!
      }
      let clonedEntryBlockArg = clonedEntryBlock.addFunctionArgument(type: clonedEntryBlockArgType, self.context)
      clonedEntryBlockArg.copyFlags(from: arg as! FunctionArgument)
    }
  }

  func cloneAndSpecializeFunctionBody(using callSite: CallSite) {
    self.cloneEntryBlockArgsWithoutOrigClosures(usingOrigCalleeAt: callSite)

    let (allSpecializedEntryBlockArgs, closureArgIndexToAllClonedReleasableClosures) = cloneAllClosures(at: callSite)

    self.cloneFunctionBody(from: callSite.applyCallee, entryBlockArguments: allSpecializedEntryBlockArgs)

    self.insertCleanupCodeForClonedReleasableClosures(
      from: callSite, closureArgIndexToAllClonedReleasableClosures: closureArgIndexToAllClonedReleasableClosures)
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
        let clonedEntryBlockArg = clonedEntryBlock.addFunctionArgument(type: clonedEntryBlockArgType, self.context)
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
    -> (allSpecializedEntryBlockArgs: [Value], 
        closureArgIndexToAllClonedReleasableClosures: [Int: [SingleValueInstruction]]) 
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
      closureArgIndexToAllClonedReleasableClosures[closureArgDesc.closureArgIndex] = allClonedReleasableClosures
    }

    return (entryBlockArgs.map { $0! }, closureArgIndexToAllClonedReleasableClosures)
  }

  private func cloneClosureChain(representedBy closureArgDesc: ClosureArgDescriptor, at callSite: CallSite) 
    -> (finalClonedReabstractedClosure: SingleValueInstruction, allClonedReleasableClosures: [SingleValueInstruction]) 
  {
    let (origToClonedValueMap, capturedArgRange) = self.addEntryBlockArgs(forValuesCapturedBy: closureArgDesc)
    let clonedFunction = self.cloned
    let clonedEntryBlock = self.entryBlock
    let clonedClosureArgs = Array(clonedEntryBlock.arguments[capturedArgRange])

    let builder = clonedEntryBlock.instructions.isEmpty
                  ? Builder(atStartOf: clonedFunction, self.context)
                  : Builder(atEndOf: clonedEntryBlock, location: clonedEntryBlock.instructions.last!.location, self.context)

    let clonedRootClosure = builder.cloneRootClosure(representedBy: closureArgDesc, capturedArguments: clonedClosureArgs)

    let (finalClonedReabstractedClosure, releasableClonedReabstractedClosures) = 
      builder.cloneRootClosureReabstractions(rootClosure: closureArgDesc.closure, clonedRootClosure: clonedRootClosure,
                                             reabstractedClosure: callSite.appliedArgForClosure(at: closureArgDesc.closureArgIndex)!,
                                             origToClonedValueMap: origToClonedValueMap,
                                             self.context)

    let allClonedReleasableClosures = [clonedRootClosure] + releasableClonedReabstractedClosures
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
      let capturedArg = clonedEntryBlock.addFunctionArgument(type: arg.type.getLoweredType(in: clonedFunction), 
                                                              self.context)
      origToClonedValueMap[arg] = capturedArg
    }

    let capturedArgRangeEnd = clonedEntryBlock.arguments.count
    let capturedArgRange = capturedArgRangeStart == capturedArgRangeEnd 
                           ? 0..<0 
                           : capturedArgRangeStart..<capturedArgRangeEnd

    return (origToClonedValueMap, capturedArgRange)
  }

  private func insertCleanupCodeForClonedReleasableClosures(from callSite: CallSite, 
                                                            closureArgIndexToAllClonedReleasableClosures: [Int: [SingleValueInstruction]])
  {
    for closureArgDesc in callSite.closureArgDescriptors {
      let allClonedReleasableClosures = closureArgIndexToAllClonedReleasableClosures[closureArgDesc.closureArgIndex]!

      // Insert a `destroy_value`, for all releasable closures, in all reachable exit BBs if the closure was passed as a
      // guaranteed parameter or its type was noescape+thick. This is b/c the closure was passed at +0 originally and we
      // need to balance the initial increment of the newly created closure(s).
      if closureArgDesc.isClosureGuaranteed || closureArgDesc.parameterInfo.isTrivialNoescapeClosure,
         !allClonedReleasableClosures.isEmpty
      {
        for exitBlock in callSite.reachableExitBBsInCallee {
          let clonedExitBlock = self.getClonedBlock(for: exitBlock)
          
          let terminator = clonedExitBlock.terminator is UnreachableInst
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

    if (self.context.needFixStackNesting) {
      self.cloned.fixStackNesting(self.context)
    }
  }
}

private extension [HashableValue: Value] {
  subscript(key: Value) -> Value? {
    get {
      self[key.hashable]
    }
    set {
      self[key.hashable] = newValue
    }
  }
}

private extension CallSite {
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

  func getArgumentsForSpecializedApply(of specializedCallee: Function) -> [NewApplyArg]
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
        let needsRetain = closureArgDesc.isCapturedArgNonTrivialObjectType(applySiteIndex: applySiteIndex, 
                                                                           specializedCallee: specializedCallee)

        newApplyArgs.append(.PreviouslyCaptured(value: capturedArg, needsRetain: needsRetain, 
                                                parentClosureArgIndex: closureArgDesc.closureArgIndex))
      }
    }

    return newApplyArgs
  }
}

private extension ClosureArgDescriptor {
  func isCapturedArgNonTrivialObjectType(applySiteIndex: Int, specializedCallee: Function) -> Bool {
    precondition(self.closure is PartialApplyInst, "ClosureArgDescriptor is not for a partial_apply closure!")

    let capturedArg = self.arguments[applySiteIndex]
    let pai = self.closure as! PartialApplyInst
    let capturedArgIndexInCallee = applySiteIndex + pai.unappliedArgumentCount
    let capturedArgConvention = self.callee.argumentConventions[capturedArgIndexInCallee]

    return !capturedArg.type.isTrivial(in: specializedCallee) && 
           !capturedArgConvention.isAllowedIndirectConvForClosureSpec
  }
}

private extension Builder {
  func cloneRootClosure(representedBy closureArgDesc: ClosureArgDescriptor, capturedArguments: [Value]) 
    -> SingleValueInstruction 
  {
    let function = self.createFunctionRef(closureArgDesc.callee)

    if let pai = closureArgDesc.closure as? PartialApplyInst {
      return self.createPartialApply(function: function, substitutionMap: SubstitutionMap(), 
                                     capturedArguments: capturedArguments, calleeConvention: pai.calleeConvention,
                                     hasUnknownResultIsolation: pai.hasUnknownResultIsolation, 
                                     isOnStack: pai.isOnStack)
    } else {
      return self.createThinToThickFunction(thinFunction: function, resultType: closureArgDesc.closure.type)
    }
  }

  func cloneRootClosureReabstractions(rootClosure: Value, clonedRootClosure: Value, reabstractedClosure: Value,
                                      origToClonedValueMap: [HashableValue: Value], _ context: FunctionPassContext) 
    -> (finalClonedReabstractedClosure: SingleValueInstruction, releasableClonedReabstractedClosures: [PartialApplyInst]) 
  {
    func inner(_ rootClosure: Value, _ clonedRootClosure: Value, _ reabstractedClosure: Value, 
               _ releasableClonedReabstractedClosures: inout [PartialApplyInst], 
               _ origToClonedValueMap: inout [HashableValue: Value]) -> Value {
      switch reabstractedClosure {
        case let reabstractedClosure where reabstractedClosure == rootClosure:
          origToClonedValueMap[reabstractedClosure] = clonedRootClosure
          return clonedRootClosure
        
        case let cvt as ConvertFunctionInst:
          let toBeReabstracted = inner(rootClosure, clonedRootClosure, cvt.fromFunction, 
                                       &releasableClonedReabstractedClosures, &origToClonedValueMap)
          let reabstracted = self.createConvertFunction(originalFunction: toBeReabstracted, resultType: cvt.type, 
                                                        withoutActuallyEscaping: cvt.withoutActuallyEscaping)
          origToClonedValueMap[cvt] = reabstracted
          return reabstracted
        
        case let cvt as ConvertEscapeToNoEscapeInst:
          let toBeReabstracted = inner(rootClosure, clonedRootClosure, cvt.fromFunction, 
                                       &releasableClonedReabstractedClosures, &origToClonedValueMap)
          let reabstracted = self.createConvertEscapeToNoEscape(originalFunction: toBeReabstracted, resultType: cvt.type,
                                                                isLifetimeGuaranteed: true)
          origToClonedValueMap[cvt] = reabstracted
          return reabstracted

        case let pai as PartialApplyInst:
          let toBeReabstracted = inner(rootClosure, clonedRootClosure, pai.arguments[0], 
                                       &releasableClonedReabstractedClosures, &origToClonedValueMap)
          
          guard let function = pai.referencedFunction else {
            log("Parent function of callSite: \(rootClosure.parentFunction)")
            log("Root closure: \(rootClosure)")
            log("Unsupported reabstraction closure: \(pai)")
            fatalError("Encountered unsupported reabstraction (via partial_apply) of root closure!")
          }

          let fri = self.createFunctionRef(function)
          let reabstracted = self.createPartialApply(function: fri, substitutionMap: SubstitutionMap(), 
                                                     capturedArguments: [toBeReabstracted], 
                                                     calleeConvention: pai.calleeConvention, 
                                                     hasUnknownResultIsolation: pai.hasUnknownResultIsolation, 
                                                     isOnStack: pai.isOnStack)
          releasableClonedReabstractedClosures.append(reabstracted)
          origToClonedValueMap[pai] = reabstracted
          return reabstracted
        
        case let mdi as MarkDependenceInst:
          let toBeReabstracted = inner(rootClosure, clonedRootClosure, mdi.value, &releasableClonedReabstractedClosures, 
                                       &origToClonedValueMap)
          let base = origToClonedValueMap[mdi.base]!
          let reabstracted = self.createMarkDependence(value: toBeReabstracted, base: base, kind: .Escaping)
          origToClonedValueMap[mdi] = reabstracted
          return reabstracted
        
        default:
          log("Parent function of callSite: \(rootClosure.parentFunction)")
          log("Root closure: \(rootClosure)")
          log("Converted/reabstracted closure: \(reabstractedClosure)")
          fatalError("Encountered unsupported reabstraction of root closure: \(reabstractedClosure)")
      }
    }

    var releasableClonedReabstractedClosures: [PartialApplyInst] = []
    var origToClonedValueMap = origToClonedValueMap
    let finalClonedReabstractedClosure = inner(rootClosure, clonedRootClosure, reabstractedClosure, 
                                               &releasableClonedReabstractedClosures, &origToClonedValueMap)
    return (finalClonedReabstractedClosure as! SingleValueInstruction, releasableClonedReabstractedClosures)
  }

  func destroyPartialApply(pai: PartialApplyInst, _ context: FunctionPassContext){
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

typealias EnumDict = Dictionary<Type, Type>

private func getSpecializedParametersCFG(basedOn callSite: CallSite, pb: Function, enumType: Type, enumDict: inout EnumDict, _ context: FunctionPassContext) -> [ParameterInfo] {
  let applySiteCallee = callSite.applyCallee
  var specializedParamInfoList: [ParameterInfo] = []

 // for (idx, arg) in pb.arguments.enumerated() {
 //   if arg.type == argType {
 //     argIdxOpt = idx
 //   }
 // }

 // let arg = pb.argument(at: argIdxOpt!)

  var found = false
  // Start by adding all original parameters except for the closure parameters.
  for (index, paramInfo) in applySiteCallee.convention.parameters.enumerated() {
    // MYTODO: is this safe to perform such check?
    if paramInfo.type.type.bridged.type == enumType.astType.type.bridged.type {
      assert(!found)
      found = true
      if enumDict[enumType] == nil {
        var rewriter = BridgedEnumRewriter()
        for closureInfoWithApplyCFG in callSite.closureInfosWithApplyCFG {
          rewriter.appendToClosuresBuffer(closureInfoWithApplyCFG.closureInfo.enumTypeAndCase.caseIdx,
                                          closureInfoWithApplyCFG.closureInfo.closure.bridged,
                                          closureInfoWithApplyCFG.closureInfo.idxInEnumPayload)
        }
        enumDict[enumType] = rewriter.rewriteBranchTracingEnum(/*enumType: */enumType.bridged,
                                                               /*topVjp: */callSite.applySite.parentFunction.bridged).type
        rewriter.clearClosuresBuffer()
      }
      let newEnumType = enumDict[enumType]!
      let newParamInfo = ParameterInfo(type: newEnumType.astType, convention: paramInfo.convention,
                                       options: paramInfo.options, hasLoweredAddresses: paramInfo.hasLoweredAddresses)
      specializedParamInfoList.append(newParamInfo)
    } else {
      specializedParamInfoList.append(paramInfo)
    }
  }
  return specializedParamInfoList
}

private extension FunctionConvention {
  func getSpecializedParameters(basedOn callSite: CallSite) -> [ParameterInfo] {
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
        let unappliedArgumentCount = closure.unappliedArgumentCount - closureCalleeConvention.indirectSILResultCount

        let prevCapturedParameters =
          closureCalleeConvention
          .parameters[unappliedArgumentCount...]
          .enumerated()
          .map { index, paramInfo in
            let argIndexOfParam = closureCallee.argumentConventions.firstParameterIndex + unappliedArgumentCount + index
            let argType = closureCallee.argumentTypes[argIndexOfParam]
            return paramInfo.withSpecializedConvention(isArgTypeTrivial: argType.isTrivial(in: closureCallee))
          }

        specializedParamInfoList.append(contentsOf: prevCapturedParameters)
      }
    }

    return specializedParamInfoList
  }
}

private extension ParameterInfo {
  func withSpecializedConvention(isArgTypeTrivial: Bool) -> Self {
    let specializedParamConvention = self.convention.isAllowedIndirectConvForClosureSpec
      ? self.convention
      : isArgTypeTrivial ? ArgumentConvention.directUnowned : ArgumentConvention.directOwned

    return ParameterInfo(type: self.type, convention: specializedParamConvention, options: self.options, 
                         hasLoweredAddresses: self.hasLoweredAddresses)
  }

  var isTrivialNoescapeClosure: Bool {
    SILFunctionType_isTrivialNoescape(type.bridged)
  }
}

private extension ArgumentConvention {
  var isAllowedIndirectConvForClosureSpec: Bool {
    switch self {
    case .indirectInout, .indirectInoutAliasable:
      return true
    default:
      return false
    }
  }
}

private extension PartialApplyInst {
  /// True, if the closure obtained from this partial_apply is the
  /// pullback returned from an autodiff VJP
  var isPullbackInResultOfAutodiffVJP: Bool {
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
       self.arguments[0].type.isFunction,
       self.arguments[0].type.isReferenceCounted(in: self.parentFunction) || self.callee.type.isThickFunction
    {
      return true
    }
    
    return false
  }

  var hasOnlyInoutIndirectArguments: Bool {
    self.argumentOperands
      .filter { !$0.value.type.isObject }
      .allSatisfy { self.convention(of: $0)!.isInout } 
  }
}

private extension Instruction {
  var asSupportedClosure: SingleValueInstruction? {
    switch self {
    case let tttf as ThinToThickFunctionInst where tttf.callee is FunctionRefInst:
      return tttf
    // TODO: figure out what to do with non-inout indirect arguments
    // https://forums.swift.org/t/non-inout-indirect-types-not-supported-in-closure-specialization-optimization/70826
    case let pai as PartialApplyInst where pai.callee is FunctionRefInst && pai.hasOnlyInoutIndirectArguments:
      return pai
    default:
      return nil
    }
  }

  var asSupportedClosureFn: Function? {
    switch self {
    case let tttf as ThinToThickFunctionInst where tttf.callee is FunctionRefInst:
      let fri = tttf.callee as! FunctionRefInst
      return fri.referencedFunction
    // TODO: figure out what to do with non-inout indirect arguments
    // https://forums.swift.org/t/non-inout-indirect-types-not-supported-in-closure-specialization-optimization/70826
    case let pai as PartialApplyInst where pai.callee is FunctionRefInst && pai.hasOnlyInoutIndirectArguments:
      let fri = pai.callee as! FunctionRefInst
      return fri.referencedFunction
    default:
      return nil
    }
  }

  var isSupportedClosure: Bool {
    asSupportedClosure != nil
  }
}

private extension ApplySite {
  var calleeIsDynamicFunctionRef: Bool {
    return !(callee is DynamicFunctionRefInst || callee is PreviousDynamicFunctionRefInst)
  }
}

private extension Function {
  var effectAllowsSpecialization: Bool {
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

  var keys: LazyMapSequence<Array<(Key, Value)>, Key> {
    entryList.lazy.map { $0.0 }
  }

  var values: LazyMapSequence<Array<(Key, Value)>, Value> {
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
  var closureInfosWithApplyCFG: [ClosureInfoWithApplyCFG] = []

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

    return context.mangle(withClosureArguments: closureArgs, closureArgIndices: closureIndices, 
                          from: applyCallee)
  }

  func specializedCalleeNameCFG(_ context: FunctionPassContext) -> String {
    // MYTODO: this should be enums and not closures
//    let enumArgs = Array(self.closureInfosWithApplyCFG.map { $0.closureInfo.closure })
//    let enumIndices = Array(self.closureInfosWithApplyCFG.map { $0.closureInfo.idxInEnumPayload })

    //return context.mangle(withEnumArguments: enumArgs, enumArgIndices: enumIndices,
    return context.mangle(withEnumArguments: [closureInfosWithApplyCFG[0].closureInfo.closure], enumArgIndices: [0],
                          from: applyCallee)
  }

  //func specializedCalleeNameCFG(enumType: Type, _ context: FunctionPassContext) -> String {
  //  let closureArgs = Array(self.closureArgDescriptors.map { $0.closure })
  //  let closureIndices = Array(self.closureArgDescriptors.map { $0.closureArgIndex })

  //  return context.mangle(withClosureArguments: closureArgs, closureArgIndices: closureIndices, 
  //                        from: applyCallee)
  //}
}

// ===================== Unit tests ===================== //

let gatherCallSiteTest = FunctionTest("closure_specialize_gather_call_site") { function, arguments, context in
  print("Specializing closures in function: \(function.name)")
  print("===============================================")
  let callSite = gatherCallSite(in: function, context)!
  // MYTODO avoid this array
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
  "closure_specialize_specialized_function_signature_and_body") { function, arguments, context in

  let callSite = gatherCallSite(in: function, context)!

  let (specializedFunction, _) = getOrCreateSpecializedFunction(basedOn: callSite, context)
  print("Generated specialized function: \(specializedFunction.name)")
  print("\(specializedFunction)\n")
}

let rewrittenCallerBodyTest = FunctionTest("closure_specialize_rewritten_caller_body") { function, arguments, context in
  let callSite = gatherCallSite(in: function, context)!

  let (specializedFunction, _) = getOrCreateSpecializedFunction(basedOn: callSite, context)
  rewriteApplyInstruction(using: specializedFunction, callSite: callSite, context)

  print("Rewritten caller body for: \(function.name):")
  print("\(function)\n")
}
