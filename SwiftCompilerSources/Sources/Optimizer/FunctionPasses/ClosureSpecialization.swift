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

import AST
import SIL

/// Closure Specialization
/// ----------------------
/// Specializes functions which take a closure (a `partial_apply` or `thin_to_thick_function` as argument.
/// The closure is created directly in the specialized function whereas the captured arguments are passed
/// as additional arguments to the specialized function. As a heuristic, this is only done if the closure
/// is actually called in the function because after specialization, the closure and its call can be
/// further optimized, e.g. the closure can be inlined.
///
/// ```
///   %3 = function_ref @closure
///   %4 = partial_apply %3(%1, %2) : (Float, Int, Bool) -> ()
///   %5 = function_ref @closure_user
///   apply %5(%4)
///   ...
///
/// sil @closure_user : ((Float) -> (), Float) -> () {
/// bb0(%0 : $(Float) -> (), %1 : $Float):
///   apply %0(%1)
///   ...
/// ```
/// ->
/// ```
///   %5 = function_ref @specialized_closure_user
///   apply %5(%1, %2)
///   ...
///
/// sil @specialized_closure_user : (Float, Int, Bool) -> () {
/// bb0(%0 : $Float, %1 : $Int, %2 : $Bool):
///   %3 = function_ref @closure
///   %4 = partial_apply %3(%1, %2)
///   apply %3(%0)
///   ...
/// ```
///
let closureSpecialization = FunctionPass(name: "closure-specialization") {
  (function: Function, context: FunctionPassContext) in

  runClosureSpecialization(function: function, isRootInvocation: true, context: context)
}

func runClosureSpecialization(function: Function, isRootInvocation: Bool, context: FunctionPassContext) {
  guard function.hasOwnership else {
    return
  }

  var remainingSpecializationRounds = 5

  repeat {
    var changed = false

    for inst in function.instructions {
      if let apply = inst as? FullApplySite {
        if trySpecialize(apply: apply, isRootInvocation: isRootInvocation, context) {
          changed = true
        }
      }
    }

    if context.needFixStackNesting {
      context.fixStackNesting(in: function)
    }
    if !changed {
      break
    }

    remainingSpecializationRounds -= 1
  } while remainingSpecializationRounds > 0
}

// Shared by the general and AutoDiff closure-specialization passes.
func trySpecialize(apply: ApplySite, isRootInvocation: Bool, _ context: FunctionPassContext) -> Bool {
  guard isCalleeSpecializable(of: apply),
        let specialization = analyzeArguments(of: apply, context)
  else {
    return false
  }

  let specializedParameters = specialization.getSpecializedParameters()

  // A function cannot have more than one "isolated" parameter.
  guard numberOfIsolatedParameters(specializedParameters) <= 1 else {
    return false
  }

  // We need to make each captured argument of all closures a unique Value. Otherwise the Cloner would
  // wrongly map added capture arguments to the re-generated closure in the specialized function:
  //
  //   %4 = partial_apply %3(%1, %1) : (Int, Int) -> ()  // %1 is captured twice
  //
  // sil @specialized_closure_user : (Int, Int) -> () {
  // bb0(%0 : $Int, %1 : $Int):
  //   %3 = function_ref @closure
  //   %4 = partial_apply %3(%0, %0)   <- instead of an argument list of `(%0, %1)`!
  //
  // This wouldn't be a problem per se - the code is correct. However, this is not reflected in the mangling.
  // And the same specialized function could be re-used at a call-site where not two identical values are
  // passed to the closure.
  //
  specialization.uniqueCaptureArguments(context)

  let (specializedFunction, isNewFunction) = specialization.getOrCreateSpecializedFunction(specializedParameters, context)

  specialization.unUniqueCaptureArguments(context)

  specialization.rewriteApply(for: specializedFunction, context)

  specialization.deleteDeadClosures(context)

  if isNewFunction && isRootInvocation {
    // A caller's closure may only become transitively applied after the cloned
    // callee has itself been specialized. Process that callee now so the next
    // caller round can see the exposed application; waiting for its worklist
    // turn would leave the already-processed caller unchanged.
    // Only root invocations do this, bounding synchronous specialization to one
    // additional level. The ordinary worklist handles deeper callees.
    context.buildSpecializedFunction(specializedFunction: specializedFunction) {
      (specializedFunction, specializedContext) in
      runClosureSpecialization(function: specializedFunction, isRootInvocation: false, context: specializedContext)
    }
  }

  return true
}

private func isCalleeSpecializable(of apply: ApplySite) -> Bool {
  if let callee = apply.referencedFunction,
     callee.isDefinition,

     // Calling `cloneRecursively` from `SpecializationInfo.cloneClosures`
     // requires the callee having ownership info. Otherwise, the cloner
     // uses `recordFoldedValue` instead of `recordClonedInstruction`, and
     // `postProcess` hook is not called, which leads to an assertion
     // failure in `BridgedClonerImpl::cloneInst`.
     callee.hasOwnership,

     // We don't support generic functions (yet)
     !apply.hasSubstitutions,

     // Don't specialize non-fragile (read as non-serialized) callees if the caller is fragile; the
     // specialized callee will have shared linkage, and thus cannot be referenced from the fragile caller.
     !(apply.parentFunction.isSerialized && !callee.isSerialized),

     // If the callee uses a dynamic Self, we cannot specialize it, since the resulting specialization
     // might no longer have 'self' as the last parameter.
     //
     // TODO: Keep the self argument the last when appending arguments.
     !callee.mayBindDynamicSelf,

     // Don't support self-recursive functions because that would result in duplicate mapping of values when cloning.
     callee != apply.parentFunction
  {
    return true
  }
  return false
}

private func analyzeArguments(of apply: ApplySite, _ context: FunctionPassContext) -> SpecializationInfo? {
  var argumentsToSpecialize = [(Operand, Closure)]()
  var rootClosures = [PartialApplyInst]()
  var capturedDependencies = [CapturedDependency]()
  var rootClosuresAdded = InstructionSet(context)
  defer { rootClosuresAdded.deinitialize() }

  for argOp in apply.argumentOperands {
    var visited = ValueSet(context)
    defer { visited.deinitialize() }
    var argumentDependencies = [CapturedDependency]()
    if let closure = findSpecializableClosure(of: argOp.value, &visited, &argumentDependencies),
       // Ok, we know that we can perform the optimization but not whether or not the optimization
       // is profitable. Check if the closure is actually called in the callee (or in a function
       // called by the callee). This opens optimization opportunities, like inlining.
       isClosureApplied(apply.calleeArgument(of: argOp, in: apply.referencedFunction!)!)
    {
      argumentsToSpecialize.append((argOp, closure))
      capturedDependencies.append(contentsOf: argumentDependencies)
      if let partialApply = closure as? PartialApplyInst,
         rootClosuresAdded.insert(partialApply)
      {
        rootClosures.append(partialApply)
      }
    }
  }
  if argumentsToSpecialize.isEmpty {
    return nil
  }
  return SpecializationInfo(apply: apply, closureArguments: argumentsToSpecialize, rootClosures: rootClosures,
                            capturedDependencies: capturedDependencies)
}

// Walks down the use-def chain of a function argument, recursively, to find a rootClosure.
private func findSpecializableClosure(of value: Value, _ visited: inout ValueSet,
                                      _ capturedDependencies: inout [CapturedDependency]) -> Closure? {
  visited.insert(value)

  let specializationLevelLimit = 2

  switch value {
  case is ConvertFunctionInst,
       is ConvertEscapeToNoEscapeInst,
       is MoveValueInst,
       is CopyValueInst:
    return findSpecializableClosure(of: (value as! UnaryInstruction).operand.value, &visited, &capturedDependencies)

  case let mdi as MarkDependenceInst:
    guard mdi.value.type.isNoEscapeFunction, mdi.value.type.isThickFunction else {
      return nil
    }
    guard let operandClosure = findSpecializableClosure(of: mdi.value, &visited, &capturedDependencies) else {
      return nil
    }
    // A base not in the closure's use-def chain must be a root-closure capture; record it for `uniqueCaptureArguments`.
    if !visited.contains(mdi.base) {
      guard let rootClosure = operandClosure as? PartialApplyInst,
            rootClosure.arguments.contains(where: { $0 == mdi.base })
      else {
        return nil
      }
      capturedDependencies.append((closure: rootClosure, markDependence: mdi))
    }
    return operandClosure

  case let partialApply as PartialApplyInst:
    // Don't specialize for re-abstractions via a partial_apply, but treat such re-abstractions like
    // closure "conversions". E.g.
    // ```
    //   %1 = partial_apply             // root closure
    //   %2 = function_ref @thunk
    //   %3 = partial_apply %2(%1)      // re-abstraction
    //   apply %f(%3)
    // ```
    // The thunk's partial_apply is cloned into the specialized function.
    // It may have substitutions that reference archetypes of the caller, bailout.
    if partialApply.isPartialApplyOfThunk,
       !partialApply.substitutionMap.replacementTypes.contains(where: { $0.hasArchetype })
    {
      // Keep the recorded dependencies only if the thunk's argument provides the root closure;
      // otherwise the thunk's partial_apply itself is tried as the root, below.
      var argumentDependencies = [CapturedDependency]()
      if let argumentClosure = findSpecializableClosure(of: partialApply.arguments[0], &visited,
                                                        &argumentDependencies) {
        capturedDependencies.append(contentsOf: argumentDependencies)
        return argumentClosure
      }
    }
    guard let callee = partialApply.referencedFunction,
          !partialApply.hasSubstitutions,

          // Captures are appended to the specialized function's parameters. A
          // dynamic-Self metadata argument must remain the last parameter.
          (!callee.isDefinition || !callee.mayBindDynamicSelf),

          // Avoid an infinite specialization loop caused by repeated runs of ClosureSpecialization and
          // ConstantCapturePropagation.
          // ConstantCapturePropagation propagates constant function-literals. Such function specializations
          // can then be optimized again by ClosureSpecialization and so on. This happens if a closure argument
          // is called _and_ referenced in another closure, which is passed to a recursive call. E.g.
          //
          // func foo(_ c: @escaping () -> ()) {
          //  c() foo({ c() })
          // }
          //
          // A limit of 2 is good enough and will not be exceed in "regular" optimization scenarios.
          callee.specializationLevel <= specializationLevelLimit,

          // Functions with a readnone, readonly or releasenone effect and a consumed captures cannot be
          // specialized because the captured arguments are passed as owned arguments to the specialized
          // function. Destroy for such arguments in the specialized function violates the effect.
          (partialApply.isOnStack || callee.effectAllowsSpecialization),

          // TODO: handle other kind of indirect arguments
          partialApply.hasOnlySupportedIndirectArguments,

          (partialApply.isOnStack || partialApply.allArgumentsCanBeCopied)
    else {
      return nil
    }
    return partialApply

  case let tttfi as ThinToThickFunctionInst:
    guard let callee = tttfi.referencedFunction,
          callee.specializationLevel <= specializationLevelLimit
    else {
      return nil
    }
    return tttfi

  default:
    return nil
  }
}

/// Either a `partial_apply` or a `thin_to_thick_function`
private typealias Closure = SingleValueInstruction

/// A `mark_dependence` whose base is a root-closure capture, paired with that capturing closure.
private typealias CapturedDependency = (closure: PartialApplyInst, markDependence: MarkDependenceInst)

/// Information about the function to be specialized and for which closure arguments.
private struct SpecializationInfo {

  // The apply which we want to specialize
  let apply: ApplySite

  // All closure arguments of the apply which we want to replace (usually there is one, but there can be
  // multiple).
  // Note that the `Closure` is not necessarily the value of the `Operand`. There can be function conventions
  // and re-abstractions (via a thunk) in-between:
  //
  //   %4 = partial_apply %3(%1) : (Int) -> ()   // %4 = rootClosure
  //   %5 = convert_function %4
  //   %6 = copy_value %5
  //   apply %5(%5)                              // %5 = closureArgument
  //
  let closureArguments: [(closureArgument: Operand, rootClosure: Closure)]

  // All rootClosures of `closureArguments` which are `partial_apply`s, and uniqued: if a rootClosure
  // appears multiple times in `closureArguments`, it's only added a single time here.
  let rootClosures: [PartialApplyInst]

  // `mark_dependence`s whose base is a root-closure capture. The base is redirected onto the
  // capture's cast in `uniqueCaptureArguments`.
  let capturedDependencies: [CapturedDependency]

  // The function to specialize
  var callee: Function { apply.referencedFunction! }

  private typealias Cloner = SIL.Cloner<FunctionPassContext>

  func getOrCreateSpecializedFunction(_ specializedParameters: [ParameterInfo],
                                      _ context: FunctionPassContext
  ) -> (Function, isNewFunction: Bool) {
    let specializedFunctionName = getSpecializedFunctionName(context)

    if let existingSpecializedFunction = context.lookupFunction(name: specializedFunctionName) {
      return (existingSpecializedFunction, false)
    }

    let specializedFunction =
      context.createSpecializedFunctionDeclaration(
        from: callee, withName: specializedFunctionName,
        withParams: specializedParameters,
        // The specialized function is always a thin function. This is important because we add additional
        // parameters after the Self parameter of witness methods. In this case the new function is not a
        // method anymore.
        withRepresentation: .thin, makeBare: true)

    context.buildSpecializedFunction(
      specializedFunction: specializedFunction,
      buildFn: { (specializedFunction, specializedContext) in
        var cloner = Cloner(cloneToEmptyFunction: specializedFunction, specializedContext)
        defer { cloner.deinitialize() }

        cloneAndSpecializeFunctionBody(using: &cloner)
        // Cloning a whole function, even if it contains an `unreachable`, doesn't require lifetime completion.
        specializedContext.setNeedCompleteLifetimes(to: false)
      })

    context.notifyNewFunction(function: specializedFunction, derivedFrom: callee)

    return (specializedFunction, true)
  }

  private func getSpecializedFunctionName(_ context: FunctionPassContext) -> String {
    var visited = Dictionary<Closure, Int>()

    let argumentManglings = closureArguments.map { (argOp, closure) in
      let argIdx = apply.calleeArgumentIndex(of: argOp)!
      if let prevArgIdx = visited[closure] {

        // If the same closure is passed multiple times to a function, we need to reflect this in the mangling:
        //
        //   %3 = function_ref @closure
        //   %4 = partial_apply %3(%1) : (Int) -> ()
        //   apply %6(%4, %4)
        //
        // sil @specialized_closure_user : (Int) -> () {
        //
        // is different than
        //
        //   %3 = function_ref @closure
        //   %4 = partial_apply %3(%1) : (Int) -> ()
        //   %5 = partial_apply %3(%1) : (Int) -> ()
        //   apply %6(%4, %5)
        //
        // sil @specialized_closure_user : (Int, Int) -> () {
        //
        return (argIdx, FunctionPassContext.ClosureArgumentMangling.previousArgumentIndex(prevArgIdx))
      } else {
        visited[closure] = argIdx
        return (argIdx, FunctionPassContext.ClosureArgumentMangling.closure(closure))
      }
    }
    return context.mangle(withClosureArguments: argumentManglings, from: callee)
  }

  func getSpecializedParameters() -> [ParameterInfo] {
    var specializedParamInfoList: [ParameterInfo] = []

    // Start by adding all original parameters except for the closure parameters.
    let firstParamIndex = callee.argumentConventions.firstParameterIndex
    for (index, paramInfo) in callee.convention.parameters.enumerated() {
      let argIndex = index + firstParamIndex
      if !closureArguments.contains(where: { apply.calleeArgumentIndex(of: $0.0) == argIndex}) {
        specializedParamInfoList.append(paramInfo)
      }
    }

    // Now, append parameters captured by each of the root closures.
    for partialApply in rootClosures {
      let closureConvention = partialApply.functionConvention
      let unappliedArgumentCount = partialApply.unappliedArgumentCount - closureConvention.indirectSILResultCount

      for (argOp, paramInfo) in zip(partialApply.argumentOperands,
                                     closureConvention.parameters[unappliedArgumentCount...]) {
        let newParamInfo = paramInfo.withSpecializedConvention(for: argOp, in: partialApply, callee: callee)
        specializedParamInfoList.append(newParamInfo)
      }
    }

    return specializedParamInfoList
  }

  private func cloneAndSpecializeFunctionBody(using cloner: inout Cloner) {
    addFunctionArgumentsWithoutClosures(using: &cloner)

    for rootClosure in rootClosures {
      addFunctionArgumentsForCaptures(of: rootClosure, using: &cloner)
    }

    let clonedClosureArguments = cloneClosures(using: &cloner)

    cloner.cloneFunctionBody(from: callee)

    addMissingDestroysAtFunctionExits(for: clonedClosureArguments, cloner.context)

    /// When we have several partial_apply instructions capturing each other in a chain,
    /// we might be able to perform several specialization rounds. See also test/SILOptimizer/closure_specialization_nested.sil.
    ///
    /// Whether a closure gets specialized depends on whether it is fully applied, possibly
    /// transitively, in the callee (the gate is in `analyzeArguments` via `isClosureApplied`).
    /// The goal across all rounds is to leave no specializable closures left in the caller.
    ///
    /// When `%15 = partial_apply @closure2(%12)` from caller is cloned into the specialized callee it comes back as a
    /// `partial_apply @closure2(%1)` serving as a callee of the further `apply`, so specialized callee argument `%1`
    /// which is by itself a closure `%12 = partial_apply @closure1(%9)` coming from caller is only a `partial_apply`
    /// operand and is therefore invisible to the applied-check.
    ///
    /// Folding `partial_apply`+`apply` into a single `apply` (via `tryOptimizeApplyOfPartialApply`) turns `partial_apply`
    /// of `closure1` in caller into a direct full-apply operand, so the next specialization round on `caller` can see
    /// that this closure is transitively applied in `closure2`. Without the fold the `partial_apply` of `closure1`
    /// closure is never recognized as applied and its `partial_apply` stays behind.
    ///
    /// We fold now rather than defer to separate sil-combiner run: the caller's specialization loop does revisit its
    /// apply each round, but unless we mutate the cloned callee now the applied-check still fails and the loop makes no
    /// progress. So, by the time the sil-combiner reaches the specialized callee (it's added to pass manager worklist
    /// via `notifyNewFunction`) and does the fold inside it, the caller's specialization has already finished and its
    /// apply of callee is never revisited at all - stranding the `partial_apply` of `closure1`.
    ///
    ///   sil @closure0 : (Float, Float) -> Float             // leaf
    ///   sil @closure1 : (Float, (Float) -> Float) -> Float  // bb0(%x, %f): return apply %f(%x)
    ///   sil @closure2 : (Float, (Float) -> Float) -> Float  // bb0(%x, %f): return apply %f(%x)
    ///   sil @callee   : (Float, (Float) -> Float) -> Float  // bb0(%x, %f): return apply %f(%x)
    ///
    ///   caller(%0 : Float, %1 : Float) -> Float:
    ///     %9  = partial_apply @closure0(%0)  : (Float) -> Float
    ///     %12 = partial_apply @closure1(%9)  : (Float) -> Float
    ///     %15 = partial_apply @closure2(%12) : (Float) -> Float
    ///     return apply @callee(%1, %15)
    ///
    /// Round 1 for caller: specialize apply of callee
    ///
    ///   caller(%0 : Float, %1 : Float) -> Float:
    ///     %9  = partial_apply @closure0(%0)  : (Float) -> Float
    ///     %12 = partial_apply @closure1(%9)  : (Float) -> Float
    ///     return apply @specialized_callee_r1(%1, %12)
    ///
    ///   specialized_callee_r1(%0 : Float, %1 : (Float) -> Float) -> Float:
    ///     return apply @closure2(%0, %1) // <-- folded from _/ %3 = partial_apply @closure2(%1)
    ///                                    //                  \ %9 = apply %3(%0)
    ///
    /// Now `%12 = partial_apply @closure1(%9)` in caller is recognized as transitively applied in
    /// `specialized_callee_r1`->`closure2`, making it subject for specialization during the 2nd round.

    for rootClosure in rootClosures {
      let clonedRootClosure = cloner.getClonedValue(of: rootClosure) as! PartialApplyInst
      let argsAreKeptAlive = cloner.context.tryOptimizeApplyOfPartialApply(closure: clonedRootClosure)
      // Unlike a regular closure (which only ever borrows, or independently copies, its captures),
      // an at-most-once closure can have consuming captures. When the fold above transfers a
      // non-Copyable capture to the new direct call, it can't copy it, so it leaves `Undef` in
      // `clonedRootClosure`'s own operand instead. Leaving such a closure's `partial_apply` behind
      // would still run its destructor at runtime, which would release whatever garbage is left
      // in that now-`Undef`'d capture slot.
      if clonedRootClosure.hasCalledAtMostOnceSemantics {
        _ = cloner.context.tryDeleteDeadClosure(closure: clonedRootClosure, needKeepArgsAlive: !argsAreKeptAlive)
      }
    }
  }

  private func addFunctionArgumentsWithoutClosures(using cloner: inout Cloner) {
    let clonedEntryBlock = cloner.getOrCreateEntryBlock()

    for originalArg in callee.arguments where !isClosureArgument(calleeArgument: originalArg) {
      let argType = originalArg.type.getLoweredType(in: cloner.targetFunction)
      let clonedArg = clonedEntryBlock.addFunctionArgument(type: argType, cloner.context)
      clonedArg.copyFlags(from: originalArg, cloner.context)
      cloner.recordFoldedValue(originalArg, mappedTo: clonedArg)
    }
  }

  private func addFunctionArgumentsForCaptures(of closure: PartialApplyInst, using cloner: inout Cloner) {
    for originalClosureArg in closure.arguments {
      let capturedArg = cloner.targetFunction.entryBlock.addFunctionArgument(
        type: originalClosureArg.type.getLoweredType(in: cloner.targetFunction),
        cloner.context)
      if !cloner.isCloned(value: originalClosureArg) {
        cloner.recordFoldedValue(originalClosureArg, mappedTo: capturedArg)
      }
    }
  }

  private func cloneClosures(using cloner: inout Cloner) -> [Value] {
    return closureArguments.map { (closureArgOp, _) in
      let clonedArg = cloner.cloneRecursively(value: closureArgOp.value)

      let originalArg = apply.calleeArgument(of: closureArgOp, in: callee)!

      let clonedValue: Value
      if clonedArg.ownership == .owned && apply.convention(of: closureArgOp) == .directGuaranteed {
        let block = cloner.targetFunction.entryBlock
        let builder = Builder(atEndOf: block, location: block.instructions.last!.location, cloner.context)
        clonedValue = builder.createBeginBorrow(of: clonedArg)
      } else {
        clonedValue = clonedArg
      }
      cloner.recordFoldedValue(originalArg, mappedTo: clonedValue)

      return clonedValue
    }
  }

  func rewriteApply(for specializedFunction: Function, _ context: FunctionPassContext) {
    insertCompensatingDestroysForOwnedClosureArguments(context)

    let newApplyArgs = getNewApplyArguments(context)

    apply.replace(withCallTo: specializedFunction, arguments: newApplyArgs, context)
  }

  private func insertCompensatingDestroysForOwnedClosureArguments(_ context: FunctionPassContext) {
    let builder = Builder(before: apply, context)
    for (argOp, _) in closureArguments where argOp.endsLifetime {
      builder.createDestroyValue(operand: argOp.value)
    }
  }

  private func getNewApplyArguments(_ context: FunctionPassContext) -> [Value] {
    let newCapturedArguments = rootClosures.flatMap { partialApply in
      partialApply.argumentOperands.map { argOp -> Value in
        let capturedArg = argOp.value
        if partialApply.hasCalledAtMostOnceSemantics {
          // An at-most-once closure can consume its captures and always gets a destructor even
          // if it's stack-promoted. So all of the arguments that are consumed have to be passed
          // the same way to the specialized version.
          if capturedArg.ownership != .none &&
              (argOp.ownership == .destroyingConsume || argOp.ownership == .forwardingConsume) {
            if capturedArg.type.isMoveOnly {
              // A move-only value cannot be copied. Transfer it directly to the specialized
              // call instead, and clear the original `partial_apply`'s operand (it isn't being
              // deleted here, only folded into a direct call) so that a later cleanup pass
              // doesn't also try to destroy the same value when it deletes the now-dead
              // `partial_apply`.
              argOp.set(to: Undef.get(type: capturedArg.type, context), context)
              return capturedArg
            }
            return capturedArg.copy(at: partialApply, andMakeAvailableIn: apply.parentBlock, context)
          }
          return capturedArg
        }
        if partialApply.isOnStack || capturedArg.ownership == .none {
          // Non-escaping closures don't consume their captures. Therefore we pass them also as "guaranteed"
          // arguments to the specialized function.
          // Note that because the non-escaping closure was passed to the original function, this guarantees
          // that the lifetime of the captured arguments also extend to at least the apply of the function.
          return capturedArg
        } else {
          // Escaping closures consume their captures. Therefore we pass them as "owned" arguments to the
          // specialized function.
          return capturedArg.copy(at: partialApply, andMakeAvailableIn: apply.parentBlock, context)
        }
      }
    }
    return nonClosureArguments.values + newCapturedArguments
  }

  func uniqueCaptureArguments(_ context: FunctionPassContext) {
    let builder = Builder(before: apply, context)

    // Insert identity cast instructions for all closure arguments to make them unique. We could use any kind
    // of forwarding instruction - we'll delete them afterwards, anyway.
    //
    //   %4 = partial_apply %3(%1, %1) : (Int, Int) -> ()  // %1 is captured twice
    // ->
    //   %2 = unchecked_value_cast %1
    //   %3 = unchecked_value_cast %1
    //   %4 = partial_apply %3(%2, %3) : (Int, Int) -> ()  // all arguments are unique values now!
    //
    for closure in rootClosures {
      for argOp in closure.argumentOperands {
        let capturedValue = argOp.value
        let cast = builder.createUncheckedValueCast(from: capturedValue, to: capturedValue.type)
        argOp.set(to: cast, context)

        // Redirect the mark_dependence base onto the same cast: the cloner only maps the cast to
        // the capture argument, and a shared base must follow its own closure's capture.
        for dependency in capturedDependencies
        where dependency.closure == closure && dependency.markDependence.base == capturedValue {
          dependency.markDependence.baseOperand.set(to: cast, context)
        }
      }
    }
  }

  func unUniqueCaptureArguments(_ context: FunctionPassContext) {
    // Remove the inserted identity casts again.
    for closure in rootClosures {
      for argOp in closure.argumentOperands {
        let cast = argOp.value as! UncheckedValueCastInst
        cast.replace(with: cast.fromValue, context)
      }
    }
  }

  func deleteDeadClosures(_ context: FunctionPassContext) {
    for  (_, closure) in closureArguments where !closure.isDeleted {
      if context.tryDeleteDeadClosure(closure: closure) {
        context.notifyInvalidatedStackNesting()
      }
    }
  }

  private var nonClosureArguments: LazyFilterSequence<OperandArray> {
    apply.argumentOperands.lazy.filter{ argOp in !closureArguments.contains{ $0.0 == argOp } }
  }

  private func isClosureArgument(calleeArgument: FunctionArgument) -> Bool {
    closureArguments.contains { apply.calleeArgument(of: $0.0, in: callee) == calleeArgument }
  }
}

private func isClosureApplied(_ closure: Value) -> Bool {
  var handledFuncs: Set<Function> = []
  return checkRecursivelyIfClosureIsApplied(closure, &handledFuncs)
}

private func checkRecursivelyIfClosureIsApplied(_ closure: Value, _ handledFuncs: inout Set<Function>) -> Bool {
  let recursionBudget = 8
  for use in closure.uses {
    switch use.instruction {

    case is ConvertFunctionInst,
         is ConvertEscapeToNoEscapeInst,
         is MoveValueInst,
         is CopyValueInst:
      if checkRecursivelyIfClosureIsApplied(use.instruction as! SingleValueInstruction, &handledFuncs) {
        return true
      }

    case let pai as PartialApplyInst:
      if pai.isPartialApplyOfThunk {
        // `pai.isPartialApplyOfThunk` implies that the captured closure (`closure` here) is applied in the thunk.
        // If the thunk closure (`pai` here) is applied by itself as well, the closure captured by
        // the thunk closure also becomes effectively applied transitively.
        return checkRecursivelyIfClosureIsApplied(pai, &handledFuncs)
      }

      guard let callee = pai.referencedFunction,
            callee.isDefinition,
            handledFuncs.insert(callee).inserted,
            handledFuncs.count <= recursionBudget,
            let calleeArg = pai.calleeArgument(of: use, in: callee)
      else {
        continue
      }

      // Check if the captured closure (`closure` here) is applied in context of wrapper closure (`pai` here).
      if checkRecursivelyIfClosureIsApplied(calleeArg, &handledFuncs) {
        // If the wrapper closure (`pai` here) is applied by itself complementary to the captured closure applied
        // in context of wrapper closure, the captured closure also becomes effectively applied transitively.
        // For example, treat `%closure1` as applied in `@foo`:
        //   sil @wrapper(%0, %1, %closure) { apply %closure(%1, %0) }
        //   sil @foo(%closure1, %arg1, %arg2) {
        //     %closure2 = partial_apply @wrapper(%closure1)
        //     apply %closure2(%arg1, %arg2)
        //   }
        return checkRecursivelyIfClosureIsApplied(pai, &handledFuncs)
      }

    case let apply as FullApplySite:
      if apply.callee == closure {
        return true
      }

      // Recurse into called function
      if let callee = apply.referencedFunction,
         callee.isDefinition,
         handledFuncs.insert(callee).inserted,
         handledFuncs.count <= recursionBudget,
         let calleeArg = apply.calleeArgument(of: use, in: callee)
      {
        if checkRecursivelyIfClosureIsApplied(calleeArg, &handledFuncs) {
          return true
        }
      }

    default:
      break
    }
  }

  return false
}

/// Add destroys for values which are not consumed, yet.
/// There are two cases of values for which we need this:
/// 1. guaranteed closure arguments
/// 2. operands of copies at the call site
///
/// ```
///   %1 = partial_apply %closure   // is cloned to the specialized function
///   %2 = copy_value %1            // is cloned to the specialized function
///   apply %function(%2) : (@guaranteed () -> ()) -> ()
///   destroy_value %1
///
/// sil @specializedFunction() -> () {
/// bb0:
///   %1 = partial_apply %closure
///   %2 = copy_value %1
///   ... // body
///   destroy_value %2              // case 1: destroy for the guaranteed closure argument
///   destroy_value %1              // case 2: destroy for the operand of the copy
///   return
/// ```
private func addMissingDestroysAtFunctionExits(for clonedArguments: [Value], _ context: FunctionPassContext) {
  var needDestroy = ValueWorklist(context)
  defer { needDestroy.deinitialize() }

  for clonedArg in clonedArguments {
    if let beginBorrow = clonedArg as? BeginBorrowInst {
      Builder.insertCleanupAtFunctionExits(of: beginBorrow.parentFunction, context) { builder in
        builder.createEndBorrow(of: beginBorrow)
      }
      completeLifetime(of: beginBorrow, context)
    }
    findValuesWhichNeedDestroyRecursively(value: clonedArg, needDestroy: &needDestroy)
  }

  while let valueToDestroy = needDestroy.pop() {
    Builder.insertCleanupAtFunctionExits(of: valueToDestroy.parentFunction, context) { builder in
      builder.createDestroyValue(operand: valueToDestroy)
    }
    completeLifetime(of: valueToDestroy, context)
  }
}

private func findValuesWhichNeedDestroyRecursively(value: Value, needDestroy: inout ValueWorklist) {
  if let svi = value as? SingleValueInstruction {
    for op in svi.operands {
      findValuesWhichNeedDestroyRecursively(value: op.value, needDestroy: &needDestroy)
    }
  }
  if value.ownership == .owned && value.uses.endingLifetime.isEmpty {
    needDestroy.pushIfNotVisited(value)
  }
}

private extension ParameterInfo {
  func withSpecializedConvention(for argOp: Operand, in partialApply: PartialApplyInst, callee: Function) -> Self {
    let argType = type.loweredType(in: partialApply.parentFunction)
    let specializedParamConvention = if self.convention.isIndirect {
      self.convention
    } else if partialApply.hasCalledAtMostOnceSemantics {
      if argType.isTrivial(in: callee) {
        ArgumentConvention.directUnowned
      } else if argOp.ownership == .destroyingConsume || argOp.ownership == .forwardingConsume {
        ArgumentConvention.directOwned
      } else {
        ArgumentConvention.directGuaranteed
      }
    } else {
      if argType.isTrivial(in: callee) {
        ArgumentConvention.directUnowned
      } else {
        if partialApply.isOnStack {
          ArgumentConvention.directGuaranteed
        } else {
          ArgumentConvention.directOwned
        }
      }
    }

    return ParameterInfo(
      type: self.type, convention: specializedParamConvention, options: self.options,
      hasLoweredAddresses: self.hasLoweredAddresses)
  }
}

private func numberOfIsolatedParameters(_ params: [ParameterInfo]) -> Int {
  var count = 0
  for param in params where param.hasOption(.isolated) {
    count += 1
  }
  return count
}

private extension PartialApplyInst {
  var isPartialApplyOfThunk: Bool {
    if self.numArguments == 1,
      let fun = self.referencedFunction,
      fun.thunkKind == .reabstractionThunk || fun.isAutodiffSubsetParametersThunk,
      self.arguments[0].type.isLoweredFunction,
      self.arguments[0].type.isReferenceCounted(in: self.parentFunction) || self.callee.type.isThickFunction
    {
      return true
    }
    return false
  }

  var hasOnlySupportedIndirectArguments: Bool {
    self.argumentOperands
      .filter { !$0.value.type.isObject }
      .allSatisfy {
        let conv = self.convention(of: $0)!
        return conv.isInout || (self.isOnStack && conv == .indirectInGuaranteed)
      }
  }

  var allArgumentsCanBeCopied: Bool {
    arguments.allSatisfy { !$0.type.isMoveOnly }
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
