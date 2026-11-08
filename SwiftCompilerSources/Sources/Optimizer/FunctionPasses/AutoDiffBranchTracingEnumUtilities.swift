import AST
import SIL

extension Type {
  func isBranchTracingEnum(in vjp: Function) -> Bool {
    return self.bridged.isAutodiffBranchTracingEnumInVJP(vjp.bridged)
  }
}

extension EnumCase {
  func enumType(in function: Function) -> Type {
    // Preserve the queried enum's generic arguments. A specialized VJP can
    // have an empty generic environment while its BTE declaration is generic.
    remapBranchTracingEnumType(self.enumType, into: function)
  }
}

extension TupleInst {
  // CFG simplification can duplicate a constructor without changing its saved
  // values. Accept that source shape, without equating different callable plans.
  func hasSamePayloadValues(as other: TupleInst) -> Bool {
    type == other.type && operands.elementsEqual(other.operands, by: { $0.value == $1.value })
  }
}

// Information required to specialize one closure stored in a payload tuple of a branch tracing enum case.
struct ClosureInBTE : Equatable {
  let closure: SingleValueInstruction
  let reabstractions: [PartialApplyInst]
  let optionalWrapper: EnumInst?
  let useInPayload: Operand
  let enumCase: EnumCase

  var payloadTuple: TupleInst { useInPayload.instruction as! TupleInst }
  var indexInPayload: Int { useInPayload.index }
}

private func getCapturedArgTypesTupleForClosure(
  closureInBTE: ClosureInBTE, context: FunctionPassContext
) -> AST.`Type` {
  context.getTupleType(elements: closureInBTE.capturedArgs.map { $0.type.rawType })
}

// For a given branch tracing enum type, return an array of branch tracing enum types
// which are used for "predecessor" elements of payload tuples of the given branch tracing enum.
// For example, for the enum below, return [_AD__$xxx_bbB__Pred__xxx, _AD__$xxx_bbC__Pred__xxx].
// Note that the "predecessor" element in the payload tuple is optional and might be not present.
//
//   enum _AD__$xxx_bbA__Pred__xxx {
//     case bbB((predecessor: _AD__$xxx_bbB__Pred__xxx, /* closure types */))
//     case bbC((predecessor: _AD__$xxx_bbC__Pred__xxx, /* closure types */))
//   }
private func getBranchTracingEnumPreds(bteType: Type, in vjp: Function) -> Set<Type> {
  guard let enumCases = bteType.getEnumCases(in: vjp) else {
    return []
  }

  var btePreds = Set<Type>()
  for enumCase in enumCases {
    guard let firstTupleElementType = enumCase.payload!.tupleElements.first else {
      continue
    }
    if firstTupleElementType.isBranchTracingEnum(in: vjp) {
      btePreds.insert(firstTupleElementType)
    }
  }

  return btePreds
}

private func iterateOverBranchTracingEnumPreds(
  bteToPredsDict: inout [Type: Set<Type>],
  currentBTEType: Type,
  in vjp: Function
) {
  let currentBTEPreds = getBranchTracingEnumPreds(bteType: currentBTEType, in: vjp)
  bteToPredsDict[currentBTEType] = currentBTEPreds
  for currentBTEPred in currentBTEPreds {
    if bteToPredsDict[currentBTEPred] == nil {
      iterateOverBranchTracingEnumPreds(
        bteToPredsDict: &bteToPredsDict, currentBTEType: currentBTEPred, in: vjp)
    }
  }
}

// Get branch tracing enum type queue for further specialization. The types in the queue
// are ordered from most nested ones (which contain no predecessor branch tracing enums in their
// case payloads) to least nested ones (which are not stored as predecessor elements in other
// branch tracing enums).
//
// For the enums below, an example of the correct order might be:
// 1. _AD__$xxx_bbD__Pred__xxx
// 2. _AD__$xxx_bbC__Pred__xxx
// 3. _AD__$xxx_bbB__Pred__xxx
// 4. _AD__$xxx_bbA__Pred__xxx
//
//   enum _AD__$xxx_bbA__Pred__xxx {
//     case bbB((predecessor: _AD__$xxx_bbB__Pred__xxx, /* closure types */))
//     case bbC((predecessor: _AD__$xxx_bbC__Pred__xxx, /* closure types */))
//   }
//   enum _AD__$xxx_bbB__Pred__xxx {
//     case bbD((predecessor: _AD__$xxx_bbD__Pred__xxx, /* closure types */))
//   }
//   enum _AD__$xxx_bbC__Pred__xxx {
//     case bbD((predecessor: _AD__$xxx_bbD__Pred__xxx, /* closure types */))
//   }
//   enum _AD__$xxx_bbD__Pred__xxx {
//     case bbE((/* closure types */))
//   }
private func getBranchTracingEnumSpecializationQueue(topBTEType: Type, in vjp: Function) -> [Type] {
  var bteToPredsDict = [Type: Set<Type>]()
  iterateOverBranchTracingEnumPreds(
    bteToPredsDict: &bteToPredsDict,
    currentBTEType: topBTEType,
    in: vjp)
  var bteSpecializationQueue = [Type]()
  let bteCount = bteToPredsDict.count

  for _ in 0..<bteCount {
    for (bteType, btePreds) in bteToPredsDict {
      guard btePreds.isEmpty else {
        continue
      }
      bteSpecializationQueue.append(bteType)
      break
    }
    bteToPredsDict.removeValue(forKey: bteSpecializationQueue.last!)
    for bteType in bteToPredsDict.keys {
      bteToPredsDict[bteType]!.remove(bteSpecializationQueue.last!)
    }
  }
  assert(bteSpecializationQueue.count == bteCount)

  return bteSpecializationQueue
}

// The queried type can originate in the pullback's generic environment. Map
// its AST type directly: lowering an intermediate interface type would lose the
// source instantiation or try to lower parameters outside any environment.
func remapBranchTracingEnumType(_ type: Type, into function: Function) -> Type {
  let interfaceType = type.rawType.mapOutOfEnvironment().getReducedType(
    of: function.loweredFunctionType.substitutedGenericSignatureOfFunctionType.genericSignature)
  return function.mapTypeIntoEnvironment(interfaceType.rawType).canonical
    .loweredTypeWithAbstractionPattern(in: function)
}

private func getBranchTracingEnumLoweredType(ed: EnumDecl, vjp: Function) -> Type {
  vjp.mapTypeIntoEnvironment(ed.declaredInterfaceType).canonical.loweredTypeWithAbstractionPattern(in: vjp)
}

private func getSourceFileFor(derivative: Function) -> SourceFile {
  if let sourceFile = derivative.sourceFile {
    return sourceFile
  }
  return derivative.bridged.getFilesForModule().withElements(ofType: FileUnit.self) {
    for fileUnit in $0 {
      if let sourceFile = fileUnit.asSourceFile {
        return sourceFile
      }
    }
    assert(false)
    return nil
  }!
}

private func cloneGenericParameters(
  canonicalGenericSig: CanonicalGenericSignature, astContext: ASTContext, declContext: DeclContext
) -> GenericParameterList {
  let params = canonicalGenericSig.genericSignature.genericParameters.map {
    assert($0.isGenericTypeParameter)
    return GenericTypeParamDecl.create(
      declContext: declContext,
      name: $0.nameOfGenericTypeParameter,
      depth: $0.depthOfGenericTypeParameter,
      index: $0.indexOfGenericTypeParameter,
      paramKind: $0.kindOfGenericTypeParameter)
  }

  return GenericParameterList.create(
    leftAngleLoc: nil, parameters: params,
    genericWhereClause: nil,
    rightAngleLoc: nil, astContext)
}

// Create specialized payload tuple type for a given enum case. Specialization implies replacing
// closures with tuples of arguments captured by these closures. Note that the last closure might be
// wrapped in an optional (this is used for supporting derivatives of throwing functions).
// Consider the enum case below:
//   case bbB((predecessor: _AD__$xxx_bbB__Pred__xxx, (Float) -> Float, Optional<() -> Float>))
//
// Assuming that the closure at index 1 is capturing one Float argument and the closure at index 2 is
// capturing two Float arguments, the specialized case would look like this:
//   case bbB((predecessor: _AD__$xxx_bbB__Pred__xxx_spec, (Float), Optional<(Float, Float)>))
private func getSpecializedParamDeclForEnumCase(
  enumCase: EnumCase,
  closuresInBTE: [ClosureInBTE],
  newEDName: inout String,
  specializedBTEDict: [Type: Type],
  topVJP: Function,
  context: FunctionPassContext
) -> (parameter: ParamDecl, payloadType: AST.`Type`) {
  let oldPayloadTupleElementTypes = enumCase.payload!.tupleElements
  let closuresInBTEForCase = closuresInBTE.filter{ $0.enumCase.index == enumCase.index }

  var nameSuffix: String = ""
  var newPayloadTupleElementTypes = [(label: Identifier, type: AST.`Type`)]()

  for (elementIndex, oldElementType) in oldPayloadTupleElementTypes.enumerated() {
    var newElementType: AST.`Type`
    let closuresInBTEForCaseAndPayloadIndex = closuresInBTEForCase.filter({ $0.indexInPayload == elementIndex })

    if let first = closuresInBTEForCaseAndPayloadIndex.first {
      // Duplicated constructors must save the same ordered SSA values.
      // Analysis rejects differing plans for the same bound case.
      let captureTypes = first.capturedArgs.map { $0.type }
      for info in closuresInBTEForCaseAndPayloadIndex.dropFirst() {
        assert(first.payloadTuple.hasSamePayloadValues(as: info.payloadTuple))
        assert(captureTypes == info.capturedArgs.map { $0.type })
      }
    }
    if let closureInBTE = closuresInBTEForCaseAndPayloadIndex.first {
      nameSuffix += "_\(elementIndex)"
      newElementType = getCapturedArgTypesTupleForClosure(
        closureInBTE: closureInBTE, context: context)
      if oldElementType.isOptional {
        assert(elementIndex + 1 == oldPayloadTupleElementTypes.count)
        newElementType = newElementType.optionalType
      }
    } else {
      newElementType = oldElementType.rawType
      if elementIndex == 0 && oldElementType.isBranchTracingEnum(in: topVJP) {
        newElementType = specializedBTEDict[oldElementType]!.rawType
      }
    }
    newPayloadTupleElementTypes.append((label: oldPayloadTupleElementTypes.label(at: elementIndex), type: newElementType))
  }

  let enumElementDecl = enumCase.enumElementDecl

  if !nameSuffix.isEmpty {
    newEDName += "_\(enumElementDecl.name)\(nameSuffix)"
  }

  let newParamDecl = enumElementDecl.parameterList.singleElement!.cloneWithoutType()
  let newPayloadType = context.getTupleType(elements: newPayloadTupleElementTypes).mapOutOfEnvironment()
  newParamDecl.setInterfaceType(type: newPayloadType)

  return (newParamDecl, newPayloadType)
}

// Create a specialized EnumDecl for a given branch tracing enum. See getSpecializedParamDeclForEnumCase
// for more info on how each enum case payload is specialized.
// The specialized enum name includes the changed payload positions followed by
// a digest of the complete canonical schema. For example:
//   _AD__$xxx_bbA__Pred__xxx_spec_bbB_1_bbC_0_1_<schema digest>
// This means that for enum case bbB closure at index 1 in payload tuple was replaced with
// its captured arguments tuple, and for enum case bbC closures at indexes 0 and 1 were
// replaced the same way.
private func autodiffSpecializeBranchTracingEnum(
  bteType: Type, topVJP: Function,
  closuresInBTE: [ClosureInBTE],
  specializedBTEDict: [Type: Type],
  context: FunctionPassContext
) -> Type {
  assert(specializedBTEDict[bteType] == nil)
  closuresInBTE.forEach { assert($0.enumCase.enumType(in: topVJP) == bteType) }

  let oldED = bteType.nominal as! EnumDecl
  let declContext = oldED.parentDeclContext!
  let astContext = declContext.astContext

  var newEDName: String = oldED.name.string + "_spec"

  let newCasePayloads = bteType.getEnumCases(in: topVJP)!.map {
    getSpecializedParamDeclForEnumCase(
      enumCase: $0, closuresInBTE: closuresInBTE, newEDName: &newEDName,
      specializedBTEDict: specializedBTEDict, topVJP: topVJP, context: context)
  }

  let canonicalGenericSig = topVJP.genericSignature.canonicalSignature
  // Include the original instantiation even when its generic arguments do not
  // occur in a payload. The complete ordered payload schema includes preserved
  // closures, optional wrappers, and already-specialized predecessor enums.
  let schema = context.getTupleType(elements:
    [bteType.rawType.mapOutOfEnvironment()] + newCasePayloads.map { $0.payloadType })
  if let existing = context.lookupAutoDiffSpecializedBranchTracingEnum(schema: schema, in: topVJP) {
    return getBranchTracingEnumLoweredType(ed: existing, vjp: topVJP)
  }
  newEDName = context.mangleAutoDiffSpecializedBranchTracingEnum(
    prefix: newEDName, schema: schema, in: topVJP)

  let genericParams =
    if canonicalGenericSig.isEmpty { GenericParameterList?(nil) }
    else { cloneGenericParameters(canonicalGenericSig: canonicalGenericSig, astContext: astContext, declContext: declContext) }

  let newED = EnumDecl.create(
    declContext: declContext,
    enumKeywordLoc: nil,
    name: newEDName,
    nameLoc: nil,
    genericParamList: genericParams,
    inheritedTypes: [],
    genericWhereClause: nil,
    braceRange: SourceRange(start: nil),
    astContext)

  newED.setImplicit()
  if !canonicalGenericSig.isEmpty {
    newED.setGenericSignature(canonicalGenericSig.genericSignature)
  }

  for (idx, enumCase) in bteType.getEnumCases(in: topVJP)!.enumerated() {
    let newEED = EnumElementDecl.create(
      declContext: newED,
      name: enumCase.enumElementDecl.baseIdentifier, nameLoc: nil,
      parameterList: ParameterList.create(
        leftParenLoc: nil, parameters: [newCasePayloads[idx].parameter],
        rightParenLoc: nil, astContext),
      equalsLoc: nil, rawValue: nil, astContext)
    newEED.setImplicit()
    newED.add(member: newEED)
  }

  // TODO: we should probably copy access level from old EnumDecl, but
  // for some reason it results in crashes for many existing tests
  newED.setAccess(swift.AccessLevel.public)
  getSourceFileFor(derivative: topVJP).addTopLevelDecl(newED.bridgedDecl)
  context.registerAutoDiffSpecializedBranchTracingEnum(newED, schema: schema, in: topVJP)

  return getBranchTracingEnumLoweredType(ed: newED, vjp: topVJP)
}

// Specialize all branch tracing enums which store control-flow graph info in topVJP.
func autodiffSpecializeBranchTracingEnums(
  topVJP: Function, topBTE: Type, closuresInBTE: [ClosureInBTE],
  context: FunctionPassContext
) -> [Type: Type] {
  let bteSpecializationQueue = getBranchTracingEnumSpecializationQueue(
    topBTEType: remapBranchTracingEnumType(topBTE, into: topVJP), in: topVJP)

  var closuresInBTEByBTE = [Type: [ClosureInBTE]]()
  for closureInBTE in closuresInBTE {
    let enumType = closureInBTE.enumCase.enumType(in: topVJP)
    closuresInBTEByBTE[enumType] = (closuresInBTEByBTE[enumType] ?? []) + [closureInBTE]
  }

  var specializedBTEDict = [Type: Type]()
  for bteType in bteSpecializationQueue {
    specializedBTEDict[bteType] = autodiffSpecializeBranchTracingEnum(
      bteType: bteType, topVJP: topVJP,
      closuresInBTE: closuresInBTEByBTE[bteType] ?? [],
      specializedBTEDict: specializedBTEDict, context: context)
  }

  return specializedBTEDict
}
