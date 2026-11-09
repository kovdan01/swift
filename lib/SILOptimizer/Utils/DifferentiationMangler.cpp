//===----- DifferentiationMangler.cpp --------- differentiation -*- C++ -*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2017 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#include "swift/SILOptimizer/Utils/DifferentiationMangler.h"
#include "swift/AST/AutoDiff.h"
#include "swift/AST/GenericSignature.h"
#include "swift/Basic/MD5Stream.h"
#include "swift/SILOptimizer/Utils/SpecializationMangler.h"

using namespace swift;
using namespace Mangle;

std::string Mangle::mangleAutoDiffSpecializedBranchTracingEnum(
    StringRef prefix, CanType schema, CanGenericSignature signature) {
  ASTMangler mangler(schema->getASTContext());
  auto key = mangler.mangleTypeForDebugger(schema, signature);
  if (signature) {
    ASTMangler signatureMangler(schema->getASTContext());
    key += signatureMangler.mangleGenericSignature(signature);
  }
  // A digest keeps nested predecessor schemas from making names grow
  // exponentially. Layout consistency is asserted when the name is reused.
  MD5Stream stream;
  stream << key;
  llvm::MD5::MD5Result digest;
  stream.final(digest);
  SmallString<32> suffix;
  llvm::MD5::stringifyResult(digest, suffix);
  return (prefix + "_" + suffix).str();
}

std::string Mangle::mangleAutoDiffSpecializedPullback(
    SILFunction *pullback, SILFunction *vjp, SILType enumType, unsigned argIndex) {
  ASTMangler typeMangler(vjp->getASTContext());
  auto type = enumType.getASTType()->mapTypeOutOfEnvironment()->getCanonicalType();
  auto typeName = typeMangler.mangleTypeForDebugger(
      type, vjp->getGenericSignature().getCanonicalSignature());
  // A VJP name already identifies its generic/function specialization choices.
  // A further ADCS round changes its tape type as function fields become captures.
  auto name = vjp->getName();
  auto key = std::to_string(name.size()) + ":" + name.str() + typeName;
  FunctionSignatureSpecializationMangler mangler(
      pullback->getASTContext(), Demangle::SpecializationPass::ClosureSpecializer,
      pullback->getSerializedKind(), pullback);
  mangler.setArgumentAutoDiffBranchTracingEnum(argIndex, key);
  return mangler.mangle();
}

/// Mangles the generic signature and get its mangling tree. This is necessary
/// because the derivative generic signature's requirements may contain names
/// which repeat the contents of the original function name. To follow Swift's
/// mangling scheme, these repetitions must be mangled as substitutions.
/// Therefore, we build mangling trees in `DifferentiationMangler` and let the
/// remangler take care of substitutions.
static NodePointer mangleGenericSignatureAsNode(GenericSignature sig,
                                                Demangler &demangler) {
  if (!sig)
    return nullptr;
  ASTMangler sigMangler(sig->getASTContext());
  auto mangledGenSig = sigMangler.mangleGenericSignature(sig);
  auto demangledGenSig = demangler.demangleSymbol(mangledGenSig);
  assert(demangledGenSig->getKind() == Node::Kind::Global);
  assert(demangledGenSig->getNumChildren() == 1);
  auto result = demangledGenSig->getFirstChild();
  assert(result->getKind() == Node::Kind::DependentGenericSignature);
  return result;
}

static NodePointer mangleAutoDiffFunctionAsNode(
    StringRef originalName, Demangle::AutoDiffFunctionKind kind,
    const AutoDiffConfig &config, Demangler &demangler) {
  assert(isMangledName(originalName));
  auto demangledOrig = demangler.demangleSymbol(originalName);
  assert(demangledOrig && "Should only be called when the original "
         "function has a mangled name");
  assert(demangledOrig->getKind() == Node::Kind::Global);
  auto derivativeGenericSignatureNode = mangleGenericSignatureAsNode(
      config.derivativeGenericSignature, demangler);
  auto *adFunc = demangler.createNode(Node::Kind::AutoDiffFunction);
  for (auto *child : *demangledOrig)
    adFunc->addChild(child, demangler);
  if (derivativeGenericSignatureNode)
    adFunc->addChild(derivativeGenericSignatureNode, demangler);
  adFunc->addChild(
      demangler.createNode(
          Node::Kind::AutoDiffFunctionKind, (Node::IndexType)kind),
      demangler);
  adFunc->addChild(
      demangler.createNode(
          Node::Kind::IndexSubset, config.parameterIndices->getString()),
      demangler);
  adFunc->addChild(
      demangler.createNode(
          Node::Kind::IndexSubset, config.resultIndices->getString()),
      demangler);
  auto root = demangler.createNode(Node::Kind::Global);
  root->addChild(adFunc, demangler);
  return root;
}

std::string DifferentiationMangler::mangleAutoDiffFunction(
    StringRef originalName, Demangle::AutoDiffFunctionKind kind,
    const AutoDiffConfig &config) {
  // If the original function is mangled, mangle the tree.
  if (isMangledName(originalName)) {
    Demangler demangler;
    auto node = mangleAutoDiffFunctionAsNode(
        originalName, kind, config, demangler);
    auto mangling = Demangle::mangleNode(node, Flavor);
    assert(mangling.isSuccess());
    return mangling.result();
  }
  // Otherwise, treat the original function symbol as a black box and just
  // mangle the other parts.
  beginManglingWithoutPrefix();
  appendOperator(originalName);
  appendAutoDiffFunctionParts("TJ", kind, config);
  return finalize();
}

// Returns the mangled name for a derivative function of the given kind.
std::string DifferentiationMangler::mangleDerivativeFunction(
    StringRef originalName, AutoDiffDerivativeFunctionKind kind,
    const AutoDiffConfig &config) {
  return mangleAutoDiffFunction(
      originalName, getAutoDiffFunctionKind(kind), config);
}

// Returns the mangled name for a derivative function of the given kind.
std::string DifferentiationMangler::mangleLinearMap(
    StringRef originalName, AutoDiffLinearMapKind kind,
    const AutoDiffConfig &config) {
  return mangleAutoDiffFunction(
      originalName, getAutoDiffFunctionKind(kind), config);
}

static NodePointer mangleDerivativeFunctionSubsetParametersThunkAsNode(
    StringRef originalName, Type toType, Demangle::AutoDiffFunctionKind kind,
    IndexSubset *fromParamIndices, IndexSubset *fromResultIndices,
    IndexSubset *toParamIndices, Demangler &demangler) {
  assert(isMangledName(originalName));
  auto demangledOrig = demangler.demangleSymbol(originalName);
  assert(demangledOrig && "Should only be called when the original "
         "function has a mangled name");
  assert(demangledOrig->getKind() == Node::Kind::Global);
  auto *thunk = demangler.createNode(Node::Kind::AutoDiffSubsetParametersThunk);
  for (auto *child : *demangledOrig)
    thunk->addChild(child, demangler);
  NodePointer toTypeNode = nullptr;
  {
    ASTMangler typeMangler(toType->getASTContext());
    toTypeNode = demangler.demangleType(
        typeMangler.mangleTypeWithoutPrefix(toType));
    assert(toTypeNode && "Cannot demangle the to-type as node");
  }
  thunk->addChild(toTypeNode, demangler);
  thunk->addChild(
      demangler.createNode(
          Node::Kind::AutoDiffFunctionKind, (Node::IndexType)kind),
      demangler);
  thunk->addChild(
      demangler.createNode(
          Node::Kind::IndexSubset, fromParamIndices->getString()),
      demangler);
  thunk->addChild(
      demangler.createNode(
          Node::Kind::IndexSubset, fromResultIndices->getString()),
      demangler);
  thunk->addChild(
      demangler.createNode(
          Node::Kind::IndexSubset, toParamIndices->getString()),
      demangler);
  auto root = demangler.createNode(Node::Kind::Global);
  root->addChild(thunk, demangler);
  return root;
}

std::string
DifferentiationMangler::mangleDerivativeFunctionSubsetParametersThunk(
    StringRef originalName, CanType toType,
    AutoDiffDerivativeFunctionKind linearMapKind,
    IndexSubset *fromParamIndices, IndexSubset *fromResultIndices,
    IndexSubset *toParamIndices) {
  beginMangling();
  auto kind = getAutoDiffFunctionKind(linearMapKind);
  // If the original function is mangled, mangle the tree.
  if (isMangledName(originalName)) {
    Demangler demangler;
    auto node = mangleDerivativeFunctionSubsetParametersThunkAsNode(
        originalName, toType, kind, fromParamIndices, fromResultIndices,
        toParamIndices, demangler);
    auto mangling = Demangle::mangleNode(node, Flavor);
    assert(mangling.isSuccess());
    return mangling.result();
  }
  // Otherwise, treat the original function symbol as a black box and just
  // mangle the other parts.
  beginManglingWithoutPrefix();
  appendOperator(originalName);
  appendType(toType, nullptr);
  auto kindCode = (char)kind;
  appendOperator("TJS", StringRef(&kindCode, 1));
  appendIndexSubset(fromParamIndices);
  appendOperator("p");
  appendIndexSubset(fromResultIndices);
  appendOperator("r");
  appendIndexSubset(toParamIndices);
  appendOperator("P");
  return finalize();
}

std::string DifferentiationMangler::mangleLinearMapSubsetParametersThunk(
    CanType fromType, AutoDiffLinearMapKind linearMapKind,
    IndexSubset *fromParamIndices, IndexSubset *fromResultIndices,
    IndexSubset *toParamIndices) {
  beginMangling();
  appendType(fromType, nullptr);
  auto functionKindCode = (char)getAutoDiffFunctionKind(linearMapKind);
  appendOperator("TJS", StringRef(&functionKindCode, 1));
  appendIndexSubset(fromParamIndices);
  appendOperator("p");
  appendIndexSubset(fromResultIndices);
  appendOperator("r");
  appendIndexSubset(toParamIndices);
  appendOperator("P");
  return finalize();
}
