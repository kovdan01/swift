//===--- GenDiffWitness.cpp - IRGen for differentiability witnesses -------===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2020 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//
//
// This file implements IR generation for SIL differentiability witnesses.
//
//===----------------------------------------------------------------------===//

#include "swift/AST/PrettyStackTrace.h"
#include "swift/SIL/SILDifferentiabilityWitness.h"

#include "ConstantBuilder.h"
#include "IRGenModule.h"

using namespace swift;
using namespace irgen;

void IRGenModule::emitSILDifferentiabilityWitness(
    SILDifferentiabilityWitness *dw) {
  PrettyStackTraceDifferentiabilityWitness _st(
      "emitting differentiability witness for", dw->getKey());
  // Don't emit declarations.
  if (dw->isDeclaration())
    return;
  // Don't emit `public_external` witnesses.
  if (dw->getLinkage() == SILLinkage::PublicExternal)
    return;
  ConstantInitBuilder builder(*this);
  auto diffWitnessContents = builder.beginStruct();
  assert(dw->getJVP() &&
         "Differentiability witness definition should have JVP");
  assert(dw->getVJP() &&
         "Differentiability witness definition should have VJP");
  // A first-class async function value points at an async function pointer
  // descriptor, not at the implementation's entry address. Witness extraction
  // uses the derivative's function type to interpret these fields, just as an
  // ordinary function_ref does. Preserve code pointers for synchronous entries.
  auto addDerivative = [&](SILFunction *derivative) {
    if (derivative->isAsync())
      diffWitnessContents.add(getAddrOfAsyncFunctionPointer(derivative));
    else
      diffWitnessContents.add(
          getAddrOfSILFunction(derivative, NotForDefinition));
  };
  addDerivative(dw->getJVP());
  addDerivative(dw->getVJP());
  getAddrOfDifferentiabilityWitness(
      dw, diffWitnessContents.finishAndCreateFuture());
}
