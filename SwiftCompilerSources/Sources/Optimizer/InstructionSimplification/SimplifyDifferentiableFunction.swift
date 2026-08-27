//===--- SimplifyDifferentiableFunction.swift -----------------------------===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2026 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

import SIL

extension DifferentiableFunctionInst : SILCombineSimplifiable {

  /// Eliminates `differentiable_function_extract`s of an owned `differentiable_function` where the
  /// `differentiable_function_extract`s are inside a borrow scope.
  /// This is done by splitting the `begin_borrow` of the whole `differentiable_function` into individual borrows of the extractees
  /// (for trivial extractees no borrow is needed). If needed, `convert_function` is emitted to cast the extractee to the
  /// ABI-compatible expected type.
  ///
  /// ```
  ///   %3 = differentiable_function [parameters X] [results X] %orig with_derivative { %jvp, %vjp }
  ///   ...
  ///   %4 = begin_borrow %3
  ///   %5 = differentiable_function_extract [original] %4
  ///   %6 = differentiable_function_extract [jvp] %4
  ///   %7 = differentiable_function_extract [vjp] %4
  ///   use %5, %6, %7
  ///   end_borrow %4
  ///   ...
  ///   end_of_lifetime %3
  /// ```
  /// ->
  /// ```
  ///   %3 = differentiable_function [parameters X] [results X] %orig with_derivative { %jvp, %vjp }
  ///   ...
  ///   %5  = begin_borrow %orig
  ///   %5x = convert_function %5 to XXX
  ///   use %5x
  ///   end_borrow %5
  ///   ...
  ///   %6  = begin_borrow %jvp
  ///   %6x = convert_function %6 to XXX
  ///   use %6x
  ///   end_borrow %6
  ///   ...
  ///   %7  = begin_borrow %vjp
  ///   %7x = convert_function %7 to XXX
  ///   use %7x
  ///   end_borrow %7
  ///   ...
  ///   end_of_lifetime %3
  /// ```
  func simplify(_ context: SimplifyContext) {
    print("[DFEOPT] simplify")
    print("[DFEOPT] instruction: \(self)")
    print("[DFEOPT] ownership: \(ownership)")

    for use in uses {
      print(
        "[DFEOPT] direct use: \(use.instruction), " +
        "endsLifetime=\(use.endsLifetime)")
    }

    if ownership == .none {
      print("[DFEOPT] taking non-OSSA noescape/dependence path")
      splitAndRemoveNoEscapeDependentExtracts(context)
      return
    }

    guard ownership == .owned else {
      print("[DFEOPT] reject: unsupported ownership")
      return
    }

    let hasBorrowScopedExtracts = hasOnlyExtractUsesInBorrowScopes()
    print("[DFEOPT] borrow-scoped pattern: \(hasBorrowScopedExtracts)")

    if hasBorrowScopedExtracts {
      print("[DFEOPT] taking existing borrow-scoped path")
      for use in uses {
        switch use.instruction {
        case let beginBorrow as BeginBorrowInst:
          splitAndRemoveExtracts(beginBorrow: beginBorrow, context)
        case is DebugValueInst:
          break
        default:
          assert(use.endsLifetime)
        }
      }
      return
    }

    print("[DFEOPT] taking noescape/dependence path")
    splitAndRemoveNoEscapeDependentExtracts(context)
  }

  private func hasOnlyExtractUsesInBorrowScopes() -> Bool {
    var hasExtract = false

    for use in uses.ignoreDebugUses {
      switch use.instruction {
      case let beginBorrow as BeginBorrowInst:
        for borrowUse in beginBorrow.uses.ignoreDebugUses {
          switch borrowUse.instruction {
          case is EndBorrowInst:
            break
          case is DifferentiableFunctionExtractInst:
            hasExtract = true
          case let convert as ConvertFunctionInst:
            for convertUse in convert.uses.ignoreDebugUses {
              switch convertUse.instruction {
              case is DifferentiableFunctionExtractInst:
                hasExtract = true
              default:
                return false
              }
            }
          default:
            return false
          }
        }
      default:
        guard use.endsLifetime else {
          return false
        }
      }
    }
    return hasExtract
  }

  private func processExtract(
      differentiableFunctionExtract: DifferentiableFunctionExtractInst,
      beginBorrow: BeginBorrowInst?,
      lifetimeEndInstructions: [Instruction] = [],
      _ context: SimplifyContext
    ) {
    guard let extractee = self.getExtractee(
      extractee: differentiableFunctionExtract.extractee
    ) else {
      print("[DFEOPT] reject: getExtractee failed")
      print("[DFEOPT] extract: \(differentiableFunctionExtract)")
      return
    }

    print("[DFEOPT] matched extractee: \(extractee)")
    // If the extractee has non-trivial ownership, it is consumed by the differentiable_function instruction.
    // We must copy it before the consumption point so the copy remains live afterward.
    let effectiveExtractee: Value
    let needsDestroy: Bool
    if extractee.ownership != .none {
      let copyBuilder = Builder(before: self, context)
      effectiveExtractee = copyBuilder.createCopyValue(operand: extractee)
      needsDestroy = true
    } else {
      effectiveExtractee = extractee
      needsDestroy = false
    }

    switch differentiableFunctionExtract.ownership {
    case .none:
      if differentiableFunctionExtract.type == effectiveExtractee.type {
        print("[DFEOPT] direct replacement; types are equal")
        differentiableFunctionExtract.replace(with: effectiveExtractee, context)
      } else if beginBorrow == nil {
        // The matched extract operates on the noescape result of:
        //
        //   convert_escape_to_noescape %differentiable_function
        //
        // Therefore, its component also has a noescape result type. A
        // convert_function cannot change escapeness.
        print("[DFEOPT] emitting convert_escape_to_noescape")
        print("[DFEOPT] source type: \(effectiveExtractee.type)")
        print("[DFEOPT] result type: \(differentiableFunctionExtract.type)")

        let convertBuilder = Builder(
          before: differentiableFunctionExtract, context)
        let newField = convertBuilder.createConvertEscapeToNoEscape(
          originalFunction: effectiveExtractee,
          resultType: differentiableFunctionExtract.type,
          isLifetimeGuaranteed: true)

        print("[DFEOPT] created replacement: \(newField)")
        differentiableFunctionExtract.replace(with: newField, context)
      } else {
        // Existing borrow-scoped optimization. Escapeness is unchanged here.
        print("[DFEOPT] emitting convert_function")
        print("[DFEOPT] source type: \(effectiveExtractee.type)")
        print("[DFEOPT] result type: \(differentiableFunctionExtract.type)")

        let convertBuilder = Builder(
          before: differentiableFunctionExtract, context)
        let newField = convertBuilder.createConvertFunction(
          originalFunction: effectiveExtractee,
          resultType: differentiableFunctionExtract.type,
          withoutActuallyEscaping: false)

        print("[DFEOPT] created replacement: \(newField)")
        differentiableFunctionExtract.replace(with: newField, context)
      }

    case .guaranteed:
      let borrowScopeStart: Instruction = beginBorrow ?? self
      let beginBuilder = Builder(before: borrowScopeStart, context)
      let borrowedField = beginBuilder.createBeginBorrow(
        of: effectiveExtractee,
        isLexical: beginBorrow?.isLexical ?? false,
        hasPointerEscape: beginBorrow?.hasPointerEscape ?? false)

      if differentiableFunctionExtract.type != effectiveExtractee.type {
        let convertBuilder = Builder(before: differentiableFunctionExtract, context)
        let newField = convertBuilder.createConvertFunction(
          originalFunction: borrowedField, resultType: differentiableFunctionExtract.type,
          withoutActuallyEscaping: false)
        differentiableFunctionExtract.replace(with: newField, context)
      } else {
        differentiableFunctionExtract.replace(with: borrowedField, context)
      }

      if let beginBorrow {
        for endBorrow in beginBorrow.endInstructions {
          let endBuilder = Builder(before: endBorrow, context)
          endBuilder.createEndBorrow(of: borrowedField)
          if needsDestroy {
            endBuilder.createDestroyValue(operand: effectiveExtractee)
          }
        }
      } else {
        for lifetimeEnd in lifetimeEndInstructions {
          let endBuilder = Builder(before: lifetimeEnd, context)
          endBuilder.createEndBorrow(of: borrowedField)
          if needsDestroy {
            endBuilder.createDestroyValue(operand: effectiveExtractee)
          }
        }
      }

    case .owned, .unowned:
      fatalError("wrong ownership of differentiable_function_extract")
    }
  }

  private func splitAndRemoveNoEscapeDependentExtracts(
    _ context: SimplifyContext
  ) {
    print("[DFEOPT] entered noescape/dependence matcher")

    // This is non-OSSA SIL. Explicit release_value instructions manage the
    // lifetime, and Operand.endsLifetime is not applicable.

    // Snapshot before replacing extracts because replacement mutates use lists.
    var extracts: [DifferentiableFunctionExtractInst] = []
    var conversionCount = 0
    var dependenceCount = 0

    for conversion in uses.users(ofType: ConvertEscapeToNoEscapeInst.self) {
      conversionCount += 1
      print("[DFEOPT] conversion: \(conversion)")

      for conversionUse in conversion.uses {
        print("[DFEOPT] conversion use: \(conversionUse.instruction)")
      }

      for dependence in conversion.uses.users(ofType: MarkDependenceInst.self) {
        dependenceCount += 1
        print("[DFEOPT] dependence: \(dependence)")
        print("[DFEOPT] value matches: \(dependence.value == conversion)")
        print("[DFEOPT] base matches: \(dependence.base == self)")

        guard dependence.value == conversion,
              dependence.base == self
        else {
          print("[DFEOPT] reject: dependence operands do not match")
          continue
        }

        for dependenceUse in dependence.uses {
          print("[DFEOPT] dependence use: \(dependenceUse.instruction)")
        }

        let matchedExtracts = Array(
          dependence.uses.users(
            ofType: DifferentiableFunctionExtractInst.self))

        print("[DFEOPT] extracts on dependence: \(matchedExtracts.count)")
        extracts.append(contentsOf: matchedExtracts)
      }
    }

    print("[DFEOPT] conversions found: \(conversionCount)")
    print("[DFEOPT] dependences found: \(dependenceCount)")
    print("[DFEOPT] total extracts found: \(extracts.count)")

    for extract in extracts {
      print("[DFEOPT] processing extract: \(extract)")
      print("[DFEOPT] extract ownership: \(extract.ownership)")

      guard extract.ownership == .none else {
        print("[DFEOPT] reject extract: expected non-OSSA ownership")
        continue
      }

      processExtract(
        differentiableFunctionExtract: extract,
        beginBorrow: nil,
        context)
    }
  }

  private func splitAndRemoveExtracts(beginBorrow: BeginBorrowInst, _ context: SimplifyContext) {
    for differentiableFunctionExtract in beginBorrow.uses.users(ofType: DifferentiableFunctionExtractInst.self) {
      processExtract(
        differentiableFunctionExtract: differentiableFunctionExtract,
        beginBorrow: beginBorrow, context)
    }

    for convertFunction in beginBorrow.uses.users(ofType: ConvertFunctionInst.self) {
      for differentiableFunctionExtract in convertFunction.uses.users(ofType: DifferentiableFunctionExtractInst.self) {
        processExtract(
          differentiableFunctionExtract: differentiableFunctionExtract,
          beginBorrow: beginBorrow, context)
      }
    }
  }
}
