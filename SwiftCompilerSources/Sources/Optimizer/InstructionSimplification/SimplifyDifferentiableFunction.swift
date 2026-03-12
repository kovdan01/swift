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

  /// Eliminates `struct_extract`s of an owned `struct` where the `struct_extract`s are inside a
  /// a borrow scope.
  /// This is done by splitting the `begin_borrow` of the whole struct into individual borrows of the fields
  /// (for trivial fields no borrow is needed). And then sinking the `struct` to it's consuming use(s).
  ///
  /// ```
  ///   %3 = struct $S(%nonTrivialField, %trivialField)  // owned
  ///   ...
  ///   %4 = begin_borrow %3
  ///   %5 = struct_extract %4, #S.nonTrivialField
  ///   %6 = struct_extract %4, #S.trivialField
  ///   use %5, %6
  ///   end_borrow %4
  ///   ...
  ///   end_of_lifetime %3
  /// ```
  /// ->
  /// ```
  ///   ...
  ///   %5 = begin_borrow %nonTrivialField
  ///   use %5, %trivialField
  ///   end_borrow %5
  ///   ...
  ///   %3 = struct $S(%nonTrivialField, %trivialField)
  ///   end_of_lifetime %3
  /// ```
  ///
  /// Similarly, split `destructure_struct` of copies:
  /// ```
  ///   %3 = struct $S(%1, %2)  // owned
  ///   ...
  ///   %4 = copy_value %3
  ///   (%5, %6) = destructure_struct %5
  /// ```
  /// ->
  /// ```
  ///   ...
  ///   %5 = copy_value %1
  ///   %6 = copy_value %2
  /// ```
  func simplify(_ context: SimplifyContext) {
    guard ownership == .owned,
          hasOnlyExtractUsesInBorrowScopes()
    else {
      return
    }

    for use in uses {
      switch use.instruction {
      case let beginBorrow as BeginBorrowInst:
        splitAndRemoveStructExtracts(beginBorrow: beginBorrow, context)
//      case let copy as CopyValueInst:
//        splitAndRemoveDestructuresOfCopy(copy: copy, context)
      case is DebugValueInst:
        break
//      case is ApplyInst:
//        break
//      case is ConvertFunctionInst:
//        break
//      case is PartialApplyInst:
//        break
      case is DestroyValueInst:
        assert(use.endsLifetime)
//        sinkToEndOfLifetime(use: use, context)
      default:
        print("DEFAULT USE BEGIN")
        print("\(use.instruction)")
        print("DEFAULT USE MIDDLE 00")
        print("\(use)")
        print("DEFAULT USE MIDDLE 01")
        print("\(self)")
        print("DEFAULT USE END")
        assert(false)
//        assert(use.endsLifetime)
//        sinkToEndOfLifetime(use: use, context)
      }
    }
//    context.erase(instructionIncludingAllUsers: self)
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

  private func splitAndRemoveStructExtracts(beginBorrow: BeginBorrowInst, _ context: SimplifyContext) {
    for differentiableFunctionExtract in beginBorrow.uses.users(ofType: DifferentiableFunctionExtractInst.self) {
      let extracteeType = differentiableFunctionExtract.extractee
      let field = self.extractee(extractee: extracteeType)
      switch differentiableFunctionExtract.ownership {
      case .none:
        if differentiableFunctionExtract.type != field.type {
          let convertBuilder = Builder(before: differentiableFunctionExtract, context)
          let newField = convertBuilder.createConvertFunction(originalFunction: field, resultType: differentiableFunctionExtract.type, withoutActuallyEscaping: false)
          differentiableFunctionExtract.replace(with: newField, context)
        } else {
          differentiableFunctionExtract.replace(with: field, context)
        }
      case .guaranteed:
        let beginBuilder = Builder(before: beginBorrow, context)
        let borrowedField = beginBuilder.createBeginBorrow(of: field,
                                                           isLexical: beginBorrow.isLexical,
                                                           hasPointerEscape: beginBorrow.hasPointerEscape)
        if differentiableFunctionExtract.type != field.type {
          let convertBuilder = Builder(before: differentiableFunctionExtract, context)
          let newField = convertBuilder.createConvertFunction(originalFunction: borrowedField, resultType: differentiableFunctionExtract.type, withoutActuallyEscaping: false)
          differentiableFunctionExtract.replace(with: newField, context)
        } else {
          differentiableFunctionExtract.replace(with: borrowedField, context)
        }
        for endBorrow in beginBorrow.endInstructions {
          let endBuilder = Builder(before: endBorrow, context)
          endBuilder.createEndBorrow(of: borrowedField)
        }
      case .owned, .unowned:
        fatalError("wrong ownership of struct_extract")
      }
    }
  }

  private func sinkToEndOfLifetime(use: Operand, _ context: SimplifyContext) {
    let builder = Builder(before: use.instruction, context)
    let delayedStruct = builder.createStruct(type: type, elements: Array(operands.values))
    use.set(to: delayedStruct, context)
  }
}
