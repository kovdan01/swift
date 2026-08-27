// REQUIRES: executable_test
//
// RUN: %empty-directory(%t)
// RUN: %target-swift-frontend -emit-sil %s -O -D SIL_CHECKS -sil-verify-all -sil-ownership-verify-all -module-name DFSpec -o %t/out.sil
// RUN: %FileCheck %s --check-prefix=CHECK1 < %t/out.sil
// RUN: %FileCheck %s --check-prefix=CHECK1-EVALUATE < %t/out.sil
// RUN: %FileCheck %s --check-prefix=CHECK2 < %t/out.sil
// RUN: %FileCheck %s --check-prefix=CHECK3 < %t/out.sil
// RUN: %FileCheck %s --check-prefix=CHECK3-EVALUATE < %t/out.sil
// RUN: %FileCheck %s --check-prefix=CHECK4 < %t/out.sil
// RUN: %FileCheck %s --check-prefix=CHECK5 < %t/out.sil
// RUN: %FileCheck %s --check-prefix=CHECK6 < %t/out.sil
// RUN: %FileCheck %s --check-prefix=CHECK7-DUPLICATE < %t/out.sil
// RUN: %FileCheck %s --check-prefix=CHECK7-DISTINCT < %t/out.sil
// RUN: %target-build-swift %s -Onone -o %t/none.out
// RUN: %target-build-swift %s -O -o %t/opt.out
// RUN: %target-codesign %t/none.out
// RUN: %target-codesign %t/opt.out
// RUN: %target-run %t/none.out
// RUN: %target-run %t/opt.out

import _Differentiation

@inline(never)
@differentiable(reverse)
public func square(_ x: Double) -> Double {
  x * x
}

// Case 1: ordinary closure specialization exposes a direct call whose
// differentiable_function has ownership none. The captured existential belongs
// to test1Evaluate; the original/JVP/VJP components of square have no captures.

public protocol Test1Protocol {
  func value() -> Double
}

@inline(never)
public func test1WithSquare(
  _ body: (@differentiable(reverse) (Double) -> Double) -> Double
) -> Double {
  body(square)
}

public func test1Entry(_ p: any Test1Protocol) -> Double {
  @inline(never)
  func test1Evaluate(
    _ function: @differentiable(reverse) (Double) -> Double
  ) -> Double {
    gradient(at: p.value(), of: function)
  }

  return test1WithSquare(test1Evaluate)
}

// Check the specialization used by the existential entry point, including its
// call to an evaluator that no longer accepts a differentiable function.
// CHECK1-LABEL: sil shared [noinline] {{.*}}test1WithSquare{{.*}} : $@convention(thin) (@in_guaranteed any Test1Protocol) -> Double {
// CHECK1-NOT: = differentiable_function
// CHECK1-NOT: {{^}}} // end sil function
// CHECK1: function_ref {{.*}} : $@convention(thin) (@in_guaranteed any Test1Protocol) -> Double
// CHECK1-NOT: = differentiable_function
// CHECK1: {{^}}} // end sil function

// Anchor the local evaluator's symbol at test1Entry, so an outer specialization
// whose mangling merely mentions test1Evaluate cannot satisfy this check.
// CHECK1-EVALUATE-LABEL: {{^}}sil {{[^@]*}}@$s6DFSpec10test1Entry{{[^ ]*Evaluate[^ ]*}} : $@convention(thin) (@in_guaranteed any Test1Protocol) -> Double {
// CHECK1-EVALUATE-NOT: = differentiable_function
// CHECK1-EVALUATE: {{^}}} // end sil function

// Case 2: each component is a convert_function of a thin_to_thick_function.
// The component conversions change the nondifferentiated class parameter from
// Test2Base to Test2Derived and must survive specialization.

public class Test2Base {
  public let value: Double

  public init(_ value: Double) {
    self.value = value
  }
}

public class Test2Derived: Test2Base {}

@inline(never)
@differentiable(reverse, wrt: x)
public func test2Evaluate(_ x: Double, _ context: Test2Base) -> Double {
  x * x + context.value
}

@inline(never)
public func test2UseConverted(
  _ x: Double,
  _ context: Test2Derived,
  _ function: @differentiable(reverse)
    (Double, @noDerivative Test2Derived) -> Double
) -> Double {
  function(x, context)
}

public func test2Entry(_ x: Double, _ context: Test2Derived) -> Double {
  test2UseConverted(x, context, test2Evaluate)
}

// CHECK2-LABEL: {{^}}// test2Entry(_:_:)
// CHECK2-NOT: = differentiable_function
// CHECK2-NOT: {{^}}} // end sil function
// CHECK2: // function_ref specialized test2UseConverted(_:_:_:)
// CHECK2-NEXT: %{{[0-9]+}} = function_ref @$s6DFSpec17test2UseConverted{{[^ ]*}} : $@convention(thin) (Double, @guaranteed Test2Derived) -> Double
// CHECK2-NOT: = differentiable_function
// CHECK2: {{^}}} // end sil function

// Case 3: specialize the generic forwarder and the reabstraction thunk for its
// body, then propagate square through that thunk into test3Evaluate.

@inline(never)
public func test3Evaluate(
  _ x: Double,
  _ function: @differentiable(reverse) (Double) -> Double
) -> Double {
  gradient(at: x, of: function)
}

@inline(never)
public func test3Forward<T>(
  _ x: T,
  _ body: (T, @differentiable(reverse) (Double) -> Double) -> T
) -> T {
  body(x, square)
}

public func test3Entry(_ x: Double) -> Double {
  test3Forward(x, test3Evaluate)
}

// Match the specialization with only its scalar parameter. The original public
// generic forwarder must remain callable with an arbitrary body.
// CHECK3-LABEL: sil shared [noinline] {{.*}}test3Forward{{.*}} : $@convention(thin) (Double) -> Double {
// CHECK3-NOT: = differentiable_function
// CHECK3-NOT: {{^}}} // end sil function
// CHECK3: function_ref {{.*}} : $@convention(thin) (Double) -> Double
// CHECK3-NOT: = differentiable_function
// CHECK3: {{^}}} // end sil function

// Also check the scalar evaluator itself, independently of emission order.
// CHECK3-EVALUATE-LABEL: {{^}}sil {{[^@]*}}@$s6DFSpec13test3Evaluate{{[^ ]*}} : $@convention(thin) (Double) -> Double {
// CHECK3-EVALUATE-NOT: = differentiable_function
// CHECK3-EVALUATE: {{^}}} // end sil function

// Case 4: retain coverage of the existing generic differentiable-function path
// with an escaping argument and component reabstraction thunks.

@inline(never)
public func test4Evaluate<T, R>(
  _ x: T,
  _ function: @differentiable(reverse) (T) -> R
) -> T.TangentVector
where
  T: Differentiable,
  R: Differentiable & FloatingPoint,
  R == R.TangentVector
{
  gradient(at: x, of: function)
}

@inline(__always)
public func test4Forward<T, R>(
  _ x: T,
  _ function: @escaping @differentiable(reverse) (T) -> R
) -> T.TangentVector
where
  T: Differentiable,
  R: Differentiable & FloatingPoint,
  R == R.TangentVector
{
  test4Evaluate(x, function)
}

public func test4Entry(_ x: Double) -> Double {
  test4Forward(x, square)
}

// CHECK4-LABEL: sil shared [noinline] {{.*}}test4Evaluate{{.*}} : $@convention(thin) (Double) -> Double {
// CHECK4-NOT: = differentiable_function
// CHECK4: {{^}}} // end sil function

// Case 5: exercise the converted VJP as well as the original component. The
// derivative depends on context.value, so retaining the class conversion and
// forwarding the correct context are observable in both results.

@inline(never)
@differentiable(reverse, wrt: x)
public func test5Evaluate(_ x: Double, _ context: Test2Base) -> Double {
  (x * x + 1) * context.value
}

@inline(never)
public func test5UseConvertedVJP(
  _ x: Double,
  _ context: Test2Derived,
  _ function: @differentiable(reverse)
    (Double, @noDerivative Test2Derived) -> Double
) -> (value: Double, gradient: Double) {
  valueWithGradient(at: x) { function($0, context) }
}

public func test5Entry(
  _ x: Double, _ context: Test2Derived
) -> (value: Double, gradient: Double) {
  test5UseConvertedVJP(x, context, test5Evaluate)
}

// CHECK5-LABEL: {{^}}// test5Entry(_:_:)
// CHECK5-NOT: = differentiable_function
// CHECK5-NOT: {{^}}} // end sil function
// CHECK5: // function_ref specialized test5UseConvertedVJP(_:_:_:)
// CHECK5-NEXT: %{{[0-9]+}} = function_ref @$s6DFSpec20test5UseConvertedVJP{{[^ ]*}} : $@convention(thin) (Double, @guaranteed Test2Derived) -> (Double, Double)
// CHECK5-NOT: = differentiable_function
// CHECK5: {{^}}} // end sil function

// Case 6: the first specialization reconstructs a differentiable_function and
// borrows it for multiple forwarding calls. Subsequent specialization must see
// through that begin_borrow and preserve the lifetime across both calls.

@inline(never)
public func test6Evaluate(
  _ x: Double,
  _ function: @differentiable(reverse) (Double) -> Double
) -> Double {
  gradient(at: x, of: function)
}

@inline(never)
public func test6Forward(
  _ x: Double,
  _ function: @differentiable(reverse) (Double) -> Double
) -> Double {
  test6Evaluate(x, function) + test6Evaluate(x + 1, function)
}

public func test6Entry(_ x: Double) -> Double {
  test6Forward(x, square)
}

// CHECK6-LABEL: sil shared [noinline] {{.*}}test6Forward{{.*}} : $@convention(thin) (Double) -> Double {
// CHECK6-NOT: = differentiable_function
// CHECK6-NOT: {{^}}} // end sil function
// CHECK6: function_ref {{.*}} : $@convention(thin) (Double) -> Double
// CHECK6-NOT: = differentiable_function
// CHECK6-NOT: {{^}}} // end sil function
// CHECK6: = apply
// CHECK6-NOT: = differentiable_function
// CHECK6-NOT: {{^}}} // end sil function
// CHECK6: = apply
// CHECK6-NOT: = differentiable_function
// CHECK6: {{^}}} // end sil function

// Case 7: the same construction can be passed to more than one formal. Also
// exercise a distinct pair of constructions with identical component types, so
// reusing a specialization must preserve which formals share a construction.

@inline(never)
@differentiable(reverse)
public func cube(_ x: Double) -> Double {
  x * x * x
}

@inline(never)
public func test7Evaluate(
  _ x: Double,
  _ first: @differentiable(reverse) (Double) -> Double,
  _ second: @differentiable(reverse) (Double) -> Double
) -> Double {
  gradient(at: x, of: first) - 3 * gradient(at: x + 2, of: second)
}

public func test7Entry(_ x: Double) -> Double {
  let function: @differentiable(reverse) (Double) -> Double = square
  return test7Evaluate(x, function, function)
}

public func test7DistinctEntry(_ x: Double) -> Double {
  test7Evaluate(x, square, cube)
}

// CHECK7-DUPLICATE-LABEL: {{^}}// test7Entry(_:)
// CHECK7-DUPLICATE-NOT: = differentiable_function
// CHECK7-DUPLICATE-NOT: {{^}}} // end sil function
// CHECK7-DUPLICATE: function_ref {{.*}} : $@convention(thin) (Double) -> Double
// CHECK7-DUPLICATE-NOT: = differentiable_function
// CHECK7-DUPLICATE: {{^}}} // end sil function

// CHECK7-DISTINCT-LABEL: {{^}}// test7DistinctEntry(_:)
// CHECK7-DISTINCT-NOT: = differentiable_function
// CHECK7-DISTINCT-NOT: {{^}}} // end sil function
// CHECK7-DISTINCT: function_ref {{.*}} : $@convention(thin) (Double) -> Double
// CHECK7-DISTINCT-NOT: = differentiable_function
// CHECK7-DISTINCT: {{^}}} // end sil function

// Exclude the harness from the SIL run so the checks select the public entry
// points' specializations without additional TestSuite closure specializations.
#if !SIL_CHECKS
import StdlibUnittest

private struct Test1Value: Test1Protocol {
  let x: Double

  func value() -> Double { x }
}

var tests = TestSuite("DifferentiableFunctionSpecializationEdgeCases")

tests.test("ExistentialCapture") {
  for x in [-3.0, 0.0, 2.0, 5.0] {
    expectEqual(2 * x, test1Entry(Test1Value(x: x)))
  }
}

tests.test("ConvertedOriginal") {
  for contextValue in [-2.0, 7.0] {
    let context = Test2Derived(contextValue)
    for x in [-3.0, 0.0, 2.0, 5.0] {
      expectEqual(x * x + contextValue, test2Entry(x, context))
    }
  }
}

tests.test("GenericBodyReabstraction") {
  for x in [-3.0, 0.0, 2.0, 5.0] {
    expectEqual(2 * x, test3Entry(x))
  }
}

tests.test("EscapingGenericComponents") {
  for x in [-3.0, 0.0, 2.0, 5.0] {
    expectEqual(2 * x, test4Entry(x))
  }
}

tests.test("ConvertedVJP") {
  for contextValue in [-2.0, 0.0, 3.0] {
    let context = Test2Derived(contextValue)
    for x in [-3.0, 0.0, 2.0, 5.0] {
      let result = test5Entry(x, context)
      expectEqual((x * x + 1) * contextValue, result.value)
      expectEqual(2 * x * contextValue, result.gradient)
    }
  }
}

tests.test("BorrowedForwarding") {
  for x in [-3.0, 0.0, 2.0, 5.0] {
    expectEqual(2 * x + 2 * (x + 1), test6Entry(x))
  }
}

tests.test("DuplicateArguments") {
  for x in [-3.0, 0.0, 2.0, 5.0] {
    expectEqual(2 * x - 6 * (x + 2), test7Entry(x))
  }
}

tests.test("DistinctArgumentsWithMatchingTypes") {
  for x in [-3.0, 0.0, 2.0, 5.0] {
    expectEqual(2 * x - 9 * (x + 2) * (x + 2), test7DistinctEntry(x))
  }
}

runAllTests()
#endif
