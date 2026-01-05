// RUN: %target-run-simple-swift

// REQUIRES: executable_test

import DifferentiationUnittest
import StdlibUnittest

var NilCoalescingTests = TestSuite("OptionalDifferentiation")

// Note: we do not support throwing closures, so use a custom overload accepting a non-throwing closure
// Test both implicit differentiation and explicit @differentiable(reverse) attr.
func ??(_ x: Float?, _ y: @autoclosure () -> Float) -> Float {
  if x == nil {
    return y()
  }
  return x!
}

@differentiable(reverse)
func ??(_ x: Double?, _ y: @autoclosure () -> Double) -> Double {
  if x == nil {
    return y()
  }
  return x!
}

// Also, test regular closures (both with and without @differentiable(reverse) attr).
func ??(_ x: Float?, _ y: () -> Float) -> Float {
  if x == nil {
    return y()
  }
  return x!
}

@differentiable(reverse)
func ??(_ x: Double?, _ y: () -> Double) -> Double {
  if x == nil {
    return y()
  }
  return x!
}

// Ensure that results are identical for lazy and non-lazy evaluation
func nilCoalescingNonLazy(_ x: Float?, _ y: Float) -> Float {
  if x == nil {
    return y
  }
  return x!
}

func nilCoalescingNonLazy(_ x: Double?, _ y: Double) -> Double {
  if x == nil {
    return y
  }
  return x!
}

NilCoalescingTests.test("Float") {
  @differentiable(reverse)
  func lazy1(_ x: Float?, _ y: Float) -> Float {
    return x ?? y * y
  }

  @differentiable(reverse)
  func lazy2(_ x: Float?, _ y: Float) -> Float {
    return x ?? {y * y}
  }

  @differentiable(reverse)
  func nonLazy(_ x: Float?, _ y: Float) -> Float {
    return nilCoalescingNonLazy(x, y * y)
  }

  let pbLazy1 = pullback(at: Float?(nil), Float(3), of: lazy1)
  let lazyResult1 = pbLazy1(Float(1))

  let pbLazy2 = pullback(at: Float?(nil), Float(3), of: lazy2)
  let lazyResult2 = pbLazy2(Float(1))

  let pbNonLazy = pullback(at: Float?(nil), Float(3), of: nonLazy)
  let nonLazyResult = pbNonLazy(Float(1))

  let expectedResult = (Optional<Float>.TangentVector(0), Float(6))
  expectEqual(lazyResult1, expectedResult)
  expectEqual(lazyResult2, expectedResult)
  expectEqual(nonLazyResult, expectedResult)
}

NilCoalescingTests.test("Double") {
  @differentiable(reverse)
  func lazy1(_ x: Double?, _ y: Double) -> Double {
    return x ?? y * y * y
  }

  @differentiable(reverse)
  func lazy2(_ x: Double?, _ y: Double) -> Double {
    return x ?? {y * y * y}
  }

  @differentiable(reverse)
  func nonLazy(_ x: Double?, _ y: Double) -> Double {
    return nilCoalescingNonLazy(x, y * y * y)
  }

  let pbLazy1 = pullback(at: Double?(nil), Double(4), of: lazy1)
  let lazyResult1 = pbLazy1(Double(1))

  let pbLazy2 = pullback(at: Double?(nil), Double(4), of: lazy2)
  let lazyResult2 = pbLazy2(Double(1))

  let pbNonLazy = pullback(at: Double?(nil), Double(4), of: nonLazy)
  let nonLazyResult = pbNonLazy(Double(1))

  let expectedResult = (Optional<Double>.TangentVector(0), Double(48))
  expectEqual(lazyResult1, expectedResult)
  expectEqual(lazyResult2, expectedResult)
  expectEqual(nonLazyResult, expectedResult)
}

runAllTests()
