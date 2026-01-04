// RUN: %target-run-simple-swift

// REQUIRES: executable_test

import DifferentiationUnittest
import StdlibUnittest

var NilCoalescingTests = TestSuite("OptionalDifferentiation")

func ??(_ x: Float?, _ y: @autoclosure () -> Float) -> Float {
  if x == nil {
    return y()
  }
  return x!
}

NilCoalescingTests.test("Test") {
  @differentiable(reverse)
  func fooClosure(_ x: Float?, _ y: Float) -> Float {
    return x ??  y * y
  }

  func coalesce(_ x: Float?, _ y: Float) -> Float {
    if x == nil {
      return y
    }
    return x!
  }

  @differentiable(reverse)
  func fooFloat(_ x: Float?, _ y: Float) -> Float {
    return coalesce(x, y * y)
  }

  expectEqual(pullback(at: nil, 3, of: fooFloat)(1),   (0.0, 6.0))
  expectEqual(pullback(at: nil, 3, of: fooClosure)(1), (0.0, 6.0))
}

runAllTests()
