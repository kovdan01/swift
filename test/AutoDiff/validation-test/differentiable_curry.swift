// RUN: %target-run-simple-swift

// REQUIRES: executable_test

import DifferentiationUnittest
import StdlibUnittest

var NilCoalescingTests = TestSuite("OptionalDifferentiation")

// Note: we do not support throwing closures, so use a custom overload accepting a non-throwing closure
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

  let pbClosure = pullback(at: Float?(nil), Float(3), of: fooClosure)
  let resultGotClosure = pbClosure(Float(1))

  let pbFloat = pullback(at: Float?(nil), Float(3), of: fooFloat)
  let resultGotFloat = pbFloat(Float(1))

  let resultExpected = (Optional<Float>.TangentVector(0), Float(6))
  expectEqual(resultGotClosure, resultExpected)
  expectEqual(resultGotFloat, resultExpected)
}

runAllTests()
