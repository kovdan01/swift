// REQUIRES: executable_test
// RUN: %target-run-simple-swift

import _Differentiation

public enum Negative: Error {
  case input
}

@differentiable(reverse)
public func positiveSquare(_ x: Float) throws -> Float {
  if x < 0 { throw Negative.input }
  return x * x
}

@derivative(of: positiveSquare)
@inline(__always)
public func positiveSquareVJP(_ x: Float) throws -> (value: Float, pullback: (Float) -> Float) {
  if x < 0 { throw Negative.input }
  return (x * x, { 2 * x * $0 })
}

@differentiable(reverse, wrt: x)
public func recoveringSquare(_ x: Float, _ flag: Bool) -> Float {
  if flag {
    do { return try positiveSquare(x) }
    catch { return x }
  }
  return 2
}

#if !DEFINITIONS_ONLY
import DifferentiationUnittest
import StdlibUnittest

var tests = TestSuite("ADCSOptionalCallback")
tests.testWithLeakChecking("NormalAndError") {
  let normal = valueWithPullback(at: Float(3), of: { recoveringSquare($0, true) })
  expectEqual(9, normal.value)
  expectEqual(12, normal.pullback(2))

  let error = valueWithPullback(at: Float(-3), of: { recoveringSquare($0, true) })
  expectEqual(-3, error.value)
  expectEqual(2, error.pullback(2))
}
runAllTests()
#endif
