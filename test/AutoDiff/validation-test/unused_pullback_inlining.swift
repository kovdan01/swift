// REQUIRES: executable_test
// RUN: %target-run-simple-swift
//
// Source control for an explicit VJP that only uses a nested VJP's primal
// result. Normal compilation keeps a projection and destroy for the discarded
// pullback until it can remove the closure. The SILOptimizer companion reduces
// the return ABI to trivial values to exercise unsupported projection shapes;
// those reduced ABIs have not been observed from this source.

import _Differentiation

@differentiable(reverse)
@inline(never)
public func cube(_ x: Float) -> Float { x * x * x }

@differentiable(reverse)
public func wrapper(_ x: Float) -> Float {
  if x > 0 { return cube(x) }
  return 2 * cube(x)
}

@derivative(of: wrapper)
public func vjpWrapper(_ x: Float) -> (value: Float, pullback: (Float) -> Float) {
  let onlyValue = valueWithPullback(at: x, of: cube).value
  if x > 0 { return (onlyValue, { 3 * x * x * $0 }) }
  return (2 * onlyValue, { 6 * x * x * $0 })
}

#if !DEFINITIONS_ONLY
import StdlibUnittest
var Tests = TestSuite("UnusedPullbackInlining")
Tests.test("ExplicitDerivativePrimalOnlyUse") {
  for x: Float in [2, -2] {
    let result = valueWithPullback(at: x, of: wrapper)
    expectEqual(x > 0 ? 8 : -16, result.value)
    let slope: Float = x > 0 ? 12 : 24
    expectEqual(slope, result.pullback(1))
    expectEqual(2 * slope, result.pullback(2))
  }
}
runAllTests()
#endif
