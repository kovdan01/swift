// REQUIRES: executable_test
// RUN: %empty-directory(%t)
// RUN: %target-build-swift %s -Onone -o %t/none.out
// RUN: %target-build-swift %s -O -o %t/opt.out
// RUN: %target-run %t/none.out
// RUN: %target-run %t/opt.out
//
// BTE specialization must preserve the existing closure analysis's termination
// for unused recursive forwarding. Parameter rotation is a positive control:
// ordinary specialization rounds can handle all three statically known closures.
// The SILOptimizer companion is reduced from these generated pullback bodies.

import _Differentiation
import StdlibUnittest

@inline(never)
public func forwardUnused(_ body: @escaping (Float) -> Float, _ depth: Int, _ seed: Float) -> Float {
  if depth == 0 { return seed }
  return forwardUnused(body, depth - 1, seed)
}

@inline(never)
public func appliedAfterRotation(
  _ first: @escaping (Float) -> Float, _ second: @escaping (Float) -> Float,
  _ third: @escaping (Float) -> Float, _ depth: Int, _ seed: Float
) -> Float {
  if depth == 0 { return third(seed) }
  return appliedAfterRotation(third, first, second, depth - 1, seed)
}

@differentiable(reverse, wrt: x)
public func identityWithForwarding(_ x: Float, _ depth: Int) -> Float { x }

@derivative(of: identityWithForwarding, wrt: x)
@inline(__always)
public func vjpIdentityWithForwarding(_ x: Float, _ depth: Int)
  -> (value: Float, pullback: (Float) -> Float) {
  (x, { seed in forwardUnused({ value in value * x }, depth, seed) })
}

@differentiable(reverse, wrt: x)
public func rotatedDerivative(_ x: Float, _ depth: Int) -> Float {
  switch depth % 3 {
  case 0: return 5 * x
  case 1: return 3 * x
  default: return x * x
  }
}

@derivative(of: rotatedDerivative, wrt: x)
@inline(__always)
public func vjpRotatedDerivative(_ x: Float, _ depth: Int)
  -> (value: Float, pullback: (Float) -> Float) {
  (rotatedDerivative(x, depth), { seed in
    appliedAfterRotation({ 2 * x * $0 }, { 3 * $0 }, { 5 * $0 }, depth, seed)
  })
}

var tests = TestSuite("ADCSRecursiveForwarding")
tests.test("FiniteForwardingWithoutApplication") {
  let result = valueWithPullback(at: Float(4), of: {
    identityWithForwarding($0, 4)
  })
  expectEqual(Float(4), result.value)
  expectEqual(Float(2), result.pullback(2))
  expectEqual(Float(5), result.pullback(5))
}
tests.test("RecursiveParameterRotation") {
  for depth in 0...5 {
    let result = valueWithPullback(at: Float(4), of: {
      rotatedDerivative($0, depth)
    })
    let slope: Float = [5, 3, 8][depth % 3]
    let value: Float = [20, 12, 16][depth % 3]
    expectEqual(value, result.value)
    expectEqual(2 * slope, result.pullback(2))
    expectEqual(5 * slope, result.pullback(5))
  }
}
runAllTests()
