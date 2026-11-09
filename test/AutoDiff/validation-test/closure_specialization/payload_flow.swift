// REQUIRES: executable_test
// RUN: %empty-directory(%t)
// RUN: %target-build-swift %s -Onone -o %t/none.out
// RUN: %target-build-swift %s -O -o %t/opt.out
// RUN: %target-codesign %t/none.out
// RUN: %target-codesign %t/opt.out
// RUN: %target-run %t/none.out
// RUN: %target-run %t/opt.out

import _Differentiation

// These source shapes combine whole tape payloads with projected predecessor
// and callback phis. Aggregate unwrapping moves their field projections across
// branches before the first ADCS invocation; mid-level optimization then folds
// specialized trivial capture tuples before the second invocation.
@inline(never)
@differentiable(reverse, wrt: x)
public func branching(_ x: Float, _ flag: Bool) -> Float {
  var result = x * x
  if flag {
    result = result + result
    result = result - x
  } else {
    result = result / x
  }
  return result
}

@inline(never)
@differentiable(reverse)
public func nested(_ x: Float, _ y: Float) -> Float {
  if x > 0 {
    if y > 10 {
      let z = x * y
      if z > 100 { return x + z }
      if y == 20 { return z + z }
    } else {
      return x + y
    }
  }
  return -y
}

@inline(never)
@differentiable(reverse, wrt: x)
public func generic<T>(_ x: Float, _ metatype: T.Type) -> Float {
  if metatype is Int.Type { return x + x }
  return x * x
}

// Library functions with inline_late semantics are preserved until the
// mid-level inliner. Keep this custom derivative opaque at the first ADCS
// invocation so the second invocation must specialize its projected callback
// phi after the closure factory has been inlined.
@inline(never)
@differentiable(reverse)
public func delayedSquare(_ x: Float) -> Float { x * x }

@_semantics("inline_late")
@derivative(of: delayedSquare)
public func delayedSquareVJP(_ x: Float) -> (value: Float, pullback: (Float) -> Float) {
  (x * x, { seed in 2 * x * seed })
}

@inline(never)
@differentiable(reverse, wrt: x)
public func lateBranch(_ x: Float, _ flag: Bool) -> Float {
  let result = delayedSquare(x)
  if flag { return result }
  return x
}

// Keep the derivative entry points alive in the source-only optimizer test.
public func branchingGradient(_ x: Float, _ flag: Bool) -> Float {
  gradient(at: x, of: { branching($0, flag) })
}
public func nestedGradient(_ x: Float, _ y: Float) -> (Float, Float) {
  gradient(at: x, y, of: nested)
}
public func genericGradient<T>(_ x: Float, _ metatype: T.Type) -> Float {
  gradient(at: x, of: { generic($0, metatype) })
}

public func lateGradient(_ x: Float, _ flag: Bool) -> Float {
  gradient(at: x, of: { lateBranch($0, flag) })
}

#if !DEFINITIONS_ONLY
import DifferentiationUnittest
import StdlibUnittest

var tests = TestSuite("PayloadFlow")
tests.testWithLeakChecking("ProjectedAndWholePayloads") {
  for x: Float in [2, 4] {
    for flag in [false, true] {
      let result = valueWithPullback(at: x, of: { branching($0, flag) })
      let value = flag ? 2 * x * x - x : x
      let slope = flag ? 4 * x - 1 : 1
      expectEqual(value, result.value)
      expectEqual(slope, result.pullback(1))
      expectEqual(2 * slope, result.pullback(2))
      expectEqual(slope, branchingGradient(x, flag))
      let late = valueWithPullback(at: x, of: { lateBranch($0, flag) })
      let lateSlope = flag ? 2 * x : 1
      expectEqual(flag ? x * x : x, late.value)
      expectEqual(lateSlope, late.pullback(1))
      expectEqual(2 * lateSlope, late.pullback(2))
      expectEqual(lateSlope, lateGradient(x, flag))
    }
  }
  expectEqual((40, 8), nestedGradient(4, 20))
  expectEqual((0, -1), nestedGradient(4, 21))
  expectEqual((1, 1), nestedGradient(4, 5))
  expectEqual((0, -1), nestedGradient(-3, -2))
  expectEqual(2, genericGradient(3, Int.self))
  expectEqual(6, genericGradient(3, Float.self))
}
runAllTests()
#endif
