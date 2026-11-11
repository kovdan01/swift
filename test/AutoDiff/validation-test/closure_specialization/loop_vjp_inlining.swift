// REQUIRES: executable_test

// RUN: %empty-directory(%t)
// RUN: %target-build-swift %s -Onone -o %t/none.out
// RUN: %target-build-swift %s -O -o %t/opt.out
// RUN: %target-codesign %t/none.out
// RUN: %target-codesign %t/opt.out
// RUN: %target-run %t/none.out
// RUN: %target-run %t/opt.out

import _Differentiation

// Keep the iteration count dynamic during optimization.
public var iterationCount = 3

@differentiable(reverse)
@inline(never)
public func square(_ x: Float) -> Float { x * x }

@differentiable(reverse)
public func loop(_ x: Float) -> Float {
  var result = x
  for _ in 0..<iterationCount {
    result = square(result)
  }
  return result
}

public func caller(_ x: Float) -> Float {
  gradient(at: x, of: loop)
}

// Keep the outer branch tape separate from the callee's runtime loop tape.
// Inlining the loop VJP before BTE specialization would make this otherwise
// loop-free VJP ineligible for specialization as well.
@differentiable(reverse, wrt: x)
public func branchingLoop(_ x: Float, _ flag: Bool) -> Float {
  if flag { return loop(x) }
  return x * x
}

// The caller has no tape of its own. Keep the callee's branch tape available
// for specialization before inlining its VJP here.
@differentiable(reverse, wrt: x)
public func throughBranchingLoop(_ x: Float, _ flag: Bool) -> Float {
  branchingLoop(x, flag)
}

public func branchingCaller(_ x: Float, _ flag: Bool) -> Float {
  gradient(at: x, of: { throughBranchingLoop($0, flag) })
}

#if !DEFINITIONS_ONLY
import DifferentiationUnittest
import StdlibUnittest

var tests = TestSuite("LoopVJPInlining")
tests.testWithLeakChecking("RuntimeTape") {
  let cases: [(Int, Float, Float)] = [(0, 2, 1), (1, 4, 4), (3, 256, 1024)]
  for (count, value, derivative) in cases {
    iterationCount = count
    let result = valueWithPullback(at: Float(2), of: loop)
    expectEqual(value, result.value)
    expectEqual(derivative, result.pullback(1))
    expectEqual(derivative, caller(2))
    for flag in [false, true] {
      let outer = valueWithPullback(at: Float(2), of: { branchingLoop($0, flag) })
      let slope: Float = flag ? derivative : 4
      expectEqual(flag ? value : 4, outer.value)
      expectEqual(slope, outer.pullback(1))
      expectEqual(2 * slope, outer.pullback(2))
      expectEqual(slope, branchingCaller(2, flag))
    }
  }
}
runAllTests()
#endif
