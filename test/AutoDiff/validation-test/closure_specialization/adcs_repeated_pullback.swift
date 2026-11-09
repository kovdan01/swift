// REQUIRES: executable_test
// RUN: %target-run-simple-swift
// RUN: %target-swift-frontend -emit-sil -O -D DEFINITIONS_ONLY -module-name adcs_repeated_pullback %s -o %t.sil
// RUN: %FileCheck %s < %t.sil

// The custom derivative captures a nontrivial inner pullback and calls it
// twice. The generator tapes the outer closure, which ADCS eliminates. Normal
// inlining later exposes repeated use inside a nested capture tuple. The
// direct taped-closure rewrite is covered separately in reduced SIL.

import _Differentiation

@differentiable(reverse, wrt: x)
@inline(never)
public func labeled(_ x: Float, _ label: String) -> Float { 2 * x }

@derivative(of: labeled, wrt: x)
@inline(__always)
public func vjpLabeled(_ x: Float, _ label: String) -> (value: Float, pullback: (Float) -> Float) {
  (2*x, { seed in withExtendedLifetime(label) { 2*seed } })
}

@differentiable(reverse, wrt: x)
@inline(never)
public func twice(_ x: Float, _ label: String) -> Float { 4*x }

@derivative(of: twice, wrt: x)
@inline(__always)
public func vjpTwice(_ x: Float, _ label: String) -> (value: Float, pullback: (Float) -> Float) {
  let (_, pullback) = vjpLabeled(x, label)
  return (4*x, { seed in pullback(seed) + pullback(seed) })
}

@differentiable(reverse, wrt: x)
public func branchTwice(_ x: Float, _ label: String, _ flag: Bool) -> Float {
  if flag { return twice(x, label) }
  return x*x
}

// CHECK-LABEL: // reverse-mode derivative of branchTwice
// CHECK-NOT: } // end sil function
// CHECK: // function_ref specialized pullback of branchTwice
// CHECK-NEXT: %[[PB:[0-9]+]] = function_ref
// CHECK-NEXT: %[[PULLBACK:[0-9]+]] = partial_apply {{.*}} %[[PB]]{{[<(]}}
// CHECK-NEXT: %[[RESULT:[0-9]+]] = tuple (%{{[0-9]+}}, %[[PULLBACK]])
// CHECK-NOT: } // end sil function
// CHECK: return %[[RESULT]]
// CHECK-NEXT: } // end sil function

// CHECK-LABEL: // specialized pullback of branchTwice
// CHECK-NOT: = partial_apply
// CHECK: } // end sil function

#if !DEFINITIONS_ONLY
import StdlibUnittest
var Tests = TestSuite("adcs_repeated_pullback")
Tests.test("Pullback") {
  for flag in [false, true] {

    let (value, pullback) = valueWithPullback(at: Float(3), of: {
      branchTwice($0, "a label longer than a small string", flag)
    })
    expectEqual(flag ? 12 : 9, value)
    expectEqual(flag ? 4 : 6, pullback(1))
    expectEqual(flag ? 8 : 12, pullback(2))

  }
}
runAllTests()
#endif
