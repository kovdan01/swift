// REQUIRES: executable_test
// RUN: %target-run-simple-swift
// RUN: %target-swift-frontend -emit-sil -O -D DEFINITIONS_ONLY -module-name adcs_discarded_pullback %s -o %t.sil
// RUN: %FileCheck %s < %t.sil

// The custom derivative captures an inner pullback but never calls it.
// ADCS eliminates the outer closure and preserves its capture for destruction.
// Normal inlining later discards the closure inside a nested tape tuple. The
// direct tape-payload destruction path is covered separately in reduced SIL.

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
public func identity(_ x: Float, _ label: String) -> Float { x }

@derivative(of: identity, wrt: x)
@inline(__always)
public func vjpIdentity(_ x: Float, _ label: String) -> (value: Float, pullback: (Float) -> Float) {
  let (_, pullback) = vjpLabeled(x, label)
  return (x, { seed in _ = pullback; return seed })
}

@differentiable(reverse, wrt: x)
public func branchIdentity(_ x: Float, _ label: String, _ flag: Bool) -> Float {
  if flag { return identity(x, label) }
  return x*x
}

@differentiable(reverse, wrt: x)
public func inactiveBranch(_ x: Float, _ label: String, _ flag: Bool) -> Float {
  _ = labeled(x, label)
  if flag { return x*x }
  return x
}

// CHECK-LABEL: // reverse-mode derivative of branchIdentity
// CHECK-NOT: } // end sil function
// CHECK: // function_ref specialized pullback of branchIdentity
// CHECK-NEXT: %[[PB:[0-9]+]] = function_ref
// CHECK-NEXT: %[[PULLBACK:[0-9]+]] = partial_apply {{.*}} %[[PB]]{{[<(]}}
// CHECK-NEXT: %[[RESULT:[0-9]+]] = tuple (%{{[0-9]+}}, %[[PULLBACK]])
// CHECK-NOT: } // end sil function
// CHECK: return %[[RESULT]]
// CHECK-NEXT: } // end sil function

// CHECK-LABEL: // specialized pullback of branchIdentity
// CHECK-NOT: = partial_apply
// CHECK: } // end sil function

#if !DEFINITIONS_ONLY
import StdlibUnittest
var Tests = TestSuite("adcs_discarded_pullback")
Tests.test("Pullback") {
  for flag in [false, true] {

    let (value, pullback) = valueWithPullback(at: Float(3), of: {
      branchIdentity($0, "a label longer than a small string", flag)
    })
    expectEqual(flag ? 3 : 9, value)
    expectEqual(flag ? 1 : 6, pullback(1))
    expectEqual(flag ? 2 : 12, pullback(2))
    expectEqual(flag ? 3 : 9, valueWithPullback(at: Float(3), of: {
      branchIdentity($0, "a label longer than a small string", flag)
    }).value)
    expectEqual(flag ? 6 : 1, gradient(at: Float(3), of: {
      inactiveBranch($0, "a label longer than a small string", flag)
    }))

  }
}
runAllTests()
#endif
