// REQUIRES: executable_test
// RUN: %target-run-simple-swift
// RUN: %target-swift-frontend -emit-sil -O -D DEFINITIONS_ONLY -module-name adcs_generic_adapter %s -o %t.sil
// RUN: %FileCheck %s < %t.sil

// Differentiating only x requires a generic subset-parameters adapter around
// the full (Float, T) pullback. ADCS must remap the root closure substitutions
// and the adapter substitutions into the cloned generic pullback environment.
// The source-derived SILOptimizer pair additionally verifies this after textual
// SIL parsing, which gives the VJP and pullback distinct generic environments.

import _Differentiation

@differentiable(reverse, wrt: (x, y))
@inline(never)
public func mixed<T: Differentiable & AdditiveArithmetic>(_ x: Float, _ y: T) -> Float
where T.TangentVector == T { 2 * x }

@derivative(of: mixed, wrt: (x, y))
@inline(__always)
public func vjpMixed<T: Differentiable & AdditiveArithmetic>(_ x: Float, _ y: T)
  -> (value: Float, pullback: (Float) -> (Float, T)) where T.TangentVector == T {
  (2 * x, { seed in (2 * seed, T.zero) })
}

@differentiable(reverse, wrt: x)
public func genericSubset<T: Differentiable & AdditiveArithmetic>(_ x: Float, _ y: T, _ flag: Bool) -> Float
where T.TangentVector == T {
  if flag { return mixed(x, y) }
  return x * x
}

// CHECK-LABEL: // reverse-mode derivative of genericSubset<A>
// CHECK-NOT: } // end sil function
// CHECK: // function_ref specialized pullback of genericSubset<A>
// CHECK-NEXT: %[[PB:[0-9]+]] = function_ref
// CHECK-NEXT: %[[PULLBACK:[0-9]+]] = partial_apply {{.*}} %[[PB]]{{[<(]}}
// CHECK-NEXT: %[[RESULT:[0-9]+]] = tuple (%{{[0-9]+}}, %[[PULLBACK]])
// CHECK-NOT: } // end sil function
// CHECK: return %[[RESULT]]
// CHECK-NEXT: } // end sil function

// CHECK-LABEL: // specialized pullback of genericSubset<A>
// CHECK-NOT: = partial_apply
// CHECK-NOT: apply {{.*}} : $@callee_guaranteed
// CHECK: } // end sil function

#if !DEFINITIONS_ONLY
import StdlibUnittest
var Tests = TestSuite("adcs_generic_adapter")
Tests.test("SubsetPullback") {
  for flag in [false, true] {
    let floating = valueWithPullback(at: Float(3), of: {
      genericSubset($0, Float(4), flag)
    })
    expectEqual(flag ? 6 : 9, floating.value)
    expectEqual(flag ? 2 : 6, floating.pullback(1))
    expectEqual(flag ? 10 : 30, floating.pullback(5))

    let double = valueWithPullback(at: Float(3), of: {
      genericSubset($0, Double(4), flag)
    })
    expectEqual(flag ? 6 : 9, double.value)
    expectEqual(flag ? 2 : 6, double.pullback(1))
    expectEqual(flag ? 4 : 12, double.pullback(2))
  }
}
runAllTests()
#endif
