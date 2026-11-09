// REQUIRES: executable_test
// RUN: %target-run-simple-swift
// RUN: %target-swift-frontend -emit-sil -O -D DEFINITIONS_ONLY -module-name adcs_covariant_helper %s -o %t.sil
// RUN: %FileCheck %s < %t.sil

// The custom derivative uses class covariance in a captured helper function.
// ADCS removes the outer taped pullback closure. A class-covariant pullback
// result itself is not emitted: TangentVector must conform to AdditiveArithmetic,
// whose inout Self requirements prevent a non-final class conformance. That
// additional SIL robustness case is covered by the separate reduced test.

import _Differentiation

public class Box { public let value: Float; public init(_ value: Float) { self.value = value } }
public final class DerivedBox: Box {}

@differentiable(reverse, wrt: x)
@inline(never)
public func covariant(_ x: Float, _ makeBox: @escaping (Float) -> DerivedBox) -> Float { 2*x }

@derivative(of: covariant, wrt: x)
@inline(__always)
public func vjpCovariant(_ x: Float, _ makeBox: @escaping (Float) -> DerivedBox)
  -> (value: Float, pullback: (Float) -> Float) {
  let base: (Float) -> Box = makeBox
  return (2*x, { seed in base(seed).value })
}

@differentiable(reverse, wrt: x)
public func branchCovariant(_ x: Float, _ flag: Bool, _ makeBox: @escaping (Float) -> DerivedBox) -> Float {
  if flag { return covariant(x, makeBox) }
  return x*x
}

// CHECK-LABEL: // reverse-mode derivative of branchCovariant
// CHECK-NOT: } // end sil function
// CHECK: // function_ref specialized pullback of branchCovariant
// CHECK-NEXT: %[[PB:[0-9]+]] = function_ref
// CHECK-NEXT: %[[PULLBACK:[0-9]+]] = partial_apply {{.*}} %[[PB]]{{[<(]}}
// CHECK-NEXT: %[[RESULT:[0-9]+]] = tuple (%{{[0-9]+}}, %[[PULLBACK]])
// CHECK-NOT: } // end sil function
// CHECK: return %[[RESULT]]
// CHECK-NEXT: } // end sil function

// CHECK-LABEL: // specialized pullback of branchCovariant
// CHECK-NOT: = partial_apply
// CHECK-NOT: apply {{.*}} : $@callee_guaranteed (Float) -> Float
// CHECK: } // end sil function

#if !DEFINITIONS_ONLY
import StdlibUnittest
var Tests = TestSuite("adcs_covariant_helper")
Tests.test("Pullback") {
  for flag in [false, true] {

    let (value, pullback) = valueWithPullback(at: Float(3), of: {
      branchCovariant($0, flag, { DerivedBox(2 * $0) })
    })
    expectEqual(flag ? 6 : 9, value)
    expectEqual(flag ? 2 : 6, pullback(1))
    expectEqual(flag ? 4 : 12, pullback(2))

  }
}
runAllTests()
#endif
