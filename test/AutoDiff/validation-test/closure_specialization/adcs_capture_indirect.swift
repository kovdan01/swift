// REQUIRES: executable_test
// RUN: %target-run-simple-swift
// RUN: %target-swift-frontend -emit-sil -O -module-name adcs_capture_indirect %s -o %t.sil
// RUN: %FileCheck %s < %t.sil

// The corresponding SILOptimizer test is reduced from the generated derivative
// before ADCS. A generic custom derivative captures T indirectly even when T is
// specialized to String at its call site.
import _Differentiation
import DifferentiationUnittest
import StdlibUnittest

@differentiable(reverse, wrt: x)
public func labeled<T>(_ x: Float, _ value: T) -> Float { 2 * x }

@derivative(of: labeled, wrt: x)
@inline(__always)
public func vjpLabeled<T>(
  _ x: Float, _ value: T
) -> (value: Float, pullback: (Float) -> Float) {
  (2 * x, { seed in withExtendedLifetime(value) { 2 * seed } })
}

@differentiable(reverse, wrt: x)
public func stringBranch(_ x: Float, _ name: String, _ flag: Bool) -> Float {
  if flag { return labeled(x, name) }
  return x * x
}

// CHECK-LABEL: // reverse-mode derivative of stringBranch
// CHECK-NOT: } // end sil function
// CHECK: // function_ref specialized pullback of stringBranch
// CHECK-NEXT: %[[PB:[0-9]+]] = function_ref
// CHECK-NEXT: %[[PULLBACK:[0-9]+]] = partial_apply {{.*}} %[[PB]]{{[<(]}}
// CHECK-NEXT: %[[RESULT:[0-9]+]] = tuple (%{{[0-9]+}}, %[[PULLBACK]])
// CHECK-NOT: } // end sil function
// CHECK: return %[[RESULT]]
// CHECK-NEXT: } // end sil function

// CHECK-LABEL: // specialized pullback of stringBranch
// CHECK-NOT: apply {{.*}} : $@callee_guaranteed
// CHECK: } // end sil function

var tests = TestSuite("ADCSIndirectCapture")
tests.testWithLeakChecking("IndirectString") {
  let name = String(repeating: "captured label", count: 7)
  let linear = valueWithPullback(at: Float(3), of: {
    stringBranch($0, name, true)
  })
  expectEqual(6, linear.value)
  expectEqual(10, linear.pullback(5))
  expectEqual(4, linear.pullback(2))

  let square = valueWithPullback(at: Float(3), of: {
    stringBranch($0, name, false)
  })
  expectEqual(9, square.value)
  expectEqual(30, square.pullback(5))
}
runAllTests()
