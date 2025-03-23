// REQUIRES: executable_test

// RUN: %empty-directory(%t)
// RUN: %target-build-swift-dylib(%t/%target-library-name(closure_spec_module)) %S/Inputs/closure_spec_module.swift \
// RUN:   -emit-module -emit-module-path %t/closure_spec_module.swiftmodule -module-name closure_spec_module
// RUN: %target-build-swift -I%t -L%t %s -lclosure_spec_module %target-rpath(%t) -o %t/none.out -Onone
// RUN: %target-build-swift -I%t -L%t %s -lclosure_spec_module %target-rpath(%t) -o %t/opt.out  -O
// RUN: %target-run %t/none.out
// RUN: %target-run %t/opt.out

/// Particular optimizations are checked in SILOptimizer tests, here we only check that optimizations occur
// RUN: %target-swift-frontend -emit-sil -I%t %s -o - -O | %FileCheck %s
// CHECK-NONE: {{^}}// pullback of myfoo3
// CHECK-NONE: {{^}}// pullback of myfoo2
// CHECK-NONE: {{^}}// pullback of myfoo2
// CHECK:      {{^}}// specialized pullback of myfoo3
// CHECK:      {{^}}// specialized pullback of myfoo2
// CHECK:      {{^}}// specialized pullback of myfoo1

import closure_spec_module
import StdlibUnittest
import DifferentiationUnittest

var AutoDiffClosureSpecializationTests = TestSuite("AutoDiffClosureSpecialization")

AutoDiffClosureSpecializationTests.testWithLeakChecking("Test") {
  func myfoo1(_ x: Float) -> Float {
    if (x > 0) {
      return mybar1(x) + mybar2(x)
    } else {
      return mybar1(x) * mybar2(x)
    }
  }

  func myfoo1_derivative(_ x: Float) -> Float {
    if (x > 0) {
      return 2 * x + 1
    } else {
      return 3 * x * x
    }
  }

  for x in -100...100 {
    expectEqual(myfoo1_derivative(Float(x)), gradient(at: Float(x), of: myfoo1))
  }

  func myfoo2(_ x: Float) -> Float {
    let y = mybar1(x) + 37 * mybar2(x)
    var z : Float = 0
    if (x > 0) {
      z = mybar1(y) * mybar2(x) + mybar2(y)
    } else {
      z = mybar1(x) * mybar1(y)
    }
    return z * x + y * y
  }

  func myfoo2_derivative(_ x: Float) -> Float {
    if (x > 0) {
      return 7030 * x * x * x * x + 5776 * x * x * x + 225 * x * x + 2 * x
    } else {
      return 5624 * x * x * x + 225 * x * x + 2 * x
    }
  }

  for x in -14...6 {
    expectEqual(myfoo2_derivative(Float(x)), gradient(at: Float(x), of: myfoo2))
  }

  func myfoo3(_ x: Float) -> Float {
    if (x > 0) {
      return mybar2(x) * mybar2(x)
    } else {
      var y = mybar1(x) + mybar2(x)
      if (x > -10) {
        return x + mybar2(y)
      } else {
        return mybar1(y) * x
      }
    }
  }

  func myfoo3_derivative(_ x: Float) -> Float {
    if (x > 0) {
      return 4 * x * x * x
    } else {
      if (x > -10) {
        return 4 * x * x * x + 6 * x * x + 2 * x + 1
      } else {
        return 3 * x * x + 2 * x
      }
    }
  }

  for x in -100...100 {
    expectEqual(myfoo3_derivative(Float(x)), gradient(at: Float(x), of: myfoo3))
  }
}

runAllTests()
