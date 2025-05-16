/// This test is work-in-progress.

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
/// TODO: support myfoo7. Now unsupported because of unexpected location of payload tuple instructions in vjp
// CHECK-NONE: {{^}}// pullback of myfoo6
// CHECK-NONE: {{^}}// pullback of myfoo5
// CHECK-NONE: {{^}}// pullback of myfoo4
// CHECK-NONE: {{^}}// pullback of myfoo3
// CHECK-NONE: {{^}}// pullback of myfoo2
// CHECK-NONE: {{^}}// pullback of myfoo1
// CHECK:      {{^}}// specialized pullback of myfoo6
// CHECK:      {{^}}// specialized pullback of myfoo5
// CHECK:      {{^}}// specialized pullback of myfoo4
// CHECK:      {{^}}// specialized pullback of myfoo3
// CHECK:      {{^}}// specialized pullback of myfoo2
// CHECK:      {{^}}// specialized pullback of myfoo1

import DifferentiationUnittest
import StdlibUnittest
import closure_spec_module

#if canImport(Glibc)
  import Glibc
#elseif canImport(Android)
  import Android
#else
  import Foundation
#endif

var AutoDiffClosureSpecializationTests = TestSuite("AutoDiffClosureSpecialization")

AutoDiffClosureSpecializationTests.testWithLeakChecking("Test") {
  func myfoo1(_ x: Float) -> Float {
    if x > 0 {
      return mybar1(x) + mybar2(x)
    } else {
      return mybar1(x) * mybar2(x)
    }
  }

  func myfoo1_derivative(_ x: Float) -> Float {
    if x > 0 {
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
    var z: Float = 0
    if x > 0 {
      z = mybar1(y) * mybar2(x) + mybar2(y)
    } else {
      z = mybar1(x) * mybar1(y)
    }
    return z * x + y * y
  }

  func myfoo2_derivative(_ x: Float) -> Float {
    if x > 0 {
      return 7030 * x * x * x * x + 5776 * x * x * x + 225 * x * x + 2 * x
    } else {
      return 5624 * x * x * x + 225 * x * x + 2 * x
    }
  }

  for x in -14...6 {
    expectEqual(myfoo2_derivative(Float(x)), gradient(at: Float(x), of: myfoo2))
  }

  func myfoo3(_ x: Float) -> Float {
    if x > 0 {
      return mybar2(x) * mybar2(x)
    } else {
      var y = mybar1(x) + mybar2(x)
      if x > -10 {
        return x + mybar2(y)
      } else {
        return mybar1(y) * x
      }
    }
  }

  func myfoo3_derivative(_ x: Float) -> Float {
    if x > 0 {
      return 4 * x * x * x
    } else {
      if x > -10 {
        return 4 * x * x * x + 6 * x * x + 2 * x + 1
      } else {
        return 3 * x * x + 2 * x
      }
    }
  }

  for x in -100...100 {
    expectEqual(myfoo3_derivative(Float(x)), gradient(at: Float(x), of: myfoo3))
  }

  func myfoo4(_ x: Float) -> Float {
    if x > 0 {
      return mybar2(x) * mybar2(x)
    } else {
      let y = mybar1(x) + mybar2(x)
      var z: Float = 37
      if x > -7 {
        z += y + mybar2(x)
      } else {
        z -= mybar1(x) * y
      }
      return mybar1(z) * mybar2(y)
    }
  }

  func myfoo4_derivative(_ x: Float) -> Float {
    if x > 0 {
      return 4 * x * x * x
    } else if x > -7 {
      return 12 * x * x * x * x * x + 25 * x * x * x * x + 164 * x * x * x + 225 * x * x + 74 * x
    } else {
      return -7 * x * x * x * x * x * x - 18 * x * x * x * x * x - 15 * x * x * x * x + 144 * x * x
        * x + 222 * x * x + 74 * x
    }
  }

  for x in -20...100 {
    expectEqual(myfoo4_derivative(Float(x)), gradient(at: Float(x), of: myfoo4))
  }

  func myfoo5(_ x: Float) -> Float {
    if x > 0 {
      return sin(x) * cos(x)
    } else {
      return sin(x) + cos(x)
    }
  }

  func myfoo5_derivative(_ x: Float) -> Float {
    if x > 0 {
      return cos(x) * cos(x) - sin(x) * sin(x)
    } else {
      return cos(x) - sin(x)
    }
  }

  for x in -100...100 {
    expectEqual(myfoo5_derivative(Float(x)), gradient(at: Float(x), of: myfoo5))
  }

  func double(x: Float) -> Float {
    return x + x
  }

  func square(x: Float) -> Float {
    return x * x
  }

  func myfoo6(_ x: Float) -> Float {
    if x > 0 {
      if (x + 1) < 5 {
        if x * 2 > 4 {
          let y = square(x: x)
          if y >= x {
            let d = double(x: x)
            return x - (d * y)
          } else {
            let e = square(x: y)
            return x + (e * y)
          }
        }
      }
    } else {
      let y = double(x: x)
      return x * y
    }

    return x * 3
  }

  func myfoo6_derivative(_ x: Float) -> Float {
    if x > 2 && x < 4 {
      return 1 - 6 * x * x
    } else if x > 0 {
      return 3
    } else {
      return 4 * x
    }
  }

  for x in -100...100 {
    expectEqual(myfoo6_derivative(Float(x)), gradient(at: Float(x), of: myfoo6))
  }

  var x : Float = -10
  while x < 10 {
    let a = myfoo6_derivative(Float(x))
    let b = gradient(at: Float(x), of: myfoo6)
    expectEqual((a * 10000).rounded(), (b * 10000).rounded())
    x += 1 / 1024
  }

  func myfoo7(_ x: Float, _ bool: Bool) -> Float {
    var result = x * x
    if bool {
      result = result + result
      result = result - x
    } else {
      result = result / x
    }
    return result
  }

  func myfoo7_derivative(_ x: Float, _ bool: Bool) -> Float {
    if bool {
      return 4 * x - 1
    }
    return 1
  }

  for x in -100...100 {
    if x == 0 {
      continue
    }
    expectEqual(gradient(at: Float(x), of: { myfoo7(Float($0), true)  }), myfoo7_derivative(Float(x), true))
    expectEqual((100000 * gradient(at: Float(x), of: { myfoo7(Float($0), false) })).rounded(), (100000 * myfoo7_derivative(Float(x), false)).rounded())
  }
}

runAllTests()
