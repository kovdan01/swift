/// Multi basic block VJP, pullback accepting branch tracing enum argument.

// REQUIRES: executable_test

// RUN: %empty-directory(%t)
// RUN: %target-build-swift %s -o %t/none.out -Onone
// RUN: %target-build-swift %s -o %t/opt.out  -O
// RUN: %target-run %t/none.out
// RUN: %target-run %t/opt.out

// RUN: %target-swift-frontend -emit-sil %s -O -o - | %FileCheck %s

import DifferentiationUnittest
import StdlibUnittest

var AutoDiffClosureSpecBTETests = TestSuite("AutoDiffClosureSpecBTE")

AutoDiffClosureSpecBTETests.testWithLeakChecking("Test") {
  // CHECK-NONE:  {{^}}// pullback of cond_tuple_var
  // CHECK-LABEL: {{^}}// specialized pullback of cond_tuple_var
  // CHECK-NEXT:  {{^}}sil private
  // CHECK-SAME:  : $@convention(thin) (Float, @owned _AD__$s12multi_bb_bteyycfU_14cond_tuple_varL_yS2fF_bb3__Pred__src_0_wrt_0) -> Float {{{$}}
  func cond_tuple_var(_ x: Float) -> Float {
    // Convoluted function returning `x + x`.
    var y: (Float, Float) = (x, x)
    var z: (Float, Float) = (x + x, x - x)
    if x > 0 {
      var w = (x, x)
      y.0 = w.1
      y.1 = w.0
      z.0 = z.0 - y.0
      z.1 = z.1 + y.0
    } else {
      z = (1 * x, x)
    }
    return y.0 + y.1 - z.0 + z.1
  }

  expectEqual((8, 2), valueWithGradient(at: 4, of: cond_tuple_var))
  expectEqual((-20, 2), valueWithGradient(at: -10, of: cond_tuple_var))
  expectEqual((-2674, 2), valueWithGradient(at: -1337, of: cond_tuple_var))
}

runAllTests()
