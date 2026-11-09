// REQUIRES: executable_test
// RUN: %target-build-swift %s -Onone -o %t.none
// RUN: %target-run %t.none
// RUN: %target-build-swift %s -O -o %t.opt
// RUN: %target-run %t.opt
// RUN: %target-swift-frontend -O -emit-sil -module-name TupleCase %s -o - | %FileCheck %s
//
// A real tuple capture is scalarized before ADCS; keep this ordinary source
// optimized while rejecting unexpected nested tape projections in the companion
// adcs_nested_tuple_rejection.sil boundary test.
// CHECK-LABEL: // reverse-mode derivative of branch
// CHECK: // function_ref specialized pullback of branch
// CHECK: return
// CHECK-LABEL: // specialized pullback of branch
// CHECK-NOT: = partial_apply
// CHECK: } // end sil function

import _Differentiation

@inline(never)
public func usePair(_ pair: (Float,Float), _ seed: Float) -> Float { pair.0 * seed + pair.1 }

@differentiable(reverse,wrt:x)
public func custom(_ x:Float, _ pair:(Float,Float)) -> Float { pair.0*x + pair.1 }
@derivative(of:custom,wrt:x)
@inline(__always)
public func vjpCustom(_ x:Float, _ pair:(Float,Float)) -> (value:Float,pullback:(Float)->Float) {
  (pair.0*x + pair.1, { seed in usePair(pair,seed) - pair.1 })
}
@differentiable(reverse,wrt:x)
public func branch(_ x:Float, _ pair:(Float,Float), _ flag:Bool) ->Float {
  if flag {return custom(x,pair)*x}
  return custom(x,pair)+x
}

let product = valueWithPullback(at: Float(4), of: { branch($0, (2, 3), true) })
precondition(product.value == 44)
precondition(product.pullback(2) == 38)
let sum = valueWithPullback(at: Float(4), of: { branch($0, (2, 3), false) })
precondition(sum.value == 15)
precondition(sum.pullback(2) == 6)
