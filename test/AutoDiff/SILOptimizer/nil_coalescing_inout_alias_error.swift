// RUN: %target-swift-frontend -emit-sil -verify %s

import _Differentiation

func ??(_ x: Float?, _ y: @autoclosure () -> Float) -> Float {
  if x == nil {
    return y()
  }
  return x!
}

// expected-error @+1 {{function is not differentiable}}
@differentiable(reverse)
// expected-note @+1 {{when differentiating this function definition}}
func o(ff: F) -> Float {
    var y = ff.i?.first { $0 >= 0.0 } ?? 0.0
    while 0.0 < y {
        // MYTODO comment
        // expected-note @+1 {{expression is not differentiable}}
	y = ff.g() ?? y
    }
    return y
}

func o2(ff: F) -> Float {
    var y = ff.i?.first { $0 >= 0.0 } ?? 0.0
    while 0.0 < y {
	y = ff.g() ?? 42
    }
    return y
}

public struct F: Differentiable {
    @noDerivative var i: [Float]? {return nil}
    func g() -> Float? {return nil}
}
