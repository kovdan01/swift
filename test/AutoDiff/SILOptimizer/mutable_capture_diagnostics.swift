// RUN: %target-swift-frontend -emit-sil -verify %s

import _Differentiation

// Nested function captures and mutates `indices` by reference (box capture).
// FILL exact strings from DiagnosticsSIL.def; pin lines with @+N after a first
// `-verify` run prints the actual emitted diagnostics.

@differentiable(reverse, wrt: x) // expected-error {{function is not differentiable}}
func foo(_ x: Double, _ cnt: Int) -> [Double] { // expected-note {{when differentiating this function definition}}
  var indices: [Double] = []
  func loopWrapper() {
    for _ in 0..<cnt {
      indices.append(x)
    }
  }
  loopWrapper() // expected-note {{cannot differentiate writes to mutable captures}}
  return indices
}

// A direct (non-nested-function) closure capture should behave identically.
@differentiable(reverse, wrt: x) // expected-error {{function is not differentiable}}
func bar(_ x: Double, _ cnt: Int) -> [Double] { // expected-note {{when differentiating this function definition}}
  var indices: [Double] = []
  let wrapper = { // expected-note {{cannot differentiate writes to mutable captures}}
    for _ in 0..<cnt {
      indices.append(x)
    }
  }
  wrapper()
  return indices
}

// Must NOT be diagnosed: `withoutDerivative` legitimately yields a
// non-varied result; its `@out`/`@in_guaranteed` addresses are not captures.
@differentiable(reverse)
func noFalsePositive(_ x: Float) -> Float {
  let y = withoutDerivative(at: x)
  return y * y
}
