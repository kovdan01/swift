// RUN: not %target-swift-frontend -emit-sil -parse-as-library -D CONTROL_FLOW -Xllvm -enable-experimental-async-autodiff %s -o %t.sil 2>&1 | %FileCheck %s --check-prefix=CFG
// RUN: not %target-swift-frontend -emit-sil -parse-as-library -D NONSCALAR -Xllvm -enable-experimental-async-autodiff %s -o %t.sil 2>&1 | %FileCheck %s --check-prefix=SCALAR
// RUN: not %target-swift-frontend -emit-sil -parse-as-library -D RECURSIVE -Xllvm -enable-experimental-async-autodiff %s -o %t.sil 2>&1 | %FileCheck %s --check-prefix=RECURSION
// RUN: not %target-swift-frontend -emit-sil -parse-as-library %S/Inputs/original.swift -o %t.sil 2>&1 | %FileCheck %s --check-prefix=FLAG
// RUN: not %target-swift-frontend -emit-sil -parse-as-library -D CUSTOM_DERIVATIVE -Xllvm -enable-experimental-async-autodiff %s -o %t.sil 2>&1 | %FileCheck %s --check-prefix=CUSTOM
// REQUIRES: concurrency
// CFG: cannot differentiate this async let: the initial implementation requires a single-block parent
// SCALAR: cannot differentiate this async let: the initial capture domain is by-value Float and Double
// RECURSION: cannot differentiate this async let: recursive child computations are not yet supported
// FLAG: cannot differentiate this async let: enable -Xllvm -enable-experimental-async-autodiff
// CUSTOM: cannot differentiate this async let: custom or pre-existing opaque derivatives
import _Differentiation

#if CONTROL_FLOW
func worker(_ x: Float) async -> Float { x * x }
@differentiable(reverse)
func rejectedControlFlow(_ x: Float) async -> Float {
  async let result = worker(x)
  if x > 0 { return await result }
  return x
}
#endif

#if NONSCALAR
struct Pair: Differentiable, Sendable { var x: Float; var y: Float }
func worker(_ p: Pair) async -> Float { p.x * p.y }
@differentiable(reverse)
func rejectedCapture(_ x: Float, _ y: Float) async -> Float {
  let pair = Pair(x: x, y: y)
  async let result = worker(pair)
  return await result
}
#endif

#if RECURSIVE
// Not executed. Ensures the residual validator terminates and diagnoses cycles.
func recursiveWorker(_ x: Float) async -> Float { await recursiveWorker(x) }
@differentiable(reverse)
func rejectedRecursion(_ x: Float) async -> Float {
  async let result = recursiveWorker(x)
  return await result
}
#endif

#if CUSTOM_DERIVATIVE
// A scalar primal signature alone must not admit an arbitrary residual closure.
func customWorker(_ x: Float) -> Float { x * x }
@derivative(of: customWorker)
func customWorkerVJP(_ x: Float) -> (value: Float, pullback: (Float) -> Float) {
  (x * x, { seed in seed * (x + x) })
}
@differentiable(reverse)
func rejectedCustomDerivative(_ x: Float) async -> Float {
  async let result = customWorker(x)
  return await result
}
#endif
