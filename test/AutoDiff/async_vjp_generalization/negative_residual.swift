// RUN: not %target-swift-frontend -emit-sil -parse-as-library %s -Xllvm -enable-experimental-async-autodiff -o /dev/null 2>&1 | %FileCheck %s
// REQUIRES: concurrency, differentiable_programming
import _Differentiation

// A synchronous derivative is numerically valid but retains a reference whose
// task-transfer provenance this series deliberately does not prove. Its source
// name is irrelevant. A future region-based proof can admit disconnected cases.
final class UnprovenReference {
  var value: Float
  init(_ value: Float) { self.value = value }
}
func ordinaryOperation(_ x: Float) -> Float { x * x }
@derivative(of: ordinaryOperation)
func ordinaryDerivative(_ x: Float) -> (value: Float, pullback: (Float) -> Float) {
  let saved = UnprovenReference(x)
  return (x*x, { seed in 2 * saved.value * seed })
}
func taskBody(_ x: Float) async -> Float { ordinaryOperation(x) }
@differentiable(reverse)
func unsupportedTransfer(_ x: Float) async -> Float {
  async let result = taskBody(x)
  return await result
}
// CHECK: error: function is not differentiable
// CHECK: cannot prove that this saved derivative value can cross the child-task boundary
