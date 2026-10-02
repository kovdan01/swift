// RUN: not %target-swift-frontend -emit-sil -parse-as-library %s -Xllvm -enable-experimental-async-autodiff -o /dev/null 2>&1 | %FileCheck %s
// REQUIRES: concurrency, differentiable_programming
import _Differentiation

func genericEcho<T: Differentiable & Sendable>(_ x: T) async -> T { x }
@differentiable(reverse)
func genericParent<T: Differentiable & Sendable>(_ x: T) async -> T {
  async let result = genericEcho(x)
  return await result
}
// CHECK: error: function is not differentiable
// CHECK: generic async-let parents require contextual helper specialization
