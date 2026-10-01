// RUN: %target-swift-frontend -emit-sil -Onone %s -Xllvm -enable-experimental-async-autodiff -sil-verify-all -sil-ownership-verify-all -o /dev/null
// RUN: %target-swift-frontend -emit-sil -O %s -Xllvm -enable-experimental-async-autodiff -sil-verify-all -sil-ownership-verify-all -o /dev/null
// REQUIRES: asserts
// REQUIRES: concurrency
// REQUIRES: differentiable_programming

import _Differentiation

func workload1(_ a: Float, _ b: Float) async -> Float { a + b }
func workload2(_ a: Float, _ b: Float) async -> Float { a * b }

@differentiable(reverse)
func doWor5k(_ a: Float, _ b: Float) async -> Float {
  async let result1 = workload1(a, b)
  async let result2 = workload2(a, b)
  return await result1 + result2
}
