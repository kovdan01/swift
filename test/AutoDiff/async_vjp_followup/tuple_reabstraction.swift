// RUN: %target-swift-frontend -emit-sil -Onone %s -Xllvm -enable-experimental-async-autodiff -Xllvm -sil-print-after=differentiation -sil-verify-all -sil-ownership-verify-all -o /dev/null 2>&1 | %FileCheck %s
// RUN: %target-swift-frontend -emit-sil -O %s -Xllvm -enable-experimental-async-autodiff -Xllvm -sil-print-after=differentiation -sil-verify-all -sil-ownership-verify-all -o /dev/null 2>&1 | %FileCheck %s
// REQUIRES: asserts
// REQUIRES: concurrency
// REQUIRES: differentiable_programming

import _Differentiation

func scalarTwo(_ a: Float, _ b: Float) async -> Float { a * b }

@differentiable(reverse)
func twoCaptures(_ a: Float, _ b: Float) async -> Float {
  async let r = scalarTwo(a, b)
  return await r
}

// Exercise both tuple packing (helper PB -> task payload) and unpacking
// (task payload -> checkpoint/direct PB), including the one-result boundary.
func scalarOne(_ x: Float) async -> Float { x * x }
func scalarThree(_ a: Float, _ b: Float, _ c: Float) async -> Float {
  a * b + c
}
func scalarThreeDouble(_ a: Double, _ b: Double, _ c: Double) async -> Double {
  a * b + c
}

@differentiable(reverse)
func oneCapture(_ x: Float) async -> Float {
  async let r = scalarOne(x)
  return await r
}

@differentiable(reverse)
func threeCaptures(_ a: Float, _ b: Float, _ c: Float) async -> Float {
  async let r = scalarThree(a, b, c)
  return await r
}

@differentiable(reverse)
func threeDoubleCaptures(_ a: Double, _ b: Double, _ c: Double) async -> Double {
  async let r = scalarThreeDouble(a, b, c)
  return await r
}

// The shared AutoDiff thunk now owns result regrouping; there is no private
// async-only scalar adapter. Check the ABI and transfers, not its mangled name.
// CHECK: sil shared {{.*}}[reabstraction_thunk]{{.*}} : $@convention(thin) (@in_guaranteed Float, @guaranteed @callee_guaranteed (@in_guaranteed Float) -> (Float, Float)) -> @out (Float, Float) {
// CHECK: apply {{.*}} : $@callee_guaranteed (@in_guaranteed Float) -> (Float, Float)
// CHECK: destructure_tuple
// CHECK: tuple_element_addr {{.*}}, 0
// CHECK: store {{.*}} to [trivial]
// CHECK: tuple_element_addr {{.*}}, 1
// CHECK: store {{.*}} to [trivial]
// CHECK: return
