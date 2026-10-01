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

// Match a real 2 -> 1 formal-result conversion, not just a successful compiler
// exit. Both numerical values must be packed into the single indirect result.
// CHECK: sil private {{.*}}@{{.*}}__swift_async_ad_pb_reabstraction_{{.*}} : $@convention(thin) (@in_guaranteed Float, @guaranteed @callee_guaranteed (@in_guaranteed Float) -> (Float, Float)) -> @out (Float, Float) {
// CHECK: apply {{.*}} : $@callee_guaranteed (@in_guaranteed Float) -> (Float, Float)
// CHECK: tuple_extract {{.*}}, 0
// CHECK: tuple_extract {{.*}}, 1
// CHECK: tuple (
// CHECK: store {{.*}} to [trivial]
// CHECK: return
