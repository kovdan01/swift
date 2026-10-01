// RUN: %target-swift-frontend -emit-sil -parse-as-library -module-name AsyncVJPTests -Xllvm -enable-experimental-async-autodiff -Xllvm -sil-print-after=differentiation -sil-verify-all -sil-ownership-verify-all %s -o %t.sil 2> %t.after
// RUN: %FileCheck %s < %t.after
// REQUIRES: concurrency
//
// Runtime and more detailed structural checks:
// python3 Inputs/run.py --frontend /path/to/swift-frontend --swiftc /path/to/swiftc
//
// The first example is the original source: neither workload is annotated and
// no custom derivative is registered.
import _Differentiation

func workload1(_ a: Float, _ b: Float) async -> Float {
  a + b
}

func workload2(_ a: Float, _ b: Float) async -> Float {
  a * b
}

// CHECK-LABEL: // reverse-mode derivative of doWor5k(_:_:)
// CHECK: sil {{.*}}@{{.*}}doWor5k{{.*}} : $@convention(thin) @async
// CHECK: alloc_stack $(Float, {{.*}}@callee_guaranteed
// CHECK: builtin "startAsyncLetWithLocalBuffer"<(Float,
// CHECK-NOT: function_ref @swift_asyncLet_get
// CHECK: builtin "startAsyncLetWithLocalBuffer"<(Float,
// CHECK: function_ref @swift_asyncLet_get
// CHECK: load [copy]
// CHECK: function_ref @swift_asyncLet_finish
// CHECK: builtin "endAsyncLetLifetime"
// CHECK: return
// CHECK-LABEL: // pullback of doWor5k(_:_:)
// CHECK-NOT: @async
// CHECK: apply
// CHECK-NOT: swift_asyncLet
// CHECK: return
@differentiable(reverse)
func doWor5k(_ a: Float, _ b: Float) async -> Float {
  async let result1 = workload1(a, b)
  async let result2 = workload2(a, b)
  return await result1 + result2
}

// A synchronous baseline using ordinary calls, NOT captures.
func syncWorkload1(_ a: Float, _ b: Float) -> Float { a + b }
func syncWorkload2(_ a: Float, _ b: Float) -> Float { a * b }
@differentiable(reverse)
func synchronous(_ a: Float, _ b: Float) -> Float {
  let r1 = syncWorkload1(a, b)
  let r2 = syncWorkload2(a, b)
  return r1 + r2
}

// No async-let machinery: an async nested VJP must be called through the
// ordinary static async-function ABI, with a synchronous saved pullback.
@differentiable(reverse)
func directCalls(_ a: Float, _ b: Float) async -> Float {
  let r = await workload2(a, b)
  return await workload1(r, a)
}

@differentiable(reverse)
func repeatedRead(_ a: Float, _ b: Float) async -> Float {
  async let result = workload2(a, b)
  let first = await result
  let second = await result
  return first + second
}

@differentiable(reverse)
func repeatedCapture(_ a: Float, _ b: Float) async -> Float {
  async let result = workload2(a, a)
  return await result + b
}

@differentiable(reverse)
func nonlinearUse(_ a: Float, _ b: Float) async -> Float {
  async let result = workload2(a, b)
  let value = await result
  return value * value
}

@differentiable(reverse)
func intermediateCapture(_ a: Float, _ b: Float) async -> Float {
  let intermediate = a + b
  async let result = workload2(intermediate, b)
  return await result
}

func composedWorkload(_ a: Float, _ b: Float) async -> Float {
  let r = await workload2(a, b)
  return await workload1(r, a)
}
@differentiable(reverse)
func nestedCalls(_ a: Float, _ b: Float) async -> Float {
  async let result = composedWorkload(a, b)
  return await result
}

@differentiable(reverse, wrt: a)
func selectedParameter(_ a: Float, _ b: Float) async -> Float {
  async let r1 = workload1(a, b)
  async let r2 = workload2(a, b)
  return await r1 + r2
}

// The result is needed in the primal and as a coefficient in the parent PB,
// but is not active for the selected differentiation parameter.
@differentiable(reverse, wrt: a)
func inactiveChild(_ a: Float, _ b: Float) async -> Float {
  async let fixed = workload2(b, b)
  return a * (await fixed)
}

// Preserve start/finish/lifetime cleanup without generating a child pullback.
@differentiable(reverse)
func unusedChild(_ a: Float, _ b: Float) async -> Float {
  async let ignored = workload2(a, b)
  return a + b
}

func doubleWorkload1(_ a: Double, _ b: Double) async -> Double { a + b }
func doubleWorkload2(_ a: Double, _ b: Double) async -> Double { a * b }
@differentiable(reverse)
func doublePrecision(_ a: Double, _ b: Double) async -> Double {
  async let r1 = doubleWorkload1(a, b)
  async let r2 = doubleWorkload2(a, b)
  return await r1 + r2
}
