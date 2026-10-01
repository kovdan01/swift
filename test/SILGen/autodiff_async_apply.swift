// RUN: %target-swift-frontend -emit-silgen -parse-stdlib %s | %FileCheck %s
// REQUIRES: differentiable_programming
// REQUIRES: concurrency

import Swift
import _Differentiation

// No differentiable body is synthesized here: these functions accept an
// already-differentiable async function. Check the async invocation and the
// synchronous returned PB independently of async-let activity analysis.

func applyOne<T: Differentiable, R: Differentiable>(
  _ f: @differentiable(reverse) (T) async -> R, _ x: T
) async -> (R, (R.TangentVector) -> T.TangentVector) {
  return await Builtin.applyDerivative_vjp_async(f, x)
}
// CHECK-LABEL: // applyOne
// CHECK: differentiable_function_extract [vjp]
// CHECK: apply {{.*}} : ${{.*}}@async
// CHECK: hop_to_executor
// CHECK: return

func applyTwo<T: Differentiable, U: Differentiable, R: Differentiable>(
  _ f: @differentiable(reverse) (T, U) async -> R, _ x: T, _ y: U
) async -> (R, (R.TangentVector) -> (T.TangentVector, U.TangentVector)) {
  return await Builtin.applyDerivative_vjp_arity2_async(f, x, y)
}
// CHECK-LABEL: // applyTwo
// CHECK: differentiable_function_extract [vjp]
// CHECK: apply {{.*}} : ${{.*}}@async
// CHECK: hop_to_executor
// CHECK: return

func applyThree<T: Differentiable, U: Differentiable, V: Differentiable,
                R: Differentiable>(
  _ f: @differentiable(reverse) (T, U, V) async -> R,
  _ x: T, _ y: U, _ z: V
) async -> (R, (R.TangentVector) ->
             (T.TangentVector, U.TangentVector, V.TangentVector)) {
  return await Builtin.applyDerivative_vjp_arity3_async(f, x, y, z)
}
// CHECK-LABEL: // applyThree
// CHECK: differentiable_function_extract [vjp]
// CHECK: apply {{.*}} : ${{.*}}@async
// CHECK: hop_to_executor
// CHECK: return

func applyConcrete(
  _ f: @differentiable(reverse) (Float, Float) async -> Float,
  _ x: Float, _ y: Float
) async -> (Float, (Float) -> (Float, Float)) {
  return await Builtin.applyDerivative_vjp_arity2_async(f, x, y)
}
// CHECK-LABEL: // applyConcrete
// CHECK: differentiable_function_extract [vjp]
// CHECK: apply {{.*}} : ${{.*}}@async
// CHECK: hop_to_executor
// CHECK: return
