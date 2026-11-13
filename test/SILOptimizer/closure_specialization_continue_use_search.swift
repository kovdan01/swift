// RUN: %target-swift-frontend -emit-sil -O -parse-as-library -sil-verify-all -module-name ClosureUseSearch %s | %FileCheck %s --check-prefixes=ORDINARY,THUNK,NEGATIVE
//
// Calling a callback now and returning a wrapper for later must still count as
// applying that callback. Searching the unapplied returned wrapper must not hide
// the earlier direct call. Consuming the callback transfers it into that wrapper.

@inline(never)
public func runAndDefer(_ value: Float, _ body: consuming @escaping (Float) -> Float) -> (Float, (Float) -> Float) {
  let result = body(value)
  let deferred = { [body] next in body(next) }
  return (result, deferred)
}

// ORDINARY-LABEL: sil {{.*}}@$s16ClosureUseSearch6update{{.*}} :
// ORDINARY-NOT: partial_apply
// ORDINARY: function_ref @{{.*}}11runAndDefer{{.*}}Tf1{{.*}} : $@convention(thin) (Float, Float)
// ORDINARY-NOT: partial_apply
// ORDINARY: apply
// ORDINARY-NEXT: return
public func update(_ value: Float, _ factor: Float) -> (Float, (Float) -> Float) {
  runAndDefer(value, { $0 * factor })
}

public struct CallbackBox<T> {
  public var body: (T) -> T
}

// A generic stored callback requires a genuine reabstraction thunk. The thunk
// is also unapplied here and must not hide the direct call to the same callback.
@inline(never)
public func runAndBox(_ value: Float, _ body: consuming @escaping (Float) -> Float) -> (Float, CallbackBox<Float>) {
  let function: (Float) -> Float = copy body
  let result = function(value)
  return (result, CallbackBox(body: function))
}

// THUNK-LABEL: sil {{.*}}@$s16ClosureUseSearch9updateBox{{.*}} :
// THUNK-NOT: partial_apply
// THUNK: function_ref @{{.*}}9runAndBox{{.*}}Tf1{{.*}} : $@convention(thin) (Float, Float)
// THUNK-NOT: partial_apply
// THUNK: apply
// THUNK-NEXT: return
public func updateBox(_ value: Float, _ factor: Float) -> (Float, CallbackBox<Float>) {
  runAndBox(value, { $0 * factor })
}

// Capturing a callback without applying it must not justify specialization.
@inline(never)
public func deferOnly(_ body: consuming @escaping (Float) -> Float) -> (Float) -> Float {
  { [body] value in body(value) }
}

// NEGATIVE-LABEL: sil {{.*}}@$s16ClosureUseSearch12unusedUpdate{{.*}} :
// NEGATIVE: partial_apply
// NEGATIVE: function_ref @{{.*}}9deferOnly{{.*}} : $@convention(thin) (@owned @callee_guaranteed (Float) -> Float)
// NEGATIVE: apply
// NEGATIVE-NEXT: return
public func unusedUpdate(_ factor: Float) -> (Float) -> Float {
  deferOnly { $0 * factor }
}
