// RUN: %target-swift-frontend -typecheck -parse-stdlib -verify %s
// REQUIRES: differentiable_programming
// REQUIRES: concurrency

import Swift
import _Differentiation

func missingAwait(_ f: @differentiable(reverse) (Float) async -> Float) async {
  _ = Builtin.applyDerivative_vjp_async(f, Float(2))
  // expected-error@-1 {{expression is 'async' but is not marked with 'await'}}
  // expected-note@-2 {{call is 'async'}}
}

func noAsyncJVP(_ f: @differentiable(reverse) (Float) async -> Float) async {
  _ = await Builtin.applyDerivative_jvp_async(f, Float(2))
  // expected-error@-1 {{module 'Builtin' has no member named 'applyDerivative_jvp_async'}}
}

func noAsyncThrows(_ f: @differentiable(reverse) (Float) async -> Float) async {
  _ = await Builtin.applyDerivative_vjp_throws_async(f, Float(2))
  // expected-error@-1 {{module 'Builtin' has no member named 'applyDerivative_vjp_throws_async'}}
}
