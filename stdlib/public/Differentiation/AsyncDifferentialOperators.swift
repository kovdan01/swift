//===--- AsyncDifferentialOperators.swift ---------------------*- swift -*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2026 the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//
//
// Experimental reverse-mode operators for nonthrowing async functions.
// Only the forward/VJP invocation is async; returned pullbacks are synchronous.
// These entry points do not relax the differentiation transform's supported
// domain or imply Sendability of arbitrary saved derivative state.
//
//===----------------------------------------------------------------------===//

import Swift

/// Evaluates an async function and returns its value and synchronous pullback.
@_alwaysEmitIntoClient
public func valueWithPullback<T: Differentiable, R: Differentiable>(
  at x: T,
  of f: @differentiable(reverse) (T) async -> R
) async -> (value: R, pullback: (R.TangentVector) -> T.TangentVector) {
  return await Builtin.applyDerivative_vjp_async(f, x)
}

/// Evaluates an async function and returns its value and synchronous pullback.
@_alwaysEmitIntoClient
public func valueWithPullback<T: Differentiable, U: Differentiable, R: Differentiable>(
  at x: T, _ y: U,
  of f: @differentiable(reverse) (T, U) async -> R
) async -> (value: R, pullback: (R.TangentVector) -> (T.TangentVector, U.TangentVector)) {
  return await Builtin.applyDerivative_vjp_arity2_async(f, x, y)
}

/// Evaluates an async function and returns its value and synchronous pullback.
@_alwaysEmitIntoClient
public func valueWithPullback<T: Differentiable, U: Differentiable, V: Differentiable, R: Differentiable>(
  at x: T, _ y: U, _ z: V,
  of f: @differentiable(reverse) (T, U, V) async -> R
) async -> (value: R, pullback: (R.TangentVector) -> (T.TangentVector, U.TangentVector, V.TangentVector)) {
  return await Builtin.applyDerivative_vjp_arity3_async(f, x, y, z)
}

/// Evaluates an async function and returns its synchronous pullback.
@_alwaysEmitIntoClient
public func pullback<T: Differentiable, R: Differentiable>(
  at x: T,
  of f: @differentiable(reverse) (T) async -> R
) async -> (R.TangentVector) -> T.TangentVector {
  let result = await valueWithPullback(at: x, of: f)
  return result.pullback
}

/// Evaluates an async function and returns its synchronous pullback.
@_alwaysEmitIntoClient
public func pullback<T: Differentiable, U: Differentiable, R: Differentiable>(
  at x: T, _ y: U,
  of f: @differentiable(reverse) (T, U) async -> R
) async -> (R.TangentVector) -> (T.TangentVector, U.TangentVector) {
  let result = await valueWithPullback(at: x, y, of: f)
  return result.pullback
}

/// Evaluates an async function and returns its synchronous pullback.
@_alwaysEmitIntoClient
public func pullback<T: Differentiable, U: Differentiable, V: Differentiable, R: Differentiable>(
  at x: T, _ y: U, _ z: V,
  of f: @differentiable(reverse) (T, U, V) async -> R
) async -> (R.TangentVector) -> (T.TangentVector, U.TangentVector, V.TangentVector) {
  let result = await valueWithPullback(at: x, y, z, of: f)
  return result.pullback
}

/// Evaluates an async scalar-valued function and applies its pullback to one.
@_alwaysEmitIntoClient
public func valueWithGradient<T: Differentiable, R: Differentiable & FloatingPoint>(
  at x: T,
  of f: @differentiable(reverse) (T) async -> R
) async -> (value: R, gradient: T.TangentVector) where R.TangentVector == R {
  let result = await valueWithPullback(at: x, of: f)
  return (result.value, result.pullback(R(1)))
}

/// Evaluates an async scalar-valued function and applies its pullback to one.
@_alwaysEmitIntoClient
public func valueWithGradient<T: Differentiable, U: Differentiable, R: Differentiable & FloatingPoint>(
  at x: T, _ y: U,
  of f: @differentiable(reverse) (T, U) async -> R
) async -> (value: R, gradient: (T.TangentVector, U.TangentVector)) where R.TangentVector == R {
  let result = await valueWithPullback(at: x, y, of: f)
  return (result.value, result.pullback(R(1)))
}

/// Evaluates an async scalar-valued function and applies its pullback to one.
@_alwaysEmitIntoClient
public func valueWithGradient<T: Differentiable, U: Differentiable, V: Differentiable, R: Differentiable & FloatingPoint>(
  at x: T, _ y: U, _ z: V,
  of f: @differentiable(reverse) (T, U, V) async -> R
) async -> (value: R, gradient: (T.TangentVector, U.TangentVector, V.TangentVector)) where R.TangentVector == R {
  let result = await valueWithPullback(at: x, y, z, of: f)
  return (result.value, result.pullback(R(1)))
}

/// Returns the gradient of an async scalar-valued function.
@_alwaysEmitIntoClient
public func gradient<T: Differentiable, R: Differentiable & FloatingPoint>(
  at x: T,
  of f: @differentiable(reverse) (T) async -> R
) async -> T.TangentVector where R.TangentVector == R {
  let result = await valueWithPullback(at: x, of: f)
  return result.pullback(R(1))
}

/// Returns the gradient of an async scalar-valued function.
@_alwaysEmitIntoClient
public func gradient<T: Differentiable, U: Differentiable, R: Differentiable & FloatingPoint>(
  at x: T, _ y: U,
  of f: @differentiable(reverse) (T, U) async -> R
) async -> (T.TangentVector, U.TangentVector) where R.TangentVector == R {
  let result = await valueWithPullback(at: x, y, of: f)
  return result.pullback(R(1))
}

/// Returns the gradient of an async scalar-valued function.
@_alwaysEmitIntoClient
public func gradient<T: Differentiable, U: Differentiable, V: Differentiable, R: Differentiable & FloatingPoint>(
  at x: T, _ y: U, _ z: V,
  of f: @differentiable(reverse) (T, U, V) async -> R
) async -> (T.TangentVector, U.TangentVector, V.TangentVector) where R.TangentVector == R {
  let result = await valueWithPullback(at: x, y, z, of: f)
  return result.pullback(R(1))
}

