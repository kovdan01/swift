// RUN: %target-swift-frontend -emit-sil -parse-as-library %s -Onone -Xllvm -enable-experimental-async-autodiff -sil-verify-all -sil-ownership-verify-all -o /dev/null
// RUN: %target-swift-frontend -emit-sil -parse-as-library %s -O -Xllvm -enable-experimental-async-autodiff -sil-verify-all -sil-ownership-verify-all -o /dev/null
// REQUIRES: concurrency, differentiable_programming
import _Differentiation
import _Concurrency

// BEGIN DIFFERENTIATED FUNCTIONS
// This type AND its tangent are nontrivial. TangentVector is NOT Self: tag has
// no derivative. Neither this name nor any library array name is compiler magic.
struct Batch: Differentiable, Sendable {
  var weights: [Float]
  var scale: Float
  @noDerivative var tag: Int
}

@inline(never)
func batchEcho(_ x: Batch) async -> Batch { x }
@inline(never)
func weighted(_ x: Batch, _ index: Int) async -> Float {
  x.weights[index] * x.scale
}

@differentiable(reverse)
func batchRoundTrip(_ x: Batch) async -> Batch {
  async let result = batchEcho(x)
  return await result
}
@differentiable(reverse)
func twoCaptures(_ x: Batch, _ y: Batch) async -> Float {
  async let a = weighted(x, 0)
  async let b = weighted(y, 1)
  return await a + b
}
@differentiable(reverse)
func batchScore(_ x: Batch) async -> Float {
  // The metadata is captured and used but has no pullback result component.
  let index = x.tag
  async let a = weighted(x, index)
  async let b = weighted(x, 0)
  return await a + b
}
@inline(never)
func combined(_ x: Batch, _ y: Batch) async -> Float {
  x.weights[0] * x.scale + y.weights[1] * y.scale
}
@differentiable(reverse, wrt: x)
func selectedCapture(_ x: Batch, _ y: Batch) async -> Float {
  async let result = combined(x, y)
  return await result
}
@differentiable(reverse)
func selectedWrapper(_ x: Batch) async -> Float {
  await selectedCapture(x, Batch(weights: [3, 4, 5], scale: 2, tag: 0))
}
// END DIFFERENTIATED FUNCTIONS

func check(_ actual: Float, _ expected: Float) {
  precondition(actual.isFinite && abs(actual - expected) < 0.0001 * max(1, abs(expected)),
               "\(actual) != \(expected)")
}
func check(_ actual: Batch.TangentVector, _ expected: [Float], _ scale: Float) {
  precondition(actual.weights.base.count == expected.count)
  for (a,b) in zip(actual.weights.base, expected) { check(a,b) }
  check(actual.scale, scale)
}

@main struct NominalTests {
  nonisolated static func main() async {
    var x = Batch(weights: [2, 4, 3], scale: 5, tag: 1)
    let identity = /*AD_AWAIT*/ await valueWithPullback(at: x, of: batchRoundTrip)
    let score = /*AD_AWAIT*/ await valueWithPullback(at: x, of: batchScore)
    let pair = /*AD_AWAIT*/ await valueWithPullback(at: x, x, of: twoCaptures)
    let selected = /*AD_AWAIT*/ await valueWithPullback(at: x, of: selectedWrapper)
    x.weights[0] = 100
    x.scale = 200
    await Task.yield()
    precondition(identity.value.tag == 1)
    check(score.value, 30)
    check(pair.value, 30)
    check(selected.value, 18)
    let seed = Batch.TangentVector(weights: [Float].TangentVector([2, -1, 3]), scale: -4)
    let synchronous: (Batch.TangentVector) -> Batch.TangentVector = identity.pullback
    for _ in 0..<5 { check(synchronous(seed), [2, -1, 3], -4) }
    for s: Float in [1, 2, 0, -3, 0.5] {
      check(score.pullback(s), [5*s, 5*s, 0], 6*s)
      let p = pair.pullback(s)
      check(p.0, [5*s, 0, 0], 2*s)
      check(p.1, [0, 5*s, 0], 4*s)
      check(selected.pullback(s), [5*s, 0, 0], 2*s)
    }
    for _ in 0..<128 {
      _ = /*AD_AWAIT*/ await valueWithPullback(at: x, of: batchRoundTrip)
    }
    print("nominal value=\(score.value) gradient=\(score.pullback(1).weights.base)")
    print("PASS nominal")
  }
}
