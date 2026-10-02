// RUN: %target-swift-frontend -emit-sil -parse-as-library %s -Onone -Xllvm -enable-experimental-async-autodiff -sil-verify-all -sil-ownership-verify-all -o /dev/null
// RUN: %target-swift-frontend -emit-sil -parse-as-library %s -O -Xllvm -enable-experimental-async-autodiff -sil-verify-all -sil-ownership-verify-all -o /dev/null
// REQUIRES: concurrency, differentiable_programming
import _Differentiation
import _Concurrency

// BEGIN DIFFERENTIATED FUNCTIONS
@inline(never)
func arrayEcho(_ x: [Float]) async -> [Float] { x }

@inline(never)
func component(_ x: [Float], _ index: Int) async -> Float { x[index] }

@differentiable(reverse)
func arrayRoundTrip(_ x: [Float]) async -> [Float] {
  async let result = arrayEcho(x)
  return await result
}

@differentiable(reverse)
func sharedArray(_ x: [Float]) async -> Float {
  async let left = component(x, 0)
  async let right = component(x, 2)
  return await left * right
}

@differentiable(reverse, wrt: x)
func arrayLoop(_ x: [Float], _ count: Int) async -> [Float] {
  var result = x
  for _ in 0..<count {
    let input = result
    async let next = arrayEcho(input)
    result = await next
  }
  return result
}
@differentiable(reverse)
func twice(_ x: [Float]) async -> [Float] { await arrayLoop(x, 2) }

@differentiable(reverse, wrt: x)
func arrayBranch(_ x: [Float], _ left: Bool) async -> [Float] {
  async let result = arrayEcho(x)
  if left { return await result }
  return x
}
@differentiable(reverse)
func arraySelected(_ x: [Float]) async -> [Float] { await arrayBranch(x, true) }
@differentiable(reverse)
func arrayUnused(_ x: [Float]) async -> [Float] { await arrayBranch(x, false) }
@inline(never)
func scaledComponent(_ x: [Float], _ scale: Float) async -> Float {
  x[0] * scale
}
@differentiable(reverse)
func mixedCaptures(_ x: [Float], _ scale: Float) async -> Float {
  async let result = scaledComponent(x, scale)
  return await result
}
// END DIFFERENTIATED FUNCTIONS

func check(_ actual: [Float], _ expected: [Float], _ label: String) {
  precondition(actual.count == expected.count, "\(label): wrong gradient shape")
  for (a, b) in zip(actual, expected) {
    precondition(a.isFinite && abs(a-b) < 0.0001 * max(1, abs(b)), "\(label): \(a) != \(b)")
  }
}

@main struct ArrayTests {
  nonisolated static func main() async {
    var input: [Float] = [2, 4, 3]
    let id = /*AD_AWAIT*/ await valueWithPullback(at: input, of: arrayRoundTrip)
    let shared = /*AD_AWAIT*/ await valueWithPullback(at: input, of: sharedArray)
    let loop = /*AD_AWAIT*/ await valueWithPullback(at: input, of: twice)
    let yes = /*AD_AWAIT*/ await valueWithPullback(at: input, of: arraySelected)
    let no = /*AD_AWAIT*/ await valueWithPullback(at: input, of: arrayUnused)
    let mixed = /*AD_AWAIT*/ await valueWithPullback(at: input, Float(5), of: mixedCaptures)
    // Mutating the original array must not change the recorded evaluation.
    input[0] = 100
    input[2] = 200
    await Task.yield()
    check(id.value, [2, 4, 3], "array primal snapshot")
    check(loop.value, [2, 4, 3], "loop primal snapshot")
    precondition(shared.value == 6)
    let reusable: ([Float].TangentVector) -> [Float].TangentVector = id.pullback
    for seed: [Float] in [[1, 1, 1], [2, -1, 3], [0, 0, 0]] {
      let tangent = [Float].TangentVector(seed)
      check(reusable(tangent).base, seed, "array identity PB")
      check(loop.pullback(tangent).base, seed, "array loop PB")
      check(yes.pullback(tangent).base, seed, "array taken branch PB")
      check(no.pullback(tangent).base, seed, "array untaken branch PB")
    }
    for seed: Float in [1, 2, 0, -3, 0.5] {
      check(shared.pullback(seed).base, [3*seed, 0, 2*seed], "array shared capture")
      let g = mixed.pullback(seed)
      check(g.0.base, [5*seed, 0, 0], "mixed nontrivial result")
      precondition(g.1 == 2*seed, "mixed scalar result")
    }
    for _ in 0..<128 {
      _ = /*AD_AWAIT*/ await valueWithPullback(at: [Float](repeating: 2, count: 3), of: twice)
    }
    print("array value=\(shared.value) gradient=\(shared.pullback(1).base)")
    print("PASS arrays")
  }
}
