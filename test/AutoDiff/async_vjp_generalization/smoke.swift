// RUN: %target-swift-frontend -emit-sil -parse-as-library %s -Onone -Xllvm -enable-experimental-async-autodiff -sil-verify-all -sil-ownership-verify-all -o /dev/null
// RUN: %target-swift-frontend -emit-sil -parse-as-library %s -O -Xllvm -enable-experimental-async-autodiff -sil-verify-all -sil-ownership-verify-all -o /dev/null
// REQUIRES: concurrency, differentiable_programming
import _Differentiation
import _Concurrency

// BEGIN DIFFERENTIATED FUNCTIONS
func workload1(_ a: Float, _ b: Float) async -> Float { a + b }
func workload2(_ a: Float, _ b: Float) async -> Float { a * b }
@differentiable(reverse)
func doWor5k(_ a: Float, _ b: Float) async -> Float {
  async let result1 = workload1(a, b)
  async let result2 = workload2(a, b)
  return await result1 + result2
}
func echo(_ x: [Float]) async -> [Float] { x }
@differentiable(reverse)
func roundTrip(_ x: [Float]) async -> [Float] {
  async let result = echo(x)
  return await result
}
// END DIFFERENTIATED FUNCTIONS

@main struct SmokeTests {
  nonisolated static func main() async {
    let scalar = /*AD_AWAIT*/ await valueWithPullback(at: Float(2), Float(3), of: doWor5k)
    let array = /*AD_AWAIT*/ await valueWithPullback(at: [Float(2), 4, 3], of: roundTrip)
    precondition(scalar.value == 11 && array.value == [2, 4, 3])
    let p: (Float) -> (Float, Float) = scalar.pullback
    let q: ([Float].TangentVector) -> [Float].TangentVector = array.pullback
    await Task.yield()
    for seed: Float in [1, 2, 0, -3, 0.5] {
      let g = p(seed)
      precondition(g.0 == 4*seed && g.1 == 3*seed)
      precondition(q([Float].TangentVector([seed, 2*seed, -seed])).base == [seed, 2*seed, -seed])
    }
    print("scalar value=\(scalar.value) gradient=\(p(1))")
    print("PASS smoke")
  }
}
