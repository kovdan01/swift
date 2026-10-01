import Swift
import _Concurrency
import _Differentiation

func workload1(_ a: Float, _ b: Float) async -> Float { a + b }
func workload2(_ a: Float, _ b: Float) async -> Float { a * b }

@differentiable(reverse)
func doWor5k(_ a: Float, _ b: Float) async -> Float {
  async let result1 = workload1(a, b)
  async let result2 = workload2(a, b)
  return await result1 + result2
}

@differentiable(reverse)
func repeatedRead(_ a: Float, _ b: Float) async -> Float {
  async let r = workload2(a, b)
  return await r + r
}

@differentiable(reverse)
func repeatedCapture(_ x: Float) async -> Float {
  async let r = workload2(x, x)
  return await r
}

func check(_ got: Float, _ expected: Float, _ what: String) {
  precondition(got.isFinite && abs(got - expected) <= 0.00001 * max(1, abs(expected)),
               "\(what): got \(got), expected \(expected)")
}

@main
struct AsyncADTaskTests {
  nonisolated static func main() async {
    let inputs: [(Float, Float)] = [(2, 3), (0, 0), (-2, 4), (0.5, -0.25)]
    let seeds: [Float] = [1, 2, 0, -3, 0.5]
    for (a, b) in inputs {
      let (value, pb) = await valueWithPullback(at: a, b, of: doWor5k)
      let synchronousPB: (Float) -> (Float, Float) = pb
      check(value, a + b + a*b, "task value")
      for seed in seeds {
        let g = synchronousPB(seed)
        check(g.0, seed * (1 + b), "task da")
        check(g.1, seed * (1 + a), "task db")
        if a == 2 && b == 3 {
          print("value=\(value), seed=\(seed), gradient=(\(g.0), \(g.1))")
        }
      }
      // The VJP has left the async-let scope. Suspend the CALLER and then reuse
      // the saved PB. This does not ask AD to differentiate Task.yield().
      await Task.yield()
      check(synchronousPB(1).0, 1 + b, "PB survives scope and caller suspension")
      let vg = await valueWithGradient(at: a, b, of: doWor5k)
      check(vg.value, value, "task valueWithGradient value")
      check(vg.gradient.0, 1 + b, "task valueWithGradient da")
      check(vg.gradient.1, 1 + a, "task valueWithGradient db")
      let g = await gradient(at: a, b, of: doWor5k)
      check(g.0, 1 + b, "task gradient da")
      check(g.1, 1 + a, "task gradient db")
      let p = await pullback(at: a, b, of: doWor5k)
      check(p(2).0, 2 * (1 + b), "task pullback API")
      let repeated = await valueWithPullback(at: a, b, of: repeatedRead)
      check(repeated.value, 2*a*b, "repeated read value")
      check(repeated.pullback(1).0, 2*b, "repeated read da")
      check(repeated.pullback(1).1, 2*a, "repeated read db")
    }
    let repeated = await valueWithPullback(at: Float(3), of: repeatedCapture)
    check(repeated.value, 9, "repeated capture value")
    check(repeated.pullback(1), 6, "repeated capture derivative")
    for i in 0..<100 {
      let result = await valueWithPullback(at: Float(i % 7), Float(3), of: doWor5k)
      if i % 2 == 0 {
        check(result.pullback(1).0, 4, "lifetime stress")
        check(result.pullback(2).0, 8, "reusable PB stress")
      }
      // Other iterations drop the saved PB without invoking it.
    }
    print("PASS async-let reverse API: original example, repeated reads/captures, reusable and dropped pullbacks")
  }
}
