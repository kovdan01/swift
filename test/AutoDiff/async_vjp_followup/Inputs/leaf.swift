import Swift
import _Concurrency
import _Differentiation

@differentiable(reverse)
func square(_ x: Float) async -> Float { x * x }

@differentiable(reverse)
func cube(_ x: Float) async -> Float { x * x * x }

@differentiable(reverse)
func combine(_ a: Float, _ b: Float) async -> Float { a + b + a * b }

@differentiable(reverse)
func combine3(_ a: Float, _ b: Float, _ c: Float) async -> Float { a * b + c }

@differentiable(reverse)
func combineDouble(_ a: Double, _ b: Double) async -> Double { a + b + a * b }

func nestedLeaf(_ a: Float, _ b: Float) async -> Float { a * b }

@differentiable(reverse)
func nested(_ a: Float, _ b: Float) async -> Float {
  let p = await nestedLeaf(a, b)
  return p + a
}

@differentiable(reverse)
func synchronous(_ x: Float) -> Float { x * x }

// Keep a genuinely higher-order call in the Onone test, including generic
// indirect results. No derivative is synthesized for these API consumers.
@inline(never)
func opaque2<T: Differentiable, U: Differentiable, R: Differentiable>(
  _ f: @differentiable(reverse) (T, U) async -> R, _ x: T, _ y: U
) async -> (value: R, pullback: (R.TangentVector) ->
           (T.TangentVector, U.TangentVector)) {
  return await valueWithPullback(at: x, y, of: f)
}

func check<T: BinaryFloatingPoint>(_ got: T, _ expected: T, _ what: String) {
  let tolerance = T(0.00001) * max(T(1), abs(expected))
  precondition(got.isFinite && abs(got - expected) <= tolerance,
               "\(what): got \(got), expected \(expected)")
}

@main
struct AsyncADLeafTests {
  nonisolated static func main() async {
    let unary = await valueWithPullback(at: Float(3), of: square)
    let unaryPB: (Float) -> Float = unary.pullback
    check(unary.value, 9, "unary value")
    check(unaryPB(2), 12, "unary pullback")
    let unaryOnlyPB = await pullback(at: Float(3), of: square)
    check(unaryOnlyPB(1), 6, "unary pullback API")
    let unaryVG = await valueWithGradient(at: Float(3), of: square)
    check(unaryVG.value, 9, "unary VG value")
    check(unaryVG.gradient, 6, "unary VG gradient")
    let unaryG = await gradient(at: Float(3), of: square)
    check(unaryG, 6, "unary gradient API")

    // Runtime choice, not a single statically-known primal function reference.
    let chooseSquare = CommandLine.arguments.count > 1
    let f: @differentiable(reverse) (Float) async -> Float
    if chooseSquare { f = square } else { f = cube }
    let dynamic = await valueWithPullback(at: Float(3), of: f)
    check(dynamic.value, chooseSquare ? 9 : 27, "function-value result")
    check(dynamic.pullback(1), chooseSquare ? 6 : 27, "function-value PB")

    let inputs: [(Float, Float)] = [(2, 3), (0, 0), (-2, 4), (0.5, -0.25)]
    let seeds: [Float] = [1, 2, 0, -3, 0.5]
    for (a, b) in inputs {
      let result = await opaque2(combine, a, b)
      check(result.value, a + b + a*b, "opaque binary value")
      let pb: (Float) -> (Float, Float) = result.pullback
      for seed in seeds {
        let g = pb(seed)
        check(g.0, seed * (1 + b), "binary da")
        check(g.1, seed * (1 + a), "binary db")
      }
      let p = await pullback(at: a, b, of: combine)
      check(p(1).0, 1 + b, "binary pullback API")
      let vg = await valueWithGradient(at: a, b, of: combine)
      check(vg.value, result.value, "binary VG value")
      check(vg.gradient.1, 1 + a, "binary VG gradient")
      let g = await gradient(at: a, b, of: combine)
      check(g.0, 1 + b, "binary gradient API")
      let n = await valueWithPullback(at: a, b, of: nested)
      check(n.value, a*b + a, "nested async value")
      check(n.pullback(1).0, b + 1, "nested async da")
      check(n.pullback(1).1, a, "nested async db")
    }

    let triple = await valueWithPullback(at: Float(2), Float(3), Float(5), of: combine3)
    let triplePB: (Float) -> (Float, Float, Float) = triple.pullback
    let tg = triplePB(2)
    check(triple.value, 11, "ternary value")
    check(tg.0, 6, "ternary da")
    check(tg.1, 4, "ternary db")
    check(tg.2, 2, "ternary dc")
    let tpb = await pullback(at: Float(2), Float(3), Float(5), of: combine3)
    check(tpb(1).2, 1, "ternary pullback API")
    let tvg = await valueWithGradient(at: Float(2), Float(3), Float(5), of: combine3)
    check(tvg.value, 11, "ternary VG value")
    check(tvg.gradient.0, 3, "ternary VG gradient")
    let tgrad = await gradient(at: Float(2), Float(3), Float(5), of: combine3)
    check(tgrad.1, 2, "ternary gradient API")

    let d = await opaque2(combineDouble, Double(2), Double(3))
    check(d.value, 11, "Double value")
    check(d.pullback(2).0, 8, "Double da")
    check(d.pullback(2).1, 6, "Double db")

    // A synchronous primal must still select the ordinary synchronous API.
    let s = valueWithPullback(at: Float(3), of: synchronous)
    check(s.value, 9, "sync value")
    check(s.pullback(1), 6, "sync PB")
    print("PASS async reverse API: arities 1/2/3, function values, generic consumer, nested calls, Double, synchronous regression")
  }
}
