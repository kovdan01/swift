// RUN: %target-swift-frontend -emit-sil -parse-as-library %s -Onone -Xllvm -enable-experimental-async-autodiff -sil-verify-all -sil-ownership-verify-all -o /dev/null
// RUN: %target-swift-frontend -emit-sil -parse-as-library %s -O -Xllvm -enable-experimental-async-autodiff -sil-verify-all -sil-ownership-verify-all -o /dev/null
// REQUIRES: concurrency, differentiable_programming
// The standalone Inputs/run.py also executes this file and its synchronous
// counterpart. Nothing rewrites compiler-generated derivative bodies.
import _Differentiation
import _Concurrency

// BEGIN DIFFERENTIATED FUNCTIONS
@inline(never)
func square(_ x: Float) async -> Float { x * x }

@inline(never)
func childBranch(_ x: Float, _ multiply: Bool) async -> Float {
  if multiply { return x * x }
  return x + x
}

@differentiable(reverse, wrt: x)
func oneArm(_ x: Float, _ useChild: Bool) async -> Float {
  async let result = square(x)
  if useChild { return await result }
  return x
}

@differentiable(reverse, wrt: x)
func bothArms(_ x: Float, _ multiply: Bool) async -> Float {
  async let result = childBranch(x, multiply)
  let value: Float
  if multiply {
    value = await result
  } else {
    value = await result
  }
  return value + x
}

@differentiable(reverse)
func selectedArm(_ x: Float) async -> Float { await oneArm(x, true) }
@differentiable(reverse)
func unusedArm(_ x: Float) async -> Float { await oneArm(x, false) }
@differentiable(reverse)
func joinedMultiply(_ x: Float) async -> Float { await bothArms(x, true) }
@differentiable(reverse)
func joinedAdd(_ x: Float) async -> Float { await bothArms(x, false) }

@differentiable(reverse, wrt: x)
func loop(_ x: Float, _ count: Int) async -> Float {
  var result = x
  for _ in 0..<count {
    let input = result
    async let next = square(input)
    result = await next
  }
  return result
}
@differentiable(reverse)
func zeroIterations(_ x: Float) async -> Float { await loop(x, 0) }
@differentiable(reverse)
func twoIterations(_ x: Float) async -> Float { await loop(x, 2) }

@inline(never)
func loopInChild(_ x: Float, _ count: Int) async -> Float {
  var result = x
  for _ in 0..<count { result = result * x }
  return result
}
@differentiable(reverse)
func childWithLoop(_ x: Float) async -> Float {
  async let result = loopInChild(x, 2)
  return await result
}

@differentiable(reverse, wrt: x)
func breakAndContinue(_ x: Float, _ stop: Int) async -> Float {
  var result: Float = 0
  for index in 0..<4 {
    if index == stop { break }
    let input = x + Float(index)
    async let term = square(input)
    if index == 1 { continue }
    result = result + (await term)
  }
  return result
}
@differentiable(reverse)
func loopControl(_ x: Float) async -> Float { await breakAndContinue(x, 3) }

@differentiable(reverse)
func readSameChildInLoop(_ x: Float) async -> Float {
  async let child = square(x)
  var result: Float = 0
  for _ in 0..<3 { result = result + (await child) }
  return result
}
// END DIFFERENTIATED FUNCTIONS

func check(_ actual: Float, _ expected: Float, _ label: String) {
  precondition(actual.isFinite && abs(actual - expected) < 0.0001 * max(1, abs(expected)),
               "\(label): \(actual) != \(expected)")
}

@main struct CFGTests {
  nonisolated static func main() async {
    // The explicit function-value type makes accidental async pullbacks a
    // source-level type error, not just a textual SIL check.
    for x: Float in [0.5, 2, -2] {
      let a = /*AD_AWAIT*/ await valueWithPullback(at: x, of: selectedArm)
      let b = /*AD_AWAIT*/ await valueWithPullback(at: x, of: unusedArm)
      let c = /*AD_AWAIT*/ await valueWithPullback(at: x, of: joinedMultiply)
      let d = /*AD_AWAIT*/ await valueWithPullback(at: x, of: joinedAdd)
      let z = /*AD_AWAIT*/ await valueWithPullback(at: x, of: zeroIterations)
      let l = /*AD_AWAIT*/ await valueWithPullback(at: x, of: twoIterations)
      let i = /*AD_AWAIT*/ await valueWithPullback(at: x, of: childWithLoop)
      let k = /*AD_AWAIT*/ await valueWithPullback(at: x, of: loopControl)
      let r = /*AD_AWAIT*/ await valueWithPullback(at: x, of: readSameChildInLoop)
      check(a.value, x*x, "selected value")
      check(b.value, x, "unused value")
      check(c.value, x*x+x, "joined multiply value")
      check(d.value, 3*x, "joined add value")
      check(z.value, x, "zero loop value")
      check(l.value, x*x*x*x, "two loop value")
      check(i.value, x*x*x, "child loop value")
      check(k.value, x*x+(x+2)*(x+2), "break/continue value")
      check(r.value, 3*x*x, "same child loop value")
      await Task.yield()
      let reusable: (Float) -> Float = l.pullback
      for seed: Float in [1, 2, 0, -3, 0.5] {
        check(a.pullback(seed), 2*x*seed, "selected derivative")
        check(b.pullback(seed), seed, "unused derivative")
        check(c.pullback(seed), (2*x+1)*seed, "joined multiply derivative")
        check(d.pullback(seed), 3*seed, "joined add derivative")
        check(z.pullback(seed), seed, "zero loop derivative")
        check(reusable(seed), 4*x*x*x*seed, "two loop derivative")
        check(i.pullback(seed), 3*x*x*seed, "child loop derivative")
        check(k.pullback(seed), (4*x+4)*seed, "break/continue derivative")
        check(r.pullback(seed), 6*x*seed, "same child loop derivative")
      }
      print("cfg x=\(x) value=\(l.value) derivative=\(reusable(1))")
    }
    for _ in 0..<64 {
      _ = /*AD_AWAIT*/ await valueWithPullback(at: Float(2), of: twoIterations)
    }
    print("PASS cfg")
  }
}
