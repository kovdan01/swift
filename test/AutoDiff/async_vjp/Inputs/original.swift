import _Differentiation

func workload1(_ a: Float, _ b: Float) async -> Float {
  a + b
}

func workload2(_ a: Float, _ b: Float) async -> Float {
  a * b
}

@differentiable(reverse)
func doWor5k(_ a: Float, _ b: Float) async -> Float {
  async let result1 = workload1(a, b)
  async let result2 = workload2(a, b)
  return await result1 + result2
}
