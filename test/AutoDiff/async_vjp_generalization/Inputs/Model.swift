import _Differentiation

// Build with library evolution. Clients must not assume this nominal value or
// its tangent is loadable or that its stored-property layout is available.
public struct Packet: Differentiable, Sendable {
  public var values: [Float]
  public var scale: Float
  @noDerivative public var tag: Int

  public init(values: [Float], scale: Float, tag: Int) {
    self.values = values
    self.scale = scale
    self.tag = tag
  }
}
