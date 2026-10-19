import _Differentiation

@inline(never)
@differentiable(reverse)
public func square(_ x: Double) -> Double {
  x * x
}

// ---------------


public protocol Test1Protocol {
  func value() -> Double
}

@inline(never)
public func test1WithSquare(_ body: (@differentiable(reverse) (Double) -> Double) -> Double) -> Double {
  body(square)
}

public func test1Entry(_ p: any Test1Protocol) -> Double {
  @inline(never)
  func test1Evaluate(_ function: @differentiable(reverse) (Double) -> Double) -> Double {
    gradient(at: p.value(), of: function)
  }

  return test1WithSquare(test1Evaluate)
}

// ------------

public class Test2Base {
  public let value: Double

  public init(_ value: Double) {
    self.value = value
  }
}

public class Test2Derived: Test2Base {}

@inline(never)
@differentiable(reverse, wrt: x)
public func test2Evaluate(_ x: Double, _ context: Test2Base) -> Double {
  x * x + context.value
}

@inline(never)
public func test2UseConverted(
  _ x: Double, _ context: Test2Derived,
  _ function: @differentiable(reverse) (Double, @noDerivative Test2Derived) -> Double
) -> Double {
  function(x, context)
}

public func test2Entry(_ x: Double, _ context: Test2Derived) -> Double {
  test2UseConverted(x, context, test2Evaluate)
}

// --------------------


@inline(never)
public func test3Evaluate(
  _ x: Double,
  _ function: @differentiable(reverse) (Double) -> Double
) -> Double {
  gradient(at: x, of: function)
}

@inline(never)
public func test3Forward<T>(
  _ x: T,
  _ body: (T, @differentiable(reverse) (Double) -> Double) -> T
) -> T {
  body(x, square)
}

public func test3Entry(_ x: Double) -> Double {
  test3Forward(x, test3Evaluate)
}


// --------------


@inline(never)
public func test4Evaluate<T, R>(
  _ x: T,
  _ function: @differentiable(reverse) (T) -> R
) -> T.TangentVector
where
  T: Differentiable,
  R: Differentiable & FloatingPoint,
  R == R.TangentVector
{
  gradient(at: x, of: function)
}

@inline(__always)
public func test4Forward<T, R>(
  _ x: T,
  _ function: @escaping @differentiable(reverse) (T) -> R
) -> T.TangentVector
where
  T: Differentiable,
  R: Differentiable & FloatingPoint,
  R == R.TangentVector
{
  test4Evaluate(x, function)
}

public func test4Entry(_ x: Double) -> Double {
  test4Forward(x, square)
}

