/// This test is work-in-progress.

// REQUIRES: executable_test

// RUN: %empty-directory(%t)
// RUN: %target-build-swift-dylib(%t/%target-library-name(closure_spec_module)) %S/Inputs/closure_spec_module.swift \
// RUN:   -emit-module -emit-module-path %t/closure_spec_module.swiftmodule -module-name closure_spec_module
// RUN: %target-build-swift -I%t -L%t %s -lclosure_spec_module %target-rpath(%t) -o %t/none.out -Onone
// RUN: %target-build-swift -I%t -L%t %s -lclosure_spec_module %target-rpath(%t) -o %t/opt.out  -O
// RUN: %target-run %t/none.out
// RUN: %target-run %t/opt.out

/// Particular optimizations are checked in SILOptimizer tests, here we only check that optimizations occur
// RUN: %target-swift-frontend -emit-sil -I%t %s -o - -O | %FileCheck %s
// CHECK-NONE: {{^}}// pullback of myfoo08
// CHECK-NONE: {{^}}// pullback of myfoo09
// CHECK-NONE: {{^}}// pullback of myfoo10
// CHECK-NONE: {{^}}// pullback of myfoo11
// CHECK-NONE: {{^}}// pullback of myfoo12
// CHECK-NONE: {{^}}// pullback of myfoo14
// CHECK-NONE: {{^}}// pullback of myfoo15
// CHECK-NONE: {{^}}// pullback of myfoo17
// CHECK-NONE: {{^}}// pullback of myfoo16
// CHECK-NONE: {{^}}// pullback of myfoo07
// CHECK-NONE: {{^}}// pullback of myfoo06
// CHECK-NONE: {{^}}// pullback of myfoo05
// CHECK-NONE: {{^}}// pullback of myfoo04
// CHECK-NONE: {{^}}// pullback of myfoo03
// CHECK-NONE: {{^}}// pullback of myfoo02
// CHECK-NONE: {{^}}// pullback of myfoo01
// CHECK:      {{^}}// specialized pullback of myfoo08
// CHECK:      {{^}}// specialized pullback of myfoo09
// CHECK:      {{^}}// specialized pullback of myfoo10
// CHECK:      {{^}}// specialized pullback of myfoo11
// CHECK:      {{^}}// specialized pullback of myfoo12
// CHECK:      {{^}}// specialized pullback of myfoo14
// CHECK:      {{^}}// specialized pullback of myfoo15
// CHECK:      {{^}}// specialized pullback of myfoo17
// CHECK:      {{^}}// specialized pullback of myfoo16
// CHECK:      {{^}}// specialized pullback of myfoo07
// CHECK:      {{^}}// specialized pullback of myfoo06
// CHECK:      {{^}}// specialized pullback of myfoo05
// CHECK:      {{^}}// specialized pullback of myfoo04
// CHECK:      {{^}}// specialized pullback of myfoo03
// CHECK:      {{^}}// specialized pullback of myfoo02
// CHECK:      {{^}}// specialized pullback of myfoo01

import DifferentiationUnittest
import StdlibUnittest
import closure_spec_module

#if canImport(Glibc)
  import Glibc
#elseif canImport(Android)
  import Android
#else
  import Foundation
#endif

var AutoDiffClosureSpecializationTests = TestSuite("AutoDiffClosureSpecialization")

extension Dictionary: Differentiable where Value: Differentiable {
    public typealias TangentVector = [Key: Value.TangentVector]
    public mutating func move(by direction: TangentVector) {
        for (componentKey, componentDirection) in direction {
            func fatalMissingComponent() -> Value {
                fatalError("missing component \(componentKey) in moved Dictionary")
            }
            self[componentKey, default: fatalMissingComponent()].move(by: componentDirection)
        }
    }

    public var zeroTangentVectorInitializer: () -> TangentVector {
        let listOfKeys = self.keys // capturing only what's needed, not the entire self, in order to not waste memory
        func initializer() -> Self.TangentVector {
            return listOfKeys.reduce(into: [Key: Value.TangentVector]()) { $0[$1] = Value.TangentVector.zero }
        }
        return initializer
    }
}

extension Dictionary: AdditiveArithmetic where Value: AdditiveArithmetic {
    public static func + (_ lhs: Self, _ rhs: Self) -> Self {
        return lhs.merging(rhs, uniquingKeysWith: +)
    }

    public static func - (_ lhs: Self, _ rhs: Self) -> Self {
        return lhs.merging(rhs.mapValues { .zero - $0 }, uniquingKeysWith: +)
    }

    public static var zero: Self { [:] }
}

extension Dictionary where Value: Differentiable {
    // get
    @usableFromInline
    @derivative(of: subscript(_:))
    func vjpSubscriptGet(key: Key) -> (value: Value?, pullback: (Optional<Value>.TangentVector) -> Dictionary<Key, Value>.TangentVector) {
        // When adding two dictionaries, nil values are equivalent to zeroes, so there is no need to manually zero-out
        // every key's value. Instead, it is faster to create a dictionary with the single non-zero entry.
        return (self[key], { v in
            if let value = v.value {
                return [key: value]
            }
            else {
                return .zero
            }
        })
    }
 }

public extension Dictionary where Value: Differentiable {
    @differentiable(reverse)
    mutating func set(_ key: Key, to newValue: Value) {
        self[key] = newValue
    }

    @derivative(of: set)
    mutating func vjpUpdated(_ key: Key, to newValue: Value) -> (value: Void, pullback: (inout TangentVector) -> (Value.TangentVector)) {
        self.set(key, to: newValue)

        let forwardCount = self.count
        let forwardKeys = self.keys // may be heavy to capture all of these, not sure how to do without them though

        return ((), { v in
            // manual zero tangent initialization
            if v.count < forwardCount {
                v = Self.TangentVector()
                forwardKeys.forEach { v[$0] = .zero }
            }

            if let dElement = v[key] {
                v[key] = .zero
                return dElement
            }
            else { // should this fail?
                v[key] = .zero
                return .zero
            }
        })
    }
}


AutoDiffClosureSpecializationTests.testWithLeakChecking("Test") {
  func myfoo01(_ x: Float) -> Float {
    if x > 0 {
      return mybar1(x) + mybar2(x)
    } else {
      return mybar1(x) * mybar2(x)
    }
  }

  func myfoo01_derivative(_ x: Float) -> Float {
    if x > 0 {
      return 2 * x + 1
    } else {
      return 3 * x * x
    }
  }

  for x in -100...100 {
    expectEqual(myfoo01_derivative(Float(x)), gradient(at: Float(x), of: myfoo01))
  }

  func myfoo02(_ x: Float) -> Float {
    let y = mybar1(x) + 37 * mybar2(x)
    var z: Float = 0
    if x > 0 {
      z = mybar1(y) * mybar2(x) + mybar2(y)
    } else {
      z = mybar1(x) * mybar1(y)
    }
    return z * x + y * y
  }

  func myfoo02_derivative(_ x: Float) -> Float {
    if x > 0 {
      return 7030 * x * x * x * x + 5776 * x * x * x + 225 * x * x + 2 * x
    } else {
      return 5624 * x * x * x + 225 * x * x + 2 * x
    }
  }

  for x in -14...6 {
    expectEqual(myfoo02_derivative(Float(x)), gradient(at: Float(x), of: myfoo02))
  }

  func myfoo03(_ x: Float) -> Float {
    if x > 0 {
      return mybar2(x) * mybar2(x)
    } else {
      var y = mybar1(x) + mybar2(x)
      if x > -10 {
        return x + mybar2(y)
      } else {
        return mybar1(y) * x
      }
    }
  }

  func myfoo03_derivative(_ x: Float) -> Float {
    if x > 0 {
      return 4 * x * x * x
    } else {
      if x > -10 {
        return 4 * x * x * x + 6 * x * x + 2 * x + 1
      } else {
        return 3 * x * x + 2 * x
      }
    }
  }

  for x in -100...100 {
    expectEqual(myfoo03_derivative(Float(x)), gradient(at: Float(x), of: myfoo03))
  }

  func myfoo04(_ x: Float) -> Float {
    if x > 0 {
      return mybar2(x) * mybar2(x)
    } else {
      let y = mybar1(x) + mybar2(x)
      var z: Float = 37
      if x > -7 {
        z += y + mybar2(x)
      } else {
        z -= mybar1(x) * y
      }
      return mybar1(z) * mybar2(y)
    }
  }

  func myfoo04_derivative(_ x: Float) -> Float {
    if x > 0 {
      return 4 * x * x * x
    } else if x > -7 {
      return 12 * x * x * x * x * x + 25 * x * x * x * x + 164 * x * x * x + 225 * x * x + 74 * x
    } else {
      return -7 * x * x * x * x * x * x - 18 * x * x * x * x * x - 15 * x * x * x * x + 144 * x * x
        * x + 222 * x * x + 74 * x
    }
  }

  for x in -20...100 {
    expectEqual(myfoo04_derivative(Float(x)), gradient(at: Float(x), of: myfoo04))
  }

  func myfoo05(_ x: Float) -> Float {
    if x > 0 {
      return sin(x) * cos(x)
    } else {
      return sin(x) + cos(x)
    }
  }

  func myfoo05_derivative(_ x: Float) -> Float {
    if x > 0 {
      return cos(x) * cos(x) - sin(x) * sin(x)
    } else {
      return cos(x) - sin(x)
    }
  }

  for x in -100...100 {
    expectEqual(myfoo05_derivative(Float(x)), gradient(at: Float(x), of: myfoo05))
  }

  func double(x: Float) -> Float {
    return x + x
  }

  func square(x: Float) -> Float {
    return x * x
  }

  func myfoo06(_ x: Float) -> Float {
    if x > 0 {
      if (x + 1) < 5 {
        if x * 2 > 4 {
          let y = square(x: x)
          if y >= x {
            let d = double(x: x)
            return x - (d * y)
          } else {
            let e = square(x: y)
            return x + (e * y)
          }
        }
      }
    } else {
      let y = double(x: x)
      return x * y
    }

    return x * 3
  }

  func myfoo06_derivative(_ x: Float) -> Float {
    if x > 2 && x < 4 {
      return 1 - 6 * x * x
    } else if x > 0 {
      return 3
    } else {
      return 4 * x
    }
  }

  for x in -100...100 {
    expectEqual(myfoo06_derivative(Float(x)), gradient(at: Float(x), of: myfoo06))
  }

  var x : Float = -10
  while x < 10 {
    let a = myfoo06_derivative(Float(x))
    let b = gradient(at: Float(x), of: myfoo06)
    expectEqual((a * 10000).rounded(), (b * 10000).rounded())
    x += 1 / 1024
  }

  func myfoo07(_ x: Float, _ bool: Bool) -> Float {
    var result = x * x
    if bool {
      result = result + result
      result = result - x
    } else {
      result = result / x
    }
    return result
  }

  func myfoo07_derivative(_ x: Float, _ bool: Bool) -> Float {
    if bool {
      return 4 * x - 1
    }
    return 1
  }

  for x in -100...100 {
    if x == 0 {
      continue
    }
    expectEqual(gradient(at: Float(x), of: { myfoo07(Float($0), true)  }), myfoo07_derivative(Float(x), true))
    expectEqual((100000 * gradient(at: Float(x), of: { myfoo07(Float($0), false) })).rounded(), (100000 * myfoo07_derivative(Float(x), false)).rounded())
  }

  // x0 > x1  && x1 < 0:  1 - x1 / x0
  // x0 > x1  && x1 >= 0: 1 + x1 / x0
  // x0 <= x1:            1
  @differentiable(reverse)
  func myfoo08(_ x0: Float, _ x1: Float) -> Float {
    let t1 = x1 + x0;
    let t2 = x0 - x1;
    let t4 = x0 > x1 ? t1 : x0;
    let t6 = 1 / x0;
    let t5 = x0 > t4 ? t2 : t4;
    let t7 = t5 * t6;
    return t7;
  }

  func myfoo08_gradient(_ x0: Float, _ x1: Float) -> (Float, Float) {
    if x0 > x1 {
      if x1 < 0 {
        return (x1 / (x0 * x0), -1 / x0)
      }
      return (-x1 / (x0 * x0), 1 / x0)
    }
    // TODO: shouldn't this be (0, 0)?
    return (1 / x0, 0)
  }

  for x0 in -10...10 {
    if x0 == 0 {
      continue
    }
    for x1 in -10...10 {
      let got = gradient(at: Float(x0), Float(x1), of: myfoo08)
      let expected = myfoo08_gradient(Float(x0), Float(x1))
      expectEqual((100000 * got.0).rounded(), (100000 * expected.0).rounded())
      expectEqual((100000 * got.1).rounded(), (100000 * expected.1).rounded())
    }
  }

  enum Enum {
    case a(Float)
    case b(Float)
  }

  @differentiable(reverse)
  func myfoo09(_ e: Enum, _ x: Float) -> Float {
    switch e {
    case let .a(a): return x * a
    case let .b(b): return x * b
    }
  }

  indirect enum Indirect {
    case e(Enum)
    case indirect(Indirect)
  }

  @differentiable(reverse)
  func myfoo10(_ indirect: Indirect, _ x: Float) -> Float {
    switch indirect {
    case let .e(e):
      switch e {
      case .a: return x * myfoo09(e, x)
      case .b: return x * myfoo09(e, x)
      }
    case let .indirect(ind): return myfoo10(ind, x)
    }
  }

  do {
    let ind: Indirect = .e(.a(3))
    expectEqual(12, gradient(at: 2, of: { x in myfoo10(ind, x) }))
    expectEqual(12, gradient(at: 2, of: { x in myfoo10(.indirect(ind), x) }))
  }

  @differentiable(reverse)
  func myfoo11(_ x: Float, _ y: Float) -> Float {
    if x > 0 {
      if y > 10 {
        let z = x * y
        if z > 100 {
          return x + z
        } else if y == 20 {
          return z + z
        }
      } else {
        return x + y
      }
    }
    return -y
  }

  expectEqual((40, 8), gradient(at: 4, 20, of: myfoo11))
  expectEqual((0, -1), gradient(at: 4, 21, of: myfoo11))
  expectEqual((1, 1), gradient(at: 4, 5, of: myfoo11))
  expectEqual((0, -1), gradient(at: -3, -2, of: myfoo11))


  @differentiable(reverse)
  func myfoo12(x0: Double, x1: Double, x2: Optional<Double>, x3: Optional<Double>) -> Double {
    let a : Double = x1 * 42
    let b : Double = a + x0
    var c : Double = 0
    if let x2NonNil = x2 {
      if x2NonNil < b {
        c = x2NonNil
      } else {
        c = b
      }
    } else {
      c = b
    }

    var d : Double = 0
    if let x3NonNil = x3 {
      if x3NonNil < c {
        d = x3NonNil
      } else {
        d = c
      }
    } else {
      d = c
    }

    return d
  }


  struct X: Differentiable {
    var a: Float
    var b: Double
  }

  @differentiable(reverse)
  func myfoo13Callee(x: X) -> (Float, Double) {
    (x.a, x.b)
  }

  @differentiable(reverse)
  func myfoo13(x: X, y: Int) -> Float {
    if y < 0 {
      return myfoo13Callee(x: x).0
    } else {
      return 37
    }
  }

  @differentiable(reverse)
  func myfoo14(_ x : Float, _ y : Float) -> (Float, Float) {
    if x > 0 {
      if y > 2 {
        return (y - 37, x + 5)
      } else {
        return (y * 42, x - 3)
      }
    }
    return (7 * x, 8 - y)
  }

  struct FloatPair : Differentiable {
    var first, second: Float
    init(_ first: Float, _ second: Float) {
      self.first = first
      self.second = second
    }
  }

  struct Pair<T : Differentiable, U : Differentiable> : Differentiable {
    var first: T
    var second: U
    init(_ first: T, _ second: U) {
      self.first = first
      self.second = second
    }
  }

  @differentiable(reverse)
  func myfoo15(_ x: Float) -> Float {
    // Convoluted function returning `x + x`.
    var y = FloatPair(x + x, x - x)
    var z = Pair(y, x)
    if x > 0 {
      var w = FloatPair(x, x)
      y.first = w.second
      y.second = w.first
      z.first.first = z.first.first - y.first
      z.first.second = z.first.second + y.first
    } else {
      z = Pair(FloatPair(y.first - x, y.second + x), x)
    }
    return y.first + y.second - z.first.first + z.first.second
  }

  expectEqual((8, 2), valueWithGradient(at: 4, of: myfoo15))
  expectEqual((-20, 2), valueWithGradient(at: -10, of: myfoo15))
  expectEqual((-2674, 2), valueWithGradient(at: -1337, of: myfoo15))

  func myfoo16_getD(from newValues: [String: Double], at key: String) -> Double? {
    if newValues.keys.contains(key) {
      return newValues[key]
    }
    return nil
  }

  @differentiable(reverse)
  func testFunctionD(newValues: [String: Double]) -> Double {
    return myfoo16_getD(from: newValues, at: "s1")!
  }

  expectEqual(pullback(at: ["s1": 1.0], of: testFunctionD)(2), ["s1" : 2.0])

  func myfoo17_getG<DataType>(from newValues: [String: DataType], at key: String) -> DataType?
  where DataType: Differentiable {
    if newValues.keys.contains(key) {
      return newValues[key]
    }
    return nil
  }

  @differentiable(reverse)
  func testFunctionG(newValues: [String: Double]) -> Double {
    return myfoo17_getG(from: newValues, at: "s1")!
  }

  expectEqual(pullback(at: ["s1": 1.0], of: testFunctionG)(2), ["s1" : 2.0])
}

runAllTests()
