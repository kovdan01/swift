/// AutoDiff Closure Specialization pass for cases when top-level pullback does
/// not have branch tracing enum argument.

// REQUIRES: executable_test

// RUN: %empty-directory(%t)
// RUN: %target-build-swift %s -o %t/none.out -Onone
// RUN: %target-build-swift %s -o %t/opt.out  -O
// RUN: %target-run %t/none.out
// RUN: %target-run %t/opt.out

// RUN: %target-swift-frontend -emit-sil %s -o - -O 1> %t/out.sil 2> %t/adcs.log

// RUN: cat %t/out.sil | %FileCheck %s --check-prefix=CHECK1
// RUN: cat %t/out.sil | %FileCheck %s --check-prefix=CHECK2
// RUN: cat %t/out.sil | %FileCheck %s --check-prefix=CHECK3
// RUN: cat %t/out.sil | %FileCheck %s --check-prefix=CHECK4

// RUN: grep \[ADCS\] %t/adcs.log | %FileCheck --check-prefix=LOG1 %s
// RUN: grep \[ADCS\] %t/adcs.log | %FileCheck --check-prefix=LOG2 %s
// RUN: grep \[ADCS\] %t/adcs.log | %FileCheck --check-prefix=LOG3 %s
// RUN: grep \[ADCS\] %t/adcs.log | %FileCheck --check-prefix=LOG4 %s

import DifferentiationUnittest
import StdlibUnittest

var AutoDiffClosureSpecNoBTETests = TestSuite("AutoDiffClosureSpecNoBTE")

typealias FloatArrayTan = Array<Float>.TangentVector

AutoDiffClosureSpecNoBTETests.testWithLeakChecking("Test1") {
  // CHECK1-NONE:  {{^}}// pullback of sumFirstThreeConcatenating1
  // CHECK1-LABEL: {{^}}// specialized pullback of sumFirstThreeConcatenating1
  // CHECK1-NEXT:  {{^}}sil private
  // CHECK1-SAME:  : $@convention(thin) (Float, @owned @callee_guaranteed (@guaranteed Array<Float>.DifferentiableView) -> (@owned Array<Float>.DifferentiableView, @owned Array<Float>.DifferentiableView), @owned Array<Float>, Int, @owned Array<Float>, Int, @owned Array<Float>, Int) -> (@owned Array<Float>.DifferentiableView, @owned Array<Float>.DifferentiableView) {{{$}}

  // LOG1-LABEL: [ADCS][2] Trying to run AutoDiff Closure Specialization pass on $s15multi_bb_no_bteyycfU_27sumFirstThreeConcatenating1L_ySfSaySfG_ACtFTJrSSpSr
  // LOG1-NEXT:  This is multi-BB case which would be handled as single-BB case

  func sumFirstThreeConcatenating1(_ a: [Float], _ b: [Float]) -> Float {
    let c = a + b
    return c[0] + c[1] + c[2]
  }

  expectEqual(
    (.init([1, 1]), .init([1, 0])),
    gradient(at: [0, 0], [0, 0], of: sumFirstThreeConcatenating1))
  expectEqual(
    (.init([1, 1, 1, 0]), .init([0, 0])),
    gradient(at: [0, 0, 0, 0], [0, 0], of: sumFirstThreeConcatenating1))
  expectEqual(
    (.init([]), .init([1, 1, 1, 0])),
    gradient(at: [], [0, 0, 0, 0], of: sumFirstThreeConcatenating1))
}

AutoDiffClosureSpecNoBTETests.testWithLeakChecking("Test2") {
  // CHECK2-NONE:  {{^}}// pullback of sumFirstThreeConcatenating2
  // CHECK2-LABEL: {{^}}// specialized pullback of sumFirstThreeConcatenating2
  // CHECK2-NEXT:  {{^}}sil private
  // CHECK2-SAME:  : $@convention(thin) (Float, Int, @owned Array<Float>, Int, @owned Array<Float>, Int, @owned Array<Float>, Int) -> (@owned Array<Float>.DifferentiableView, @owned Array<Float>.DifferentiableView) {{{$}}

  // LOG2-LABEL: [ADCS][2] Trying to run AutoDiff Closure Specialization pass on $s15multi_bb_no_bteyycfU0_27sumFirstThreeConcatenating2L_ySfSaySfG_ACtFTJrSSpSr
  // LOG2-NEXT:  This is multi-BB case which would be handled as single-BB case

  func sumFirstThreeConcatenating2(_ a: [Float], _ b: [Float]) -> Float {
    var c = a
    c += b
    return c[0] + c[1] + c[2]
  }

  expectEqual(
    (.init([1, 1]), .init([1, 0])),
    gradient(at: [0, 0], [0, 0], of: sumFirstThreeConcatenating2))
  expectEqual(
    (.init([1, 1, 1, 0]), .init([0, 0])),
    gradient(at: [0, 0, 0, 0], [0, 0], of: sumFirstThreeConcatenating2))
  expectEqual(
    (.init([]), .init([1, 1, 1, 0])),
    gradient(at: [], [0, 0, 0, 0], of: sumFirstThreeConcatenating2))
}

AutoDiffClosureSpecNoBTETests.testWithLeakChecking("Test3") {
  // From: https://github.com/apple/swift-evolution/blob/master/proposals/0258-property-wrappers.md#proposed-solution
  // Tests the following functionality:
  // - Enum property wrapper.
  @propertyWrapper
  enum Lazy<Value> {
    case uninitialized(() -> Value)
    case initialized(Value)

    init(wrappedValue: @autoclosure @escaping () -> Value) {
      self = .uninitialized(wrappedValue)
    }

    var wrappedValue: Value {
      // TODO(TF-1250): Replace with actual mutating getter implementation.
      // Requires differentiation to support functions with multiple results.
      get {
        switch self {
        case .uninitialized(let initializer):
          let value = initializer()
          // NOTE: Actual implementation assigns to `self` here.
          return value
        case .initialized(let value):
          return value
        }
      }
      set {
        self = .initialized(newValue)
      }
    }
  }

  // From: https://github.com/apple/swift-evolution/blob/master/proposals/0258-property-wrappers.md#clamping-a-value-within-bounds
  @propertyWrapper
  struct Clamping<V: Comparable> {
    var value: V
    let min: V
    let max: V

    init(wrappedValue: V, min: V, max: V) {
      value = wrappedValue
      self.min = min
      self.max = max
      assert(value >= min && value <= max)
    }

    var wrappedValue: V {
      get { return value }
      set {
        if newValue < min {
          value = min
        } else if newValue > max {
          value = max
        } else {
          value = newValue
        }
      }
    }
  }

  struct RealPropertyWrappers: Differentiable {
    @Lazy var x: Float = 3

    @Clamping(min: -10, max: 10)
    var y: Float = 4
  }

  // CHECK3-NONE:  {{^}}// pullback of multiply
  // CHECK3-LABEL: {{^}}// specialized pullback of multiply
  // CHECK3-NEXT:  {{^}}sil private
  // CHECK3-SAME:  : $@convention(thin) (Float, Float, Float) -> RealPropertyWrappers.TangentVector {{{$}}

  // LOG3-LABEL: [ADCS][2] Trying to run AutoDiff Closure Specialization pass on $s15multi_bb_no_bteyycfU1_8multiplyL_ySfAAyycfU1_20RealPropertyWrappersL_VFTJrSpSr
  // LOG3-NEXT:  This is multi-BB case which would be handled as single-BB case

  @differentiable(reverse)
  func multiply(_ s: RealPropertyWrappers) -> Float {
    return s.x * s.y
  }

  expectEqual(
    .init(x: 4, y: 3),
    gradient(at: RealPropertyWrappers(x: 3, y: 4), of: multiply))
}

AutoDiffClosureSpecNoBTETests.testWithLeakChecking("Test4") {
  struct Class: Differentiable {
    var stored: Float
    var optional: Float?

    init(stored: Float, optional: Float?) {
      self.stored = stored
      self.optional = optional
    }

    @differentiable(reverse)
    func method() -> Float {
      let c: Class
      do {
        let tmp = Class(stored: stored, optional: optional)
        let tuple = (tmp, tmp)
        c = tuple.0
      }
      if let x = c.optional {
        return x * c.stored
      }
      return c.stored
    }
  }

  /// TODO: even though branch tracing enum is not passed to top-level pullback
  /// directly, it is captured by one of the closures which was specialized.
  /// Because of that, this enum argument is now an argument of specialized top-level
  /// pullback. Specializing closures passed as payload tuple elements of the enum
  /// is currently not supported.

  // CHECK4-NONE:  {{^}}// pullback of methodWrapper
  // CHECK4-LABEL: {{^}}// specialized pullback of methodWrapper
  // CHECK4-NEXT:  {{^}}sil private
  // CHECK4-SAME:  : $@convention(thin) (Float, @owned _AD__$s15multi_bb_no_bteyycfU2_5ClassL_V6methodSfyF_bb3__Pred__src_0_wrt_0) -> Class.TangentVector {{{$}}

  // LOG4-LABEL: [ADCS][1] Trying to run AutoDiff Closure Specialization pass on $s15multi_bb_no_bteyycfU2_13methodWrapperL_ySfAAyycfU2_5ClassL_VFTJrSpSr
  // LOG4-NEXT:  This is multi-BB case which would be handled as single-BB case

  @differentiable(reverse)
  func methodWrapper(_ x: Class) -> Float {
    x.method()
  }

  expectEqual(
    valueWithGradient(at: Class(stored: 3, optional: 4), of: methodWrapper),
    (12, .init(stored: 4, optional: .init(3))))
  expectEqual(
    valueWithGradient(at: Class(stored: 3, optional: nil), of: methodWrapper),
    (3, .init(stored: 1, optional: .init(0))))
}

runAllTests()
