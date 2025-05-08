// REQUIRES: executable_test

// RUN: %empty-directory(%t)
// RUN: %target-build-swift %s -o %t/none.out -Onone
// RUN: %target-build-swift %s -o %t/opt.out  -O
// RUN: %target-run %t/none.out
// RUN: %target-run %t/opt.out

/// TODO: make this agnostic to functions order in SIL
// RUN: %target-swift-frontend -emit-sil %s -o - -O 2> %t/adcs.log | %FileCheck %s
// CHECK-NONE: {{^}}// pullback of multiply
// CHECK-NONE: {{^}}// pullback of methodWrapper
// CHECK-NONE: {{^}}// pullback of sumFirstThreeConcatenating2
// CHECK-NONE: {{^}}// pullback of sumFirstThreeConcatenating1

// CHECK:      {{^}}// specialized pullback of multiply
// CHECK-NEXT: {{^}}sil private
// CHECK-SAME: : $@convention(thin) (Float, Float, Float) -> RealPropertyWrappers.TangentVector {{{$}}

/// TODO: even though branch tracing enum was not passed to top-level pullback
/// directly, it was captured by one of the closures which was specialized.
/// Because of that, this enum argument is now an argument of top-level pullback.
/// Specializing closures passed as payload tuple elements of the enum is currently
/// not supported.
// CHECK:      {{^}}// specialized pullback of methodWrapper
// CHECK-NEXT: {{^}}sil private
// CHECK-SAME: : $@convention(thin) (Float, @owned _AD__$s40closure_spec_without_branch_tracing_enumyycfU2_5ClassL_V6methodSfyF_bb3__Pred__src_0_wrt_0) -> Class.TangentVector {{{$}}

// CHECK:      {{^}}// specialized pullback of sumFirstThreeConcatenating2
// CHECK-NEXT: {{^}}sil private
// CHECK-SAME: : $@convention(thin) (Float, Int, @owned Array<Float>, Int, @owned Array<Float>, Int, @owned Array<Float>, Int) -> (@owned Array<Float>.DifferentiableView, @owned Array<Float>.DifferentiableView) {{{$}}

// CHECK:      {{^}}// specialized pullback of sumFirstThreeConcatenating1
// CHECK-NEXT: {{^}}sil private
// CHECK-SAME: : $@convention(thin) (Float, @owned @callee_guaranteed (@guaranteed Array<Float>.DifferentiableView) -> (@owned Array<Float>.DifferentiableView, @owned Array<Float>.DifferentiableView), @owned Array<Float>, Int, @owned Array<Float>, Int, @owned Array<Float>, Int) -> (@owned Array<Float>.DifferentiableView, @owned Array<Float>.DifferentiableView) {{{$}}

// RUN: cat %t/adcs.log | grep \[ADCS\] | %FileCheck --check-prefix=LOG %s
// LOG:      [ADCS][1] Trying to run AutoDiff Closure Specialization pass on $s40closure_spec_without_branch_tracing_enumyycfU2_13methodWrapperL_ySfAAyycfU2_5ClassL_VFTJrSpSr
// LOG-NEXT: This is multi-BB case which would be handled as single-BB case
// LOG:      [ADCS][2] Trying to run AutoDiff Closure Specialization pass on $s40closure_spec_without_branch_tracing_enumyycfU1_8multiplyL_ySfAAyycfU1_20RealPropertyWrappersL_VFTJrSpSr
// LOG-NEXT: This is multi-BB case which would be handled as single-BB case
// LOG:      [ADCS][2] Trying to run AutoDiff Closure Specialization pass on $s40closure_spec_without_branch_tracing_enumyycfU0_27sumFirstThreeConcatenating2L_ySfSaySfG_ACtFTJrSSpSr
// LOG-NEXT: This is multi-BB case which would be handled as single-BB case
// LOG:      [ADCS][2] Trying to run AutoDiff Closure Specialization pass on $s40closure_spec_without_branch_tracing_enumyycfU_27sumFirstThreeConcatenating1L_ySfSaySfG_ACtFTJrSSpSr
// LOG-NEXT: This is multi-BB case which would be handled as single-BB case

import DifferentiationUnittest
import StdlibUnittest

var AutoDiffClosureSpecializationWOBranchTracingEnumTests = TestSuite(
  "AutoDiffClosureSpecializationWOBranchTracingEnum")

typealias FloatArrayTan = Array<Float>.TangentVector

AutoDiffClosureSpecializationWOBranchTracingEnumTests.testWithLeakChecking("Test1") {
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

  func identity(_ array: [Float]) -> [Float] {
    var results: [Float] = []
    for i in withoutDerivative(at: array.indices) {
      results = results + [array[i]]
    }
    return results
  }
  let v = FloatArrayTan([4, -5, 6])
  expectEqual(v, pullback(at: [1, 2, 3], of: identity)(v))

  let v1: [Float] = [1, 1]
  let v2: [Float] = [1, 1, 1]
  expectEqual((.zero, .zero), pullback(at: v1, v2, of: +)(.zero))
}

AutoDiffClosureSpecializationWOBranchTracingEnumTests.testWithLeakChecking("Test2") {
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

AutoDiffClosureSpecializationWOBranchTracingEnumTests.testWithLeakChecking("Test3") {
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

  @differentiable(reverse)
  func multiply(_ s: RealPropertyWrappers) -> Float {
    return s.x * s.y
  }

  expectEqual(
    .init(x: 4, y: 3),
    gradient(at: RealPropertyWrappers(x: 3, y: 4), of: multiply))
}

AutoDiffClosureSpecializationWOBranchTracingEnumTests.testWithLeakChecking("Test4") {
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
