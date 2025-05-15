/// Multi basic block VJP, pullback not accepting branch tracing enum argument.

// REQUIRES: executable_test

// RUN: %empty-directory(%t)
// RUN: %target-build-swift %s -o %t/none.out -Onone
// RUN: %target-build-swift %s -o %t/opt.out  -O
// RUN: %target-run %t/none.out
// RUN: %target-run %t/opt.out

// RUN: %target-swift-frontend -emit-sil %s -O -o %t/out.sil
// RUN: cat %t/out.sil | %FileCheck %s --check-prefix=CHECK1
// RUN: cat %t/out.sil | %FileCheck %s --check-prefix=CHECK2
// RUN: cat %t/out.sil | %FileCheck %s --check-prefix=CHECK3
// RUN: cat %t/out.sil | %FileCheck %s --check-prefix=CHECK4

import DifferentiationUnittest
import StdlibUnittest

var AutoDiffClosureSpecNoBTETests = TestSuite("AutoDiffClosureSpecNoBTE")

typealias FloatArrayTan = Array<Float>.TangentVector

AutoDiffClosureSpecNoBTETests.testWithLeakChecking("Test1") {
  // CHECK1-NONE:  {{^}}// pullback of sumFirstThreeConcatenating1
  // CHECK1-LABEL: {{^}}// specialized pullback of sumFirstThreeConcatenating1
  // CHECK1-NEXT:  {{^}}sil private
  // CHECK1-SAME:  : $@convention(thin) (Float, @owned @callee_guaranteed (@guaranteed Array<Float>.DifferentiableView) -> (@owned Array<Float>.DifferentiableView, @owned Array<Float>.DifferentiableView), @owned Array<Float>, Int, @owned Array<Float>, Int, @owned Array<Float>, Int) -> (@owned Array<Float>.DifferentiableView, @owned Array<Float>.DifferentiableView) {{{$}}
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
  @propertyWrapper
  enum Wrapper {
    case case1(Float)
    case case2(Float)

    init(wrappedValue: Float) {
      self = .case1(wrappedValue)
    }

    var wrappedValue: Float {
      get {
        switch self {
        case .case1(let val):
          return val
        case .case2(let val):
          return val * 2
        }
      }
      set {
        self = .case2(wrappedValue)
      }
    }
  }

  struct RealPropertyWrappers: Differentiable {
    @Wrapper var x: Float = 3
    var y: Float = 4
  }

  // CHECK3-NONE:  {{^}}// pullback of multiply
  // CHECK3-LABEL: {{^}}// specialized pullback of multiply
  // CHECK3-NEXT:  {{^}}sil private
  // CHECK3-SAME:  : $@convention(thin) (Float, Float, Float) -> RealPropertyWrappers.TangentVector {{{$}}
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
        let tmp = Class(stored: 1 * stored, optional: optional)
        let tuple = (tmp, tmp)
        c = tuple.0
      }
      if let x = c.optional {
        return x * c.stored
      }
      return 1 * c.stored
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
  // CHECK4-SAME:  : $@convention(thin) (Float, @owned _AD__$s3outyycfU2_5ClassL_V6methodSfyF_bb3__Pred__src_0_wrt_0) -> Class.TangentVector {{{$}}
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
