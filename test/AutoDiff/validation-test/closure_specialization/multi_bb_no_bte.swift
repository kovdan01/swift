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
  // CHECK1-LABEL: // reverse-mode derivative of sumFirstThreeConcatenating1 #1 (_:_:) in closure #1 in
  // CHECK1-NEXT:  sil private @$s3outyycfU_27sumFirstThreeConcatenating1L_ySfSaySfG_ACtFTJrSSpSr : $@convention(thin) (@guaranteed Array<Float>, @guaranteed Array<Float>) -> (Float, @owned @callee_guaranteed (Float) -> (@owned Array<Float>.DifferentiableView, @owned Array<Float>.DifferentiableView)) {
  // CHECK1:       bb0(%0 : $Array<Float>, %1 : $Array<Float>):
  // CHECK1:       bb8:
  // CHECK1-NOT:     = function_ref
  // CHECK1-NOT:     = partial_apply
  // CHECK1:         %[[#E52:]] = function_ref @$s3outyycfU_27sumFirstThreeConcatenating1L_ySfSaySfG_ACtFTJpSSpSr055$sSfSa16_DifferentiationAA14DifferentiableRzlE0B4ViewVyg8_GIegno_G10AEIegyo_TRSfSa01_I0AE0K0RzlE0kO0VySf_GIegno_ADSfAIIegno_0f5Sf16_i26E7_vjpAdd3lhs3rhsSf5value_g17_SftSfc8pullbacktg1_y5FZSf_Y6SfcfU_ADSfAIIegno_AJTf1nnccccc_n0fh1_ijkl4E13_v32Subscript5indexx5value_SaA2aBRzlmnO59Vy13TangentVectorQz_GAIc8pullbacktSi_tFAKL_yAjiaBRzlFSf_TG5ACSiAkCSiAkCSiTf1nnccc_n : $@convention(thin) (Float, @owned @callee_guaranteed (@guaranteed Array<Float>.DifferentiableView) -> (@owned Array<Float>.DifferentiableView, @owned Array<Float>.DifferentiableView), @owned Array<Float>, Int, @owned Array<Float>, Int, @owned Array<Float>, Int) -> (@owned Array<Float>.DifferentiableView, @owned Array<Float>.DifferentiableView)
  // CHECK1:         %[[#E53:]] = partial_apply [callee_guaranteed] %[[#E52]](%[[#]], %[[#]], %[[#]], %[[#]], %[[#]], %[[#]], %[[#]]) : $@convention(thin) (Float, @owned @callee_guaranteed (@guaranteed Array<Float>.DifferentiableView) -> (@owned Array<Float>.DifferentiableView, @owned Array<Float>.DifferentiableView), @owned Array<Float>, Int, @owned Array<Float>, Int, @owned Array<Float>, Int) -> (@owned Array<Float>.DifferentiableView, @owned Array<Float>.DifferentiableView)
  // CHECK1:         %[[#E55:]] = tuple (%[[#]], %[[#E53]])
  // CHECK1:         return %[[#E55]]
  // CHECK1:       } // end sil function '$s3outyycfU_27sumFirstThreeConcatenating1L_ySfSaySfG_ACtFTJrSSpSr'

  // CHECK1-NONE:  {{^}}// pullback of sumFirstThreeConcatenating1
  // CHECK1:       // specialized pullback of sumFirstThreeConcatenating1 #1 (_:_:) in closure #1 in
  // CHECK1:       sil private @$s3outyycfU_27sumFirstThreeConcatenating1L_ySfSaySfG_ACtFTJpSSpSr055$sSfSa16_DifferentiationAA14DifferentiableRzlE0B4ViewVyg8_GIegno_G10AEIegyo_TRSfSa01_I0AE0K0RzlE0kO0VySf_GIegno_ADSfAIIegno_0f5Sf16_i26E7_vjpAdd3lhs3rhsSf5value_g17_SftSfc8pullbacktg1_y5FZSf_Y6SfcfU_ADSfAIIegno_AJTf1nnccccc_n0fh1_ijkl4E13_v32Subscript5indexx5value_SaA2aBRzlmnO59Vy13TangentVectorQz_GAIc8pullbacktSi_tFAKL_yAjiaBRzlFSf_TG5ACSiAkCSiAkCSiTf1nnccc_nTf4ngnnnnnn_n : $@convention(thin) (Float, @guaranteed @callee_guaranteed (@guaranteed Array<Float>.DifferentiableView) -> (@owned Array<Float>.DifferentiableView, @owned Array<Float>.DifferentiableView), @owned Array<Float>, Int, @owned Array<Float>, Int, @owned Array<Float>, Int) -> (@owned Array<Float>.DifferentiableView, @owned Array<Float>.DifferentiableView) {
  // CHECK1-NOT:     = partial_apply
  // CHECK1-NOT:     = thin_to_thick_function
  // CHECK1:         %[[#D17:]] = apply %1(%[[#D14:]]) : $@callee_guaranteed (@guaranteed Array<Float>.DifferentiableView) -> (@owned Array<Float>.DifferentiableView, @owned Array<Float>.DifferentiableView)
  // CHECK1:         return %[[#D17]]
  // CHECK1:       } // end sil function '$s3outyycfU_27sumFirstThreeConcatenating1L_ySfSaySfG_ACtFTJpSSpSr055$sSfSa16_DifferentiationAA14DifferentiableRzlE0B4ViewVyg8_GIegno_G10AEIegyo_TRSfSa01_I0AE0K0RzlE0kO0VySf_GIegno_ADSfAIIegno_0f5Sf16_i26E7_vjpAdd3lhs3rhsSf5value_g17_SftSfc8pullbacktg1_y5FZSf_Y6SfcfU_ADSfAIIegno_AJTf1nnccccc_n0fh1_ijkl4E13_v32Subscript5indexx5value_SaA2aBRzlmnO59Vy13TangentVectorQz_GAIc8pullbacktSi_tFAKL_yAjiaBRzlFSf_TG5ACSiAkCSiAkCSiTf1nnccc_nTf4ngnnnnnn_n'
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
  // CHECK2-LABEL: // reverse-mode derivative of sumFirstThreeConcatenating2 #1 (_:_:) in closure #2 in
  // CHECK2-NEXT:  sil private @$s3outyycfU0_27sumFirstThreeConcatenating2L_ySfSaySfG_ACtFTJrSSpSr : $@convention(thin) (@guaranteed Array<Float>, @guaranteed Array<Float>) -> (Float, @owned @callee_guaranteed (Float) -> (@owned Array<Float>.DifferentiableView, @owned Array<Float>.DifferentiableView)) {
  // CHECK2:       bb10:
  // CHECK2-NOT:     = function_ref
  // CHECK2-NOT:     = partial_apply
  // CHECK2:         %[[#E52:]] = function_ref @$s3outyycfU0_27sumFirstThreeConcatenating2L_ySfSaySfG_ACtFTJpSSpSr055$sSfSa16_DifferentiationAA14DifferentiableRzlE0B4ViewVyg8_GIegno_G10AEIegyo_TRSfSa01_I0AE0K0RzlE0kO0VySf_GIegno_ADSfAIIegno_0f5Sf16_i26E7_vjpAdd3lhs3rhsSf5value_g17_SftSfc8pullbacktg1_y5FZSf_Y6SfcfU_ADSfAIIegno_AJTf1nnccccc_n0fh1_ijkl4E10_v25Appendyyt5value_SaA2aBRzlmno55Vy13TangentVectorQz_GAIzc8pullbacktSayxGz_AKtFZA2IzcfU_G4_Tg5Si0fh1_ijkl4E13_v32Subscript5indexx5value_SaA2aBRzlmnO59Vy13TangentVectorQz_GAIc8pullbacktSi_tFAKL_yAjiaBRzlFSf_TG5ACSiAlCSiAlCSiTf1ncccc_n : $@convention(thin) (Float, Int, @owned Array<Float>, Int, @owned Array<Float>, Int, @owned Array<Float>, Int) -> (@owned Array<Float>.DifferentiableView, @owned Array<Float>.DifferentiableView)
  // CHECK2:         %[[#E53:]] = partial_apply [callee_guaranteed] %[[#E52]](%[[#]], %[[#]], %[[#]], %[[#]], %[[#]], %[[#]], %[[#]]) : $@convention(thin) (Float, Int, @owned Array<Float>, Int, @owned Array<Float>, Int, @owned Array<Float>, Int) -> (@owned Array<Float>.DifferentiableView, @owned Array<Float>.DifferentiableView)
  // CHECK2:         %[[#E55:]] = tuple (%[[#]], %[[#E53]])
  // CHECK2:         return %[[#E55]]
  // CHECK2:       } // end sil function '$s3outyycfU0_27sumFirstThreeConcatenating2L_ySfSaySfG_ACtFTJrSSpSr'

  // CHECK2-NONE:  {{^}}// pullback of sumFirstThreeConcatenating2
  // CHECK2:       {{^}}// specialized pullback of sumFirstThreeConcatenating2
  // CHECK2:       sil private @$s3outyycfU0_27sumFirstThreeConcatenating2L_ySfSaySfG_ACtFTJpSSpSr055$sSfSa16_DifferentiationAA14DifferentiableRzlE0B4ViewVyg8_GIegno_G10AEIegyo_TRSfSa01_I0AE0K0RzlE0kO0VySf_GIegno_ADSfAIIegno_0f5Sf16_i26E7_vjpAdd3lhs3rhsSf5value_g17_SftSfc8pullbacktg1_y5FZSf_Y6SfcfU_ADSfAIIegno_AJTf1nnccccc_n0fh1_ijkl4E10_v25Appendyyt5value_SaA2aBRzlmno55Vy13TangentVectorQz_GAIzc8pullbacktSayxGz_AKtFZA2IzcfU_G4_Tg5Si0fh1_ijkl4E13_v32Subscript5indexx5value_SaA2aBRzlmnO59Vy13TangentVectorQz_GAIc8pullbacktSi_tFAKL_yAjiaBRzlFSf_TG5ACSiAlCSiAlCSiTf1ncccc_n : $@convention(thin) (Float, Int, @owned Array<Float>, Int, @owned Array<Float>, Int, @owned Array<Float>, Int) -> (@owned Array<Float>.DifferentiableView, @owned Array<Float>.DifferentiableView) {
  // CHECK2-NOT:     = partial_apply
  // CHECK2-NOT:     = thin_to_thick_function
  // CHECK2:       } // end sil function '$s3outyycfU0_27sumFirstThreeConcatenating2L_ySfSaySfG_ACtFTJpSSpSr055$sSfSa16_DifferentiationAA14DifferentiableRzlE0B4ViewVyg8_GIegno_G10AEIegyo_TRSfSa01_I0AE0K0RzlE0kO0VySf_GIegno_ADSfAIIegno_0f5Sf16_i26E7_vjpAdd3lhs3rhsSf5value_g17_SftSfc8pullbacktg1_y5FZSf_Y6SfcfU_ADSfAIIegno_AJTf1nnccccc_n0fh1_ijkl4E10_v25Appendyyt5value_SaA2aBRzlmno55Vy13TangentVectorQz_GAIzc8pullbacktSayxGz_AKtFZA2IzcfU_G4_Tg5Si0fh1_ijkl4E13_v32Subscript5indexx5value_SaA2aBRzlmnO59Vy13TangentVectorQz_GAIc8pullbacktSi_tFAKL_yAjiaBRzlFSf_TG5ACSiAlCSiAlCSiTf1ncccc_n'

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

  // CHECK3:       // reverse-mode derivative of multiply #1 (_:) in closure #3 in
  // CHECK3-NEXT:  sil private @$s3outyycfU1_8multiplyL_ySfAAyycfU1_20RealPropertyWrappersL_VFTJrSpSr : $@convention(thin) (RealPropertyWrappers) -> (Float, @owned @callee_guaranteed (Float) -> RealPropertyWrappers.TangentVector) {
  // CHECK3:       bb3(%[[#A16:]] : $Float):
  // CHECK3:         %[[#A22:]] = function_ref @$s3outyycfU1_8multiplyL_ySfAAyycfU1_20RealPropertyWrappersL_VFTJpSpSr073$sSf16_DifferentiationE12_vjpMultiply3lhs3rhsSf5value_Sf_SftSfc8pullbacktm1_n5FZSf_N6SfcfU_S2fTf1nnc_n015$s3outyycfU1_20cdE16L_V1xSfvgTJpSpSrTf1ncnn_n : $@convention(thin) (Float, Float, Float) -> RealPropertyWrappers.TangentVector
  // CHECK3:         %[[#A23:]] = partial_apply [callee_guaranteed] %[[#A22]](%[[#]], %[[#A16]]) : $@convention(thin) (Float, Float, Float) -> RealPropertyWrappers.TangentVector
  // CHECK3:         %[[#A24:]] = tuple (%[[#]], %[[#A23]])
  // CHECK3:         return %[[#A24]]
  // CHECK3:       } // end sil function '$s3outyycfU1_8multiplyL_ySfAAyycfU1_20RealPropertyWrappersL_VFTJrSpSr'

  // CHECK3-NONE:  {{^}}// pullback of multiply
  // CHECK3:       {{^}}// specialized pullback of multiply
  // CHECK3:       sil private @$s3outyycfU1_8multiplyL_ySfAAyycfU1_20RealPropertyWrappersL_VFTJpSpSr073$sSf16_DifferentiationE12_vjpMultiply3lhs3rhsSf5value_Sf_SftSfc8pullbacktm1_n5FZSf_N6SfcfU_S2fTf1nnc_n015$s3outyycfU1_20cdE16L_V1xSfvgTJpSpSrTf1ncnn_n : $@convention(thin) (Float, Float, Float) -> RealPropertyWrappers.TangentVector {
  // CHECK3-NOT:     = function_ref
  // CHECK3-NOT:     = partial_apply
  // CHECK3-NOT:     = thin_to_thick_function
  // CHECK3:       } // end sil function '$s3outyycfU1_8multiplyL_ySfAAyycfU1_20RealPropertyWrappersL_VFTJpSpSr073$sSf16_DifferentiationE12_vjpMultiply3lhs3rhsSf5value_Sf_SftSfc8pullbacktm1_n5FZSf_N6SfcfU_S2fTf1nnc_n015$s3outyycfU1_20cdE16L_V1xSfvgTJpSpSrTf1ncnn_n'

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

  // CHECK4-LABEL: // reverse-mode derivative of methodWrapper #1 (_:) in closure #4 in 
  // CHECK4-NEXT:  sil private @$s3outyycfU2_13methodWrapperL_ySfAAyycfU2_5ClassL_VFTJrSpSr : $@convention(thin) (Class) -> (Float, @owned @callee_guaranteed (Float) -> Class.TangentVector) {
  // CHECK4:       bb3(%[[#C33:]] : $Float, %[[#C34:]] : $_AD__$s3outyycfU2_5ClassL_V6methodSfyF_bb3__Pred__src_0_wrt_0):
  // CHECK4-NOT:     = function_ref @$s4test5ClassV6methodSfyFTJpSpSr
  // CHECK4-NOT:     = partial_apply
  // CHECK4-NOT:     = function_ref @$s4test13methodWrapperySfAA5ClassVFTJpSpSr
  // CHECK4:         %[[#C39:]] = function_ref @$s3outyycfU2_13methodWrapperL_ySfAAyycfU2_5ClassL_VFTJpSpSr014$s3outyycfU2_5D21L_V6methodSfyFTJpSpSrAA05_AD__ef2_5d2L_gH24F_bb3__Pred__src_0_wrt_033_E588B908471A5F020CF23EC392ADD7D3LLOTf1nc_n : $@convention(thin) (Float, @owned _AD__$s3outyycfU2_5ClassL_V6methodSfyF_bb3__Pred__src_0_wrt_0) -> Class.TangentVector 
  // CHECK4:         %[[#C40:]] = partial_apply [callee_guaranteed] %50(%49) : $@convention(thin) (Float, @owned _AD__$s3outyycfU2_5ClassL_V6methodSfyF_bb3__Pred__src_0_wrt_0) -> Class.TangentVector
  // CHECK4:         %[[#C42:]] = tuple (%[[#C33]], %[[#C40]])
  // CHECK4:         return %[[#C42]]
  // CHECK4:       } // end sil function '$s3outyycfU2_13methodWrapperL_ySfAAyycfU2_5ClassL_VFTJrSpSr'

  /// TODO: even though branch tracing enum is not passed to top-level pullback
  /// directly, it is captured by one of the closures which was specialized.
  /// Because of that, this enum argument is now an argument of specialized top-level
  /// pullback. Specializing closures passed as payload tuple elements of the enum
  /// is currently not supported.

  // CHECK4-NONE:  {{^}}// pullback of methodWrapper
  // CHECK4:       {{^}}// specialized pullback of methodWrapper
  // CHECK4:       sil private @$s3outyycfU2_13methodWrapperL_ySfAAyycfU2_5ClassL_VFTJpSpSr014$s3outyycfU2_5D21L_V6methodSfyFTJpSpSrAA05_AD__ef2_5d2L_gH24F_bb3__Pred__src_0_wrt_033_E588B908471A5F020CF23EC392ADD7D3LLOTf1nc_n : $@convention(thin) (Float, @owned _AD__$s3outyycfU2_5ClassL_V6methodSfyF_bb3__Pred__src_0_wrt_0) -> Class.TangentVector {
  // CHECK4-NOT:     = function_ref
  // CHECK4-NOT:     = partial_apply
  // CHECK4-NOT:     = thin_to_thick_function
  // CHECK4:       } // end sil function '$s3outyycfU2_13methodWrapperL_ySfAAyycfU2_5ClassL_VFTJpSpSr014$s3outyycfU2_5D21L_V6methodSfyFTJpSpSrAA05_AD__ef2_5d2L_gH24F_bb3__Pred__src_0_wrt_033_E588B908471A5F020CF23EC392ADD7D3LLOTf1nc_n'

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
