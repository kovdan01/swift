// RUN: %target-swift-frontend -emit-sil -Onone %s -Xllvm -enable-experimental-async-autodiff -Xllvm -sil-print-after=differentiation -sil-verify-all -sil-ownership-verify-all -o %t.Onone.sil 2> %t.Onone.after.txt
// RUN: %python %S/Inputs/check_helper_debug.py %t.Onone.after.txt
// RUN: %target-swift-frontend -emit-sil -O %s -Xllvm -enable-experimental-async-autodiff -Xllvm -sil-print-after=differentiation -sil-verify-all -sil-ownership-verify-all -o %t.O.sil 2> %t.O.after.txt
// RUN: %python %S/Inputs/check_helper_debug.py %t.O.after.txt
// RUN: %target-swift-frontend -emit-sil -g -Onone %s -Xllvm -enable-experimental-async-autodiff -Xllvm -sil-print-after=differentiation -sil-verify-all -sil-ownership-verify-all -o %t.g.Onone.sil 2> %t.g.Onone.after.txt
// RUN: %python %S/Inputs/check_helper_debug.py %t.g.Onone.after.txt
// RUN: %target-swift-frontend -emit-sil -g -O %s -Xllvm -enable-experimental-async-autodiff -Xllvm -sil-print-after=differentiation -sil-verify-all -sil-ownership-verify-all -o %t.g.O.sil 2> %t.g.O.after.txt
// RUN: %python %S/Inputs/check_helper_debug.py %t.g.O.after.txt
// RUN: %target-swift-frontend -emit-ir -g -Onone %s -Xllvm -enable-experimental-async-autodiff -sil-verify-all -sil-ownership-verify-all -o /dev/null
// RUN: %target-swift-frontend -emit-ir -g -O %s -Xllvm -enable-experimental-async-autodiff -sil-verify-all -sil-ownership-verify-all -o /dev/null
// REQUIRES: asserts
// REQUIRES: concurrency
// REQUIRES: differentiable_programming

// The task initializer has an error-result convention, even though the
// numerical workload does not throw. Its special $error debug_value must stay
// in the original initializer and must NOT survive in the nonthrowing helper
// or helper VJP. Keep the ordinary a/b capture debug records instead of deleting
// all debug instructions. Check immediately after AD, before debug stripping or
// inlining can hide a bad helper. The IR runs also exercise debug-info lowering.

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
