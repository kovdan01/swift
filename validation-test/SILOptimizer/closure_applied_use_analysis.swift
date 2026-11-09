// REQUIRES: executable_test
// RUN: %empty-directory(%t)
// RUN: %target-build-swift %s -Onone -o %t/none.out
// RUN: %target-build-swift %s -O -o %t/opt.out
// RUN: %target-run %t/none.out
// RUN: %target-run %t/opt.out
// RUN: %target-swift-frontend -O -emit-sil -D DEFINITIONS_ONLY -module-name closure_applied_use_analysis %s -o - | %FileCheck %s
//
// Closure-use analysis must terminate for unused recursive forwarding and
// follow distinct parameters when recursion rotates a closure into an applied
// position. A long finite forwarding chain must expose the callback invocation
// without relying on a distinct-function count as a recursion bound.
// The SILOptimizer companion isolates the generated closure-specialization shapes.

@inline(never)
public func forwardUnused(_ body: @escaping (Float) -> Float, _ depth: Int, _ seed: Float) -> Float {
  if depth == 0 { return seed }
  return forwardUnused(body, depth - 1, seed)
}

@inline(never)
public func appliedAfterRotation(
  _ first: @escaping (Float) -> Float, _ second: @escaping (Float) -> Float,
  _ third: @escaping (Float) -> Float, _ depth: Int, _ seed: Float
) -> Float {
  if depth == 0 { return third(seed) }
  return appliedAfterRotation(third, first, second, depth - 1, seed)
}

public func unusedCaller(_ x: Float, _ depth: Int, _ seed: Float) -> Float {
  forwardUnused({ $0 * x }, depth, seed)
}

public func rotatedCaller(_ x: Float, _ depth: Int, _ seed: Float) -> Float {
  appliedAfterRotation({ 2 * x * $0 }, { 3 * $0 }, { 5 * $0 }, depth, seed)
}

// Only the first callback is statically known in this public entry point.
// Function-only visitation cannot reach its application through the second
// and third parameters, and the two runtime callbacks cannot start a round.
// CHECK-LABEL: sil {{.*}}@$s28closure_applied_use_analysis18mixedRotatedCalleryS2f_SiS3fcS2fctF :
// CHECK-NOT: partial_apply
// CHECK: function_ref @{{[^ ]*AfterRotation[^ ]*}} : $@convention(thin) (@guaranteed @callee_guaranteed (Float) -> Float, @guaranteed @callee_guaranteed (Float) -> Float, Int, Float, Float) -> Float
// CHECK-NOT: partial_apply
// CHECK: return
// CHECK-NEXT: } // end sil function
public func mixedRotatedCaller(
  _ x: Float, _ depth: Int, _ seed: Float,
  _ second: @escaping (Float) -> Float, _ third: @escaping (Float) -> Float
) -> Float {
  appliedAfterRotation({ 2 * x * $0 }, second, third, depth, seed)
}

@inline(never)
public func forward0(_ body: (Float) -> Float, _ seed: Float) -> Float {
  forward1(body, seed)
}

@inline(never)
public func forward1(_ body: (Float) -> Float, _ seed: Float) -> Float {
  forward2(body, seed)
}

@inline(never)
public func forward2(_ body: (Float) -> Float, _ seed: Float) -> Float {
  forward3(body, seed)
}

@inline(never)
public func forward3(_ body: (Float) -> Float, _ seed: Float) -> Float {
  forward4(body, seed)
}

@inline(never)
public func forward4(_ body: (Float) -> Float, _ seed: Float) -> Float {
  forward5(body, seed)
}

@inline(never)
public func forward5(_ body: (Float) -> Float, _ seed: Float) -> Float {
  forward6(body, seed)
}

@inline(never)
public func forward6(_ body: (Float) -> Float, _ seed: Float) -> Float {
  forward7(body, seed)
}

@inline(never)
public func forward7(_ body: (Float) -> Float, _ seed: Float) -> Float {
  forward8(body, seed)
}

@inline(never)
public func forward8(_ body: (Float) -> Float, _ seed: Float) -> Float {
  forward9(body, seed)
}

@inline(never)
public func forward9(_ body: (Float) -> Float, _ seed: Float) -> Float {
  forward10(body, seed)
}

@inline(never)
public func forward10(_ body: (Float) -> Float, _ seed: Float) -> Float {
  forward11(body, seed)
}

@inline(never)
public func forward11(_ body: (Float) -> Float, _ seed: Float) -> Float {
  body(seed)
}

// Main's eight-function applied-use limit leaves a stack closure here. The
// bounded value traversal reaches the invocation through all twelve functions.
// CHECK-LABEL: sil {{.*}}@$s28closure_applied_use_analysis10deepCalleryS2f_SftF :
// CHECK-NOT: partial_apply
// CHECK: function_ref @{{[^ ]+}} : $@convention(thin) (Float, Float) -> Float
// CHECK-NOT: partial_apply
// CHECK: return
// CHECK-NEXT: } // end sil function
public func deepCaller(_ x: Float, _ seed: Float) -> Float {
  forward0({ x * $0 }, seed)
}

#if !DEFINITIONS_ONLY
for depth in 0...8 {
  for seed: Float in [-2, 2, 5] {
    precondition(unusedCaller(4, depth, seed) == seed)
    let slope: Float = [5, 3, 8][depth % 3]
    precondition(rotatedCaller(4, depth, seed) == slope * seed)
    precondition(mixedRotatedCaller(4, depth, seed, { 3 * $0 }, { 5 * $0 }) == slope * seed)
  }
}
for seed: Float in [-2, 2, 5] {
  precondition(deepCaller(4, seed) == 4 * seed)
}
#endif
