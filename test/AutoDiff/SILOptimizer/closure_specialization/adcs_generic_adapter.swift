// RUN: %target-swift-frontend -O -sil-verify-all -emit-sil -D DEFINITIONS_ONLY -module-name adcs_generic_adapter %S/../../validation-test/closure_specialization/adcs_generic_adapter.swift -Xllvm -sil-print-around=autodiff-closure-specialization -o %t.sil 2> %t.passes
// RUN: %FileCheck %s < %t.passes
// RUN: %FileCheck %S/../../validation-test/closure_specialization/adcs_generic_adapter.swift < %t.sil

// Inside the new pullback, both the recreated inner closure and the direct
// call of the outer subset thunk use the pullback's generic parameter.
// CHECK-LABEL: *** New SIL function in {{.*}}pass: AutodiffClosureSpecialization
// CHECK-NEXT: // specialized pullback of genericSubset<A>(_:_:_:)
// CHECK: // specialized pullback of genericSubset<A>(_:_:_:)
// CHECK-NOT: } // end sil function
// CHECK: // function_ref closure #1 in vjpMixed<A>(_:_:)
// CHECK-NEXT: %[[ROOT:[0-9]+]] = function_ref
// CHECK-NEXT: %[[INNER:[0-9]+]] = partial_apply {{.*}} %[[ROOT]]<τ_0_0>()
// CHECK-NEXT: // function_ref autodiff subset parameters thunk for pullback
// CHECK-NEXT: %[[THUNK:[0-9]+]] = function_ref
// CHECK-NEXT: {{.*}} = apply %[[THUNK]]<τ_0_0>(%{{[0-9]+}}, %[[INNER]])
// CHECK: destroy_value %[[INNER]]
// CHECK: } // end sil function
