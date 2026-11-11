// RUN: %target-swift-frontend -O -sil-verify-all -emit-sil -D DEFINITIONS_ONLY -module-name adcs_optional_callback %S/../../validation-test/closure_specialization/adcs_optional_callback.swift -Xllvm -sil-disable-skipping-passes -Xllvm -sil-print-after=autodiff-closure-specialization -o %t.sil 2> %t.passes
// RUN: %FileCheck %s < %t.passes

// Exercise both ADCS visits even when the mid-level pipeline leaves the VJP
// unchanged. The success and error cases save different optional wrappers in
// whole payloads that flow to the same pullback argument. ADCS declines this
// merge, even though the success edge has a statically known callback.
// CHECK: stage HighLevel,Module+StackPromote, pass
// CHECK-NOT: stage MidLevel,Function
// CHECK: // reverse-mode derivative of recoveringSquare
// CHECK: %[[NONE:[0-9]+]] = enum $Optional<@callee_guaranteed (Float) -> Float>, #Optional.none!enumelt
// CHECK-NEXT: %[[ERROR_PAYLOAD:[0-9]+]] = tuple $(predecessor: {{.*}}, Optional<@callee_guaranteed (Float) -> Float>) ({{%[0-9]+}}, %[[NONE]])
// CHECK-NEXT: enum ${{[^,]+}}, #[[ERROR_TAPE:[^.]+]].[[CASE:bb[0-9]+]]!enumelt, %[[ERROR_PAYLOAD]]
// CHECK: // function_ref closure #1 in positiveSquareVJP
// CHECK-NEXT: %[[ROOT:[0-9]+]] = function_ref
// CHECK-NEXT: %[[CLOSURE:[0-9]+]] = partial_apply {{.*}} %[[ROOT]](
// CHECK-NEXT: %[[SOME:[0-9]+]] = enum $Optional<@callee_guaranteed (Float) -> Float>, #Optional.some!enumelt, %[[CLOSURE]]
// CHECK-NEXT: %[[NORMAL_PAYLOAD:[0-9]+]] = tuple $(predecessor: {{.*}}, Optional<@callee_guaranteed (Float) -> Float>) ({{%[0-9]+}}, %[[SOME]])
// CHECK-NEXT: enum ${{[^,]+}}, #[[NORMAL_TAPE:[^.]+]].[[CASE]]!enumelt, %[[NORMAL_PAYLOAD]]
// CHECK-NOT: } // end sil function
// CHECK: // function_ref pullback of recoveringSquare
// CHECK-LABEL: // pullback of recoveringSquare
// CHECK: %[[ERROR:[0-9]+]] = unchecked_enum_data {{.*}}, #[[ERROR_TAPE]].[[CASE]]!enumelt
// CHECK-NEXT: br [[JOIN:bb[0-9]+]]({{%[0-9]+}}, %[[ERROR]])
// CHECK: %[[NORMAL:[0-9]+]] = unchecked_enum_data {{.*}}, #[[NORMAL_TAPE]].[[CASE]]!enumelt
// CHECK-NEXT: br [[JOIN]]({{%[0-9]+}}, %[[NORMAL]])
// CHECK: [[JOIN]]({{%[0-9]+}} : $Float, %[[PAYLOAD:[0-9]+]] : @owned $(predecessor: {{.*}}, Optional<@callee_guaranteed (Float) -> Float>))
// CHECK-NEXT: ({{%[0-9]+}}, %[[OPTIONAL:[0-9]+]]) = destructure_tuple %[[PAYLOAD]]
// CHECK-NEXT: switch_enum %[[OPTIONAL]], case #Optional.some!enumelt: [[APPLY:bb[0-9]+]], case #Optional.none!enumelt:
// CHECK: [[APPLY]](%[[CALLBACK:[0-9]+]] : @owned $@callee_guaranteed (Float) -> Float)
// CHECK-NEXT: {{%[0-9]+}} = apply %[[CALLBACK]](

// CHECK: stage MidLevel,Function, pass
// CHECK: // reverse-mode derivative of recoveringSquare
// CHECK-NOT: } // end sil function
// CHECK: // function_ref pullback of recoveringSquare
