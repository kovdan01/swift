// RUN: %target-swift-frontend -O -sil-verify-all -emit-sil -D DEFINITIONS_ONLY -module-name payload_flow %S/../../validation-test/closure_specialization/payload_flow.swift -Xllvm -sil-print-around=autodiff-closure-specialization -o %t.sil 2> %t.passes
// RUN: %FileCheck %s --check-prefix=INPUT < %t.passes
// RUN: %FileCheck %s --check-prefix=EARLY < %t.passes
// RUN: %FileCheck %s --check-prefix=LATE < %t.passes
// RUN: %FileCheck %s --check-prefix=FINAL < %t.sil

// Compile the exact validation source. Textual SIL cannot faithfully round-trip
// its compiler-generated singleton tuple AST types, so source compilation also
// keeps the real ownership, enum-case and phi shapes intact.
//
// Both invocation points perform BTE specialization: branching is handled in
// HighLevel; lateBranch remains opaque until MidLevel inlining exposes its
// custom pullback. This checks actual late specialization, not just a second
// visit to a function already specialized by the first invocation.

// The first input has both a whole multi-field payload argument and a scalar
// callback phi fed by singleton payload projections from two incoming edges.
// INPUT-LABEL: // pullback of branching(_:_:)
// INPUT: bb{{[0-9]+}}(%{{[0-9]+}} : @owned $(predecessor: {{.*}}, @callee_guaranteed (Float) -> (Float, Float))):
// INPUT: %[[PAYLOAD1:[0-9]+]] = unchecked_enum_data
// INPUT-NEXT: %[[FIELD1:[0-9]+]] = destructure_tuple %[[PAYLOAD1]]
// INPUT-NEXT: br [[JOIN:bb[0-9]+]](%{{[0-9]+}}, %[[FIELD1]], %{{[0-9]+}})
// INPUT: %[[PAYLOAD2:[0-9]+]] = unchecked_enum_data
// INPUT-NEXT: %[[FIELD2:[0-9]+]] = destructure_tuple %[[PAYLOAD2]]
// INPUT-NEXT: br [[JOIN]](%{{[0-9]+}}, %[[FIELD2]], %{{[0-9]+}})
// INPUT: [[JOIN]](%{{[0-9]+}} : $Builtin.FPIEEE32, %[[CALLBACK:[0-9]+]] : @owned $@callee_guaranteed (Float) -> (Float, Float), %{{[0-9]+}} : $Builtin.FPIEEE32):
// INPUT: apply %[[CALLBACK]]

// A projected predecessor-enum phi is trivial already at the first invocation;
// its incoming fields use tuple_extract rather than destructure_tuple.
// INPUT-LABEL: // pullback of generic<A>(_:_:)
// INPUT: %[[PREDPAYLOAD1:[0-9]+]] = unchecked_enum_data
// INPUT-NEXT: %[[PRED1:[0-9]+]] = tuple_extract %[[PREDPAYLOAD1]], 0
// INPUT-NEXT: br [[PREDJOIN:bb[0-9]+]](%{{[0-9]+}}, %[[PRED1]])
// INPUT: %[[PREDPAYLOAD2:[0-9]+]] = unchecked_enum_data
// INPUT-NEXT: %[[PRED2:[0-9]+]] = tuple_extract %[[PREDPAYLOAD2]], 0
// INPUT-NEXT: br [[PREDJOIN]](%{{[0-9]+}}, %[[PRED2]])
// INPUT: [[PREDJOIN]](%{{[0-9]+}} : $Float, %{{[0-9]+}} : $_AD__{{[^,]+}}):

// Creating this specialized pullback during the high-level invocation proves
// the first phase rewrote the projected phi. Its subsequent pass input has a
// capture-tuple phi and direct calls, rather than opaque callback applications.
// EARLY: *** New SIL function in {{.*}}stage HighLevel,Module+StackPromote, pass: AutodiffClosureSpecialization {{.*}}, Function: $s12payload_flow9branchingyS2f_SbtFTJp
// EARLY-NEXT: // specialized pullback of branching(_:_:)
// EARLY: // specialized pullback of branching(_:_:)
// EARLY: bb{{[0-9]+}}(%{{[0-9]+}} : $Builtin.FPIEEE32, %{{[0-9]+}} : $(Float, Float), %{{[0-9]+}} : $Builtin.FPIEEE32):
// EARLY-NOT: apply {{.*}} : $@callee_guaranteed
// EARLY: } // end sil function

// The first invocation leaves the custom derivative call and the original
// pullback in place. The second invocation sees a known captured closure and
// creates a new specialized pullback in the actual mid-level stage.
// LATE-LABEL: // reverse-mode derivative of lateBranch(_:_:)
// LATE: // function_ref delayedSquareVJP(_:)
// LATE: apply
// LATE-NOT: function_ref specialized pullback of lateBranch
// LATE: // function_ref pullback of lateBranch(_:_:)
// LATE: } // end sil function
// LATE-LABEL: // pullback of lateBranch(_:_:)
// LATE: bb{{[0-9]+}}(%{{[0-9]+}} : $Builtin.FPIEEE32, %{{[0-9]+}} : $Builtin.FPIEEE32, %[[OPAQUE:[0-9]+]] : @owned $@callee_guaranteed (Float) -> Float):
// LATE: apply %[[OPAQUE]]
// LATE: } // end sil function
// LATE: *** New SIL function in {{.*}}stage MidLevel,Function, pass: AutodiffClosureSpecialization {{.*}}, Function: $s12payload_flow10lateBranchyS2f_SbtFTJp
// LATE-NEXT: // specialized pullback of lateBranch(_:_:)
// LATE: // specialized pullback of lateBranch(_:_:)
// LATE: bb{{[0-9]+}}(%{{[0-9]+}} : $Builtin.FPIEEE32, %{{[0-9]+}} : $Builtin.FPIEEE32, %[[CAPTURES:[0-9]+]] : $(_: Float)):
// LATE: tuple_extract %[[CAPTURES]], 0
// LATE-NOT: apply {{.*}} : $@callee_guaranteed
// LATE: } // end sil function

// The final source-generated VJPs return specialized pullbacks. Their bodies
// have no materialized closures or opaque callback applications.
// FINAL-LABEL: // reverse-mode derivative of branching(_:_:)
// FINAL-NOT: } // end sil function
// FINAL: // function_ref specialized pullback of branching(_:_:)
// FINAL: return
// FINAL: } // end sil function
// FINAL-LABEL: // specialized pullback of branching(_:_:)
// FINAL-NOT: = partial_apply
// FINAL-NOT: apply {{.*}} : $@callee_guaranteed
// FINAL: } // end sil function
// FINAL-LABEL: // reverse-mode derivative of nested(_:_:)
// FINAL-NOT: } // end sil function
// FINAL: // function_ref specialized pullback of nested(_:_:)
// FINAL: return
// FINAL: } // end sil function
// FINAL-LABEL: // specialized pullback of nested(_:_:)
// FINAL-NOT: = partial_apply
// FINAL-NOT: apply {{.*}} : $@callee_guaranteed
// FINAL: } // end sil function
// FINAL-LABEL: // reverse-mode derivative of generic<A>(_:_:)
// FINAL-NOT: } // end sil function
// FINAL: // function_ref specialized pullback of generic<A>(_:_:)
// FINAL: return
// FINAL: } // end sil function
// FINAL-LABEL: // specialized pullback of generic<A>(_:_:)
// FINAL-NOT: = partial_apply
// FINAL-NOT: apply {{.*}} : $@callee_guaranteed
// FINAL: } // end sil function
// FINAL-LABEL: // reverse-mode derivative of lateBranch(_:_:)
// FINAL-NOT: } // end sil function
// FINAL: // function_ref specialized pullback of lateBranch(_:_:)
// FINAL: return
// FINAL: } // end sil function
// FINAL-LABEL: // specialized pullback of lateBranch(_:_:)
// FINAL-NOT: = partial_apply
// FINAL-NOT: apply {{.*}} : $@callee_guaranteed
// FINAL: } // end sil function
