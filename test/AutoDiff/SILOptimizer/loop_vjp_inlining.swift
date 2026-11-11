// REQUIRES: asserts

// Compile the exact validation source. Raising the ordinary call benefit makes
// the loop callee profitable, so the AutoDiff policy must preserve the outer
// branch tape until BTE specialization has finished. The loop itself remains
// supported by the existing runtime tape, independently of BTE specialization.
// RUN: %target-swift-frontend -O -sil-verify-all -emit-sil -D DEFINITIONS_ONLY -module-name loop_vjp_inlining -Xllvm -sil-inline-removed-call-benefit=100 -Xllvm -sil-print-around=autodiff-closure-specialization -Xllvm '-sil-print-function=$s17loop_vjp_inlining13branchingLoopyS2f_SbtFTJrSUpSr,$s17loop_vjp_inlining20throughBranchingLoopyS2f_SbtFTJrSUpSr' %S/../validation-test/closure_specialization/loop_vjp_inlining.swift -o %t.sil 2> %t.log
// RUN: %FileCheck %s --check-prefix=PHASE < %t.log
// RUN: %FileCheck %s --check-prefix=FINAL < %t.sil

// Before the first ADCS invocation, the non-looping wrapper still calls the
// loop VJP. Pulling that callee's runtime loop context into the wrapper early
// would make the wrapper ineligible for BTE specialization as well.
// PHASE: *** SIL function before {{.*}}stage HighLevel,Module+StackPromote, pass {{[0-9]+}}: AutodiffClosureSpecialization
// PHASE-NEXT: // reverse-mode derivative of branchingLoop(_:_:)
// PHASE-NOT: autoDiffProjectTopLevelSubcontext
// PHASE: // function_ref reverse-mode derivative of loop(_:)
// PHASE: apply
// PHASE-NOT: autoDiffProjectTopLevelSubcontext
// PHASE: } // end sil function

// The outer branch tape is specialized while the loop callee remains opaque.
// Low-level inlining after ADCS can still inline the loop VJP normally.
// PHASE: *** SIL function after {{.*}}stage HighLevel,Module+StackPromote, pass {{[0-9]+}}: AutodiffClosureSpecialization
// PHASE-NEXT: // reverse-mode derivative of branchingLoop(_:_:)
// PHASE-NOT: autoDiffProjectTopLevelSubcontext
// PHASE: // function_ref specialized pullback of branchingLoop(_:_:)
// PHASE-NOT: autoDiffProjectTopLevelSubcontext
// PHASE: } // end sil function

// A straight-line VJP also leaves a branching callee's tape intact until ADCS.
// PHASE: *** SIL function before {{.*}}stage HighLevel,Module+StackPromote, pass {{[0-9]+}}: AutodiffClosureSpecialization
// PHASE-NEXT: // reverse-mode derivative of throughBranchingLoop(_:_:)
// PHASE-NOT: } // end sil function
// PHASE: // function_ref reverse-mode derivative of branchingLoop(_:_:)
// PHASE-NEXT: %[[BRANCH_VJP:[0-9]+]] = function_ref
// PHASE-NEXT: %{{[0-9]+}} = apply %[[BRANCH_VJP]]
// PHASE: } // end sil function

// The final pullback keeps the branch specialization; it does not construct
// any closure contexts to invoke the static square pullback.
// FINAL-LABEL: // specialized pullback of branchingLoop(_:_:)
// FINAL-NOT: = partial_apply
// FINAL: } // end sil function

// Later inlining uses the already-specialized branching pullback.
// FINAL-LABEL: // reverse-mode derivative of throughBranchingLoop(_:_:)
// FINAL-NOT: function_ref reverse-mode derivative of branchingLoop
// FINAL-NOT: } // end sil function
// FINAL: // function_ref specialized pullback of branchingLoop(_:_:)
// FINAL-NOT: function_ref reverse-mode derivative of branchingLoop
// FINAL: } // end sil function
