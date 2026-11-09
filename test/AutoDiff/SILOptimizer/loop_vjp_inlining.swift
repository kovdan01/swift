// REQUIRES: asserts

// Raise the ordinary call benefit so this test isolates the AutoDiff inlining
// policy. Disable ADCS to show that runtime loop tapes need no specialization
// before their VJPs can be inlined into a non-VJP caller.
// RUN: %target-swift-frontend -O -emit-sil -D DEFINITIONS_ONLY -module-name loop_vjp_inlining -Xllvm -sil-disable-pass=autodiff-closure-specialization -Xllvm -sil-inline-removed-call-benefit=100 -Xllvm -debug-only=sil-inliner -Xllvm -sil-print-pass-name %S/../validation-test/closure_specialization/loop_vjp_inlining.swift -o %t.sil 2> %t.log
// RUN: %FileCheck %s --check-prefix=INLINE < %t.log
// RUN: %FileCheck %s --check-prefix=TAPE < %t.sil

// The source-generated VJP is admitted in the high-level pipeline, before the
// low-level pipeline's unconditional VJP inlining rule can apply.
// INLINE-LABEL: stage HighLevel,Function+EarlyLoopOpt, pass {{[0-9]+}}: EarlyPerfInliner (early-inline), Function: $s17loop_vjp_inlining6calleryS2fF
// INLINE-NOT: stage MidLevel
// INLINE-NOT: stage LowLevel
// INLINE: "reverse-mode derivative of loop_vjp_inlining.loop(_:)" inlined into "loop_vjp_inlining.caller(_:)"

// The loop still contains taped callbacks, so isTrivialVJP cannot admit it.
// Its returned pullback captures only the runtime context object.
// TAPE-LABEL: // reverse-mode derivative of loop(_:)
// TAPE: partial_apply
// TAPE: builtin "autoDiffAllocateSubcontextWithType"
// TAPE: %[[PB:[0-9]+]] = function_ref {{.*}} : $@convention(thin) (Float, @guaranteed Builtin.NativeObject) -> Float
// TAPE: builtin "autoDiffProjectTopLevelSubcontext"
// TAPE: %[[CLOSURE:[0-9]+]] = partial_apply [callee_guaranteed] %[[PB]](%{{[0-9]+}}) : $@convention(thin) (Float, @guaranteed Builtin.NativeObject) -> Float
// TAPE: %[[RESULT:[0-9]+]] = tuple (%{{[0-9]+}}, %[[CLOSURE]])
// TAPE: return %[[RESULT]]
// TAPE: } // end sil function
