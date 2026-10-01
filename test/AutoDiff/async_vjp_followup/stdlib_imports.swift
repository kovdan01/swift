// RUN: %target-swift-frontend -emit-silgen -parse-as-library -parse-stdlib -disable-implicit-concurrency-module-import -enable-library-evolution -enable-upcoming-feature MemberImportVisibility -module-name _Differentiation %S/../../../stdlib/public/Differentiation/Differentiable.swift %S/../../../stdlib/public/Differentiation/AsyncDifferentialOperators.swift -o %t.sil
// RUN: %FileCheck %s --input-file=%t.sil
// RUN: %target-swift-frontend -emit-module -parse-as-library -parse-stdlib -disable-implicit-concurrency-module-import -enable-library-evolution -enable-upcoming-feature MemberImportVisibility -experimental-skip-non-inlinable-function-bodies -module-name _Differentiation %S/../../../stdlib/public/Differentiation/Differentiable.swift %S/../../../stdlib/public/Differentiation/AsyncDifferentialOperators.swift -o %t.swiftmodule -emit-module-interface-path %t.swiftinterface
// REQUIRES: concurrency
// REQUIRES: differentiable_programming

// Compile actual stdlib sources, not wrappers importing _Differentiation. The
// failure was in emitExpectedExecutorProlog, so -typecheck alone is insufficient.
// There is deliberately no import in this file and no experimental AD flag:
// generic reverse operators invoke a VJP; they do not synthesize async lets.
// This small module is a build test only. Do not use it for numerical tests,
// because it deliberately does not contain the concrete Float conformances.

// CHECK: // valueWithPullback(at:of:)
// CHECK: sil {{.*}}@async
// CHECK: differentiable_function_extract [vjp]
// CHECK: apply {{.*}} : ${{.*}}@async
// CHECK: hop_to_executor
// CHECK: return
