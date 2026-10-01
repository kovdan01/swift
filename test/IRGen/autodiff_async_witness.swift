// RUN: %target-swift-frontend -emit-ir -parse-as-library -module-name AsyncWitness %s | %FileCheck %s
// REQUIRES: concurrency
// REQUIRES: differentiable_programming

import _Differentiation

@differentiable(reverse)
public func asyncIdentity(_ x: Float) async -> Float { x }

@differentiable(reverse)
public func syncIdentity(_ x: Float) -> Float { x }

// Check the witness GLOBAL, not a descriptor definition or llvm.used entry.
// CHECK-DAG: @"{{.*}}asyncIdentity{{.*}}WJrSpSr" = {{.*}}{ ptr, ptr } { ptr @"$s12AsyncWitness13asyncIdentityyS2fYaFTJfSpSrTu", ptr @"$s12AsyncWitness13asyncIdentityyS2fYaFTJrSpSrTu" }
// CHECK-DAG: @"{{.*}}syncIdentity{{.*}}WJrSpSr" = {{.*}}{ ptr, ptr } { ptr @"$s12AsyncWitness12syncIdentityyS2fFTJfSpSr", ptr @"$s12AsyncWitness12syncIdentityyS2fFTJrSpSr" }
