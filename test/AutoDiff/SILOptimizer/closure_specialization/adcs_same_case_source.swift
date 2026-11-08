// RUN: %target-swift-frontend -emit-sil -O -D DEFINITIONS_ONLY -module-name adcs_same_case_source %S/../../validation-test/closure_specialization/adcs_same_case_source.swift -Xllvm -sil-print-before=autodiff-closure-specialization -o %t.sil 2> %t.before
// RUN: %FileCheck %S/../../validation-test/closure_specialization/adcs_same_case_source.swift < %t.sil
// RUN: %FileCheck %s --check-prefix=BEFORE < %t.before

// Use the actual source so the synthesized one-element tuple payloads retain
// their AST types. The first ADCS input has distinct tuple instructions saving
// the same callback for the same branch-tracing case. The validation pair also
// checks that the optimized VJP returns the specialized pullback and that all
// opaque callback applications disappear from its body.

// BEFORE-LABEL: // reverse-mode derivative of twoSelections
// BEFORE: bb3:
// BEFORE: %[[FIRST:[0-9]+]] = tuple (%[[CALLBACK:[0-9]+]])
// BEFORE-NEXT: %{{[0-9]+}} = enum [[ENUMTYPE:\$_AD__[^, ]+]], [[ENUMCASE:#[^ ]+]], %[[FIRST]]
// BEFORE-NOT: } // end sil function
// BEFORE: bb4:
// BEFORE: %[[SECOND:[0-9]+]] = tuple (%[[CALLBACK]])
// BEFORE-NEXT: %{{[0-9]+}} = enum [[ENUMTYPE]], [[ENUMCASE]], %[[SECOND]]
// BEFORE-NOT: } // end sil function
// BEFORE: return
// BEFORE-NEXT: } // end sil function
