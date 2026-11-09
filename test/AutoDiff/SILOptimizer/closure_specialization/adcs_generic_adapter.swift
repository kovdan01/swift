// RUN: %target-swift-frontend -O -sil-verify-all -emit-sil -D DEFINITIONS_ONLY -module-name adcs_generic_adapter %S/../../validation-test/closure_specialization/adcs_generic_adapter.swift -o %t.sil
// RUN: %FileCheck %S/../../validation-test/closure_specialization/adcs_generic_adapter.swift < %t.sil
