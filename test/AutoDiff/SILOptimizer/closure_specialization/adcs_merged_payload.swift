// RUN: %target-swift-frontend -O -sil-verify-all -emit-sil -D DEFINITIONS_ONLY -module-name ADCSMP %S/../../validation-test/closure_specialization/adcs_merged_payload.swift -o %t.sil
// RUN: %FileCheck %S/../../validation-test/closure_specialization/adcs_merged_payload.swift --check-prefix=OPT < %t.sil
