// RUN: %empty-directory(%t)
// RUN: %{python} %S/Inputs/make_api_bootstrap.py %S/../../../stdlib/public/Differentiation/AsyncDifferentialOperators.swift %S/Inputs/leaf.swift %t/leaf.swift
// RUN: %target-build-swift -parse-stdlib -parse-as-library -Onone %t/leaf.swift -Xllvm -enable-experimental-async-autodiff -o %t/leaf
// RUN: %target-run %t/leaf | %FileCheck %s --check-prefix=LEAF
// RUN: %target-build-swift -parse-stdlib -parse-as-library -O %t/leaf.swift -Xllvm -enable-experimental-async-autodiff -o %t/leaf-O
// RUN: %target-run %t/leaf-O choose-square | %FileCheck %s --check-prefix=LEAF
// RUN: %{python} %S/Inputs/make_api_bootstrap.py %S/../../../stdlib/public/Differentiation/AsyncDifferentialOperators.swift %S/Inputs/tasks.swift %t/tasks.swift
// RUN: %target-build-swift -parse-stdlib -parse-as-library -Onone %t/tasks.swift -Xllvm -enable-experimental-async-autodiff -o %t/tasks
// RUN: %target-run %t/tasks | %FileCheck %s --check-prefix=TASKS
// RUN: %target-build-swift -parse-stdlib -parse-as-library -O %t/tasks.swift -Xllvm -enable-experimental-async-autodiff -o %t/tasks-O
// RUN: %target-run %t/tasks-O | %FileCheck %s --check-prefix=TASKS
// REQUIRES: executable_test
// REQUIRES: concurrency_runtime
// REQUIRES: differentiable_programming
// UNSUPPORTED: use_os_stdlib

// LEAF: PASS async reverse API:
// TASKS: value=11.0, seed=1.0, gradient=(4.0, 3.0)
// TASKS: PASS async-let reverse API:
