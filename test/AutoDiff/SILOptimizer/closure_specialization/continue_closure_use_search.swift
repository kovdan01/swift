// RUN: %target-swift-frontend -emit-sil -O -parse-as-library -sil-verify-all -module-name closure_use_search %s | %FileCheck %s
//
// A thermal update composes three generated pullbacks. A pullback can be
// captured by an unused adapter as well as applied through another use. A
// failed recursive search through that adapter must not hide the live use.
// This example is reduced from a building-inferencer heat-step workload.
//
// CHECK-LABEL: sil hidden {{.*}}@{{.*}}10regimeStep{{.*}}TJrSSSpSr :
// CHECK: [[PULLBACK:%[0-9]+]] = partial_apply
// CHECK-NOT: partial_apply
// CHECK: [[RESULT:%[0-9]+]] = tuple ({{.*}}, [[PULLBACK]])
// CHECK-NEXT: return [[RESULT]]
// CHECK-NEXT: } // end sil function

import _Differentiation

@differentiable(reverse)
func filmCoefficient(_ conductance: Float, _ regime: Float) -> Float {
    let corrected = conductance + 0.01 * conductance * conductance
    return regime * corrected
}

@differentiable(reverse)
func heatFlow(_ temperature: Float, _ surface: Float, _ coefficient: Float) -> Float {
    coefficient * (surface - temperature)
}

@differentiable(reverse)
func eulerStep(_ temperature: Float, _ flow: Float, _ inverseMassStep: Float) -> Float {
    temperature + flow * inverseMassStep
}

@differentiable(reverse)
func regimeStep(_ temperature: Float, _ surface: Float, _ conductance: Float) -> Float {
    let regime: Float
    if temperature > 21 { regime = 1.31 } else { regime = 1.81 }
    let coefficient = filmCoefficient(conductance, regime)
    let flow = heatFlow(temperature, surface, coefficient)
    return eulerStep(temperature, flow, 0.03125)
}


@differentiable(reverse)
func loss(_ p: [Float]) -> Float {
    var temperature = p[0]
    let surface: Float = 25
    let conductance = p[1]
    var objective: Float = 0
    for _ in 0..<512 {
        temperature = regimeStep(temperature, surface, conductance)
        let error = temperature - 21.5
        objective += error * error
    }
    return objective / 512
}

public func heatGradient(_ parameters: [Float]) -> [Float].TangentVector {
  gradient(at: parameters, of: loss)
}
