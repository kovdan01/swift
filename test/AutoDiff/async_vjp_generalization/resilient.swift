// RUN: %empty-directory(%t)
// RUN: %target-swift-frontend -emit-module -enable-library-evolution -module-name AsyncADValueModel %S/Inputs/Model.swift -o %t/AsyncADValueModel.swiftmodule
// RUN: %target-swift-frontend -emit-sil -parse-as-library -I %t %s -Onone -Xllvm -enable-experimental-async-autodiff -sil-verify-all -sil-ownership-verify-all -o /dev/null
// RUN: %target-swift-frontend -emit-sil -parse-as-library -I %t %s -O -Xllvm -enable-experimental-async-autodiff -sil-verify-all -sil-ownership-verify-all -o /dev/null
// REQUIRES: concurrency, differentiable_programming
import _Differentiation
import _Concurrency
import AsyncADValueModel

// BEGIN DIFFERENTIATED FUNCTIONS
@inline(never)
func echoPacket(_ x: Packet) async -> Packet { x }
@differentiable(reverse)
func packetRoundTrip(_ x: Packet) async -> Packet {
  async let value = echoPacket(x)
  return await value
}
// END DIFFERENTIATED FUNCTIONS

@main struct ResilientTests {
  nonisolated static func main() async {
    let x = Packet(values: [2, 4, 3], scale: 5, tag: 9)
    let result = /*AD_AWAIT*/ await valueWithPullback(at: x, of: packetRoundTrip)
    let synchronous: (Packet.TangentVector) -> Packet.TangentVector = result.pullback
    precondition(result.value.values == [2, 4, 3] && result.value.tag == 9)
    // Use an existing tangent value to avoid requiring a public synthesized
    // memberwise initializer in the separately-built module.
    var seed = Packet.TangentVector.zero
    seed.values = [Float].TangentVector([2, -1, 3])
    seed.scale = 7
    await Task.yield()
    for _ in 0..<64 {
      let g = synchronous(seed)
      precondition(g.values.base == [2, -1, 3] && g.scale == 7)
    }
    print("resilient gradient=\(synchronous(seed).values.base)")
    print("PASS resilient")
  }
}
