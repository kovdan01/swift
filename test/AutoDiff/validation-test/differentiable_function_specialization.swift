// RUN: %target-swift-frontend -emit-sil %s -O -o %t/out.sil
// RUN: cat %t/out.sil | %FileCheck %s

import _Differentiation

extension Array.DifferentiableView:
    @retroactive Sequence,
    @retroactive Collection,
    @retroactive RangeReplaceableCollection,
    @retroactive RandomAccessCollection
    where Element: Differentiable
{
    public typealias Element = Array.Element
    public typealias Index = Array.Index
    public typealias SubSequence = Array.SubSequence

    @inlinable
    public subscript(position: Index) -> Element {
        _read { yield base[position] }
        set(newValue) { base[position] = newValue }
    }

    @inlinable
    public subscript(bounds: Range<Index>) -> SubSequence {
        get { base[bounds] }
        set(newValue) { base[bounds] = newValue }
    }

    @inlinable
    public var startIndex: Index { base.startIndex }

    @inlinable
    public var endIndex: Index { base.endIndex }

    @inlinable
    public init() {
        self.init(Array<Element>())
    }

    @inlinable
    public mutating func replaceSubrange<C>(_ subrange: Range<Index>, with newElements: C)
        where C: Collection, Element == C.Element
    {
        base.replaceSubrange(subrange, with: newElements)
    }
}

public protocol DifferentiableCollection: Differentiable & Collection where
    Element: Differentiable,
    TangentVector: DifferentiableCollectionTangentVector,
    TangentVector.Element == Element.TangentVector
{
    associatedtype Element
    associatedtype TangentVector

    var tangentCount: Int { get }

    func tangentIndex(for i: Index) -> TangentVector.Index
}

extension DifferentiableCollection where Index == TangentVector.Index {
    @inlinable public var tangentCount: Int { count }
    @inlinable public func tangentIndex(for i: Index) -> TangentVector.Index { i }
}

public protocol DifferentiableCollectionTangentVector: DifferentiableCollection {
    init()
    init(repeating value: Element, count: Int)
    mutating func reserveCapacity(_ capacity: Int)
    mutating func writeTangentContribution(of value: Element, at index: Index)
}

extension Array: DifferentiableCollection where Element: Differentiable & AdditiveArithmetic {}

extension Array.DifferentiableView: DifferentiableCollection where Element: AdditiveArithmetic {}

extension Array.DifferentiableView: DifferentiableCollectionTangentVector where Element: AdditiveArithmetic {
    @inlinable public mutating func writeTangentContribution(of value: Element, at index: Index) {
        self[index] += value
    }
}

@inline(never)
@differentiable(reverse)
public func fusedScalarZip<C1>(
    _ c1: C1,
    with transform: @differentiable(reverse) (Double) -> Double
) -> [Double] where
    C1: DifferentiableCollection, C1.Element == Double
{
    let n = c1.count
    if n == 0 { return [] }
    var out = [Double](repeating: 0, count: n)
    var i1 = c1.startIndex
    for k in 0 ..< n {
        out[k] = transform(c1[i1])
        c1.formIndex(after: &i1)
    }
    return out
}

@inline(never)
func valueWithPullbackWrapper(at: Double, of: @differentiable(reverse) (Double) -> Double) -> (Double, (Double) -> Double) {
    return valueWithPullback(at: at, of: of)
}

@inline(never)
@derivative(of: fusedScalarZip)
public func _vjpFusedScalarZip<C1>(
    _ c1: C1,
    with transform: @differentiable(reverse) (Double) -> Double
) -> (
    value: [Double],
    pullback: ([Double].TangentVector) -> (C1.TangentVector)
) where
    C1: DifferentiableCollection, C1.Element == Double
{
    let n = c1.count
    let tangentCount1 = c1.tangentCount

    var out = [Double](repeating: 0, count: n)
    var partials1 = [Double](repeating: 0, count: n)
    var tangentIndices1 = [C1.TangentVector.Index]()
    tangentIndices1.reserveCapacity(n)

    var i1 = c1.startIndex
    for k in 0 ..< n {
        let (value, pullback) = valueWithPullbackWrapper(at: c1[i1], of: transform)
        out[k] = value
        let (d1) = pullback(1.0)
        partials1[k] = d1
        tangentIndices1.append(c1.tangentIndex(for: i1))
        c1.formIndex(after: &i1)
    }

    return (
        value: out,
        pullback: { v in
            var results1 = C1.TangentVector(repeating: .zero, count: tangentCount1)
            guard v.count != 0 else { return (results1) }
            precondition(v.count == n)
            var vi = v.startIndex
            for k in 0 ..< n {
                let dOut = v[vi]
                results1.writeTangentContribution(of: dOut * partials1[k], at: tangentIndices1[k])
                v.formIndex(after: &vi)
            }
            return (results1)
        }
    )
}

@differentiable(reverse)
public func caller(_ arr: [Double]) -> [Double] {
    return fusedScalarZip(arr) { 37 * $0 }
}

// CHECK: // caller(_:)
// CHECK: sil @$s3out6callerySaySdGACF : $@convention(thin) (@guaranteed Array<Double>) -> @owned Array<Double> {
// CHECK: bb0(%0 : $Array<Double>):
// CHECK:   // function_ref specialized fusedScalarZip<A>(_:with:)
// CHECK:   %2 = function_ref @$s3out14fusedScalarZip_4withSaySdGx_S2dYjrXEtAA24DifferentiableCollectionRzSd7ElementRtzlFAD_Tg5S2dIgyd_S4dIegyd_Igydo_S4dIegyd_Igydo_Tf1na_n30$s3out6callerySaySdGACFS2dcfU_0ijkL9U_TJfSpSr0ijkl5U_TJrnO0Tf1nccc_n : $@convention(thin) (@guaranteed Array<Double>) -> @owned Array<Double>
// CHECK:   %3 = apply %2(%0) : $@convention(thin) (@guaranteed Array<Double>) -> @owned Array<Double>
// CHECK:   return %3
// CHECK: } // end sil function '$s3out6callerySaySdGACF'
