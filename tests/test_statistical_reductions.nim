# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
import std/[math, unittest]
import UniAccurate

template reductionSuite(T: typedesc) =
  suite "statistical reductions [" & $T & "]":
    test "scaled norm preserves extreme representable magnitudes":
      let largest = when T is float64:
          cast[float64](0x7FEF_FFFF_FFFF_FFFF'u64)
        else:
          cast[float32](0x7F7F_FFFF'u32)
      check scaledEuclideanNorm([largest]) == largest
      check scaledEuclideanNorm([T(3), T(4)]) == T(5)

    test "scaled norm avoids subnormal square underflow":
      let tiny = when T is float64: T(1e-300) else: T(1e-30)
      check scaledEuclideanNorm([tiny]) == tiny

    test "centered sums recover cancellation":
      let base = when T is float64: T(1e10) else: T(1e5)
      let values = [base, base + T(1), base + T(2)]
      check centeredSumSquares(values, base + T(1)) == T(2)
      check centeredCrossProduct(values, [T(2), T(4), T(6)],
        base + T(1), T(4)) == T(4)

    test "empty and non-finite inputs follow IEEE semantics":
      let empty: seq[T] = @[]
      check scaledEuclideanNorm(empty) == T(0)
      check centeredSumSquares(empty, T(0)) == T(0)
      check classify(scaledEuclideanNorm([T(NaN)])) == fcNan
      check classify(scaledEuclideanNorm([T(Inf)])) == fcInf

reductionSuite(float64)
reductionSuite(float32)
