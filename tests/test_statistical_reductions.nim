# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
import std/[math, unittest]
import UniAccurate
when not defined(release) and not defined(danger):
  import contracts

template reductionSuite(T: typedesc) =
  suite "statistical reductions [" & $T & "]":
    test "scaled mean avoids sum overflow and preserves cancellation":
      let largest = when T is float64:
          cast[float64](0x7FEF_FFFF_FFFF_FFFF'u64)
        else:
          cast[float32](0x7F7F_FFFF'u32)
      check scaledMean([largest, largest]) == largest
      let base = when T is float64: T(1e16) else: T(1e7)
      check scaledMean([base, T(1), -base]) == T(1) / T(3)
      let tiny = when T is float64:
          cast[float64](1'u64)
        else:
          cast[float32](1'u32)
      check scaledMean([tiny, tiny]) == tiny

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

    test "centered cosine avoids overflowing sums of squares":
      let large = when T is float64: T(1e308) else: T(1e30)
      let tolerance = when T is float64: T(2e-15) else: T(2e-6)
      check abs(centeredCosineSimilarity([large, -large], [large, -large],
        T(0), T(0)) - T(1)) <= tolerance
      check abs(centeredCosineSimilarity([large, -large], [-large, large],
        T(0), T(0)) + T(1)) <= tolerance
      check classify(centeredCosineSimilarity([large, large], [T(1), T(2)],
        large, T(1.5))) == fcNan

    test "empty and non-finite inputs follow IEEE semantics":
      let empty: seq[T] = @[]
      check scaledEuclideanNorm(empty) == T(0)
      check centeredSumSquares(empty, T(0)) == T(0)
      when not defined(release) and not defined(danger):
        expect PreConditionDefect: discard scaledMean(empty)
      else:
        expect ValueError: discard scaledMean(empty)
      check classify(scaledEuclideanNorm([T(NaN)])) == fcNan
      check classify(scaledEuclideanNorm([T(Inf)])) == fcInf

reductionSuite(float64)
reductionSuite(float32)
