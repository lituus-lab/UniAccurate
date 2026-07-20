# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
import std/unittest
import std/math
import UniAccurate

## Correctness of `kahanSum`. Rigorous forward-error-bound checks (vs an MPFR
## oracle) belong with the oracles module; here we lock the structural
## behavior: empty, single, exact integer sums, the canonical compensation win
## on `0.1·10`, equality with `naiveSum` on exact data, the magnitude-robustness
## limitation, and non-finite propagation including the opposite-sign overflow
## artifact.

suite "kahanSum":
  test "empty is zero":
    check kahanSum(newSeq[float64](0)) == 0.0
    check kahanSum(newSeq[float32](0)) == 0.0'f32

  test "single and small exact sums":
    check kahanSum([1.0]) == 1.0
    check kahanSum([1.0, 2.0]) == 3.0
    check kahanSum([1.0, 2.0, 3.0, 4.0]) == 10.0
    check kahanSum([1.0'f32, 2.0'f32, 3.0'f32]) == 6.0'f32

  test "integer-valued sums are exact":
    var x: seq[float64] = @[]
    for i in 1 .. 100:
      x.add float64(i)
    check kahanSum(x) == 5050.0

  test "recovers the lost low-order bit (0.1 x 10)":
    # Naive left-to-right gives 0.9999999999999999; Kahan recovers 1.0.
    let xs = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    check kahanSum(xs) == 1.0
    check naiveSum(xs) != 1.0

  test "matches naiveSum on exact integer data":
    var x: seq[float64] = @[]
    for i in 0 ..< 500:
      x.add float64(i)
    check kahanSum(x) == naiveSum(x)

  test "magnitude-robustness limitation":
    # A later addend dominates the running sum; Kahan's form loses the small
    # addends (the motivation for the magnitude-robust variant). Lock the
    # known weakness: the exact sum is 2.0 but Kahan does not recover it.
    check kahanSum([1.0, 1e100, 1.0, -1e100]) != 2.0

  test "non-finite propagates":
    check classify(kahanSum([1.0, NaN, 2.0])) == fcNan
    check classify(kahanSum([1.0, Inf])) == fcInf
    check classify(kahanSum([1.0, Inf, -Inf])) == fcNan # opposite-sign overflow

  test "finite input never yields NaN":
    # Overflow to a single-sign Inf, not NaN (the guard diverts before any
    # Inf - Inf). 1e308 x 256 overflows float64.
    var x: seq[float64] = @[]
    for _ in 0 ..< 256:
      x.add 1e308
    let r = kahanSum(x)
    check classify(r) == fcInf
