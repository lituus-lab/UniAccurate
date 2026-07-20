# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
import std/unittest
import std/math
import UniAccurate

## Correctness of `naiveSum` against exact-integer and std-sum references.
## Rigorous forward-error-bound checks (vs an MPFR oracle) belong with the
## oracles module; here we lock the structural behavior: empty, single, exact
## integer sums, equality with Nim's plain sequential `sum`, and non-finite
## propagation including the opposite-sign overflow artifact.

suite "naiveSum":
  test "empty is zero":
    check naiveSum(newSeq[float64](0)) == 0.0
    check naiveSum(newSeq[float32](0)) == 0.0'f32

  test "single and small exact sums":
    check naiveSum([1.0]) == 1.0
    check naiveSum([1.0, 2.0]) == 3.0
    check naiveSum([1.0, 2.0, 3.0, 4.0]) == 10.0
    check naiveSum([1.0'f32, 2.0'f32, 3.0'f32]) == 6.0'f32

  test "integer-valued sums are exact":
    var x: seq[float64] = @[]
    for i in 1 .. 100:
      x.add float64(i)
    check naiveSum(x) == 5050.0
    check naiveSum([1.0, 2, 3, 4, 5, 6, 7, 8]) == 36.0

  test "matches std sequential sum (finite)":
    var x: seq[float64] = @[]
    for i in 0 ..< 500:
      x.add float64(i) * 0.1 + float64(i mod 13)
    var y: seq[float32] = @[]
    for i in 0 ..< 500:
      y.add float32(i) * 0.1'f32 + float32(i mod 13)
    check naiveSum(x) == sum(x)
    check naiveSum(y) == sum(y)

  test "non-finite propagates":
    check classify(naiveSum([1.0, NaN, 2.0])) == fcNan
    check classify(naiveSum([1.0, Inf])) == fcInf
    check classify(naiveSum([1.0, Inf, -Inf])) == fcNan # opposite-sign overflow
