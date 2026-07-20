# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
import std/unittest
import std/math
import UniAccurate

## Structural correctness of `pairwiseSum`: empty/edge cases, exact-integer
## sums that exercise the recursion at sizes around and above the base-case
## threshold, equality with `naiveSum` for exactly representable sums, and
## non-finite propagation including a merge-node opposite-sign overflow.
## Worst-case/RMS accuracy ordering against an MPFR oracle belongs with the
## oracles module.

proc iotaSumF64(n: int): float64 =
  ## Exact integer sum `1 + 2 + ... + n` as a float64 (exact while the result
  ## stays below 2^53, which holds for every `n` used here).
  var s: float64 = 0
  for i in 1 .. n:
    s += float64(i)
  result = s

suite "pairwiseSum":
  test "empty is zero":
    check pairwiseSum(newSeq[float64](0)) == 0.0
    check pairwiseSum(newSeq[float32](0)) == 0.0'f32

  test "single and small exact sums":
    check pairwiseSum([1.0]) == 1.0
    check pairwiseSum([1.0, 2.0, 3.0, 4.0]) == 10.0
    check pairwiseSum([1.0'f32, 2.0'f32, 3.0'f32]) == 6.0'f32

  test "integer sums are exact across the recursion":
    # Sizes spanning the threshold (128): below, at, just above, power of two.
    for n in [1, 64, 128, 129, 200, 256, 257, 1000]:
      var x: seq[float64] = @[]
      for i in 1 .. n:
        x.add float64(i)
      check pairwiseSum(x) == iotaSumF64(n)

  test "matches naiveSum for exactly representable sums":
    var x: seq[float64] = @[]
    for i in 1 .. 300:
      x.add float64(i)
    check pairwiseSum(x) == naiveSum(x) # both exact for integer-valued data

  test "finite input stays finite (recursion does not crash)":
    var x: seq[float64] = @[]
    for i in 0 ..< 500:
      x.add float64(i) * 0.1 + float64(i mod 17)
    check classify(pairwiseSum(x)) in {fcNormal, fcZero, fcNegZero}

  test "non-finite propagates":
    check classify(pairwiseSum([1.0, NaN, 2.0])) == fcNan
    check classify(pairwiseSum([1.0, Inf])) == fcInf
    check classify(pairwiseSum([1.0, Inf, -Inf])) == fcNan # opposite-sign overflow

  test "opposite-sign overflow at a merge node":
    # Two finite blocks each overflow to opposite infinities in their base
    # case; the merge node computes +Inf + -Inf = NaN. Locks the deferred
    # fallback limitation documented in the module.
    var x: seq[float64] = @[]
    for _ in 0 ..< 128:
      x.add 1e308
    for _ in 0 ..< 128:
      x.add -1e308
    check classify(pairwiseSum(x)) == fcNan
