# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
import std/unittest
import std/math
import UniAccurate

## Correctness of the compensated sums (`kahanSum`, `neumaierSum`, `kleinSum`).
## Rigorous forward-error-bound checks (vs an exact rational oracle) live in
## `py/tests/test_oracle.py`; here we lock the structural behavior: empty,
## single, exact integer sums, the canonical compensation win on `0.1·10`, the
## magnitude-robustness distinction (Kahan loses it; Neumaier/Klein recover it),
## equality with `naiveSum` on exact data, and non-finite propagation including
## the opposite-sign overflow artifact.

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

suite "neumaierSum":
  test "empty is zero":
    check neumaierSum(newSeq[float64](0)) == 0.0
    check neumaierSum(newSeq[float32](0)) == 0.0'f32

  test "single and small exact sums":
    check neumaierSum([1.0]) == 1.0
    check neumaierSum([1.0, 2.0, 3.0, 4.0]) == 10.0
    check neumaierSum([1.0'f32, 2.0'f32, 3.0'f32]) == 6.0'f32

  test "integer-valued sums are exact":
    var x: seq[float64] = @[]
    for i in 1 .. 100:
      x.add float64(i)
    check neumaierSum(x) == 5050.0

  test "recovers the lost low-order bit (0.1 x 10)":
    let xs = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    check neumaierSum(xs) == 1.0

  test "magnitude-robust: recovers the dominated addend":
    # Where Kahan loses the small addends, Neumaier recovers the exact 2.0.
    check neumaierSum([1.0, 1e100, 1.0, -1e100]) == 2.0

  test "matches naiveSum on exact integer data":
    var x: seq[float64] = @[]
    for i in 0 ..< 500:
      x.add float64(i)
    check neumaierSum(x) == naiveSum(x)

  test "non-finite propagates":
    check classify(neumaierSum([1.0, NaN, 2.0])) == fcNan
    check classify(neumaierSum([1.0, Inf])) == fcInf
    check classify(neumaierSum([1.0, Inf, -Inf])) == fcNan

  test "finite input never yields NaN":
    var x: seq[float64] = @[]
    for _ in 0 ..< 256:
      x.add 1e308
    let r = neumaierSum(x)
    check classify(r) == fcInf

suite "kleinSum":
  test "empty is zero":
    check kleinSum(newSeq[float64](0)) == 0.0
    check kleinSum(newSeq[float32](0)) == 0.0'f32

  test "single and small exact sums":
    check kleinSum([1.0]) == 1.0
    check kleinSum([1.0, 2.0, 3.0, 4.0]) == 10.0
    check kleinSum([1.0'f32, 2.0'f32, 3.0'f32]) == 6.0'f32

  test "integer-valued sums are exact":
    var x: seq[float64] = @[]
    for i in 1 .. 100:
      x.add float64(i)
    check kleinSum(x) == 5050.0

  test "recovers the lost low-order bit (0.1 x 10)":
    let xs = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    check kleinSum(xs) == 1.0

  test "magnitude-robust: recovers the dominated addend":
    check kleinSum([1.0, 1e100, 1.0, -1e100]) == 2.0

  test "matches naiveSum on exact integer data":
    var x: seq[float64] = @[]
    for i in 0 ..< 500:
      x.add float64(i)
    check kleinSum(x) == naiveSum(x)

  test "non-finite propagates":
    check classify(kleinSum([1.0, NaN, 2.0])) == fcNan
    check classify(kleinSum([1.0, Inf])) == fcInf
    check classify(kleinSum([1.0, Inf, -Inf])) == fcNan

  test "finite input never yields NaN":
    var x: seq[float64] = @[]
    for _ in 0 ..< 256:
      x.add 1e308
    let r = kleinSum(x)
    check classify(r) == fcInf

suite "compensated sums agree on exact data":
  test "all three agree with naiveSum on integers":
    var x: seq[float64] = @[]
    for i in 1 .. 300:
      x.add float64(i)
    let n = naiveSum(x)
    check kahanSum(x) == n
    check neumaierSum(x) == n
    check kleinSum(x) == n

  test "neumaier and klein agree on the magnitude case":
    let xs = [1.0, 1e100, 1.0, -1e100]
    check neumaierSum(xs) == kleinSum(xs)
    check kleinSum(xs) == 2.0

suite "assumeFinite opt-in is bit-identical on finite non-overflowing input":
  # The opt-in strips the `isFin` guards; on inputs where no guard ever fires
  # (all finite, no partial-sum overflow) both paths run the same EFT, so the
  # result is bit-for-bit identical. The magnitude case (1e100, no overflow) is
  # the strongest witness — the guard is live but never trips there.
  template parity(T: typedesc) =
    let ints = [T(1.0), T(2.0), T(3.0), T(4.0), T(5.0)]
    let tenths = [T(0.1), T(0.1), T(0.1), T(0.1), T(0.1),
                  T(0.1), T(0.1), T(0.1), T(0.1), T(0.1)]
    # 1e20 is finite in both float32 and float64 (max f32 ~3.4e38); 1e100 would
    # overflow f32 to Inf and trip the opt-in's `allFin` precondition.
    let mag = [T(1.0), T(1e20), T(1.0), T(-1e20)]
    check kahanSum(ints, assumeFinite = true) == kahanSum(ints)
    check kahanSum(tenths, assumeFinite = true) == kahanSum(tenths)
    check neumaierSum(ints, assumeFinite = true) == neumaierSum(ints)
    check neumaierSum(tenths, assumeFinite = true) == neumaierSum(tenths)
    check neumaierSum(mag, assumeFinite = true) == neumaierSum(mag)
    check kleinSum(ints, assumeFinite = true) == kleinSum(ints)
    check kleinSum(tenths, assumeFinite = true) == kleinSum(tenths)
    check kleinSum(mag, assumeFinite = true) == kleinSum(mag)

  test "float64 parity": parity(float64)
  test "float32 parity": parity(float32)

  test "the guard is load-bearing (guarded path localizes overflow to Inf)":
    # The opt-in strips the `isFin` guards; on overflow its output is undefined
    # (a `twoSum(Inf, v)` precondition violation in debug, garbage in release),
    # so it is not pinned. The guarded default localizes the overflow to a
    # single-sign ±Inf — the safety property the guard exists for — checked in
    # every sum's own "finite input never yields NaN" test and restated here as
    # the contrast that makes the opt-in meaningful.
    var x: seq[float64] = @[]
    for _ in 0 ..< 256:
      x.add 1e308
    check classify(kahanSum(x)) == fcInf
    check classify(neumaierSum(x)) == fcInf
    check classify(kleinSum(x)) == fcInf
