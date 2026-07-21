# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
## Structural tests for the ORO/Rump family in `orosum.nim`.
##
## `sum2`/`sumK` (ORO online cascaded transform) and `accSum`/`nearSum`/
## `conditionNumber` (Rump transform), for float32 and float64:
##
##   * `sum2` — magnitude-robust compensated sum, value-identical to
##     `neumaierSum` (the ORO Vecsum-with-final-correction *is* Neumaier's
##     scheme).
##   * `sumK` — K-fold cascaded transform; K=1 the naive `twoSum` chain, K=2 the
##     first-order compensated sum (≈ `neumaierSum`), K<1 treated as 1.
##   * `accSum` — *faithful* rounding (within 1 ulp of the exact sum); checked
##     against `superSum` (correctly rounded) as the exact-round reference.
##   * `nearSum` — *correctly rounded*; bit-exact == `superSum` on every case.
##   * `conditionNumber` — `Σ|xᵢ|/|Σxᵢ|`; `1` on no cancellation, large under
##     cancellation, `Inf` when the sum is exactly `0`, `0` for empty input.
##
## Edge cases: empty/single, `0.1×10`, catastrophic cancellation, magnitude
## robustness, NaN/±Inf propagation (delegated to `superSum`, so finite input
## never yields NaN), finite overflow → ±Inf, subnormal recovery, order-
## invariance (correctly-rounded `nearSum`; integer-exact `sum2`/`sumK`). The
## exact-`Fraction` forward-bound tier lives in `py/tests/test_oracle.py`.
import std/unittest
import std/math
import UniAccurate

proc next(r: var uint64): uint64 =
  ## Deterministic xorshift64 so every run reproduces (no RNG state).
  var x = r
  x = x xor (x shl 13)
  x = x xor (x shr 7)
  x = x xor (x shl 17)
  r = x
  x

proc shuffle[T](r: var uint64, a: var seq[T]) =
  ## Fisher–Yates over the xorshift64 stream — a deterministic permutation.
  for i in countdown(a.high, 1):
    let j = int(next(r) mod uint64(i + 1))
    swap(a[i], a[j])

func maxFin(T: typedesc): T =
  ## Largest finite value of `T` (`high` returns Inf for floats in Nim).
  when T is float64: cast[float64](0x7FEF_FFFF_FFFF_FFFF'u64)
  else: cast[float32](0x7F7F_FFFF'u32)

func subnormal(T: typedesc): T =
  ## Smallest positive subnormal of `T` (one unit in the last place of zero).
  when T is float64: cast[float64](1'u64)
  else: cast[float32](1'u32)

func within1Ulp[T](a, refn: T): bool =
  ## `a` is a faithful rounding of the exact sum iff it is one of the floats
  ## adjacent to that sum. `superSum` is the correctly-rounded (nearest) float,
  ## so a faithful `a` lies in `{refn - ulp, refn, refn + ulp}`: the check is
  ## `|a - refn| <= ulp(refn)`. `ulp(0)` is the min subnormal, so the bound is
  ## safe when the reference is small. An exact match short-circuits first,
  ## covering the both-overflow case (`+Inf == +Inf`, where `|Inf - Inf|` would
  ## wrongly yield `NaN`).
  if a == refn:
    return true
  if classify(a) in {fcNan, fcInf, fcNegInf} or
      classify(refn) in {fcNan, fcInf, fcNegInf}:
    return false # one overflowed and the other did not — not faithful-adjacent
  abs(a - refn) <= ulp(refn)

proc mixedMag(T: typedesc, r: var uint64, n: int): seq[T] =
  ## `n` finite values with exponents spread across the normal range, so the
  ## sum rounds at every magnitude and exercises the cascaded transform.
  result = newSeq[T](n)
  for i in 0 ..< n:
    when T is float64:
      let expField = uint64(next(r) mod 2044) + 1 # biased exp in [1, 2044]: finite
      var bits = expField shl 52
      bits = bits or (next(r) and 0x000F_FFFF_FFFF_FFFF'u64)
      result[i] = cast[float64](bits)
    else:
      let expField = uint32(next(r) mod 252) + 2           # exp in [-125, 126]
      var bits = expField shl 23
      bits = bits or (uint32(next(r)) and 0x007F_FFFF'u32)
      result[i] = cast[float32](bits)

template oroSuite(T: typedesc) =
  suite "ORO/Rump summation [" & $T & "]":

    test "sum2: empty array sums to zero":
      let x: seq[T] = @[]
      check sum2(x) == T(0.0)

    test "sum2: single element is that element":
      check sum2([T(3.5)]) == T(3.5)

    test "sum2: ten 0.1 sum to exactly 1.0":
      let x = [T(0.1), T(0.1), T(0.1), T(0.1), T(0.1),
               T(0.1), T(0.1), T(0.1), T(0.1), T(0.1)]
      check sum2(x) == T(1.0)

    test "sum2: magnitude-robust, value-identical to neumaierSum":
      let mag = [T(1.0), T(1e20), T(1.0), T(-1e20)]
      check sum2(mag) == neumaierSum(mag)
      check sum2(mag) == T(2.0) # Kahan loses this; sum2 (Neumaier) recovers it

    test "sum2: NaN propagates (does not raise)":
      check classify(sum2([T(1.0), T(NaN), T(2.0)])) == fcNan

    test "sum2: finite overflow yields Inf, never NaN":
      let m = maxFin(T)
      check classify(sum2([m, m])) == fcInf

    test "sumK: K=1 is the naive twoSum chain (integer-exact)":
      check sumK([T(1.0), T(2.0), T(3.0), T(4.0)], 1) == T(10.0)

    test "sumK: K=2 is first-order compensated (recovers 0.1x10)":
      let x = [T(0.1), T(0.1), T(0.1), T(0.1), T(0.1),
               T(0.1), T(0.1), T(0.1), T(0.1), T(0.1)]
      check sumK(x, 2) == T(1.0)

    test "sumK: K=2 is magnitude-robust (recovers the dominated addends)":
      let mag = [T(1.0), T(1e20), T(1.0), T(-1e20)]
      check sumK(mag, 2) == T(2.0)

    test "sumK: K<1 is treated as 1":
      check sumK([T(1.0), T(2.0), T(3.0)], 0) == sumK([T(1.0), T(2.0), T(3.0)], 1)

    test "sumK: empty and single":
      let e: seq[T] = @[]
      check sumK(e, 3) == T(0.0)
      check sumK([T(7.0)], 3) == T(7.0)

    test "sumK: NaN propagates":
      check classify(sumK([T(1.0), T(NaN)], 2)) == fcNan

    test "sumK: finite overflow yields Inf, never NaN":
      let m = maxFin(T)
      check classify(sumK([m, m, m], 2)) == fcInf

    test "sumK: useFastTwoSum is bit-identical to the default (the EFT swap)":
      # twoSumFast yields the same (s, e) as twoSum, so the cascaded transform is
      # bit-identical — checked on integer-exact, 0.1×10, magnitude-robust, and
      # mixed-magnitude random data, for K in {1, 2, 3}.
      let ints = [T(1.0), T(2.0), T(3.0), T(4.0), T(5.0), T(6.0)]
      let tenths = [T(0.1), T(0.1), T(0.1), T(0.1), T(0.1),
                    T(0.1), T(0.1), T(0.1), T(0.1), T(0.1)]
      let mag = [T(1.0), T(1e20), T(1.0), T(-1e20), T(3.0), T(-3.0)]
      var r = 0x5EED'u64
      for K in 1 .. 3:
        check sumK(ints, K, useFastTwoSum = true) == sumK(ints, K)
        check sumK(tenths, K, useFastTwoSum = true) == sumK(tenths, K)
        check sumK(mag, K, useFastTwoSum = true) == sumK(mag, K)
      for n in [10, 100, 1000]:
        let x = mixedMag(T, r, n)
        for K in 1 .. 3:
          check sumK(x, K, useFastTwoSum = true) == sumK(x, K)

    test "sumK: assumeFinite is bit-identical on finite non-overflowing input":
      # The opt-in strips transform's per-step `isFin` guard and the cascade's
      # `isFin(result)` guard; neither fires on finite non-overflowing data, so
      # the cascade is bit-for-bit identical. Checked with and without
      # useFastTwoSum (the two levers compose independently). Bounded data only
      # — the opt-in's overflow output is undefined (a `twoSum(Inf, ·)`
      # precondition violation in debug), so `mixedMag` (which can overflow) is
      # not exercised here; `useFastTwoSum`'s own parity test covers that data.
      let ints = [T(1.0), T(2.0), T(3.0), T(4.0), T(5.0), T(6.0)]
      let tenths = [T(0.1), T(0.1), T(0.1), T(0.1), T(0.1),
                    T(0.1), T(0.1), T(0.1), T(0.1), T(0.1)]
      let mag = [T(1.0), T(1e20), T(1.0), T(-1e20), T(3.0), T(-3.0)]
      for K in 1 .. 3:
        check sumK(ints, K, assumeFinite = true) == sumK(ints, K)
        check sumK(tenths, K, assumeFinite = true) == sumK(tenths, K)
        check sumK(mag, K, assumeFinite = true) == sumK(mag, K)
        check sumK(ints, K, useFastTwoSum = true, assumeFinite = true) ==
          sumK(ints, K)

    test "sum2: assumeFinite is bit-identical to the default":
      let mag = [T(1.0), T(1e20), T(1.0), T(-1e20)]
      check sum2(mag, assumeFinite = true) == sum2(mag)
      check sum2(mag, assumeFinite = true) == T(2.0)

    test "accSum: empty and single":
      let e: seq[T] = @[]
      check accSum(e) == T(0.0)
      check accSum([T(3.5)]) == T(3.5)

    test "accSum: ten 0.1 is faithful (within 1 ulp of the exact round)":
      let x = [T(0.1), T(0.1), T(0.1), T(0.1), T(0.1),
               T(0.1), T(0.1), T(0.1), T(0.1), T(0.1)]
      check accSum(x) == T(1.0) # the exact sum 1.0 is a float → faithful picks it

    test "accSum: faithful under catastrophic cancellation":
      let mag = [T(1.0), T(1e20), T(1.0), T(-1e20)]
      check accSum(mag) == T(2.0)

    test "accSum: faithful on mixed-magnitude random data":
      var r = 0xBEEF'u64
      for n in [10, 100, 1000]:
        let x = mixedMag(T, r, n)
        check within1Ulp(accSum(x), superSum(x))

    test "accSum: NaN/Inf fall back to the exact superaccumulator":
      check classify(accSum([T(1.0), T(NaN), T(2.0)])) == fcNan
      check classify(accSum([T(1.0), T(Inf), T(2.0)])) == fcInf

    test "accSum: finite overflow yields Inf, never NaN":
      let m = maxFin(T)
      check classify(accSum([m, m])) == fcInf

    test "accSum: recovers a subnormal lost by naive summation":
      let eta = subnormal(T)
      check accSum([T(1.0), eta, T(-1.0)]) == eta

    test "nearSum: empty and single":
      let e: seq[T] = @[]
      check nearSum(e) == T(0.0)
      check nearSum([T(3.5)]) == T(3.5)

    test "nearSum: correctly rounded (bit-exact == superSum)":
      # The exact round of the sum, bit-for-bit, on cases that round and cancel.
      let x = [T(0.1), T(0.1), T(0.1), T(0.1), T(0.1),
               T(0.1), T(0.1), T(0.1), T(0.1), T(0.1)]
      check nearSum(x) == superSum(x)
      check nearSum(x) == T(1.0)
      let mag = [T(1.0), T(1e20), T(1.0), T(-1e20)]
      check nearSum(mag) == superSum(mag)
      check nearSum(mag) == T(2.0)

    test "nearSum: bit-exact == superSum on mixed-magnitude data":
      var r = 0xF00D'u64
      for n in [10, 100, 1000]:
        let x = mixedMag(T, r, n)
        check nearSum(x) == superSum(x)

    test "nearSum: recovers a subnormal lost by naive summation":
      let eta = subnormal(T)
      check nearSum([T(1.0), eta, T(-1.0)]) == eta

    test "nearSum: NaN/Inf fall back to the exact superaccumulator":
      check classify(nearSum([T(1.0), T(NaN), T(2.0)])) == fcNan
      check classify(nearSum([T(1.0), T(Inf), T(2.0)])) == fcInf

    test "nearSum: finite overflow yields Inf, never NaN":
      let m = maxFin(T)
      check classify(nearSum([m, m])) == fcInf

    test "nearSum: order-invariance (correctly rounded ⇒ permutation-invariant)":
      # nearSum rounds the exact real sum; integer addition is exact and
      # commutative, so the exact sum (and its single rounding) is invariant.
      var r = 0xCAFE'u64
      var x = mixedMag(T, r, 300)
      let ref0 = nearSum(x)
      for _ in 0 ..< 30:
        shuffle(r, x)
        check nearSum(x) == ref0

    test "conditionNumber: no cancellation ⇒ 1":
      check conditionNumber([T(1.0), T(1.0), T(1.0), T(1.0)]) == T(1.0)

    test "conditionNumber: cancellation ⇒ large":
      # Σ|x| ≈ 2e20, |Σx| ≈ 1 ⇒ cond ≈ 2e20. 1e20 stays inside the float32
      # range, so this is a finite-input cancellation for both precisions.
      let c = [T(1e20), T(1.0), T(-1e20)]
      check conditionNumber(c) > T(1e19)

    test "conditionNumber: exact-zero sum ⇒ Inf":
      check classify(conditionNumber([T(1.0), T(-1.0)])) == fcInf

    test "conditionNumber: empty ⇒ 0":
      let e: seq[T] = @[]
      check conditionNumber(e) == T(0.0)

    test "conditionNumber: finite input never NaN":
      let m = maxFin(T)
      check classify(conditionNumber([m, m, -m])) != fcNan

oroSuite(float64)
oroSuite(float32)

suite "ORO/Rump contracts (postconditions)":
  test "postconditions hold (no raise in debug, compiled away in release)":
    check sum2([1.0, 2.0, 3.0]) == 6.0
    check sumK([1.0, 2.0, 3.0], 2) == 6.0
    check accSum([1.0, 2.0, 3.0]) == 6.0
    check nearSum([1.0, 2.0, 3.0]) == 6.0
    check conditionNumber([1.0, 2.0, 3.0]) == 1.0
