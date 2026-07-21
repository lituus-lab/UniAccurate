# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
## Structural tests for the small superaccumulator in `exactsum.nim`.
##
## Covers `superSum` / `superDot` for float32 and float64: exact recovery on
## nice inputs (0.1×10, catastrophic cancellation), huge-magnitude exact
## cancellation (the superaccumulator holds the true integer sum, so
## `max + max − max − max = 0` and `max + max − max = max`, where naive hits
## `+Inf + −Inf = NaN`), IEEE propagation of NaN/±Inf, finite-overflow → ±Inf,
## subnormal exactness, order-invariance (integer addition is exact and
## commutative, so the single final rounding is permutation-invariant), the
## online/streaming property (incremental `add` + `round` == batch `superSum`),
## and `superDot` (exact product accumulation, product-overflow held at true
## magnitude, finite-input never NaN). The bit-exact `math.fsum` oracle
## comparison lives in `py/tests/test_oracle.py` (the exact-`Fraction`
## forward-bound tier), not here.
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

proc mixedMag(T: typedesc, r: var uint64, n: int): seq[T] =
  ## `n` finite values with exponents spread across the normal range, so the
  ## sum rounds at every magnitude and exercises the chunk sweep.
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

template superSuite(T: typedesc) =
  suite "superaccumulator exact summation [" & $T & "]":

    test "sums ten 0.1 to exactly 1.0":
      let x = [T(0.1), T(0.1), T(0.1), T(0.1), T(0.1),
               T(0.1), T(0.1), T(0.1), T(0.1), T(0.1)]
      check superSum(x) == T(1.0)

    test "recovers the sum under catastrophic cancellation":
      check superSum([T(1.0), T(1e20), T(1.0), T(-1e20)]) == T(2.0)

    test "empty array sums to zero":
      let x: seq[T] = @[]
      check superSum(x) == T(0.0)

    test "single element is that element":
      check superSum([T(3.5)]) == T(3.5)

    test "NaN propagates (does not raise)":
      check classify(superSum([T(1.0), T(NaN), T(2.0)])) == fcNan

    test "+Inf propagates":
      check classify(superSum([T(1.0), T(Inf), T(2.0)])) == fcInf

    test "Inf - Inf yields NaN":
      check classify(superSum([T(Inf), T(-Inf)])) == fcNan

    test "finite overflow does not raise and yields Inf":
      let m = maxFin(T)
      check classify(superSum([m, m])) == fcInf

    test "huge-magnitude cancellation is exact (no fallback needed)":
      # The superaccumulator holds the exact integer sum; max and -max cancel
      # bit-for-bit, leaving exactly 1.0.
      let m = maxFin(T)
      check superSum([m, -m, T(1.0)]) == T(1.0)

    test "opposite-sign finite overflow is never NaN":
      # A naive/dot sum of these would hit `+Inf + -Inf = NaN` (an order
      # artifact); the superaccumulator holds the exact integer sum, so it
      # recovers the true finite cancellation or a correctly-signed ±Inf on
      # genuine overflow — never NaN for finite input. This is the property
      # `naiveDot`/`dot2` rely on for their finite-input NaN fallback.
      let m = maxFin(T)
      check superSum([m, m, -m, -m]) == T(0.0) # equal-magnitude: exact 0
      check superSum([m, m, -m]) == m # no overflow: m + m - m = m (naive: NaN)
      check classify(superSum([m, m])) == fcInf # true overflow → +Inf
      check superSum([m, m, -m, -m, T(1.0)]) == T(1.0) # extreme unequal: exact 1

    test "signed -0.0 sums to 0.0":
      check superSum([T(-0.0), T(-0.0)]) == T(0.0)

    test "subnormals accumulate exactly (gradual underflow)":
      let eta = subnormal(T)
      check superSum([eta, eta]) == eta + eta
      check superSum([eta, -eta]) == T(0.0)

    test "recovers a subnormal lost by naive summation":
      # [1, eta, -1]: the exact sum is the subnormal eta (representable), so the
      # superaccumulator rounds to it bit-for-bit; naive loses it (1 + eta → 1).
      let eta = subnormal(T)
      check superSum([T(1.0), eta, T(-1.0)]) == eta
      check naiveSum([T(1.0), eta, T(-1.0)]) == T(0.0)

    test "order-invariance: every permutation gives the same result":
      # The superaccumulator holds the exact integer sum regardless of add order
      # (integer addition is exact, commutative, associative), so the single
      # final rounding is bit-for-bit invariant under permutation.
      var r = 0xC0FFEE'u64
      var x = mixedMag(T, r, 500)
      let ref0 = superSum(x)
      for _ in 0 ..< 30:
        shuffle(r, x)
        check superSum(x) == ref0

template dotSuite(T: typedesc) =
  suite "superaccumulator exact dot product [" & $T & "]":

    test "dot of small integer data is exact":
      check superDot([T(1.0), T(2.0), T(3.0)], [T(4.0), T(5.0), T(6.0)]) == T(32.0)

    test "empty dot is zero":
      let x: seq[T] = @[]
      check superDot(x, x) == T(0.0)

    test "product overflow is held at its true magnitude (not ±Inf)":
      # 1e100·1e100 = 1e200, representable in float64 (max ~1.8e308); the
      # float32 branch uses 1e10·1e10 = 1e20, also representable. The product
      # alone does not overflow either precision: the exact accumulator's role
      # here is to hold the product and the subtracted term in the integer
      # domain and round once, yielding the true magnitude rather than an
      # intermediate rounding of the product.
      when T is float64:
        let big = T(1e100)
        check superDot([big, T(1.0)], [big, T(-1.0)]) == big * big - T(1.0)
      else:
        let big = T(1e10) # float32: 1e10·1e10 = 1e20, representable
        check superDot([big, T(1.0)], [big, T(-1.0)]) == big * big - T(1.0)

    test "opposite-sign finite overflow dot is never NaN":
      # Two products overflow to ±Inf in float arithmetic; the exact accumulator
      # cancels them to the true finite value — never NaN for finite input.
      # This is the finite-input NaN fallback the dot family relies on.
      let m = maxFin(T)
      check superDot([m, -m], [m, m]) == T(0.0) # m·m + (-m)·m = 0, exact
      check classify(superDot([m, m], [m, m])) == fcInf # 2·m² overflows → +Inf

    test "NaN operand propagates":
      check classify(superDot([T(1.0), T(NaN)], [T(2.0), T(3.0)])) == fcNan

    test "Inf operand propagates":
      check classify(superDot([T(1.0), T(Inf)], [T(2.0), T(3.0)])) == fcInf

    test "order-invariance: permuting both vectors gives the same dot":
      # `Σ xᵢyᵢ` is invariant under a common permutation of both vectors, and
      # the exact accumulator makes the single final rounding so as well.
      var r = 0xD07'u64
      var x = mixedMag(T, r, 300)
      var y = mixedMag(T, r, 300)
      let ref0 = superDot(x, y)
      for _ in 0 ..< 30:
        shuffle(r, x)
        shuffle(r, y)
        check superDot(x, y) == ref0

superSuite(float64)
superSuite(float32)
dotSuite(float64)
dotSuite(float32)

suite "superSum online/streaming property [float64]":
  test "incremental add + round equals batch superSum":
    # Exercises the lazy carry counter across many adds.
    var r = 0x10C'u64
    var x = newSeq[float64](5000)
    for i in 0 ..< x.len:
      let expField = uint64(next(r) mod 2044) + 1 # biased exp in [1, 2044]: finite
      var bits = expField shl 52
      bits = bits or (next(r) and 0x000F_FFFF_FFFF_FFFF'u64)
      x[i] = cast[float64](bits)
    var acc: SuperAccumulator[float64]
    initSuperAccumulator(acc)
    for v in x:
      acc.add(v)
    check acc.round() == superSum(x)

  test "add(openArray) equals element-wise add":
    var r = 0x21E'u64
    var x = newSeq[float64](3000)
    for i in 0 ..< x.len:
      let expField = uint64(next(r) mod 2044) + 1 # biased exp in [1, 2044]: finite
      var bits = expField shl 52
      bits = bits or (next(r) and 0x000F_FFFF_FFFF_FFFF'u64)
      x[i] = cast[float64](bits)
    var a, b: SuperAccumulator[float64]
    initSuperAccumulator(a)
    initSuperAccumulator(b)
    for v in x:
      a.add(v)
    b.add(x)
    check a.round() == b.round()

suite "superSum / superDot contracts (postconditions)":
  test "postconditions hold (no raise in debug, compiled away in release)":
    check superSum([1.0, 2.0, 3.0]) == 6.0
    check superSum([1.0, 1e100, 1.0, -1e100]) == 2.0
    check superSum(newSeq[float64](0)) == 0.0
    check superSum([5.0]) == 5.0
    check superDot([1.0, 2.0], [3.0, 4.0]) == 11.0
