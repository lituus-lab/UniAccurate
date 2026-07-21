# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
## Structural tests for the correctly-rounded summation in `shewchuksum.nim`.
##
## Covers `shewchukSum` / `fsum` for float32 and float64: exact recovery on
## nice inputs (0.1×10, catastrophic cancellation, magnitude absorption), the
## round-half-to-even tie, IEEE propagation of NaN/±Inf, overflow robustness
## (finite inputs never raise, never NaN), subnormal exactness, and
## order-invariance (every permutation gives the same correctly-rounded result).
## The bit-exact `math.fsum` oracle comparison lives in `py/tests/test_oracle.py`
## (the exact-`Fraction` forward-bound tier), not here.
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

template shewSuite(T: typedesc) =
  suite "correctly-rounded summation [" & $T & "]":

    test "sums ten 0.1 to exactly 1.0":
      let x = [T(0.1), T(0.1), T(0.1), T(0.1), T(0.1),
               T(0.1), T(0.1), T(0.1), T(0.1), T(0.1)]
      check shewchukSum(x) == T(1.0)
      check fsum(x) == T(1.0)

    test "recovers the sum under catastrophic cancellation":
      check shewchukSum([T(1.0), T(1e20), T(1.0), T(-1e20)]) == T(2.0)

    test "recovers a value absorbed by naive summation":
      let x = [T(1e16), T(1.0), T(-1e16)]
      check shewchukSum(x) == T(1.0)
      check naiveSum(x) != T(1.0)

    test "empty array sums to zero":
      let x: seq[T] = @[]
      check shewchukSum(x) == T(0.0)
      check fsum(x) == T(0.0)

    test "single element is that element":
      check shewchukSum([T(3.5)]) == T(3.5)
      check fsum([T(3.5)]) == T(3.5)

    test "NaN propagates (does not raise)":
      check classify(shewchukSum([T(1.0), T(NaN), T(2.0)])) == fcNan

    test "+Inf propagates":
      check classify(shewchukSum([T(1.0), T(Inf), T(2.0)])) == fcInf

    test "Inf - Inf yields NaN":
      check classify(shewchukSum([T(Inf), T(-Inf)])) == fcNan

    test "overflow from finite inputs does not raise and yields Inf":
      let m = maxFin(T)
      check classify(shewchukSum([m, m, -m])) == fcInf

    test "opposite-sign finite overflow is never NaN":
      # The naive fallback is sign-stable; the exact-expansion path recovers
      # true cancellation when it does not overflow. Both keep finite-input
      # results finite or ±Inf, never NaN.
      let m = maxFin(T)
      check classify(shewchukSum([m, m, -m])) != fcNan
      check classify(fsum([m, m, -m])) != fcNan
      check shewchukSum([m, -m, m, -m]) == T(0.0) # exact, no overflow
      check fsum([m, -m, m, -m]) == T(0.0)
      check classify(shewchukSum([m, m, -m, -m])) != fcNan # overflow bail
      check classify(fsum([m, m, -m, -m])) != fcNan

    test "signed -0.0 sums to 0.0":
      check shewchukSum([T(-0.0), T(-0.0)]) == T(0.0)

    test "subnormals accumulate exactly (gradual underflow)":
      let eta = subnormal(T)
      check shewchukSum([eta, eta, eta, eta]) == T(4) * eta
      check fsum([eta, eta, eta, eta]) == T(4) * eta
      check shewchukSum([eta, -eta]) == T(0.0)

    test "recovers a subnormal lost by naive summation":
      let eta = subnormal(T)
      check shewchukSum([T(1.0), eta, T(-1.0)]) == eta
      check naiveSum([T(1.0), eta, T(-1.0)]) == T(0.0)

    test "order-invariance: every permutation gives the same result":
      # Correctly rounded ⇒ the exact real sum (order-independent) rounded
      # once, so both shewchukSum and fsum are bit-for-bit invariant under
      # permutation.
      var r = 0xC0FFEE'u64
      var x = newSeq[T](500)
      for i in 0 ..< x.len:
        when T is float64:
          let expField = uint64(next(r) mod 1022) + 1
          var bits = expField shl 52
          bits = bits or (next(r) and 0x000F_FFFF_FFFF_FFFF'u64)
          x[i] = cast[float64](bits)
        else:
          let expField = uint32(next(r) mod 126) + 1
          var bits = expField shl 23
          bits = bits or (uint32(next(r)) and 0x007F_FFFF'u32)
          x[i] = cast[float32](bits)
      let ref0 = shewchukSum(x)
      let ref1 = fsum(x)
      for _ in 0 ..< 30:
        shuffle(r, x)
        check shewchukSum(x) == ref0
        check fsum(x) == ref1

shewSuite(float64)
shewSuite(float32)

suite "correctly-rounded summation [float64] tie-break":
  test "round-half-to-even: sum([1e-16, 1, 1e16]) rounds up, not down":
    # The CPython `fsum` example: the exact sum straddles a tie at ulp(1e16) =
    # 2 and rounds up (to even); naive summation absorbs the 1 and yields 1e16.
    # Below the float32 ulp of 1e16, so this is a float64-only property.
    let x = [1e-16, 1.0, 1e16]
    check shewchukSum(x) == 1.0000000000000002e16
    check shewchukSum(x) != naiveSum(x)

suite "shewchukSum contracts (postconditions)":
  test "postconditions hold (no raise in debug, compiled away in release)":
    check shewchukSum([1.0, 2.0, 3.0]) == 6.0
    check fsum([1.0, 2.0, 3.0]) == 6.0
    check shewchukSum(newSeq[float64](0)) == 0.0
    check fsum(newSeq[float64](0)) == 0.0
