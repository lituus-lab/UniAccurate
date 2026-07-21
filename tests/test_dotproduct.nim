# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
## Structural tests for the dot family in `dotproduct.nim`.
##
## `naiveDot`, `dot2` (ORO Alg. 5.3, K = 2; Graillat Dot2FMA), `dotK` (the
## online (K−1)-fold cascade), for float32 and float64:
##
##   * `naiveDot` — left-to-right `Σ xᵢyᵢ`; finite inputs never yield NaN (an
##     opposite-sign ±Inf product overflow → NaN is recovered via `superDot`).
##   * `dot2` — twice-precision compensated dot; `dot2(x, y) == dotK(x, y, 2)`
##     bit-for-bit; recovers the compensation win on cancellation; falls back to
##     `naiveDot` (hence `superDot` on the rare NaN) on non-finite/overflow.
##   * `dotK` — K-fold; `K = 1` the naive dot, `K < 1` treated as 1.
##
## Edge cases: empty/single, integer-exact, catastrophic cancellation (the
## dot-product analogue of `0.1×10`), magnitude robustness, NaN/±Inf
## propagation, finite product overflow → ±Inf (never NaN), `assumeFinite`
## bit-identical to the guarded path on finite non-overflowing input. The
## exact-`Fraction` forward-bound tier lives in `py/tests/test_oracle.py`.
import std/unittest
import std/math
import UniAccurate

func maxFin(T: typedesc): T =
  ## Largest finite value of `T` (`high` returns Inf for floats in Nim).
  when T is float64: cast[float64](0x7FEF_FFFF_FFFF_FFFF'u64)
  else: cast[float32](0x7F7F_FFFF'u32)

func withinRel[T](a, refn: T; tol: T): bool =
  ## `a` agrees with `refn` to a relative tolerance — the structural proxy for
  ## the published forward bound (the exact bound check is the oracle tier). An
  ## exact match short-circuits first, so an all-zero dot (`0 == 0`) passes.
  if a == refn:
    return true
  if classify(a) in {fcNan, fcInf, fcNegInf} or
      classify(refn) in {fcNan, fcInf, fcNegInf}:
    return false
  abs(a - refn) <= tol * max(abs(a), abs(refn))

template dotSuite(T: typedesc) =
  suite "dot product [" & $T & "]":

    test "naiveDot: empty arrays dot to zero":
      let x: seq[T] = @[]
      let y: seq[T] = @[]
      check naiveDot(x, y) == T(0.0)

    test "naiveDot: single pair is the product":
      check naiveDot([T(3.0)], [T(4.0)]) == T(12.0)

    test "naiveDot: integer data is exact":
      let x = [T(1.0), T(2.0), T(3.0), T(4.0)]
      let y = [T(5.0), T(6.0), T(7.0), T(8.0)]
      check naiveDot(x, y) == T(70.0) # 5 + 12 + 21 + 32

    test "naiveDot: NaN propagates (does not raise)":
      check classify(naiveDot([T(1.0), T(NaN)], [T(2.0), T(1.0)])) == fcNan

    test "naiveDot: Inf operand propagates":
      check classify(naiveDot([T(1.0), T(Inf)], [T(2.0), T(1.0)])) == fcInf

    test "naiveDot: finite product overflow yields Inf, never NaN":
      let m = maxFin(T)
      # Same-sign overflow: m·m → +Inf, summed with another +Inf → +Inf.
      check classify(naiveDot([m, m], [m, m])) == fcInf

    test "naiveDot: finite opposite-sign overflow recovered (no NaN)":
      # m·m → +Inf and m·(-m) → -Inf; a naive order can yield +Inf + -Inf = NaN.
      # The all-finite guard recovers the true value via superDot (finite or a
      # correctly-signed ±Inf), so finite input never yields NaN.
      let m = maxFin(T)
      let z = naiveDot([m, m], [m, -m])
      check classify(z) != fcNan

    test "dot2: empty and single":
      let e: seq[T] = @[]
      check dot2(e, e) == T(0.0)
      check dot2([T(3.0)], [T(4.0)]) == T(12.0)

    test "dot2: integer data is exact":
      let x = [T(1.0), T(2.0), T(3.0), T(4.0)]
      let y = [T(5.0), T(6.0), T(7.0), T(8.0)]
      check dot2(x, y) == T(70.0)

    test "dot2: recovers cancellation lost by naiveDot":
      # The dot analogue of the 0.1×10 win: products that cancel to a small
      # result, where naive accumulation drops the low order. dot2 keeps the
      # product and sum errors at twice precision, so it tracks superDot.
      let x = [T(1e20), T(1.0), T(-1e20)]
      let y = [T(1.0), T(1.0), T(1.0)]
      check dot2(x, y) == T(1.0)
      check naiveDot(x, y) != T(1.0) # naive loses the 1.0 beside the 1e20

    test "dot2: == dotK(x, y, 2) bit-for-bit":
      let x = [T(0.1), T(0.2), T(0.3), T(1e20), T(-1e20)]
      let y = [T(0.3), T(0.2), T(0.1), T(1.0), T(1.0)]
      check dot2(x, y) == dotK(x, y, 2)

    test "dot2: NaN/Inf fall back to naiveDot (propagate)":
      check classify(dot2([T(1.0), T(NaN)], [T(2.0), T(1.0)])) == fcNan
      check classify(dot2([T(1.0), T(Inf)], [T(2.0), T(1.0)])) == fcInf

    test "dot2: finite overflow yields Inf, never NaN":
      let m = maxFin(T)
      check classify(dot2([m, m], [m, m])) == fcInf
      check classify(dot2([m, m], [m, -m])) != fcNan

    test "dot2: accurate vs the exact dot on mixed data":
      # A structural proxy for the published bound |res - xᵀy| <=
      # 2u|xᵀy| + 2γ²_{n+1}(2u)·Σ|xᵢyᵢ|: dot2 tracks the exact dot to within a
      # loose relative tolerance, where naiveDot drifts. The exact-`Fraction`
      # bound is the oracle tier (`py/tests/test_oracle.py`).
      let x = [T(1.0), T(1e8), T(1.0), T(-1e8), T(1.0), T(1e8), T(-1e8)]
      let y = [T(1.0), T(1.0), T(1.0), T(1.0), T(1.0), T(1.0), T(1.0)]
      let refn = superDot(x, y)
      check withinRel(dot2(x, y), refn, T(1e-6))
      # dot2 is no farther from the exact dot than naiveDot (its bound is below
      # naiveDot's on the same data): compare the absolute errors directly.
      check abs(dot2(x, y) - refn) <= abs(naiveDot(x, y) - refn)

    test "dotK: K=1 is naiveDot (integer-exact)":
      let x = [T(1.0), T(2.0), T(3.0)]
      let y = [T(4.0), T(5.0), T(6.0)]
      check dotK(x, y, 1) == naiveDot(x, y)
      check dotK(x, y, 1) == T(32.0)

    test "dotK: K<1 is treated as 1":
      let x = [T(1.0), T(2.0)]
      let y = [T(3.0), T(4.0)]
      check dotK(x, y, 0) == dotK(x, y, 1)

    test "dotK: K=3 is at least as accurate as K=2":
      let x = [T(1e20), T(1.0), T(-1e20)]
      let y = [T(1.0), T(1.0), T(1.0)]
      check dotK(x, y, 3) == T(1.0)

    test "dotK: empty and single":
      let e: seq[T] = @[]
      check dotK(e, e, 3) == T(0.0)
      check dotK([T(7.0)], [T(6.0)], 3) == T(42.0)

    test "dotK: K>3 runtime cascade matches K=3 on integer data":
      # The runtime-K path (K > 3) is semantically identical to the
      # monomorphized K=3 path; on integer data both hit the exact integer.
      let x = [T(1.0), T(2.0), T(3.0), T(4.0), T(5.0)]
      let y = [T(6.0), T(7.0), T(8.0), T(9.0), T(10.0)]
      check dotK(x, y, 5) == dotK(x, y, 3)
      check dotK(x, y, 5) == T(130.0) # 6 + 14 + 24 + 36 + 50

    test "dotK: NaN propagates":
      check classify(dotK([T(1.0), T(NaN)], [T(2.0), T(1.0)], 2)) == fcNan

    test "dotK: finite overflow yields Inf, never NaN":
      let m = maxFin(T)
      check classify(dotK([m, m], [m, m], 2)) == fcInf
      check classify(dotK([m, m], [m, -m], 3)) != fcNan

    test "assumeFinite: bit-identical to the guarded path on finite input":
      # The opt-in strips the per-element isFin guards; on finite
      # non-overflowing input it must be bit-identical to the default path.
      let x = [T(0.1), T(0.2), T(0.3), T(1e8), T(-1e8), T(1.0)]
      let y = [T(0.3), T(0.2), T(0.1), T(1.0), T(1.0), T(1.0)]
      check dot2(x, y, assumeFinite = true) == dot2(x, y)
      check dotK(x, y, 3, assumeFinite = true) == dotK(x, y, 3)

dotSuite(float64)
dotSuite(float32)

suite "dot product contracts (postconditions)":
  test "postconditions hold (no raise in debug, compiled away in release)":
    let x = [1.0, 2.0, 3.0]
    let y = [4.0, 5.0, 6.0]
    check naiveDot(x, y) == 32.0
    check dot2(x, y) == 32.0
    check dotK(x, y, 2) == 32.0
    check dotK(x, y, 3) == 32.0
