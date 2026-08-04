# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
## Direct tests for `simd_dispatch.nim`'s public API (ADR-0008): `c_api.nim`
## reaches these through `ua_dot_naive`/`ua_dot2`/`ua_dot_k`, exercised
## end-to-end by the existing `tests/c` suite, but this file targets
## `dispatchNaiveDot`/`dispatchDot2`/`dispatchDotK3` by name so a regression
## in the dispatch module itself (not just its `c_api.nim` wrapping) shows up
## here directly.
##
## Expected values, not bit-for-bit comparisons against `naiveDot`/`dot2`/
## `dotK`: on amd64 the AVX2/AVX-512 kernels sum in a different lane order
## than the scalar body, so a cancellation-recovering case can legitimately
## differ in its last bit from the scalar implementation while still meeting
## the algorithm's own accuracy contract (dot2/dotK3 must still land on the
## mathematically exact answer for these particular inputs). On a non-amd64
## target `dispatch*` delegates straight to the scalar body (no float64 NEON
## path), so every check here also holds bit-for-bit there.
##
## `dispatchNaiveDot` alone does not guarantee no-NaN on opposite-sign
## overflow (`naiveDot`'s scalar body self-recovers via its own internal
## `classify`+`allFin`+`superDot` guard, dotproduct.nim, but the raw
## AVX2/AVX-512 kernel it delegates to on amd64 does not -- matching
## `naiveDotSimd`'s pre-existing, already-shipped contract, ADR-0005: no
## reliability flag for the naive kernel). That recovery is applied
## externally, by `ua_dot_naive` in `c_api.nim`; the test below verifies that
## composition, not a guarantee `dispatchNaiveDot` does not itself make.
## `dispatchDot2`/`dispatchDotK3` DO self-recover on this input, because their
## `isFin(r)` reliability check catches the NaN and falls back to the scalar
## `dot2`/`dotK`, which in turn fall back to `naiveDot`'s own recovery.
import std/unittest
import std/math
import UniAccurate/algorithms/exactsum
import UniAccurate/simd_dispatch
import UniAccurate/simd # LaneConcentrationFallback

proc next(r: var uint64): uint64 =
  ## Deterministic xorshift64 so every run is reproducible (no RNG state).
  var x = r
  x = x xor (x shl 13)
  x = x xor (x shr 7)
  x = x xor (x shl 17)
  r = x
  x

proc unit(r: var uint64): float64 =
  ## Uniform in [0, 1).
  float64(next(r) shr 11) * (1.0 / 9007199254740992.0)

suite "simd_dispatch: naiveDot":
  test "empty arrays dot to zero":
    let e: seq[float64] = @[]
    check dispatchNaiveDot(e, e) == 0.0

  test "single pair is the product":
    check dispatchNaiveDot([3.0], [4.0]) == 12.0

  test "integer data is exact":
    let x = [1.0, 2.0, 3.0, 4.0]
    let y = [5.0, 6.0, 7.0, 8.0]
    check dispatchNaiveDot(x, y) == 70.0

  test "NaN operand propagates":
    check classify(dispatchNaiveDot([1.0, NaN], [2.0, 1.0])) == fcNan

  test "Inf operand propagates":
    check classify(dispatchNaiveDot([1.0, Inf], [2.0, 1.0])) == fcInf

  test "finite opposite-sign overflow: the c_api.nim recovery composition works":
    # dispatchNaiveDot alone may yield NaN here (see the file header); this
    # reproduces the exact classify+allFin+superDot wrap ua_dot_naive applies
    # in c_api.nim, verifying what a real caller actually gets.
    let m = cast[float64](0x7FEF_FFFF_FFFF_FFFF'u64) # largest finite float64
    let x = [m, m]
    let y = [m, -m]
    let r = dispatchNaiveDot(x, y)
    let recovered = if classify(r) == fcNan: superDot(x, y) else: r
    # m*m and m*(-m) are exact negatives of each other regardless of
    # magnitude, so the exact real dot product -- what superDot recovers --
    # is precisely 0.0, not just "some finite value".
    check recovered == 0.0

suite "simd_dispatch: dot2":
  test "empty and single":
    let e: seq[float64] = @[]
    check dispatchDot2(e, e) == 0.0
    check dispatchDot2([3.0], [4.0]) == 12.0

  test "integer data is exact":
    let x = [1.0, 2.0, 3.0, 4.0]
    let y = [5.0, 6.0, 7.0, 8.0]
    check dispatchDot2(x, y) == 70.0

  test "recovers cancellation lost by naiveDot":
    let x = [1e20, 1.0, -1e20]
    let y = [1.0, 1.0, 1.0]
    check dispatchDot2(x, y) == 1.0
    check dispatchNaiveDot(x, y) != 1.0

  test "NaN/Inf propagate":
    check classify(dispatchDot2([1.0, NaN], [2.0, 1.0])) == fcNan
    check classify(dispatchDot2([1.0, Inf], [2.0, 1.0])) == fcInf

  test "finite opposite-sign overflow recovered (no NaN)":
    let m = cast[float64](0x7FEF_FFFF_FFFF_FFFF'u64)
    check classify(dispatchDot2([m, m], [m, -m])) != fcNan

suite "simd_dispatch: dotK3":
  test "empty and single":
    let e: seq[float64] = @[]
    check dispatchDotK3(e, e) == 0.0
    check dispatchDotK3([7.0], [6.0]) == 42.0

  test "at least as accurate as dot2 on the same cancellation case":
    let x = [1e20, 1.0, -1e20]
    let y = [1.0, 1.0, 1.0]
    check dispatchDotK3(x, y) == 1.0

  test "NaN/Inf propagate":
    check classify(dispatchDotK3([1.0, NaN], [2.0, 1.0])) == fcNan
    check classify(dispatchDotK3([1.0, Inf], [2.0, 1.0])) == fcInf

suite "simd_dispatch: past the vector width":
  ## The suites above top out at n = 4, which never reaches the strided loop of
  ## an 8-lane AVX-512 kernel -- only its zero-padded tail step. These cases run
  ## on both sides of both widths (L = 4 AVX2, L = 8 AVX-512).

  test "integer data is exact for every n across both vector widths":
    for n in 0 .. 40:
      var x, y: seq[float64]
      for i in 0 ..< n:
        x.add float64((i mod 7) - 3)
        y.add float64((i mod 5) - 2)
      let want = superDot(x, y)
      check dispatchNaiveDot(x, y) == want
      check dispatchDot2(x, y) == want
      check dispatchDotK3(x, y) == want

  test "cancellation past the vector width stays within twice precision":
    # Near-cancelling pairs: sum|xi*yi| >> |result|, so the lane running sums
    # are far larger than the result. A lane merge that rounds its own
    # compensation away (`sl[j] + el[j]` before the merge, rather than merging
    # the 2L parts) costs ~eps*max|sl| here -- orders past the twice-precision
    # bound, and low enough in lane concentration that the reliability check
    # still reads "reliable" and never falls back. ADR-0008.
    const eps = 2.220446049250313e-16
    var r: uint64 = 0x2545F4914F6CDD1D'u64
    for n in [8, 16, 64, 500]:
      var x, y: seq[float64]
      for _ in 0 ..< n div 2:
        let a = 1.0 + unit(r)
        x.add a
        x.add -a * (0.99 + 0.02 * unit(r))
        y.add 0.5 + unit(r)
        y.add 0.5 + unit(r)
      let want = superDot(x, y)
      check abs(dispatchDot2(x, y) - want) <= 4.0 * eps * abs(want)
      check abs(dispatchDotK3(x, y) - want) <= 4.0 * eps * abs(want)

  test "lane concentration past the vector width forces the scalar fallback":
    # Alternating 1e16/-1e16 products: each lane always lands the same sign
    # (period 2 divides both L = 4 and L = 8), so every per-lane running sum
    # grows to ~1e16 while the terms cancel exactly pair-by-pair (a - a == 0
    # is always exact in IEEE754), leaving a small exact residual. maxLane
    # (~1e16) blows past `LaneConcentrationFallback * abs(r)` (1024 * 3) by
    # ~13 orders of magnitude, so `reliable = false` here -- unlike the milder
    # case above, this actually exercises the scalar dot2/dotK3 recovery
    # branch, not just the reliability check reading "reliable". ADR-0008.
    var x, y: seq[float64]
    for _ in 0 ..< 10:
      x.add 1e16
      y.add 1.0
      x.add -1e16
      y.add 1.0
    x.add 3.0
    y.add 1.0
    let want = superDot(x, y)
    check want == 3.0
    # Independently confirm the fallback precondition rather than trusting
    # the comment: for both AVX2 (L = 4) and AVX-512 (L = 8), lane j
    # accumulates indices i ≡ j mod L; since L is even and the data
    # alternates sign with period 2, every lane sees only same-sign terms
    # and its running sum grows to ~1e16, well past
    # LaneConcentrationFallback * |want|.
    for L in [4, 8]:
      var lane = newSeq[float64](L)
      for i in 0 ..< x.len: lane[i mod L] += x[i] * y[i]
      var maxLane = 0.0
      for v in lane: maxLane = max(maxLane, abs(v))
      check maxLane > LaneConcentrationFallback * abs(want)
    check dispatchDot2(x, y) == want
    check dispatchDotK3(x, y) == want
