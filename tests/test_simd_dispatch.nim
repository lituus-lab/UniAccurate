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
## mathematically exact answer for these particular inputs). On this
## development machine (arm64) `dispatch*` delegates straight to the scalar
## body (no float64 NEON path), so every check here also holds bit-for-bit
## today -- see ADR-0008's Verification section for what still needs an
## amd64 hardware run.
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
