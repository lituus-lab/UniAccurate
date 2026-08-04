# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
## EXPERIMENTAL -- not imported by UniAccurate.nim, not wired into c_api.nim,
## not part of the shipped library or the Python wheel. A prototype for one
## question: can a single compiled binary contain both AVX2 and AVX-512
## dot-product kernels and pick between them at runtime via nimsimd's real
## CPUID check (`nimsimd/runtimecheck.checkInstructionSets`), instead of the
## existing `-d:avx2`/`-d:avx512` picking exactly one ISA at compile time?
##
## Validated on FreeBSD/Zen4 (real AVX-512 hardware) for `naiveDot`: compiles,
## correct (diffs ~1e-12, expected for reordered summation), dispatches to
## AVX-512 via real CPUID, negligible dispatch overhead (~0.4% over a direct
## call). Measured: scalar 0.5864 ns/elem, AVX2 0.2042 (2.87x), AVX-512
## 0.1869 (3.14x). This revision extends the same technique to `dot2`/`dotK3`
## (K=3) -- untested past `nim check`-level review on arm64, which cannot
## compile or type-check amd64 SIMD intrinsics at all. Re-run
## `nimble simdRuntimeExperiment` on the AVX-512 machine and paste the output
## (or the compiler error, if it fails to build).
##
## `dot2Avx2`/`dot2Avx512`/`dotK3Avx2`/`dotK3Avx512` mirror the exact EFT
## recurrence from `simd.nim`'s `defineDot2V`/`defineDotK3V` templates
## (twoSum-based product/sum error tracking, zero-padded tail step), but as
## hand-written, individually `target`-attributed procs rather than through
## the shared templates -- the templates don't parametrize a per-function
## codegen attribute, and duplicating the (short, already-proven) recurrence
## by hand is lower-risk than modifying the templates blind. The scalar
## baselines reuse the library's own `dot2`/`dotK` directly (proven, tested
## code) rather than reimplementing them.
##
## Known gap: nimsimd's `InstructionSet` enum has no separate FMA entry, so
## `checkInstructionSets({AVX512F})`/`{AVX2}` are used as a proxy for
## "AVX-512F+FMA"/"AVX2+FMA" -- true on every shipping CPU with those ISAs
## to date, but not a CPUID-verified guarantee of FMA specifically.
when defined(amd64):
  import nimsimd/avx2
  import nimsimd/fma
  import nimsimd/avx512/f
  import nimsimd/runtimecheck
  import ./algorithms/dotproduct
  import ./algorithms/compensatedsum

  # --------------------------------------------------------------------
  # naiveDot -- validated on real AVX-512 hardware (see header).
  # --------------------------------------------------------------------

  func naiveDotScalar(x, y: openArray[float64]): float64 =
    result = 0.0
    for i in 0 ..< x.len: result += x[i] * y[i]

  func naiveDotAvx2(x, y: openArray[float64]): float64
      {.codegenDecl: "__attribute__((target(\"avx2,fma\"))) $# $#$#".} =
    if x.len == 0: return 0.0
    var acc = mm256_setzero_pd()
    let n = x.len
    var i = 0
    while i + 4 <= n:
      acc = mm256_fmadd_pd(mm256_loadu_pd(cast[pointer](unsafeAddr x[i])),
                            mm256_loadu_pd(cast[pointer](unsafeAddr y[i])), acc)
      i += 4
    var lanes: array[4, float64]
    mm256_storeu_pd(cast[pointer](addr lanes[0]), acc)
    result = 0.0
    for j in 0 ..< 4: result += lanes[j]
    while i < n: result += x[i] * y[i]; inc i

  func naiveDotAvx512(x, y: openArray[float64]): float64
      {.codegenDecl: "__attribute__((target(\"avx512f,fma\"))) $# $#$#".} =
    if x.len == 0: return 0.0
    var acc = mm512_setzero_pd()
    let n = x.len
    var i = 0
    while i + 8 <= n:
      acc = mm512_fmadd_pd(mm512_loadu_pd(cast[pointer](unsafeAddr x[i])),
                            mm512_loadu_pd(cast[pointer](unsafeAddr y[i])), acc)
      i += 8
    var lanes: array[8, float64]
    mm512_storeu_pd(cast[pointer](addr lanes[0]), acc)
    result = 0.0
    for j in 0 ..< 8: result += lanes[j]
    while i < n: result += x[i] * y[i]; inc i

  # --------------------------------------------------------------------
  # dot2 (ORO Alg 5.3, K=2) -- NOT yet validated on real hardware.
  # --------------------------------------------------------------------

  func dot2Scalar(x, y: openArray[float64]): float64 =
    ## Default (assumeFinite=false): 4-5 isFin guards per element. This is
    ## what a typical caller experiences today -- NOT a fair SIMD-only
    ## comparison, since the AVX2/AVX-512 kernels below have no per-element
    ## guards at all (matching the shipped dot2Simd's own no-guard
    ## recurrence). See dot2ScalarUnguarded for the apples-to-apples one.
    dot2(x, y)

  func dot2ScalarUnguarded(x, y: openArray[float64]): float64 =
    ## assumeFinite=true: same unguarded EFT recurrence as the SIMD kernels
    ## below, scalar. The fair baseline for isolating the SIMD-width gain
    ## alone, separate from the (also real, already documented in
    ## ADR-0007 Lever 3) gain from skipping the isFin guards.
    dot2(x, y, assumeFinite = true)

  func dot2Avx2(x, y: openArray[float64]): float64
      {.codegenDecl: "__attribute__((target(\"avx2,fma\"))) $# $#$#".} =
    if x.len == 0: return 0.0
    let zero = mm256_setzero_pd()
    var s = zero
    var e = zero
    let n = x.len
    var i = 0
    while i + 4 <= n:
      let xv = mm256_loadu_pd(cast[pointer](unsafeAddr x[i]))
      let yv = mm256_loadu_pd(cast[pointer](unsafeAddr y[i]))
      let h = mm256_mul_pd(xv, yv)
      let r1 = mm256_fmadd_pd(xv, yv, mm256_sub_pd(zero, h))
      let s2 = mm256_add_pd(s, h)
      let z = mm256_sub_pd(s2, s)
      let r2 = mm256_add_pd(mm256_sub_pd(s, mm256_sub_pd(s2, z)), mm256_sub_pd(h, z))
      s = s2
      e = mm256_add_pd(mm256_add_pd(e, r1), r2)
      i += 4
    if i < n:
      var px, py: array[4, float64]
      var k = 0
      while i < n: px[k] = x[i]; py[k] = y[i]; inc i; inc k
      let xv = mm256_loadu_pd(cast[pointer](addr px[0]))
      let yv = mm256_loadu_pd(cast[pointer](addr py[0]))
      let h = mm256_mul_pd(xv, yv)
      let r1 = mm256_fmadd_pd(xv, yv, mm256_sub_pd(zero, h))
      let s2 = mm256_add_pd(s, h)
      let z = mm256_sub_pd(s2, s)
      let r2 = mm256_add_pd(mm256_sub_pd(s, mm256_sub_pd(s2, z)), mm256_sub_pd(h, z))
      s = s2
      e = mm256_add_pd(mm256_add_pd(e, r1), r2)
    var sl, el: array[4, float64]
    mm256_storeu_pd(cast[pointer](addr sl[0]), s)
    mm256_storeu_pd(cast[pointer](addr el[0]), e)
    var vals: array[4, float64]
    for j in 0 ..< 4: vals[j] = sl[j] + el[j]
    neumaierSum(vals)

  func dot2Avx512(x, y: openArray[float64]): float64
      {.codegenDecl: "__attribute__((target(\"avx512f,fma\"))) $# $#$#".} =
    if x.len == 0: return 0.0
    let zero = mm512_setzero_pd()
    var s = zero
    var e = zero
    let n = x.len
    var i = 0
    while i + 8 <= n:
      let xv = mm512_loadu_pd(cast[pointer](unsafeAddr x[i]))
      let yv = mm512_loadu_pd(cast[pointer](unsafeAddr y[i]))
      let h = mm512_mul_pd(xv, yv)
      let r1 = mm512_fmadd_pd(xv, yv, mm512_sub_pd(zero, h))
      let s2 = mm512_add_pd(s, h)
      let z = mm512_sub_pd(s2, s)
      let r2 = mm512_add_pd(mm512_sub_pd(s, mm512_sub_pd(s2, z)), mm512_sub_pd(h, z))
      s = s2
      e = mm512_add_pd(mm512_add_pd(e, r1), r2)
      i += 8
    if i < n:
      var px, py: array[8, float64]
      var k = 0
      while i < n: px[k] = x[i]; py[k] = y[i]; inc i; inc k
      let xv = mm512_loadu_pd(cast[pointer](addr px[0]))
      let yv = mm512_loadu_pd(cast[pointer](addr py[0]))
      let h = mm512_mul_pd(xv, yv)
      let r1 = mm512_fmadd_pd(xv, yv, mm512_sub_pd(zero, h))
      let s2 = mm512_add_pd(s, h)
      let z = mm512_sub_pd(s2, s)
      let r2 = mm512_add_pd(mm512_sub_pd(s, mm512_sub_pd(s2, z)), mm512_sub_pd(h, z))
      s = s2
      e = mm512_add_pd(mm512_add_pd(e, r1), r2)
    var sl, el: array[8, float64]
    mm512_storeu_pd(cast[pointer](addr sl[0]), s)
    mm512_storeu_pd(cast[pointer](addr el[0]), e)
    var vals: array[8, float64]
    for j in 0 ..< 8: vals[j] = sl[j] + el[j]
    neumaierSum(vals)

  # --------------------------------------------------------------------
  # dotK3 (ORO Alg 5.3, K=3) -- NOT yet validated on real hardware.
  # --------------------------------------------------------------------

  func dotK3Scalar(x, y: openArray[float64]): float64 =
    ## Default (assumeFinite=false) -- see dot2Scalar's comment, same caveat.
    dotK(x, y, 3)

  func dotK3ScalarUnguarded(x, y: openArray[float64]): float64 =
    ## assumeFinite=true -- see dot2ScalarUnguarded's comment, same reasoning.
    dotK(x, y, 3, assumeFinite = true)

  func dotK3Avx2(x, y: openArray[float64]): float64
      {.codegenDecl: "__attribute__((target(\"avx2,fma\"))) $# $#$#".} =
    if x.len == 0: return 0.0
    let zero = mm256_setzero_pd()
    var s = zero
    var es = zero
    var ec = zero
    let n = x.len
    var i = 0
    while i + 4 <= n:
      let xv = mm256_loadu_pd(cast[pointer](unsafeAddr x[i]))
      let yv = mm256_loadu_pd(cast[pointer](unsafeAddr y[i]))
      let h = mm256_mul_pd(xv, yv)
      let r1 = mm256_fmadd_pd(xv, yv, mm256_sub_pd(zero, h))
      let s2 = mm256_add_pd(s, h)
      let z = mm256_sub_pd(s2, s)
      let r2 = mm256_add_pd(mm256_sub_pd(s, mm256_sub_pd(s2, z)), mm256_sub_pd(h, z))
      s = s2
      let es2 = mm256_add_pd(es, r1)
      let zr1 = mm256_sub_pd(es2, es)
      let er1 = mm256_add_pd(mm256_sub_pd(es, mm256_sub_pd(es2, zr1)), mm256_sub_pd(r1, zr1))
      es = es2
      let es3 = mm256_add_pd(es, r2)
      let zr2 = mm256_sub_pd(es3, es)
      let er2 = mm256_add_pd(mm256_sub_pd(es, mm256_sub_pd(es3, zr2)), mm256_sub_pd(r2, zr2))
      es = es3
      ec = mm256_add_pd(ec, mm256_add_pd(er1, er2))
      i += 4
    if i < n:
      var px, py: array[4, float64]
      var k = 0
      while i < n: px[k] = x[i]; py[k] = y[i]; inc i; inc k
      let xv = mm256_loadu_pd(cast[pointer](addr px[0]))
      let yv = mm256_loadu_pd(cast[pointer](addr py[0]))
      let h = mm256_mul_pd(xv, yv)
      let r1 = mm256_fmadd_pd(xv, yv, mm256_sub_pd(zero, h))
      let s2 = mm256_add_pd(s, h)
      let z = mm256_sub_pd(s2, s)
      let r2 = mm256_add_pd(mm256_sub_pd(s, mm256_sub_pd(s2, z)), mm256_sub_pd(h, z))
      s = s2
      let es2 = mm256_add_pd(es, r1)
      let zr1 = mm256_sub_pd(es2, es)
      let er1 = mm256_add_pd(mm256_sub_pd(es, mm256_sub_pd(es2, zr1)), mm256_sub_pd(r1, zr1))
      es = es2
      let es3 = mm256_add_pd(es, r2)
      let zr2 = mm256_sub_pd(es3, es)
      let er2 = mm256_add_pd(mm256_sub_pd(es, mm256_sub_pd(es3, zr2)), mm256_sub_pd(r2, zr2))
      es = es3
      ec = mm256_add_pd(ec, mm256_add_pd(er1, er2))
    var sl, esl, ecl: array[4, float64]
    mm256_storeu_pd(cast[pointer](addr sl[0]), s)
    mm256_storeu_pd(cast[pointer](addr esl[0]), es)
    mm256_storeu_pd(cast[pointer](addr ecl[0]), ec)
    var vals: array[4, float64]
    for j in 0 ..< 4: vals[j] = sl[j] + esl[j] + ecl[j]
    neumaierSum(vals)

  func dotK3Avx512(x, y: openArray[float64]): float64
      {.codegenDecl: "__attribute__((target(\"avx512f,fma\"))) $# $#$#".} =
    if x.len == 0: return 0.0
    let zero = mm512_setzero_pd()
    var s = zero
    var es = zero
    var ec = zero
    let n = x.len
    var i = 0
    while i + 8 <= n:
      let xv = mm512_loadu_pd(cast[pointer](unsafeAddr x[i]))
      let yv = mm512_loadu_pd(cast[pointer](unsafeAddr y[i]))
      let h = mm512_mul_pd(xv, yv)
      let r1 = mm512_fmadd_pd(xv, yv, mm512_sub_pd(zero, h))
      let s2 = mm512_add_pd(s, h)
      let z = mm512_sub_pd(s2, s)
      let r2 = mm512_add_pd(mm512_sub_pd(s, mm512_sub_pd(s2, z)), mm512_sub_pd(h, z))
      s = s2
      let es2 = mm512_add_pd(es, r1)
      let zr1 = mm512_sub_pd(es2, es)
      let er1 = mm512_add_pd(mm512_sub_pd(es, mm512_sub_pd(es2, zr1)), mm512_sub_pd(r1, zr1))
      es = es2
      let es3 = mm512_add_pd(es, r2)
      let zr2 = mm512_sub_pd(es3, es)
      let er2 = mm512_add_pd(mm512_sub_pd(es, mm512_sub_pd(es3, zr2)), mm512_sub_pd(r2, zr2))
      es = es3
      ec = mm512_add_pd(ec, mm512_add_pd(er1, er2))
      i += 8
    if i < n:
      var px, py: array[8, float64]
      var k = 0
      while i < n: px[k] = x[i]; py[k] = y[i]; inc i; inc k
      let xv = mm512_loadu_pd(cast[pointer](addr px[0]))
      let yv = mm512_loadu_pd(cast[pointer](addr py[0]))
      let h = mm512_mul_pd(xv, yv)
      let r1 = mm512_fmadd_pd(xv, yv, mm512_sub_pd(zero, h))
      let s2 = mm512_add_pd(s, h)
      let z = mm512_sub_pd(s2, s)
      let r2 = mm512_add_pd(mm512_sub_pd(s, mm512_sub_pd(s2, z)), mm512_sub_pd(h, z))
      s = s2
      let es2 = mm512_add_pd(es, r1)
      let zr1 = mm512_sub_pd(es2, es)
      let er1 = mm512_add_pd(mm512_sub_pd(es, mm512_sub_pd(es2, zr1)), mm512_sub_pd(r1, zr1))
      es = es2
      let es3 = mm512_add_pd(es, r2)
      let zr2 = mm512_sub_pd(es3, es)
      let er2 = mm512_add_pd(mm512_sub_pd(es, mm512_sub_pd(es3, zr2)), mm512_sub_pd(r2, zr2))
      es = es3
      ec = mm512_add_pd(ec, mm512_add_pd(er1, er2))
    var sl, esl, ecl: array[8, float64]
    mm512_storeu_pd(cast[pointer](addr sl[0]), s)
    mm512_storeu_pd(cast[pointer](addr esl[0]), es)
    mm512_storeu_pd(cast[pointer](addr ecl[0]), ec)
    var vals: array[8, float64]
    for j in 0 ..< 8: vals[j] = sl[j] + esl[j] + ecl[j]
    neumaierSum(vals)

  # --------------------------------------------------------------------
  # Runtime dispatch, once per kernel family, cached.
  # --------------------------------------------------------------------

  type DotKernel = proc(x, y: openArray[float64]): float64 {.nimcall.}

  var isaName: string
  var naiveDotImpl, dot2Impl, dotK3Impl: DotKernel

  proc initDispatch() =
    if checkInstructionSets({AVX512F}):
      isaName = "avx512f+fma"
      naiveDotImpl = naiveDotAvx512
      dot2Impl = dot2Avx512
      dotK3Impl = dotK3Avx512
    elif checkInstructionSets({AVX2}):
      isaName = "avx2+fma"
      naiveDotImpl = naiveDotAvx2
      dot2Impl = dot2Avx2
      dotK3Impl = dotK3Avx2
    else:
      isaName = "scalar"
      naiveDotImpl = naiveDotScalar
      dot2Impl = dot2Scalar
      dotK3Impl = dotK3Scalar

  proc naiveDotRuntime*(x, y: openArray[float64]): float64 =
    if naiveDotImpl == nil: initDispatch()
    naiveDotImpl(x, y)

  proc dot2Runtime*(x, y: openArray[float64]): float64 =
    if dot2Impl == nil: initDispatch()
    dot2Impl(x, y)

  proc dotK3Runtime*(x, y: openArray[float64]): float64 =
    if dotK3Impl == nil: initDispatch()
    dotK3Impl(x, y)

  when isMainModule:
    import std/[random, monotimes, times, strformat]

    initDispatch()
    echo "AVX2 detected:    ", checkInstructionSets({AVX2})
    echo "AVX512F detected: ", checkInstructionSets({AVX512F})
    echo "Dispatched to:    ", isaName
    echo ""

    var rng = initRand(42)
    const n = 1_000_000
    var x = newSeq[float64](n)
    var y = newSeq[float64](n)
    for i in 0 ..< n:
      x[i] = rng.rand(2.0) - 1.0
      y[i] = rng.rand(2.0) - 1.0

    var sinkAcc: float64 = 0.0
    proc keep(v: float64) {.inline: false.} =
      # Non-inline sink -- without it the release optimizer proved an
      # unused scalar-loop result dead and deleted the whole loop on the
      # first version of this file (measured 0.0000 ns/elem on FreeBSD/
      # Zen4, not physically possible for a real O(n) computation).
      sinkAcc += v

    proc timeit(fn: DotKernel, reps: int): float =
      result = 1e18
      for _ in 1 .. reps:
        let t0 = getMonoTime()
        keep(fn(x, y))
        let dt = (getMonoTime() - t0).inNanoseconds.float
        if dt < result: result = dt
      result /= n.float

    proc checkAndTime(name: string, scalarFn, avx2Fn, avx512Fn, runtimeFn: DotKernel,
                       unguardedFn: DotKernel = nil) =
      let rScalar = scalarFn(x, y)
      let rAvx2 = avx2Fn(x, y)
      let rAvx512 = avx512Fn(x, y)
      echo name, ":"
      echo &"  scalar (default) = {rScalar:.6f}"
      echo &"  avx2             = {rAvx2:.6f}  (|diff| = {abs(rAvx2 - rScalar):.3e})"
      echo &"  avx512           = {rAvx512:.6f}  (|diff| = {abs(rAvx512 - rScalar):.3e})"

      if unguardedFn != nil:
        # Two different comparisons: default-guarded scalar (what a caller
        # gets today) vs SIMD conflates the SIMD-width gain with the
        # separate, already-documented (ADR-0007 Lever 3) gain from
        # skipping dot2/dotK's per-element isFin guards -- the SIMD kernels
        # have no such guards, matching the shipped dot2Simd/dotK3Simd.
        # unguardedFn (assumeFinite=true) isolates the SIMD-only effect.
        let tScalar = timeit(scalarFn, 20)
        let tUnguarded = timeit(unguardedFn, 20)
        let tAvx2 = timeit(avx2Fn, 20)
        let tAvx512 = timeit(avx512Fn, 20)
        let tDispatch = timeit(runtimeFn, 20)
        echo "  timing (best of 20, ns/elem):"
        echo &"    scalar, default (assumeFinite=false, guarded) : {tScalar:.4f}"
        echo &"    scalar, unguarded (assumeFinite=true)         : {tUnguarded:.4f}"
        echo &"    avx2                                          : {tAvx2:.4f}"
        echo &"    avx512                                        : {tAvx512:.4f}"
        echo &"    dispatch (picked {isaName})                    : {tDispatch:.4f}"
        echo &"    SIMD-only gain (unguarded scalar / avx512): {(tUnguarded / tAvx512):.2f}x"
        echo &"    guard-skip + SIMD gain (default scalar / avx512): {(tScalar / tAvx512):.2f}x"
      else:
        echo &"  timing (best of 20, ns/elem): scalar {timeit(scalarFn, 20):.4f}" &
          &" | avx2 {timeit(avx2Fn, 20):.4f} | avx512 {timeit(avx512Fn, 20):.4f}" &
          &" | dispatch({isaName}) {timeit(runtimeFn, 20):.4f}"
      echo ""

    checkAndTime("naiveDot", naiveDotScalar, naiveDotAvx2, naiveDotAvx512,
      proc(x, y: openArray[float64]): float64 = naiveDotRuntime(x, y))
    checkAndTime("dot2", dot2Scalar, dot2Avx2, dot2Avx512,
      proc(x, y: openArray[float64]): float64 = dot2Runtime(x, y),
      dot2ScalarUnguarded)
    checkAndTime("dotK3", dotK3Scalar, dotK3Avx2, dotK3Avx512,
      proc(x, y: openArray[float64]): float64 = dotK3Runtime(x, y),
      dotK3ScalarUnguarded)

    echo "sink = ", sinkAcc # keeps every timed call live across the whole run

else:
  when isMainModule:
    echo "amd64 only -- nothing to test on this architecture."
