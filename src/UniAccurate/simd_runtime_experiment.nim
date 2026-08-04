# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
## EXPERIMENTAL -- not imported by UniAccurate.nim, not wired into c_api.nim,
## not part of the shipped library or the Python wheel. A prototype for one
## question: can a single compiled binary contain both AVX2 and AVX-512
## dot-product kernels and pick between them at runtime via nimsimd's real
## CPUID check (`nimsimd/runtimecheck.checkInstructionSets`), instead of the
## existing `-d:avx2`/`-d:avx512` picking exactly one ISA at compile time?
##
## Written and reviewed on arm64 (Apple Silicon), which cannot compile amd64
## SIMD intrinsics at all -- `when defined(amd64)` means none of this file's
## logic has been type-checked or run anywhere yet. Build and run it
## (`nimble simdRuntimeExperiment`) on real amd64 hardware, ideally
## AVX-512-capable, before deciding whether to fold this technique into
## simd.nim/c_api.nim. Expect at least one fix-and-retry round from real
## compiler output -- paste it back verbatim if it fails.
##
## Scope: only the naive FMA dot-reduce (the simplest of the three SIMD dot
## kernels, and the one with the largest measured AVX-512 win per
## ADR-0007: dot2_simd/naive_dot_simd, Zen4). If this technique works, the
## same target-attribute mechanism extends to simd.nim's other templates.
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

  func naiveDotScalar(x, y: openArray[float64]): float64 =
    result = 0.0
    for i in 0 ..< x.len: result += x[i] * y[i]

  func naiveDotAvx2(x, y: openArray[float64]): float64
      {.codegenDecl: "__attribute__((target(\"avx2,fma\"))) $# $#$#".} =
    ## Compiled with the `target` attribute, not a global -mavx2/-mfma flag,
    ## so this coexists in the same translation unit as the AVX-512 variant
    ## below and the plain scalar one above.
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

  type DotKernel = proc(x, y: openArray[float64]): float64 {.nimcall.}

  var dispatched: DotKernel
  var dispatchedName: string

  proc initDispatch() =
    if checkInstructionSets({AVX512F}):
      dispatched = naiveDotAvx512
      dispatchedName = "avx512f+fma"
    elif checkInstructionSets({AVX2}):
      dispatched = naiveDotAvx2
      dispatchedName = "avx2+fma"
    else:
      dispatched = naiveDotScalar
      dispatchedName = "scalar"

  proc naiveDotRuntime*(x, y: openArray[float64]): float64 =
    ## The only "product-shaped" entry point in this prototype: dispatches
    ## once (cached), then reuses the decision on every call.
    if dispatched == nil: initDispatch()
    dispatched(x, y)

  when isMainModule:
    import std/[random, monotimes, times, strformat]

    initDispatch()
    echo "AVX2 detected:    ", checkInstructionSets({AVX2})
    echo "AVX512F detected: ", checkInstructionSets({AVX512F})
    echo "Dispatched to:    ", dispatchedName
    echo ""

    var rng = initRand(42)
    const n = 1_000_000
    var x = newSeq[float64](n)
    var y = newSeq[float64](n)
    for i in 0 ..< n:
      x[i] = rng.rand(2.0) - 1.0
      y[i] = rng.rand(2.0) - 1.0

    let rScalar = naiveDotScalar(x, y)
    let rAvx2 = naiveDotAvx2(x, y)
    let rAvx512 = naiveDotAvx512(x, y)
    echo "Correctness (all three should match closely -- naive dot is not"
    echo "correctly-rounded, so exact equality across reduction orders isn't"
    echo "expected, but the gap should be tiny relative to the values):"
    echo &"  scalar = {rScalar:.6f}"
    echo &"  avx2   = {rAvx2:.6f}  (|diff| = {abs(rAvx2 - rScalar):.3e})"
    echo &"  avx512 = {rAvx512:.6f}  (|diff| = {abs(rAvx512 - rScalar):.3e})"
    echo ""

    var sinkAcc: float64 = 0.0
    proc keep(v: float64) {.inline: false.} =
      # Non-inline sink forces every timed call's result to stay live, so
      # the release optimizer cannot prove `discard fn(x, y)` is dead code
      # and delete the whole loop -- confirmed on FreeBSD/Zen4 that it did
      # exactly that for the plain scalar loop (measured 0.0000 ns/elem,
      # which is not physically possible for a real O(n) computation).
      sinkAcc += v

    proc timeit(fn: DotKernel, reps: int): float =
      result = 1e18
      for _ in 1 .. reps:
        let t0 = getMonoTime()
        keep(fn(x, y))
        let dt = (getMonoTime() - t0).inNanoseconds.float
        if dt < result: result = dt
      result /= n.float

    echo "Timing (best of 20, ns/elem):"
    echo &"  scalar : {timeit(naiveDotScalar, 20):.4f}"
    echo &"  avx2   : {timeit(naiveDotAvx2, 20):.4f}"
    echo &"  avx512 : {timeit(naiveDotAvx512, 20):.4f}"
    echo &"  runtime dispatch (picked {dispatchedName}): ",
      &"{timeit(proc(x, y: openArray[float64]): float64 = naiveDotRuntime(x, y), 20):.4f}"
    echo "sink = ", sinkAcc # keeps every timed call live across the whole run

else:
  when isMainModule:
    echo "amd64 only -- nothing to test on this architecture."
