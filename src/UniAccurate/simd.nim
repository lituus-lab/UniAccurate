# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
## SIMD summation kernels, gated by `-d:simd`.
##
## Vectorized naive and compensated sums over the ISAs nimsimd exposes:
## AVX-512 and AVX2 (float64) on amd64, NEON (float32) on arm64. The kernels
## keep L per-lane running sums (one vector accumulator), then scalar-reduce
## the lanes: `storev` to a stack array, no horizontal-add intrinsic. The
## compensated kernels merge the L lane partials with the scalar compensated
## recurrence (`kahanSum` / `neumaierSum` / `kleinSum`) — a SIMD-block,
## scalar-merge scheme.
##
## Reliability: a compensated SIMD result is `reliable` when the lane partials
## are not concentrated relative to the result, `maxLane <=
## LaneConcentrationFallback * abs(r)`; the caller falls back to the scalar
## algorithm otherwise (bit-identical to the no-`-d:simd` build). Naive and
## pairwise carry no reliability flag: the naive forward-error bound holds
## regardless of lane concentration.
##
## FMA is never used in the compensated recurrence (ADR-0005); v1 has no dot
## product, so the layer uses no FMA at all. nimsimd is imported inside the ISA
## branches only, so a scalar build (no `-d:simd`) never pulls it.
when defined(simd):
  import ./algorithms/naivesum
  import ./algorithms/pairwisesum
  import ./algorithms/compensatedsum
  when defined(avx512) or defined(avx2):
    import std/math
    import ./twosum

  const LaneConcentrationFallback* = 1024.0
    ## Lane-concentration threshold for the compensated reliability check.

  const
    simdF64Enabled* = defined(avx2) or defined(avx512)
      ## float64 SIMD is available (AVX2 or AVX-512); NEON has no float64 path.
    simdF32Enabled* = simdF64Enabled or defined(arm64)
      ## float32 SIMD is available (float64 ISAs or arm64 NEON).

  template defineNaiveV(name, zero, loadv, addv, storev, L: untyped) =
    func name(x: openArray[float64]): float64 =
      if x.len == 0: return 0.0
      var acc = zero
      var i = 0
      let n = x.len
      while i + L <= n:
        acc = addv(acc, loadv(cast[pointer](unsafeAddr x[i])))
        i += L
      var lanes: array[L, float64]
      storev(cast[pointer](addr lanes[0]), acc)
      result = 0.0
      for j in 0 ..< L: result += lanes[j]
      while i < n: result += x[i]; inc i

  template defineCompV(name, zero, loadv, addv, storev, L: untyped,
                      merge: proc(v: openArray[float64]): float64) =
    func name(x: openArray[float64]): (float64, bool) =
      if x.len == 0: return (0.0, true)
      var acc = zero
      var i = 0
      let n = x.len
      while i + L <= n:
        acc = addv(acc, loadv(cast[pointer](unsafeAddr x[i])))
        i += L
      var lanes: array[L, float64]
      storev(cast[pointer](addr lanes[0]), acc)
      var tail = 0.0
      while i < n: tail += x[i]; inc i
      var vals: array[L + 1, float64]
      for j in 0 ..< L: vals[j] = lanes[j]
      vals[L] = tail
      let r = merge(vals.toOpenArray(0, L))
      var maxLane = 0.0
      for v in vals: maxLane = max(maxLane, abs(v))
      (r, isFin(r) and maxLane <= LaneConcentrationFallback * abs(r))

  when defined(avx512):
    import nimsimd/avx512/f

    defineNaiveV(naiveSumSimdAvx512, mm512_setzero_pd(), mm512_loadu_pd,
                 mm512_add_pd, mm512_storeu_pd, 8)
    defineCompV(kahanSumSimdAvx512, mm512_setzero_pd(), mm512_loadu_pd,
                mm512_add_pd, mm512_storeu_pd, 8,
                proc(v: openArray[float64]): float64 = kahanSum(v))
    defineCompV(neumaierSumSimdAvx512, mm512_setzero_pd(), mm512_loadu_pd,
                mm512_add_pd, mm512_storeu_pd, 8,
                proc(v: openArray[float64]): float64 = neumaierSum(v))
    defineCompV(kleinSumSimdAvx512, mm512_setzero_pd(), mm512_loadu_pd,
                mm512_add_pd, mm512_storeu_pd, 8,
                proc(v: openArray[float64]): float64 = kleinSum(v))

  elif defined(avx2):
    import nimsimd/avx2

    defineNaiveV(naiveSumSimdAvx2, mm256_setzero_pd(), mm256_loadu_pd,
                 mm256_add_pd, mm256_storeu_pd, 4)
    defineCompV(kahanSumSimdAvx2, mm256_setzero_pd(), mm256_loadu_pd,
                mm256_add_pd, mm256_storeu_pd, 4,
                proc(v: openArray[float64]): float64 = kahanSum(v))
    defineCompV(neumaierSumSimdAvx2, mm256_setzero_pd(), mm256_loadu_pd,
                mm256_add_pd, mm256_storeu_pd, 4,
                proc(v: openArray[float64]): float64 = neumaierSum(v))
    defineCompV(kleinSumSimdAvx2, mm256_setzero_pd(), mm256_loadu_pd,
                mm256_add_pd, mm256_storeu_pd, 4,
                proc(v: openArray[float64]): float64 = kleinSum(v))

  when defined(arm64):
    import nimsimd/neon

    func naiveSumSimdNeonF32(x: openArray[float32]): float32 =
      if x.len == 0: return 0.0'f32
      const L = 4
      var acc = vmovq_n_f32(0.0'f32)
      var i = 0
      let n = x.len
      while i + L <= n:
        acc = vaddq_f32(acc, vld1q_f32(cast[pointer](unsafeAddr x[i])))
        i += L
      var lanes: array[L, float32]
      vst1q_f32(cast[pointer](addr lanes[0]), acc)
      result = 0.0'f32
      for j in 0 ..< L: result += lanes[j]
      while i < n: result += x[i]; inc i

  func naiveSumSimd*[T: SomeFloat](x: openArray[T]): T =
    when T is float64 and defined(avx512):
      naiveSumSimdAvx512(x)
    elif T is float64 and defined(avx2):
      naiveSumSimdAvx2(x)
    elif T is float32 and defined(arm64):
      naiveSumSimdNeonF32(x)
    else:
      naiveSum(x)

  func pairwiseSimdRec[T: SomeFloat](x: openArray[T], lo, hi: int): T =
    let n = hi - lo + 1
    if n <= PairwiseThreshold:
      return naiveSumSimd(toOpenArray(x, lo, hi))
    let m = lo + n div 2
    result = pairwiseSimdRec(x, lo, m - 1) + pairwiseSimdRec(x, m, hi)

  func pairwiseSumSimd*[T: SomeFloat](x: openArray[T]): T =
    when (T is float64 and (defined(avx2) or defined(avx512))) or
         (T is float32 and defined(arm64)):
      if x.len == 0: return T(0)
      pairwiseSimdRec(x, 0, x.high)
    else:
      pairwiseSum(x)

  func kahanSumSimd*[T: SomeFloat](x: openArray[T]): (T, bool) =
    when T is float64 and defined(avx512):
      kahanSumSimdAvx512(x)
    elif T is float64 and defined(avx2):
      kahanSumSimdAvx2(x)
    else:
      (kahanSum(x), true)

  func neumaierSumSimd*[T: SomeFloat](x: openArray[T]): (T, bool) =
    when T is float64 and defined(avx512):
      neumaierSumSimdAvx512(x)
    elif T is float64 and defined(avx2):
      neumaierSumSimdAvx2(x)
    else:
      (neumaierSum(x), true)

  func kleinSumSimd*[T: SomeFloat](x: openArray[T]): (T, bool) =
    when T is float64 and defined(avx512):
      kleinSumSimdAvx512(x)
    elif T is float64 and defined(avx2):
      kleinSumSimdAvx2(x)
    else:
      (kleinSum(x), true)
