# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
## SIMD summation and dot-product kernels, gated by `-d:simd`.
##
## Vectorized naive and compensated sums over the ISAs nimsimd exposes:
## AVX-512 and AVX2 (float64) on amd64, NEON (float32) on arm64. The sum kernels
## keep L per-lane running sums (one vector accumulator), then scalar-reduce
## the lanes: `storev` to a stack array, no horizontal-add intrinsic. The
## compensated sum kernels merge the L lane partials with the scalar compensated
## recurrence (`kahanSum` / `neumaierSum` / `kleinSum`) — a SIMD-block,
## scalar-merge scheme.
##
## The dot kernels (`naiveDotSimd` / `dot2Simd` / `dotK3Simd`) vectorize the ORO
## Dot2 / DotK recurrence per lane with FMA: each lane keeps a running product
## sum `s` and an error cascade (`e` for Dot2, `(es, ec)` for DotK3), advanced by
## `h = xᵢ·yᵢ`, `r₁ = fma(xᵢ, yᵢ, −h)` (the exact product error) and a vector
## `twoSum` for the sum error `r₂`. The L lane K-fold estimates are merged
## scalarly with `neumaierSum` (its `γ_K·Σ|xᵢyᵢ|` merge error is far below the
## K-fold bound, so a 2-fold merge suffices). The tail is a zero-padded final
## vector step so it stays K-fold without a scalar FMA. `naiveDotSimd` is an FMA
## dot reduce with a naive scalar merge (within the naive bound, no reliability
## flag).
##
## Reliability: a compensated SIMD result is `reliable` when the lane partials
## are not concentrated relative to the result, `maxLane <=
## LaneConcentrationFallback * abs(r)`; the caller falls back to the scalar
## algorithm otherwise (bit-identical to the no-`-d:simd` build). Naive sum and
## naive dot carry no reliability flag: their forward-error bound holds
## regardless of lane concentration.
##
## FMA is used only in the dot product (the product `xᵢ·yᵢ` and its error
## extraction) — never in the compensated sum recurrence (ADR-0004); the
## FMA-in-SIMD-dot piece ADR-0005 deferred lands here (ADR-0007 Lever 1). nimsimd is
## imported inside the ISA branches only, so a scalar build (no `-d:simd`) never
## pulls it.
when defined(simd):
  import contracts
  import std/math
  import ./twosum
  import ./algorithms/naivesum
  import ./algorithms/pairwisesum
  import ./algorithms/compensatedsum
  import ./algorithms/dotproduct

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

  template defineNaiveDotV(name, zero, loadv, fmaddv, storev, L: untyped) =
    ## K-lane strided FMA dot reduce; scalar naive merge of the K lane dots and
    ## the tail. FMA fuses `xᵢ·yᵢ` into the running lane sum (one rounding
    ## instead of two — more accurate than scalar naive dot, still within the
    ## naive bound). No reliability flag.
    func name(x, y: openArray[float64]): float64 =
      if x.len == 0: return 0.0
      var acc = zero
      let n = x.len
      var i = 0
      while i + L <= n:
        acc = fmaddv(loadv(cast[pointer](unsafeAddr x[i])),
                     loadv(cast[pointer](unsafeAddr y[i])), acc)
        i += L
      var lanes: array[L, float64]
      storev(cast[pointer](addr lanes[0]), acc)
      result = 0.0
      for j in 0 ..< L: result += lanes[j]
      while i < n: result += x[i] * y[i]; inc i

  template defineDot2V(name, zero, loadv, mulv, fmaddv, addv, subv, storev,
                       L: untyped) =
    ## K lanes of the Dot2 (ORO Alg. 5.3, K = 2) recurrence: per term
    ## `h = xᵢ·yᵢ`, `r₁ = fma(xᵢ, yᵢ, −h)`, `(s₂, r₂) = twoSum(s, h)`,
    ## `e = (e + r₁) + r₂`. Scalar `neumaierSum` merge of the K lane estimates
    ## `sₖ + eₖ`. The tail is a zero-padded final vector step so it stays
    ## twice-precision (zero lanes contribute nothing). Returns
    ## `(result, reliable)`: false on a non-finite result (→ scalar IEEE
    ## fallback) or when a lane running sum concentrates beyond
    ## `LaneConcentrationFallback · |result|` (cancellation data) — then the
    ## caller falls back to the scalar `dot2` body for a scalar-exact result.
    func name(x, y: openArray[float64]): (float64, bool) =
      if x.len == 0: return (0.0, true)
      var s = zero
      var e = zero
      let n = x.len
      var i = 0
      while i + L <= n:
        let xv = loadv(cast[pointer](unsafeAddr x[i]))
        let yv = loadv(cast[pointer](unsafeAddr y[i]))
        let h = mulv(xv, yv)
        let r1 = fmaddv(xv, yv, subv(zero, h)) # xᵢ·yᵢ − h (exact product error)
        let s2 = addv(s, h)
        let z = subv(s2, s)
        let r2 = addv(subv(s, subv(s2, z)), subv(h, z)) # twoSum(s, h) error
        s = s2
        e = addv(addv(e, r1), r2)
        i += L
      if i < n:
        # Zero-padded final step: tail in lanes 0 ..< r, zero elsewhere. Zero
        # lanes give h = r1 = r2 = 0 and leave s, e unchanged, so only the tail
        # lanes advance — twice precision without a scalar FMA.
        var px: array[L, float64]
        var py: array[L, float64]
        var k = 0
        while i < n:
          px[k] = x[i]; py[k] = y[i]; inc i; inc k
        let xv = loadv(cast[pointer](addr px[0]))
        let yv = loadv(cast[pointer](addr py[0]))
        let h = mulv(xv, yv)
        let r1 = fmaddv(xv, yv, subv(zero, h))
        let s2 = addv(s, h)
        let z = subv(s2, s)
        let r2 = addv(subv(s, subv(s2, z)), subv(h, z))
        s = s2
        e = addv(addv(e, r1), r2)
      var sl: array[L, float64]
      var el: array[L, float64]
      storev(cast[pointer](addr sl[0]), s)
      storev(cast[pointer](addr el[0]), e)
      var vals: array[L, float64]
      for j in 0 ..< L: vals[j] = sl[j] + el[j]
      let r = neumaierSum(vals.toOpenArray(0, L - 1))
      var maxLane = 0.0
      for v in sl: maxLane = max(maxLane, abs(v))
      (r, isFin(r) and maxLane <= LaneConcentrationFallback * abs(r))

  template defineDotK3V(name, zero, loadv, mulv, fmaddv, addv, subv, storev,
                        L: untyped) =
    ## K lanes of the DotK (ORO Alg. 5.3, K = 3) recurrence. Per lane the running
    ## dot sum `s` is advanced by `twoSum(s, h)` (`h = xᵢ·yᵢ`, `r₁ = fma` product
    ## error), and the per-term errors `(r₁, r₂)` are folded into a second-order
    ## compensated accumulator `(es, ec)` — an online `sum2` of the error stream,
    ## the K = 3 analog of `dot2`'s `e = (e + r₁) + r₂`. The tail is a zero-padded
    ## final vector step (zero lanes contribute `r₁ = r₂ = 0` and leave `es`,
    ## `ec` unchanged), keeping 3-fold precision without a scalar FMA. The K lane
    ## 3-fold estimates `sₖ + esₖ + ecₖ` are merged scalarly with `neumaierSum`
    ## (its `γ_K·Σ|xᵢyᵢ|` merge error is far below the K = 3 bound, so a 2-fold
    ## merge suffices). Same reliability contract as `dot2`.
    func name(x, y: openArray[float64]): (float64, bool) =
      if x.len == 0: return (0.0, true)
      var s = zero
      var es = zero
      var ec = zero
      let n = x.len
      var i = 0
      while i + L <= n:
        let xv = loadv(cast[pointer](unsafeAddr x[i]))
        let yv = loadv(cast[pointer](unsafeAddr y[i]))
        let h = mulv(xv, yv)
        let r1 = fmaddv(xv, yv, subv(zero, h)) # xᵢ·yᵢ − h (exact product error)
        let s2 = addv(s, h)
        let z = subv(s2, s)
        let r2 = addv(subv(s, subv(s2, z)), subv(h, z)) # twoSum(s, h) error
        s = s2
        # Fold (r1, r2) into (es, ec) as an online sum2 (two twoSum steps; the
        # second-level errors er1, er2 accumulate naively into ec).
        let es2 = addv(es, r1)
        let zr1 = subv(es2, es)
        let er1 = addv(subv(es, subv(es2, zr1)), subv(r1, zr1))
        es = es2
        let es3 = addv(es, r2)
        let zr2 = subv(es3, es)
        let er2 = addv(subv(es, subv(es3, zr2)), subv(r2, zr2))
        es = es3
        ec = addv(ec, addv(er1, er2))
        i += L
      if i < n:
        var px: array[L, float64]
        var py: array[L, float64]
        var k = 0
        while i < n:
          px[k] = x[i]; py[k] = y[i]; inc i; inc k
        let xv = loadv(cast[pointer](addr px[0]))
        let yv = loadv(cast[pointer](addr py[0]))
        let h = mulv(xv, yv)
        let r1 = fmaddv(xv, yv, subv(zero, h))
        let s2 = addv(s, h)
        let z = subv(s2, s)
        let r2 = addv(subv(s, subv(s2, z)), subv(h, z))
        s = s2
        let es2 = addv(es, r1)
        let zr1 = subv(es2, es)
        let er1 = addv(subv(es, subv(es2, zr1)), subv(r1, zr1))
        es = es2
        let es3 = addv(es, r2)
        let zr2 = subv(es3, es)
        let er2 = addv(subv(es, subv(es3, zr2)), subv(r2, zr2))
        es = es3
        ec = addv(ec, addv(er1, er2))
      var sl: array[L, float64]
      var esl: array[L, float64]
      var ecl: array[L, float64]
      storev(cast[pointer](addr sl[0]), s)
      storev(cast[pointer](addr esl[0]), es)
      storev(cast[pointer](addr ecl[0]), ec)
      var vals: array[L, float64]
      for j in 0 ..< L: vals[j] = sl[j] + esl[j] + ecl[j]
      let r = neumaierSum(vals.toOpenArray(0, L - 1))
      var maxLane = 0.0
      for v in sl: maxLane = max(maxLane, abs(v))
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
    defineNaiveDotV(naiveDotSimdAvx512, mm512_setzero_pd(), mm512_loadu_pd,
                    mm512_fmadd_pd, mm512_storeu_pd, 8)
    defineDot2V(dot2SimdAvx512, mm512_setzero_pd(), mm512_loadu_pd,
                mm512_mul_pd, mm512_fmadd_pd, mm512_add_pd, mm512_sub_pd,
                mm512_storeu_pd, 8)
    defineDotK3V(dotK3SimdAvx512, mm512_setzero_pd(), mm512_loadu_pd,
                 mm512_mul_pd, mm512_fmadd_pd, mm512_add_pd, mm512_sub_pd,
                 mm512_storeu_pd, 8)

  elif defined(avx2):
    import nimsimd/avx2
    import nimsimd/fma

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
    defineNaiveDotV(naiveDotSimdAvx2, mm256_setzero_pd(), mm256_loadu_pd,
                    mm256_fmadd_pd, mm256_storeu_pd, 4)
    defineDot2V(dot2SimdAvx2, mm256_setzero_pd(), mm256_loadu_pd,
                mm256_mul_pd, mm256_fmadd_pd, mm256_add_pd, mm256_sub_pd,
                mm256_storeu_pd, 4)
    defineDotK3V(dotK3SimdAvx2, mm256_setzero_pd(), mm256_loadu_pd,
                 mm256_mul_pd, mm256_fmadd_pd, mm256_add_pd, mm256_sub_pd,
                 mm256_storeu_pd, 4)

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

    func naiveDotSimdNeonF32(x, y: openArray[float32]): float32 =
      if x.len == 0: return 0.0'f32
      const L = 4
      var acc = vmovq_n_f32(0.0'f32)
      let n = x.len
      var i = 0
      while i + L <= n:
        acc = vfmaq_f32(acc, vld1q_f32(cast[pointer](unsafeAddr x[i])),
                        vld1q_f32(cast[pointer](unsafeAddr y[i]))) # acc + xᵢ·yᵢ
        i += L
      var lanes: array[L, float32]
      vst1q_f32(cast[pointer](addr lanes[0]), acc)
      result = 0.0'f32
      for j in 0 ..< L: result += lanes[j]
      while i < n: result += x[i] * y[i]; inc i

    func dot2SimdNeonF32(x, y: openArray[float32]): (float32, bool) =
      if x.len == 0: return (0.0'f32, true)
      const L = 4
      var s = vmovq_n_f32(0.0'f32)
      var e = vmovq_n_f32(0.0'f32)
      let n = x.len
      var i = 0
      while i + L <= n:
        let xv = vld1q_f32(cast[pointer](unsafeAddr x[i]))
        let yv = vld1q_f32(cast[pointer](unsafeAddr y[i]))
        let h = vmulq_f32(xv, yv)
        let r1 = vfmaq_f32(vsubq_f32(vmovq_n_f32(0.0'f32), h), xv,
            yv) # xᵢ·yᵢ − h (exact product error)
        let s2 = vaddq_f32(s, h)
        let z = vsubq_f32(s2, s)
        let r2 = vaddq_f32(vsubq_f32(s, vsubq_f32(s2, z)), vsubq_f32(h, z))
        s = s2
        e = vaddq_f32(vaddq_f32(e, r1), r2)
        i += L
      if i < n:
        var px: array[L, float32]
        var py: array[L, float32]
        var k = 0
        while i < n:
          px[k] = x[i]; py[k] = y[i]; inc i; inc k
        let xv = vld1q_f32(cast[pointer](addr px[0]))
        let yv = vld1q_f32(cast[pointer](addr py[0]))
        let h = vmulq_f32(xv, yv)
        let r1 = vfmaq_f32(vsubq_f32(vmovq_n_f32(0.0'f32), h), xv, yv)
        let s2 = vaddq_f32(s, h)
        let z = vsubq_f32(s2, s)
        let r2 = vaddq_f32(vsubq_f32(s, vsubq_f32(s2, z)), vsubq_f32(h, z))
        s = s2
        e = vaddq_f32(vaddq_f32(e, r1), r2)
      var sl: array[L, float32]
      var el: array[L, float32]
      vst1q_f32(cast[pointer](addr sl[0]), s)
      vst1q_f32(cast[pointer](addr el[0]), e)
      var vals: array[L, float32]
      for j in 0 ..< L: vals[j] = sl[j] + el[j]
      let r = neumaierSum(vals.toOpenArray(0, L - 1))
      var maxLane = 0.0'f32
      for v in sl: maxLane = max(maxLane, abs(v))
      (r, isFin(r) and maxLane <= float32(LaneConcentrationFallback) * abs(r))

    func dotK3SimdNeonF32(x, y: openArray[float32]): (float32, bool) =
      if x.len == 0: return (0.0'f32, true)
      const L = 4
      var s = vmovq_n_f32(0.0'f32)
      var es = vmovq_n_f32(0.0'f32)
      var ec = vmovq_n_f32(0.0'f32)
      let n = x.len
      var i = 0
      while i + L <= n:
        let xv = vld1q_f32(cast[pointer](unsafeAddr x[i]))
        let yv = vld1q_f32(cast[pointer](unsafeAddr y[i]))
        let h = vmulq_f32(xv, yv)
        let r1 = vfmaq_f32(vsubq_f32(vmovq_n_f32(0.0'f32), h), xv, yv)
        let s2 = vaddq_f32(s, h)
        let z = vsubq_f32(s2, s)
        let r2 = vaddq_f32(vsubq_f32(s, vsubq_f32(s2, z)), vsubq_f32(h, z))
        s = s2
        let es2 = vaddq_f32(es, r1)
        let zr1 = vsubq_f32(es2, es)
        let er1 = vaddq_f32(vsubq_f32(es, vsubq_f32(es2, zr1)), vsubq_f32(r1, zr1))
        es = es2
        let es3 = vaddq_f32(es, r2)
        let zr2 = vsubq_f32(es3, es)
        let er2 = vaddq_f32(vsubq_f32(es, vsubq_f32(es3, zr2)), vsubq_f32(r2, zr2))
        es = es3
        ec = vaddq_f32(ec, vaddq_f32(er1, er2))
        i += L
      if i < n:
        var px: array[L, float32]
        var py: array[L, float32]
        var k = 0
        while i < n:
          px[k] = x[i]; py[k] = y[i]; inc i; inc k
        let xv = vld1q_f32(cast[pointer](addr px[0]))
        let yv = vld1q_f32(cast[pointer](addr py[0]))
        let h = vmulq_f32(xv, yv)
        let r1 = vfmaq_f32(vsubq_f32(vmovq_n_f32(0.0'f32), h), xv, yv)
        let s2 = vaddq_f32(s, h)
        let z = vsubq_f32(s2, s)
        let r2 = vaddq_f32(vsubq_f32(s, vsubq_f32(s2, z)), vsubq_f32(h, z))
        s = s2
        let es2 = vaddq_f32(es, r1)
        let zr1 = vsubq_f32(es2, es)
        let er1 = vaddq_f32(vsubq_f32(es, vsubq_f32(es2, zr1)), vsubq_f32(r1, zr1))
        es = es2
        let es3 = vaddq_f32(es, r2)
        let zr2 = vsubq_f32(es3, es)
        let er2 = vaddq_f32(vsubq_f32(es, vsubq_f32(es3, zr2)), vsubq_f32(r2, zr2))
        es = es3
        ec = vaddq_f32(ec, vaddq_f32(er1, er2))
      var sl: array[L, float32]
      var esl: array[L, float32]
      var ecl: array[L, float32]
      vst1q_f32(cast[pointer](addr sl[0]), s)
      vst1q_f32(cast[pointer](addr esl[0]), es)
      vst1q_f32(cast[pointer](addr ecl[0]), ec)
      var vals: array[L, float32]
      for j in 0 ..< L: vals[j] = sl[j] + esl[j] + ecl[j]
      let r = neumaierSum(vals.toOpenArray(0, L - 1))
      var maxLane = 0.0'f32
      for v in sl: maxLane = max(maxLane, abs(v))
      (r, isFin(r) and maxLane <= float32(LaneConcentrationFallback) * abs(r))

  func naiveSumSimd*[T: SomeFloat](x: openArray[T]): T {.contractual.} =
    ensure:
      x.len != 0 or result == T(0)
    body:
      when T is float64 and defined(avx512):
        result = naiveSumSimdAvx512(x)
      elif T is float64 and defined(avx2):
        result = naiveSumSimdAvx2(x)
      elif T is float32 and defined(arm64):
        result = naiveSumSimdNeonF32(x)
      else:
        result = naiveSum(x)

  func pairwiseSimdRec[T: SomeFloat](x: openArray[T], lo, hi: int): T =
    let n = hi - lo + 1
    if n <= PairwiseThreshold:
      return naiveSumSimd(toOpenArray(x, lo, hi))
    let m = lo + n div 2
    result = pairwiseSimdRec(x, lo, m - 1) + pairwiseSimdRec(x, m, hi)

  func pairwiseSumSimd*[T: SomeFloat](x: openArray[T]): T {.contractual.} =
    ensure:
      x.len != 0 or result == T(0)
    body:
      when (T is float64 and (defined(avx2) or defined(avx512))) or
           (T is float32 and defined(arm64)):
        if x.len == 0: return T(0)
        result = pairwiseSimdRec(x, 0, x.high)
      else:
        result = pairwiseSum(x)

  func kahanSumSimd*[T: SomeFloat](x: openArray[T]): (T, bool) {.contractual.} =
    ensure:
      x.len != 0 or (result[0] == T(0) and result[1])
    body:
      when T is float64 and defined(avx512):
        result = kahanSumSimdAvx512(x)
      elif T is float64 and defined(avx2):
        result = kahanSumSimdAvx2(x)
      else:
        result = (kahanSum(x), true)

  func neumaierSumSimd*[T: SomeFloat](x: openArray[T]): (T,
      bool) {.contractual.} =
    ensure:
      x.len != 0 or (result[0] == T(0) and result[1])
    body:
      when T is float64 and defined(avx512):
        result = neumaierSumSimdAvx512(x)
      elif T is float64 and defined(avx2):
        result = neumaierSumSimdAvx2(x)
      else:
        result = (neumaierSum(x), true)

  func kleinSumSimd*[T: SomeFloat](x: openArray[T]): (T, bool) {.contractual.} =
    ensure:
      x.len != 0 or (result[0] == T(0) and result[1])
    body:
      when T is float64 and defined(avx512):
        result = kleinSumSimdAvx512(x)
      elif T is float64 and defined(avx2):
        result = kleinSumSimdAvx2(x)
      else:
        result = (kleinSum(x), true)

  func naiveDotSimd*[T: SomeFloat](x, y: openArray[T]): T {.contractual.} =
    ## SIMD naive dot (FMA reduce). Falls back to the scalar `naiveDot` off the
    ## SIMD ISAs. No reliability flag — the naive forward-error bound holds
    ## regardless of lane concentration.
    require:
      x.len == y.len
    ensure:
      x.len != 0 or result == T(0)
    body:
      when T is float64 and defined(avx512):
        result = naiveDotSimdAvx512(x, y)
      elif T is float64 and defined(avx2):
        result = naiveDotSimdAvx2(x, y)
      elif T is float32 and defined(arm64):
        result = naiveDotSimdNeonF32(x, y)
      else:
        result = naiveDot(x, y)

  func dot2Simd*[T: SomeFloat](x, y: openArray[T]): (T, bool) {.contractual.} =
    ## SIMD Dot2 (K = 2) kernel — `(result, reliable)`. The C ABI dispatches to
    ## this on a SIMD ISA and falls back to the scalar `dot2` when `reliable` is
    ## false (lane concentration or a non-finite result); off the SIMD ISAs it
    ## returns `(dot2(x, y), true)` so the caller's fallback path is a no-op.
    require:
      x.len == y.len
    ensure:
      x.len != 0 or (result[0] == T(0) and result[1])
    body:
      when T is float64 and defined(avx512):
        result = dot2SimdAvx512(x, y)
      elif T is float64 and defined(avx2):
        result = dot2SimdAvx2(x, y)
      elif T is float32 and defined(arm64):
        result = dot2SimdNeonF32(x, y)
      else:
        result = (dot2(x, y), true)

  func dotK3Simd*[T: SomeFloat](x, y: openArray[T]): (T, bool) {.contractual.} =
    ## SIMD DotK (K = 3) kernel — `(result, reliable)`. See `dot2Simd`; the
    ## scalar fallback is `dotK(x, y, 3)`.
    require:
      x.len == y.len
    ensure:
      x.len != 0 or (result[0] == T(0) and result[1])
    body:
      when T is float64 and defined(avx512):
        result = dotK3SimdAvx512(x, y)
      elif T is float64 and defined(avx2):
        result = dotK3SimdAvx2(x, y)
      elif T is float32 and defined(arm64):
        result = dotK3SimdNeonF32(x, y)
      else:
        result = (dotK(x, y, 3), true)
