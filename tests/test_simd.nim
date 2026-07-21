# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
## SIMD-layer parity and safety tests. Mirrors `test_property.nim`'s xorshift64
## RNG and generators. Skipped (a message, no suites) on a build without
## `-d:simd`.

when defined(simd):
  import std/unittest
  import std/math
  import UniAccurate

  proc next(r: var uint64): uint64 =
    var x = r
    x = x xor (x shl 13)
    x = x xor (x shr 7)
    x = x xor (x shl 17)
    r = x
    x

  proc randomF64(r: var uint64): float64 =
    let expField = uint64(next(r) mod 1022 + 512)
    var bits = expField shl 52
    bits = bits or (next(r) and 0x000F_FFFF_FFFF_FFFF'u64)
    cast[float64](bits)

  proc randomDotF64(r: var uint64): float64 =
    ## Finite float64 for dot suites: exponent halved (unbiased [-255, 244]) so
    ## products stay finite (unbiased <= 488) and the running sum of up to 1e5
    ## of them (unbiased <= 505) never overflows. The sum generators double the
    ## exponent on the product, so they are unsafe for dot at scale.
    let expField = uint64(next(r) mod 500 + 768)
    var bits = expField shl 52
    bits = bits or (next(r) and 0x000F_FFFF_FFFF_FFFF'u64)
    cast[float64](bits)

  proc randomSignedBigF64(r: var uint64): float64 =
    let expField = uint64(next(r) mod 7 + 2040)
    var bits = expField shl 52
    bits = bits or (next(r) and 0x000F_FFFF_FFFF_FFFF'u64)
    if (next(r) and 1) == 1:
      bits = bits or (1'u64 shl 63)
    cast[float64](bits)

  proc randomF32(r: var uint64): float32 =
    let expField = uint32(next(r) mod 126 + 65)
    var bits = expField shl 23
    bits = bits or (uint32(next(r)) and 0x007F_FFFF'u32)
    cast[float32](bits)

  proc randomDotF32(r: var uint64): float32 =
    ## Finite float32 for dot suites: exponent halved (unbiased [-25, 34]) so
    ## products stay finite (unbiased <= 68) and the running sum of up to 1e4 of
    ## them (unbiased <= 82) never overflows the float32 range.
    let expField = uint32(next(r) mod 60 + 98)
    var bits = expField shl 23
    bits = bits or (uint32(next(r)) and 0x007F_FFFF'u32)
    cast[float32](bits)

  proc randomIntF64(r: var uint64): float64 =
    let v = int64(next(r) mod 2_000_001) - 1_000_000
    float64(v)

  const Seed = 0x9E37_79B9_7F4A_7C15'u64

  when simdF64Enabled:
    suite "SIMD == scalar bit-exact on integer data (float64)":
      var r = Seed
      test "naive/pairwise/compensated SIMD match the scalar sums":
        for n in [1, 2, 3, 10, 100, 500, 1000]:
          var x = newSeq[float64](n)
          for i in 0 ..< n:
            x[i] = randomIntF64(r)
          check naiveSumSimd(x) == naiveSum(x)
          check pairwiseSumSimd(x) == pairwiseSum(x)
          let (rk, relk) = kahanSumSimd(x)
          check relk and rk == kahanSum(x)
          let (rn, reln) = neumaierSumSimd(x)
          check reln and rn == neumaierSum(x)
          let (rl, rell) = kleinSumSimd(x)
          check rell and rl == kleinSum(x)

    suite "compensated SIMD reliable matches scalar, finite never NaN (float64)":
      var r = Seed xor 1
      test "bounded random float64 up to n=100000":
        template checkComp(simdP, scalarP: untyped) =
          let (res, rel) = simdP(x)
          check classify(res) != fcNan
          if rel:
            let sc = scalarP(x)
            check abs(res - sc) <= 1e-9 * max(1.0, abs(sc))
        for n in [1, 10, 100, 1000, 10000, 100000]:
          var x = newSeq[float64](n)
          for i in 0 ..< n:
            x[i] = randomF64(r)
          checkComp(kahanSumSimd, kahanSum)
          checkComp(neumaierSumSimd, neumaierSum)
          checkComp(kleinSumSimd, kleinSum)

    suite "SIMD finite never NaN (overflow-prone float64)":
      var r = Seed xor 2
      test "naive and compensated, signed big float64 up to n=10000":
        template checkFinite(simdP: untyped) =
          let (res, _) = simdP(x)
          check classify(res) != fcNan
        for n in [1, 10, 100, 1000, 10000]:
          var x = newSeq[float64](n)
          for i in 0 ..< n:
            x[i] = randomSignedBigF64(r)
          check classify(naiveSumSimd(x)) != fcNan
          checkFinite(kahanSumSimd)
          checkFinite(neumaierSumSimd)
          checkFinite(kleinSumSimd)

    suite "SIMD dot == scalar bit-exact on integer data (float64)":
      var r = Seed xor 8
      test "naive/dot2/dotK3 SIMD match the scalar dots":
        # Integer products and their running sum stay below 2^53, so the dot is
        # exact at every precision: the FMA reduce, the Dot2/DotK3 recurrence,
        # and the scalar algorithms all hit the exact integer bit-for-bit,
        # regardless of the reliability flag (cancellation can flip it, but the
        # value is still exact).
        for n in [1, 2, 3, 10, 100, 500, 1000]:
          var x = newSeq[float64](n)
          var y = newSeq[float64](n)
          for i in 0 ..< n:
            x[i] = randomIntF64(r)
            y[i] = randomIntF64(r)
          check naiveDotSimd(x, y) == naiveDot(x, y)
          let (rd2, _) = dot2Simd(x, y)
          check rd2 == dot2(x, y)
          let (rd3, _) = dotK3Simd(x, y)
          check rd3 == dotK(x, y, 3)

    suite "SIMD dot reliable matches scalar, finite never NaN (float64)":
      var r = Seed xor 9
      test "bounded random float64 up to n=100000":
        template checkDot2 =
          let (res, rel) = dot2Simd(x, y)
          check classify(res) != fcNan
          if rel:
            let sc = dot2(x, y)
            check abs(res - sc) <= 1e-9 * max(1.0, abs(sc))
        template checkDotK3 =
          let (res, rel) = dotK3Simd(x, y)
          check classify(res) != fcNan
          if rel:
            let sc = dotK(x, y, 3)
            check abs(res - sc) <= 1e-9 * max(1.0, abs(sc))
        for n in [1, 10, 100, 1000, 10000, 100000]:
          var x = newSeq[float64](n)
          var y = newSeq[float64](n)
          for i in 0 ..< n:
            x[i] = randomDotF64(r)
            y[i] = randomDotF64(r)
          let ns = naiveDot(x, y)
          check classify(naiveDotSimd(x, y)) != fcNan
          check abs(naiveDotSimd(x, y) - ns) <= 1e-9 * max(1.0, abs(ns))
          checkDot2()
          checkDotK3()

  when simdF32Enabled:
    suite "float32 NEON path":
      var r = Seed xor 3
      test "naive/pairwise close to scalar, compensated bit-exact, never NaN":
        for n in [1, 2, 10, 100, 1000, 10000]:
          var x = newSeq[float32](n)
          for i in 0 ..< n:
            x[i] = randomF32(r)
          let ns = naiveSum(x)
          check classify(naiveSumSimd(x)) != fcNan
          check abs(naiveSumSimd(x) - ns) <= 1e-2'f32 * max(1.0'f32, abs(ns))
          let ps = pairwiseSum(x)
          check classify(pairwiseSumSimd(x)) != fcNan
          check abs(pairwiseSumSimd(x) - ps) <= 1e-2'f32 * max(1.0'f32, abs(ps))
          # f32 compensated falls back to the scalar algorithm on arm64.
          check kahanSumSimd(x) == (kahanSum(x), true)
          check neumaierSumSimd(x) == (neumaierSum(x), true)
          check kleinSumSimd(x) == (kleinSum(x), true)

    suite "float32 NEON dot path":
      var r = Seed xor 10
      test "naive close to scalar, dot2/dotK3 close when reliable, never NaN":
        # The f32 NEON dot kernels (FMA reduce, Dot2/DotK3 recurrence) are real
        # kernels, not scalar fallbacks. naive is an FMA dot — within the naive
        # bound of scalar naiveDot. dot2/dotK3 return (res, reliable); when
        # reliable they track the scalar compensated dot, and the C ABI falls
        # back to the scalar body otherwise (exercised in ctestSimd).
        for n in [1, 2, 10, 100, 1000, 10000]:
          var x = newSeq[float32](n)
          var y = newSeq[float32](n)
          for i in 0 ..< n:
            x[i] = randomDotF32(r)
            y[i] = randomDotF32(r)
          let ns = naiveDot(x, y)
          check classify(naiveDotSimd(x, y)) != fcNan
          check abs(naiveDotSimd(x, y) - ns) <= 1e-2'f32 * max(1.0'f32, abs(ns))
          let (rd2, rel2) = dot2Simd(x, y)
          check classify(rd2) != fcNan
          if rel2:
            let sc = dot2(x, y)
            check abs(rd2 - sc) <= 1e-2'f32 * max(1.0'f32, abs(sc))
          let (rd3, rel3) = dotK3Simd(x, y)
          check classify(rd3) != fcNan
          if rel3:
            let sc = dotK(x, y, 3)
            check abs(rd3 - sc) <= 1e-2'f32 * max(1.0'f32, abs(sc))

else:
  echo "test_simd: skipped (build without -d:simd)"
