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

  when defined(arm64):
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

else:
  echo "test_simd: skipped (build without -d:simd)"
