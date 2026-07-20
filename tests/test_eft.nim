import std/unittest
import std/math
import UniAccurate

# Deterministic xorshift64 so the sweeps are reproducible (no RNG state).
proc next(r: var uint64): uint64 =
  var x = r
  x = x xor (x shl 13)
  x = x xor (x shr 7)
  x = x xor (x shl 17)
  r = x
  x

proc randomF64(r: var uint64): float64 =
  # Finite, well-scaled float64 kept in the EFT-exact range: exponent in
  # [1, 500] so that `a*b` (exp <= 1000) and the Veltkamp split `2^27 * a`
  # (exp <= 527) both stay finite and normal. Above exp ~2019 the split
  # overflows (a documented Dekker limitation, not an algorithm bug), so we
  # stay clear of it here and exercise the exact regime the identity holds in.
  let expField = uint64(next(r) mod 500 + 1)
  var bits = expField shl 52
  bits = bits or (next(r) and 0x000F_FFFF_FFFF_FFFF'u64) # random significand
  cast[float64](bits)

suite "ulp":
  test "known binades (float64)":
    check ulp(1.0) == 2.220446049250313e-16 # 2^-52
    check ulp(1.5) == 2.220446049250313e-16 # same binade as 1.0
    check ulp(2.0) == 4.440892098500626e-16 # 2^-51, spacing doubles at the binade
    check ulp(4.0) == 8.881784197001252e-16
    check ulp(0.0) == 5e-324 # smallest subnormal
    check ulp(-3.0) == ulp(3.0) # sign-agnostic

  test "non-finite":
    check classify(ulp(Inf)) == fcNan
    check classify(ulp(-Inf)) == fcNan
    check classify(ulp(NaN)) == fcNan

  test "float32":
    check ulp(1.0'f32) == 1.1920929e-7'f32 # 2^-23
    check ulp(0.0'f32) == 1.0e-45'f32

suite "split":
  test "reconstructs exactly over the normal range":
    var r = 0x9E37_79B9_7F4A_7C15'u64
    for _ in 1 .. 5000:
      let a = randomF64(r)
      let (hi, lo) = split(a)
      check hi + lo == a
      check isNonOverlapping(hi, lo)

  test "powers of two split to (p, 0)":
    let (hi, lo) = split(8.0)
    check hi == 8.0
    check lo == 0.0

suite "twoSum":
  test "representable sum recovers e = 0":
    let (s, e) = twoSum(1.0, 2.0)
    check s == 3.0
    check e == 0.0

  test "lost addend is recovered in e":
    let halfUlp = pow(2.0, -53) # half-ulp of 1.0, ties-to-even → s = 1.0
    let (s, e) = twoSum(1.0, halfUlp)
    check s == 1.0
    check e == halfUlp

  test "non-overlap and bound over a sweep":
    var r = 0x2545_F491_4F6C_DD1D'u64
    for _ in 1 .. 5000:
      let a = randomF64(r)
      let b = randomF64(r)
      let (s, e) = twoSum(a, b)
      check isNonOverlapping(s, e)
      check abs(e) <= ulp(s)

  test "same (s, e) as twoSumFast":
    var r = 0x243F_6A88_85A3_08D3'u64
    for _ in 1 .. 5000:
      let a = randomF64(r)
      let b = randomF64(r)
      let (s1, e1) = twoSum(a, b)
      let (s2, e2) = twoSumFast(a, b)
      check s1 == s2 # value-identical (signed zero tolerated)
      check e1 == e2

  test "matches fastTwoSum when |a| >= |b|":
    var r = 0xB7E1_5162_8AED_2A6B'u64
    for _ in 1 .. 5000:
      let ra = abs(randomF64(r))
      let rb = abs(randomF64(r))
      let (a, b) = if ra >= rb: (ra, rb) else: (rb, ra) # enforce |a| >= |b|
      let (s1, e1) = twoSum(a, b)
      let (s2, e2) = fastTwoSum(a, b)
      check s1 == s2
      check e1 == e2

suite "twoDiff":
  test "representable diff recovers e = 0":
    let (s, e) = twoDiff(5.0, 2.0)
    check s == 3.0
    check e == 0.0

  test "non-overlap over a sweep":
    var r = 0xCBBB_9B44_6920_9AC4'u64
    for _ in 1 .. 5000:
      let a = randomF64(r)
      let b = randomF64(r)
      let (s, e) = twoDiff(a, b)
      check isNonOverlapping(s, e)
      check abs(e) <= ulp(s)

suite "twoProduct":
  test "integer product recovers e = 0":
    let (p, e) = twoProduct(3.0, 5.0)
    check p == 15.0
    check e == 0.0

  test "non-overlap and bound over a sweep":
    var r = 0x9E37_79B9_7F4A_7C15'u64
    for _ in 1 .. 5000:
      let a = randomF64(r)
      let b = randomF64(r)
      let (p, e) = twoProduct(a, b)
      check isNonOverlapping(p, e)
      check abs(e) <= ulp(p)

  test "matches twoProductFMA (Dekker path agrees with the FMA path)":
    var r = 0xD1B5_4A91_1A6E_F3C0'u64
    for _ in 1 .. 5000:
      let a = randomF64(r)
      let b = randomF64(r)
      let (p1, e1) = twoProduct(a, b)
      let (p2, e2) = twoProductFMA(a, b)
      check p1 == p2 # value-identical (signed zero tolerated)
      check e1 == e2

suite "isNonOverlapping":
  test "disjoint bits":
    check isNonOverlapping(1.0, pow(2.0, -53)) # twoSum output of (1.0, 2^-53)
    check isNonOverlapping(2.0, 1.0) # adjacent powers of two, no shared bit
    check isNonOverlapping(1.0, 0.0)

  test "overlapping":
    check not isNonOverlapping(3.0, 1.0) # 3 = 11b, 1 = 1b share the 1-bit
    check not isNonOverlapping(NaN, 1.0)
    check not isNonOverlapping(Inf, 1.0)
