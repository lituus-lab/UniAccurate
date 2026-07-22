# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
## Structural and property tests for the Shewchuk expansion primitives in
## `algorithms/expansions.nim`.
##
## The exact-real oracle is `shewchukSum` (correctly rounded = round of the
## exact real sum): every primitive here produces an expansion whose real sum is
## a known function of its inputs, so rounding that real sum once must match
## `shewchukSum` of the expansion's components. The representation invariant
## (non-overlapping, ascending magnitude, zero-eliminated where advertised) is
## checked directly. The decimal-EFT cross-check lives in `py/tests` (needs the
## C ABI); this tier stays in Nim.
import std/unittest
import std/math
import contracts
import UniAccurate

proc next(r: var uint64): uint64 =
  ## Deterministic xorshift64 so the sweeps reproduce (no RNG state).
  var x = r
  x = x xor (x shl 13)
  x = x xor (x shr 7)
  x = x xor (x shl 17)
  r = x
  x

proc randomF64(r: var uint64): float64 =
  ## Same EFT-exact range as `test_eft`: products stay normal, splits stay
  ## finite, so the expansion primitives' preconditions hold.
  let expField = uint64(next(r) mod 1022 + 512)
  var bits = expField shl 52
  bits = bits or (next(r) and 0x000F_FFFF_FFFF_FFFF'u64)
  cast[float64](bits)

proc randomF64Bounded(r: var uint64): float64 =
  ## Exp in [-200, 200]. Used where a product of two such values (or a component
  ## times a scalar) is formed: the product stays in [2^-400, 2^400], and scaling
  ## by another stays in [2^-600, 2^600], all normal — so `twoProduct`'s
  ## no-total-underflow precondition holds for `scaleExpansionZeroElim`.
  let expField = uint64(next(r) mod 400 + 823)
  var bits = expField shl 52
  bits = bits or (next(r) and 0x000F_FFFF_FFFF_FFFF'u64)
  cast[float64](bits)

proc p2(e: int): float64 =
  ## Exact `2^e` for `e` in [-1074, 1023] (covers subnormals and normals).
  if e >= -1022:
    cast[float64](uint64(e + 1023) shl 52)
  else:
    cast[float64](1'u64 shl (e + 1074))

proc zeroFree(e: openArray[float64]): bool =
  ## True when no component is zero (the ZeroElim invariant).
  for v in e:
    if v == 0.0:
      return false
  result = true

proc nonOverlappingAll(e: openArray[float64]): bool =
  ## True when every pair of components is non-overlapping.
  for i in 0 ..< e.len:
    for j in (i + 1) ..< e.len:
      if not isNonOverlapping(e[i], e[j]):
        return false
  result = true

proc ascendingMag(e: openArray[float64]): bool =
  ## True when components are in nondecreasing order of magnitude.
  for i in 1 ..< e.len:
    if abs(e[i - 1]) > abs(e[i]):
      return false
  result = true

proc validZeroElim(e: openArray[float64]): bool =
  ## A zero-eliminated expansion: a single component (the value, possibly
  ## `0.0` when the sum is exactly zero), or several non-overlapping components
  ## ascending in magnitude with no zeros.
  if e.len <= 1:
    return true
  for v in e:
    if v == 0.0:
      return false
  result = nonOverlappingAll(e) and ascendingMag(e)

proc randExpansion(r: var uint64): seq[float64] =
  ## Build a valid (non-overlapping, ascending, zero-eliminated) expansion by
  ## merging `twoProduct` EFT outputs. Each `(p, e)` pair is a non-overlapping
  ## ascending 2-expansion — `|e| <= ulp(p) < |p|`, so `[e, p]` is smallest-first
  ## — and `fastExpansionSumZeroElim` preserves the invariant across merges.
  var acc: seq[float64] = @[]
  let n = int(next(r) mod 3) + 2
  for _ in 0 ..< n:
    let (p, e) = twoProduct(randomF64Bounded(r), randomF64Bounded(r))
    let pair: seq[float64] = if e != 0.0: @[e, p] else: @[p]
    acc = if acc.len == 0: pair else: fastExpansionSumZeroElim(acc, pair)
  result = acc

let emptyF64: seq[float64] = @[]

suite "twoSquare":
  test "exact square carries no error":
    let (p, e) = twoSquare(2.0)
    check p == 4.0
    check e == 0.0

  test "matches twoProduct(a, a) over the normal range":
    var r = 0x243F_6A88_85A3_08D3'u64
    for _ in 1 ..< 5000:
      let a = randomF64(r)
      let (p1, e1) = twoSquare(a)
      let (p2, e2) = twoProduct(a, a)
      check p1 == p2
      check e1 == e2

  test "error bounded by ulp over a sweep":
    # Non-overlap of (p, e) is a Dekker guarantee away from the subnormal
    # boundary, but `a*a` can land p where ulp(p) = 2^-1074 and the error term
    # shares that lowest bit, so isNonOverlapping does not hold universally
    # (twoProduct(a, a) hits the same edge). The EFT bound |e| <= ulp(p) does,
    # and the exact identity is covered by the twoProduct equivalence above.
    var r = 0x9E37_79B9_7F4A_7C15'u64
    for _ in 1 ..< 5000:
      let a = randomF64(r)
      let (p, e) = twoSquare(a)
      check abs(e) <= ulp(p)

suite "twoDiffTail":
  test "matches the tail of twoDiff over a sweep":
    var r = 0xCBBB_9B44_6920_9AC4'u64
    for _ in 1 ..< 5000:
      let a = randomF64(r)
      let b = randomF64(r)
      let s = a - b
      check twoDiffTail(a, b, s) == twoDiff(a, b)[1]

  test "exact diff carries no tail":
    check twoDiffTail(5.0, 2.0, 3.0) == 0.0

suite "zeroElim":
  test "drops zeros, preserves order":
    check zeroElim(@[0.0, 1.0, 0.0, 2.0, 0.0]) == @[1.0, 2.0]
    check zeroElim(@[0.0, 0.0]).len == 0
    check zeroElim(@[3.0]) == @[3.0]

suite "growExpansion":
  test "real sum is b + sum(e), exactly":
    var r = 0xB7E1_5162_8AED_2A6B'u64
    for _ in 1 ..< 3000:
      let e = randExpansion(r)
      let b = randomF64(r)
      let h = growExpansion(e, b)
      check shewchukSum(h) == shewchukSum(e & @[b])
      check nonOverlappingAll(h)

  test "empty expansion grows to the single addend":
    check growExpansion(emptyF64, 7.0) == @[7.0]

suite "fastExpansionSumZeroElim":
  test "real sum is sum(f) + sum(g), exactly":
    var r = 0xD1B5_4A91_1A6E_F3C0'u64
    for _ in 1 ..< 3000:
      let f = randExpansion(r)
      let g = randExpansion(r)
      let h = fastExpansionSumZeroElim(f, g)
      check shewchukSum(h) == shewchukSum(f & g)
      check validZeroElim(h)

  test "empty operands pass through zero-eliminated":
    let e = @[p2(-53), 1.0]
    check fastExpansionSumZeroElim(emptyF64, e) == e
    check fastExpansionSumZeroElim(e, emptyF64) == e
    check fastExpansionSumZeroElim(emptyF64, emptyF64).len == 0

suite "scaleExpansionZeroElim":
  test "real sum is b * sum(e), exactly":
    var r = 0x2545_F491_4F6C_DD1D'u64
    for _ in 1 ..< 3000:
      let e = randExpansion(r)
      let b = randomF64Bounded(r)
      let h = scaleExpansionZeroElim(e, b)
      # Independent exact oracle: scale each component exactly with twoProduct
      # and merge the 2-expansions with fastExpansionSumZeroElim (both tested
      # independently above). The real sum of that expansion is b * sum(e),
      # exactly as h represents it, so the two correctly-rounded reads match.
      var acc: seq[float64] = @[]
      for v in e:
        let (p, pt) = twoProduct(v, b)
        let pair: seq[float64] = if pt != 0.0: @[pt, p] else: @[p]
        acc = if acc.len == 0: pair else: fastExpansionSumZeroElim(acc, pair)
      check shewchukSum(h) == shewchukSum(acc)
      check validZeroElim(h)

  test "empty expansion scales to empty":
    check scaleExpansionZeroElim(emptyF64, 3.0).len == 0

  test "scaling by one preserves the real value":
    let e = @[p2(-53), 1.0]
    let h = scaleExpansionZeroElim(e, 1.0)
    check shewchukSum(h) == shewchukSum(e)

suite "estimate":
  test "empty is zero, single is the element":
    check estimate(emptyF64) == 0.0
    check estimate(@[3.5]) == 3.5

  test "two-expansion is correctly rounded (one rounding)":
    var r = 0x9E37_79B9_7F4A_7C15'u64
    for _ in 1 ..< 3000:
      let (p, e) = twoProduct(randomF64(r), randomF64(r))
      let exp: seq[float64] = if e != 0.0: @[e, p] else: @[p]
      check estimate(exp) == shewchukSum(exp)

  test "within two ulp of the correctly-rounded value over a sweep":
    # estimate is a naive smallest-first sum: within one ulp of the true value
    # (Shewchuk 1997). shewchukSum is within half an ulp, so the two agree to
    # within 1.5 ulp; the 2-ulp bound is the safe, documented slack. Skip the
    # Inf/NaN/zero corners where ulp is not a meaningful scale.
    var r = 0x2545_F491_4F6C_DD1D'u64
    for _ in 1 ..< 3000:
      let e = randExpansion(r)
      let est = estimate(e)
      let cr = shewchukSum(e)
      if classify(cr) in {fcNormal, fcSubnormal} and
         classify(est) in {fcNormal, fcSubnormal, fcZero, fcNegZero}:
        check abs(est - cr) <= 2 * ulp(cr)
